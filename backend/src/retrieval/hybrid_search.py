"""Hybrid retrieval: dense (BAAI/bge-small-en-v1.5) + sparse (BM25) + RRF fusion + cross-encoder reranking.

The pipeline:
  1. Build two independent indices from the policy chunk corpus (dense numpy, sparse bm25).
  2. For a given query, compute dense scores (cosine similarity) and sparse scores (BM25).
  3. Fuse both result lists using Reciprocal Rank Fusion (RRF, K=60).
  4. Re-score the fused candidates with a cross-encoder reranker.
  5. Return the top-k PolicyEvidence objects with full scoring metadata.
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.models.schemas import PolicyChunk, PolicyEvidence
from src.retrieval.embeddings import EmbeddingService, cosine_similarity_matrix

logger = logging.getLogger(__name__)

# RRF constant
RRF_K = 60


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


class HybridSearch:
    def __init__(
        self,
        chunks: List[PolicyChunk],
        emb_service: EmbeddingService,
        rerank_top_n: int = 12,
        final_top_k: int = 8,
        force_rebuild_embeddings: bool = False,
    ) -> None:
        self.chunks = chunks
        self.emb_service = emb_service
        self.rerank_top_n = rerank_top_n
        self.final_top_k = final_top_k

        self._texts: List[str] = [c.text for c in chunks]
        self._chunk_ids: List[str] = [c.chunk_id for c in chunks]

        t0 = time.time()
        self._build_dense_index(force_rebuild_embeddings)
        self._build_sparse_index()
        self._build_time_ms = (time.time() - t0) * 1000
        # Cache of search results keyed by (normalized_query, top_k, with_rerank).
        # Dimension queries repeat heavily across cases, so this is a large win.
        self._search_cache: Dict[tuple, List[PolicyEvidence]] = {}
        logger.info("HybridSearch built in %.0fms (%d chunks)", self._build_time_ms, len(chunks))

    # ------------------------------------------------------------------
    def _build_dense_index(self, force_rebuild: bool):
        self._dense_vecs = self.emb_service.compute_corpus_embeddings(
            self._texts, force_rebuild=force_rebuild,
        )

    def _build_sparse_index(self):
        tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    def dense_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        qv = self.emb_service.encode_one(query)
        sims = cosine_similarity_matrix(qv.reshape(1, -1), self._dense_vecs)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top_idx]

    def sparse_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------
    @staticmethod
    def reciprocal_rank_fusion(
        lists: List[List[Tuple[int, float]]],
        k: int = RRF_K,
    ) -> List[Tuple[int, float]]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
        score_map: Dict[int, float] = defaultdict(float)
        for ranked in lists:
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                score_map[doc_id] += 1.0 / (k + rank)
        fused = sorted(score_map.items(), key=lambda x: -x[1])
        return fused

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        with_rerank: bool = True,
    ) -> List[PolicyEvidence]:
        top_k = top_k or self.final_top_k
        cache_key = (query.strip().lower(), top_k, with_rerank)
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return [ev.model_copy(deep=True) for ev in cached]

        dense = self.dense_search(query, top_k=min(self.rerank_top_n, len(self.chunks)))
        sparse = self.sparse_search(query, top_k=min(self.rerank_top_n, len(self.chunks)))
        fused = self.reciprocal_rank_fusion([dense, sparse], k=RRF_K)[: self.rerank_top_n]

        # dense_score and sparse_score maps for metadata
        dense_map = {i: s for i, s in dense}
        sparse_map = {i: s for i, s in sparse}
        fused_map = {i: rrf for i, rrf in fused}

        # Extract docs for reranking
        cand_idx = [i for i, _ in fused]
        cand_texts = [self._texts[i] for i in cand_idx]
        cand_chunk_ids = [self._chunk_ids[i] for i in cand_idx]

        # Reranking
        if with_rerank and self.emb_service.rerank_enabled and cand_texts:
            rerank_scores = self.emb_service.rerank(query, cand_texts, top_k=len(cand_texts))
            # sort by rerank score descending
            reranked_order = np.argsort(rerank_scores)[::-1]
        else:
            rerank_scores = [0.0] * len(cand_texts)
            reranked_order = np.arange(len(cand_texts))

        evidence: List[PolicyEvidence] = []
        for new_rank, orig_list_pos in enumerate(reranked_order[:top_k], start=1):
            chunk_idx = cand_idx[orig_list_pos]
            chunk = self.chunks[chunk_idx]
            evidence.append(
                PolicyEvidence(
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    section=chunk.section,
                    heading=chunk.heading,
                    text=chunk.text,
                    score=float(rerank_scores[orig_list_pos]),
                    dense_score=float(dense_map.get(chunk_idx, 0.0)),
                    sparse_score=float(sparse_map.get(chunk_idx, 0.0)),
                    rerank_score=float(rerank_scores[orig_list_pos]),
                    rank=new_rank,
                )
            )

        self._search_cache[cache_key] = [ev.model_copy(deep=True) for ev in evidence]
        return evidence

    def multi_query_search(
        self,
        queries: List[str],
        top_k: int = 8,
        with_rerank: bool = True,
    ) -> List[PolicyEvidence]:
        """Run search for each query, pool results, deduplicate, and
        return top-k merged by average rerank score.
        """
        all_evidence: Dict[str, PolicyEvidence] = {}
        for q in queries:
            for ev in self.search(q, top_k=top_k, with_rerank=with_rerank):
                if ev.chunk_id not in all_evidence:
                    all_evidence[ev.chunk_id] = ev
                else:
                    existing = all_evidence[ev.chunk_id]
                    # average rerank scores for same chunk across queries
                    avg = (existing.rerank_score + ev.rerank_score) / 2
                    existing.rerank_score = avg
                    existing.score = avg
        pooled = list(all_evidence.values())
        pooled.sort(key=lambda e: -e.rerank_score)
        return pooled[:top_k]