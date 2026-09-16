"""Dense embedding service built on fastembed (ONNX, no torch required).

The service lazily loads the ONNX model, caches the computed corpus
embeddings on disk, and exposes an encode() API used by the hybrid
search index. It also hosts a cross-encoder reranker for the rerank
stage.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
EMBED_CACHE = Path("data/policy/embeddings.npy")

# 384-dim for bge-small-en-v1.5
EMBED_DIM = 384


class EmbeddingService:
    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        cache_path: Path = EMBED_CACHE,
        device: str = "cpu",
        force_rebuild: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_path = cache_path
        self.device = device
        self.force_rebuild = force_rebuild
        self._model = None
        self._reranker = None

    # ------------------------------------------------------------------
    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        return self._model

    def _get_reranker(self):
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(RERANK_MODEL)
        return self._reranker

    # ------------------------------------------------------------------
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into a (n, dim) float32 matrix."""
        model = self._get_model()
        # FastEmbed returns numpy arrays already.
        vecs = list(model.embed(texts, batch_size=batch_size))
        return np.asarray(vecs, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    # ------------------------------------------------------------------
    def compute_corpus_embeddings(self, texts: List[str], force_rebuild: bool = False):
        """Compute + cache corpus embeddings keyed by whole corpus.

        Cache validity depends only on the text content hash, so we
        store a sibling .json with the hash to know whether to rebuild.
        """
        import hashlib

        digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        meta_path = self.cache_path.with_suffix(".json")

        if (
            not force_rebuild and not self.force_rebuild
            and self.cache_path.exists()
            and meta_path.exists()
        ):
            cfg = json_load(meta_path)
            if cfg.get("digest") == digest and cfg.get("dim") == EMBED_DIM:
                emb = np.load(self.cache_path)
                logger.info("Loaded cached embeddings %s", self.cache_path)
                if emb.shape[0] != len(texts):
                    logger.warning("Embedding count mismatch, rebuilding.")
                else:
                    return emb

        emb = self.encode(texts)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path, emb)
        json_dump({"digest": digest, "dim": EMBED_DIM, "count": len(texts)}, meta_path)
        logger.info("Computed + cached %d embeddings (dim=%d)", len(texts), EMBED_DIM)
        return emb

    # ------------------------------------------------------------------
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[float]:
        """Cross-encoder scores for (query, doc) pairs.

        fastembed returns scores index-aligned with the input documents;
        we request all of them and return the list in the same order.
        """
        if not documents:
            return []
        rr = self._get_reranker()
        k = len(documents) if top_k is None else min(top_k, len(documents))
        scores = list(rr.rerank(query=query, documents=documents, top_k=k))
        return scores


def json_load(path: Path):
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def json_dump(obj, path: Path):
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def cosine_similarity_matrix(qv: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """(nq, dim) x (nc, dim) -> (nq, nc) cosine matrix."""
    qn = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
    cn = cv / (np.linalg.norm(cv, axis=1, keepdims=True) + 1e-12)
    return qn @ cn.T