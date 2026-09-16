"""Shared lightweight text-matching utilities.

Used to (a) select the policy chunk that best supports a finding and
(b) validate that a citation is supported by its chunk. Keeping both
sides on the same token-overlap metric makes the citation selection and
the validation consistent.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Set

STOPWORDS: Set[str] = set(
    "a an the and or of in to for with on at by from is are was were be been has have had it its "
    "this that these those said as not no will would can could should may might must "
    "policy finding assessment treatment type covered coverage applicable dimension clause "
    "applies apply condition conditions case claim".split()
)


def significant_tokens(text: str) -> Set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def overlap_ratio(source: str, target: str) -> float:
    st = significant_tokens(source)
    if not st:
        return 0.0
    tt = significant_tokens(target)
    return len(st & tt) / len(st)


def best_matching(
    text: str,
    candidates: Iterable[tuple],
    min_overlap: float = 0.30,
    max_results: int = 2,
) -> List:
    """Return up to max_results (chunk_id, score) ranked by overlap.

    Each candidate is expected to be a (chunk_id, chunk_text) tuple.
    Only candidates with overlap >= min_overlap are returned.
    """
    scored = []
    for cid, ctext in candidates:
        ov = overlap_ratio(text, ctext)
        if ov >= min_overlap:
            scored.append((cid, ov))
    scored.sort(key=lambda x: -x[1])
    return scored[:max_results]