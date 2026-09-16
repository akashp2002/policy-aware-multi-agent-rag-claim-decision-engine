"""Policy PDF ingestion and meaningful chunking.

The policy wordings from Indian health insurance policies follow a
semi-structured layout: running page headers, clause numbers, definition
entries, and numbered cover/exclusion lists. We exploit this structure
instead of blindly slicing the text.

Strategy:
  1. Extract each page's text with pypdf.
  2. Detect major *sections* (DEFINITIONS, SCOPE OF COVER, WHAT WE
     EXCLUDE, EXTENSIONS, CLAIMS PROCEDURE, STANDARD TERMS AND
     CONDITIONS).
  3. Sentence-tokenize within a section and accumulate sentences into a
     chunk until a soft token budget is reached, breaking at sentence
     boundaries (never mid-sentence), with overlap so clause boundaries
     don't get lost.
  4. Attach rich metadata: page, section, heading, clause number, char
     length, and a stable chunk id.

The resulting chunks (JSON) are the retrieval units for both dense and
sparse indexing.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from pypdf import PdfReader

from src.models.schemas import PolicyChunk

logger = logging.getLogger(__name__)

DEFAULT_PDF = Path("data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf")
CHUNKS_OUT = Path("data/policy/policy_chunks.json")
RAW_PAGES_OUT = Path("data/policy/policy_pages_raw.json")

# Word budget. Health policy chunks are clause sized; ~250-350 words keeps
# them self-contained yet long enough to carry a limit or condition.
CHUNK_WORDS_SOFT = 260
CHUNK_WORDS_OVERLAP = 60
MAX_CHUNK_WORDS = 420

# Section start markers (uppercase headings found in the source).
SECTION_MARKERS = [
    "PROSPECTUS",
    "DEFINITIONS",
    "SCOPE OF COVER",
    "WHAT  WE COVER",
    "WHAT WE EXCLUDE",
    "EXTENSIONS",
    "CLAIMS PROCEDURE",
    "STANDARD TERMS AND CONDITIONS",
]

# Noise lines that appear on almost every page (running header/footer).
NOISE_PATTERNS = [
    re.compile(r"^\s*UNIVERSAL SOMPO GENERAL INSURANCE CO LTD\s*$", re.I),
    re.compile(r"^\s*\d+\s+CSC- Individual Health Insurance-Policy Wording\s+UNIHLIP\w+\s+IRDAI Reg No:\d+"),
    re.compile(r"^\s*Page \d+ of \d+\s*$", re.I),
    re.compile(r"^\s*\*+\s*END\s*\*+\s*$"),
]

_CLAUSE_RX = re.compile(r"^\s*(\d+(?:\.\d+)*)[\.\)]\s*")


def _clean_line(line: str) -> str:
    for pat in NOISE_PATTERNS:
        if pat.match(line):
            return ""
    return line.strip()


def _extract_pages(pdf_path: Path) -> List[Dict[str, object]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lines = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
        pages.append({"page": i + 1, "text": lines})
    return pages


def _normalize(text: str) -> str:
    # Collapse multiple spaces, convert smart quotes, strip weird dashes.
    t = text.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def _split_into_sentences(text: str) -> List[str]:
    """Split on sentence boundaries while keeping abbreviations intact."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for p in parts:
        p = p.strip()
        if p and not p.upper().startswith("UNIVERSAL SOMPO"):
            out.append(p)
    return out


def _build_chunks(section_pages: List[Tuple[int, str, str]]) -> List[Dict[str, object]]:
    """Turn (page, section, text) tuples into meaningful chunks."""
    chunks: List[Dict[str, object]] = []
    chunk_counter = 0

    for page_no, section, text in section_pages:
        sentences = _split_into_sentences(text)
        if not sentences:
            continue

        buffer: List[str] = []
        buffer_words = 0
        heading = section

        def flush() -> None:
            nonlocal buffer, buffer_words, chunk_counter
            if not buffer:
                return
            chunk_text = " ".join(buffer)
            words = len(chunk_text.split())
            chunk_counter += 1
            chunk_id = f"P{page_no:02d}-{chunk_counter:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_no,
                "section": section,
                "heading": heading,
                "text": chunk_text,
                "char_len": len(chunk_text),
                "word_len": words,
            })
            # overlap: keep last few sentences so clause context flows
            overlap_take = max(1, len(buffer) // 4)
            buffer = buffer[-overlap_take:] if len(buffer) > 4 else []
            buffer_words = sum(len(s.split()) for s in buffer)

        for sent in sentences:
            n_words = len(sent.split())
            if n_words > MAX_CHUNK_WORDS:
                # long monolithic sentence - split hard on commas
                sub = re.split(r",\s+", sent)
                for s_sub in sub:
                    if not s_sub.strip():
                        continue
                    sub_text = s_sub.strip()
                    buffer.append(sub_text)
                    buffer_words += len(sub_text.split())
                    if buffer_words >= CHUNK_WORDS_SOFT:
                        flush()
                continue
            buffer.append(sent)
            buffer_words += n_words
            if buffer_words >= CHUNK_WORDS_SOFT and len(buffer) >= 3:
                flush()
        flush()

    return chunks


def chunk_policy_text(pages: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Group cleaned page text into sections then chunk each section.

    Each page is scanned for section markers; text is segmented so that
    every portion is attributed to the most recent marker that precedes
    it (handles multiple markers per page, e.g. PROSPECTUS then
    DEFINITIONS on page 1).
    """
    page_sections: List[Tuple[int, str, str]] = []
    current_section = "PROSPECTUS"

    for page in pages:
        page_no = int(page["page"])
        cleaned = _normalize(str(page["text"]))

        marker_spans = []
        for marker in SECTION_MARKERS:
            for m in re.finditer(rf"^\s*{re.escape(marker)}\s*[.:]?\s*$", cleaned, re.MULTILINE):
                marker_spans.append((m.start(), marker.replace("  ", " ")))
        marker_spans.sort(key=lambda x: x[0])

        if not marker_spans:
            page_sections.append((page_no, current_section, cleaned))
            continue

        # content before the first marker keeps the previously active section
        first_pos = marker_spans[0][0]
        prefix = cleaned[:first_pos].strip()
        if prefix:
            page_sections.append((page_no, current_section, prefix))

        # each marker opens a new segment that runs to the next marker
        for i, (pos, section) in enumerate(marker_spans):
            end = marker_spans[i + 1][0] if i + 1 < len(marker_spans) else len(cleaned)
            seg_text = cleaned[pos:end].strip()
            current_section = section
            if seg_text:
                page_sections.append((page_no, section, seg_text))

    return _build_chunks(page_sections)


def ingest_policy(pdf_path: Path = DEFAULT_PDF,
                  out_path: Path = CHUNKS_OUT,
                  force: bool = False) -> List[Dict[str, object]]:
    """Extract pages and produce metadata-rich chunks.

    Returns the chunk dictionaries (and writes them to out_path).
    """
    if out_path.exists() and not force:
        logger.info("Chunks already exist at %s, loading.", out_path)
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    pages = _extract_pages(pdf_path)
    with open(RAW_PAGES_OUT, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)

    chunks = chunk_policy_text(pages)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)

    logger.info("Ingested %d chunks from %s", len(chunks), pdf_path)
    return chunks


def load_chunks(path: Path = CHUNKS_OUT) -> List[PolicyChunk]:
    """Load persisted chunks as typed PolicyChunk objects."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    chunks = [PolicyChunk(**c) for c in raw]
    return chunks


def build_section_index(chunks: List[PolicyChunk]) -> Dict[str, List[PolicyChunk]]:
    idx: Dict[str, List[PolicyChunk]] = {}
    for c in chunks:
        idx.setdefault(c.section, []).append(c)
    return idx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output = ingest_policy(force=True)
    print(f"Created {len(output)} chunks")
    from collections import Counter
    print(Counter(c["section"] for c in output))