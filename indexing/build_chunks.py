"""
Passage-level (chunked) BM25 index.

Instead of one vector per whole document, split each document's text into
overlapping windows ("chunks") and index those. Retrieval then matches the
*relevant passage* rather than a whole article, which fixes the dilution where a
short query is a tiny fraction of a long doc, and (bonus) tells you which part of
the document matched — useful if results feed an LLM later.

Each chunk = the parent's weighted title tokens + a window of the body/PDF
tokens, so title matches still count on every chunk. At query time we rank
chunks with BM25 (bm25.py) and collapse to the best chunk per parent document.

Artifacts in index/chunks/ (BM25 layout + chunk_meta.json mapping chunk->parent).

Usage (from ./indexing, after build_index.py):
    python build_chunks.py
"""

import json
import time
from pathlib import Path

from build_index import (CORPUS_DIR, W_PAGE_TITLE, MIN_DF, MAX_DF,
                          url_excluded, log)
from bm25 import build_bm25, save_bm25, load_bm25, bm25_search
from greek_text import tokenize

# Absolute so load works regardless of the caller's CWD (evaluate.py runs from
# the project root; build_index's INDEX_DIR is relative to ./indexing).
CHUNK_DIR = Path(__file__).resolve().parent / "index" / "chunks"
CHUNK_TOKENS = 120          # window size in tokens
STRIDE = 80                 # step between windows (overlap = CHUNK_TOKENS - STRIDE)


def doc_chunks(doc: dict) -> list[list[str]]:
    """Token-window chunks of a doc: each = weighted title + a body/PDF window."""
    parts = [doc.get("body", "") or ""]
    i = 1
    while f"component_file_{i}" in doc:
        comp = doc[f"component_file_{i}"]
        if isinstance(comp, dict):
            parts.append(comp.get("body", "") or "")
        i += 1
    body = tokenize("\n".join(parts))
    title = tokenize(doc.get("title", "")) * W_PAGE_TITLE

    if not body:
        return [title] if title else []
    chunks = []
    for start in range(0, len(body), STRIDE):
        chunks.append(title + body[start:start + CHUNK_TOKENS])
        if start + CHUNK_TOKENS >= len(body):
            break
    return chunks


def build():
    t0 = time.time()
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CORPUS_DIR.glob("*.json"))
    log.info("Chunking %d docs...", len(files))

    chunk_docs: list[str] = []      # token strings, one per chunk (BM25 input)
    chunk_meta: list[dict] = []     # parent info per chunk
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if url_excluded(doc.get("url", "")):
            continue
        meta = {"file": path.name, "title": doc.get("title", ""),
                "url": doc.get("url", ""), "section": doc.get("section", "")}
        for k, toks in enumerate(doc_chunks(doc)):
            if not toks:
                continue
            chunk_docs.append(" ".join(toks))
            chunk_meta.append({**meta, "chunk": k})

    log.info("Built %d chunks from %d docs. Fitting BM25...",
             len(chunk_docs), len(files))
    save_bm25(build_bm25(chunk_docs, MIN_DF, MAX_DF), CHUNK_DIR)
    (CHUNK_DIR / "chunk_meta.json").write_text(
        json.dumps(chunk_meta, ensure_ascii=False), encoding="utf-8")
    log.info("Saved chunk index to %s (%d chunks). Done in %.1fs",
             CHUNK_DIR, len(chunk_docs), time.time() - t0)


# --------------------------------------------------------------------------
# Search: rank chunks, collapse to the best chunk per parent document
# --------------------------------------------------------------------------
def load_chunk_index():
    index = load_bm25(CHUNK_DIR)
    chunk_meta = json.loads((CHUNK_DIR / "chunk_meta.json").read_text(encoding="utf-8"))
    return index, chunk_meta


def chunk_search(query, index, chunk_meta, top_k: int = 5):
    # pull a generous pool of chunks, then keep each document's best chunk
    pool = bm25_search(query, index, chunk_meta, top_k=max(200, top_k * 20))
    best: dict[str, tuple[float, dict]] = {}
    for score, meta in pool:
        f = meta["file"]
        if f not in best or score > best[f][0]:
            best[f] = (score, meta)
    ranked = sorted(best.values(), key=lambda x: -x[0])[:top_k]
    return ranked


if __name__ == "__main__":
    build()
