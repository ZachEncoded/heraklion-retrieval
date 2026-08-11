"""
Build a TF-IDF inverted index over the corpus JSON files.

Reads every corpus/*.json, applies field weighting (a term in the page title
counts more than one in a body), fits a scikit-learn TF-IDF model with our
Greek-aware tokenizer, and saves everything search.py needs:

    index/tfidf_matrix.npz     sparse doc-term matrix (TF-IDF, L2-normalized)
    index/vectorizer.pkl       fitted TfidfVectorizer (vocabulary + idf)
    index/doc_meta.json        per-row metadata (title, url, section, file)
    index/inverted_index.json  term -> [row ids]  (the "reversed dictionary")

Only human content is indexed: page title, page body, and each attached
file's title + extracted text (component_file_N). url / section / timestamp
are kept as metadata only, never indexed. JSON structure is never indexed.

Field weighting is done by REPEATING a field's tokens N times before the
vectorizer sees them, so a title term gets N x the term-frequency of a body
term. Weights are configurable below.

Usage (run from the 'indexing' folder; corpus is one level up):
    python build_index.py
"""

import json
import logging
import pickle
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from greek_text import tokenize

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Anchored to THIS file so it works from any working directory: corpus is at the
# project root (one level up), the index sits next to this script in ./index.
CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
INDEX_DIR  = Path(__file__).resolve().parent / "index"

# Field weights: how many times each field's tokens are repeated.
# Higher => that field dominates relevance. Tune against real queries.
W_PAGE_TITLE      = 3
W_COMPONENT_TITLE = 3
W_PAGE_BODY       = 1
W_COMPONENT_BODY  = 1

# Set to a tuple of URL path prefixes to exclude, e.g. ("/en/",) for Greek-only.
# The /en/ mirror is excluded from crawling (broken/empty), so exclude it here
# too, and any leftover /en/ docs in the corpus stay out of the search index.
EXCLUDE_URL_PREFIXES: tuple[str, ...] = ("/en/",)

# HARD REQUIREMENT: never return files from the municipality e-services portal.
# The upstream chatbot/RAG routes e-service intents to eservices.heraklion.gr
# itself; by the time THIS retriever is called, an e-service is NOT wanted. Any
# doc whose host is listed here is skipped at index time, so it can never be
# retrieved. (The crawler's strict-domain policy already never crawls this
# subdomain — this is a second, explicit guarantee.)
EXCLUDE_HOSTS: frozenset[str] = frozenset({"eservices.heraklion.gr"})

# Vectorizer knobs
MIN_DF = 2            # ignore terms appearing in fewer than 2 docs (typos/noise)
MAX_DF = 0.5          # ignore terms in >50% of docs (corpus-specific stopwords)
NGRAM_RANGE = (1, 1)  # unigrams; (1, 2) adds bigrams (bigger, sometimes better)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("indexer")


def url_excluded(url: str) -> bool:
    p = urlparse(url)
    if p.netloc.lower() in EXCLUDE_HOSTS:      # e-services portal — never index
        return True
    return any(p.path.startswith(pref) for pref in EXCLUDE_URL_PREFIXES)


_YEAR_RE_URL = re.compile(r"(?:press-releases-|announcements-|/)(20[0-2]\d)(?:/|\b)")
_YEAR_RE_TITLE = re.compile(r"\|\s*(20[0-2]\d)\s*\|")

def extract_year(url: str, title: str) -> int | None:
    """Best-effort publish YEAR of a doc, for the recency boost. The stored
    'timestamp' is crawl-time (useless), so we parse the year from the URL
    (e.g. .../press-releases-2015/...) or the title (e.g. '... | 2015 | ...').
    ~87% of docs yield a year; the rest get None (no boost, no penalty)."""
    m = _YEAR_RE_URL.search(url) or _YEAR_RE_TITLE.search(title)
    if m:
        return int(m.group(1))
    m = re.search(r"20[0-2]\d", url)          # any year-looking token in the URL
    return int(m.group()) if m else None


def year_of(doc: dict) -> int | None:
    """Publish year for the recency boost. Prefer the crawler's parsed
    'published' release date; fall back to parsing the URL/title."""
    pub = doc.get("published") or ""
    if pub[:4].isdigit():
        return int(pub[:4])
    return extract_year(doc.get("url", ""), doc.get("title", ""))


def weighted_tokens(doc: dict) -> list[str]:
    """Concatenate field tokens with repetition according to field weights.
    Indexes ONLY: page title, page body, and each component's title + body."""
    toks: list[str] = []
    toks += tokenize(doc.get("title", "")) * W_PAGE_TITLE
    toks += tokenize(doc.get("body", "")) * W_PAGE_BODY
    i = 1
    while f"component_file_{i}" in doc:
        comp = doc[f"component_file_{i}"]
        if isinstance(comp, dict):
            toks += tokenize(comp.get("title", "")) * W_COMPONENT_TITLE
            toks += tokenize(comp.get("body", "")) * W_COMPONENT_BODY
        i += 1
    return toks


def load_corpus():
    """Return (list_of_token_strings, list_of_metadata) for usable files."""
    files = sorted(CORPUS_DIR.glob("*.json"))
    log.info("Found %d JSON files in %s", len(files), CORPUS_DIR.resolve())
    skipped = 0
    docs, metas = [], []
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        url = doc.get("url", "")
        if url_excluded(url):
            skipped += 1
            continue
        toks = weighted_tokens(doc)
        if not toks:
            skipped += 1
            continue
        docs.append(" ".join(toks))
        metas.append({
            "file": path.name,
            "title": doc.get("title", ""),
            "url": url,
            "section": doc.get("section", ""),
            "timestamp": doc.get("timestamp", ""),
            "published": doc.get("published"),
            "year": year_of(doc),
        })
    log.info("Loaded %d documents (%d skipped)", len(docs), skipped)
    return docs, metas


def build_inverted_index(matrix: csr_matrix,
                         vocab_inv: list[str]) -> dict[str, list[int]]:
    """term -> sorted list of doc row-ids that contain it."""
    csc = matrix.tocsc()
    indptr = np.asarray(csc.indptr)
    indices = np.asarray(csc.indices)
    inv: dict[str, list[int]] = {}
    for term_id, term in enumerate(vocab_inv):
        start, end = int(indptr[term_id]), int(indptr[term_id + 1])
        rows = indices[start:end]
        if len(rows):
            inv[term] = sorted(int(r) for r in rows)
    return inv


def main():
    t0 = time.time()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    docs, metas = load_corpus()
    if not docs:
        log.error("No documents to index. Check CORPUS_DIR (%s).",
                  CORPUS_DIR.resolve())
        return

    # Our tokens are already normalized+filtered; tell sklearn to just split
    # on whitespace and keep them verbatim (no extra lowercasing/tokenizing).
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\S+",
        lowercase=False,
        preprocessor=None,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,      # 1+log(tf): dampens very frequent terms
        norm="l2",              # so dot product == cosine similarity
    )

    log.info("Fitting TF-IDF over %d documents...", len(docs))
    try:
        matrix: csr_matrix = csr_matrix(vectorizer.fit_transform(docs))
    except ValueError:
        # Small-corpus fallback: min_df/max_df pruning can wipe the vocabulary.
        log.warning("Relaxing min_df/max_df for small corpus")
        vectorizer.set_params(min_df=1, max_df=1.0)
        matrix = csr_matrix(vectorizer.fit_transform(docs))

    vocab_inv = vectorizer.get_feature_names_out().tolist()
    log.info("Matrix: %d docs x %d terms (%d non-zeros)",
             matrix.shape[0], matrix.shape[1], matrix.nnz)

    log.info("Building inverted index...")
    inverted = build_inverted_index(matrix, vocab_inv)

    # --- persist everything --------------------------------------------------
    sp.save_npz(INDEX_DIR / "tfidf_matrix.npz", matrix)
    with open(INDEX_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    (INDEX_DIR / "doc_meta.json").write_text(
        json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "inverted_index.json").write_text(
        json.dumps(inverted, ensure_ascii=False), encoding="utf-8")

    # --- BM25 index (alternative scorer, same weighted docs) -----------------
    from bm25 import build_bm25, save_bm25
    log.info("Building BM25 index...")
    save_bm25(build_bm25(docs, MIN_DF, MAX_DF), INDEX_DIR)

    log.info("Saved index to %s/", INDEX_DIR.resolve())
    log.info("Vocabulary size: %d terms", len(vocab_inv))
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()