"""
BM25 ranking — an alternative scorer to the TF-IDF cosine in search.py.

BM25 is the modern lexical-retrieval standard. Unlike TF-IDF cosine (which
L2-normalizes, so a long document's matching terms get "diluted"), BM25 has an
explicit, tunable document-length normalization (b) and term-frequency
saturation (k1). It usually ranks better, especially over documents of varying
length.

Built from the SAME weighted token strings build_index.py feeds its TF-IDF
model (title repeated for field weighting), so field weights carry over. Row
order matches doc_meta.json.

Artifacts written next to the TF-IDF index (index/):
    bm25_counts.npz   sparse doc-term COUNT matrix
    bm25_idf.npy      BM25 idf per term
    bm25_doclen.npy   token count per document
    bm25_meta.pkl     {avgdl, CountVectorizer}   (vectorizer holds the vocab)

build_index.py builds and saves these alongside the TF-IDF index; this module
also provides load + search used by search.py / evaluate.py.
"""

import pickle
from datetime import datetime

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

# Standard BM25 hyper-parameters. k1 controls term-frequency saturation, b the
# strength of length normalization (0 = none, 1 = full).
K1 = 1.5
B = 0.75

# Recency boost. A doc's BM25 score is multiplied by 1 + LAMBDA * 0.5^(age/HALFLIFE),
# where age = current_year - publish_year (from meta["year"]). This lets recent
# docs edge out equally-relevant older ones WITHOUT overriding relevance (a much
# stronger match still wins). Docs with no known year get multiplier 1.0.
#   RECENCY_LAMBDA   = 0 disables it; higher = stronger recency preference.
#   RECENCY_HALFLIFE = years for the recency weight to halve.
# Tuned to 0.05 (mild): fixes stale results on broad topical queries (e.g.
# "ανακύκλωση" now surfaces 2023+ instead of 2010) while keeping the cost to
# specific-page ("known-item") accuracy small — this component's main job.
# Measured on the cross-form eval: Hit@1 0.412 (off) -> 0.353 at 0.05. Raise it
# toward 0.10-0.20 if citizens mostly want the *latest* news over a specific page.
RECENCY_LAMBDA = 0.08
RECENCY_HALFLIFE = 4.0
_recency_cache: dict = {}


def _recency_mult(metas):
    """Per-doc score multiplier from meta['year']; cached per metas object."""
    key = id(metas)
    cached = _recency_cache.get(key)
    if cached is not None and len(cached) == len(metas):
        return cached
    mult = np.ones(len(metas), dtype=np.float32)
    if RECENCY_LAMBDA > 0:
        cur = datetime.now().year
        for i, m in enumerate(metas):
            y = m.get("year")
            if y:
                mult[i] = 1.0 + RECENCY_LAMBDA * (0.5 ** ((cur - y) / RECENCY_HALFLIFE))
    _recency_cache[key] = mult
    return mult


def build_bm25(docs: list[str], min_df, max_df) -> dict:
    """Fit a count model + BM25 statistics over the weighted token strings."""
    cv = CountVectorizer(analyzer="word", token_pattern=r"\S+", lowercase=False,
                         min_df=min_df, max_df=max_df)
    try:
        counts = cv.fit_transform(docs)
    except ValueError:
        cv.set_params(min_df=1, max_df=1.0)
        counts = cv.fit_transform(docs)
    counts = counts.tocsr().astype(np.float32)

    n_docs = counts.shape[0]
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    doc_len = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(doc_len.mean()) if n_docs else 0.0
    return {"counts": counts, "idf": idf, "doc_len": doc_len, "avgdl": avgdl, "cv": cv}


def save_bm25(art: dict, index_dir):
    sp.save_npz(index_dir / "bm25_counts.npz", art["counts"])
    np.save(index_dir / "bm25_idf.npy", art["idf"])
    np.save(index_dir / "bm25_doclen.npy", art["doc_len"])
    with open(index_dir / "bm25_meta.pkl", "wb") as f:
        pickle.dump({"avgdl": art["avgdl"], "cv": art["cv"]}, f)


def load_bm25(index_dir) -> dict:
    """Load BM25 artifacts and precompute the per-document length norm."""
    counts = sp.load_npz(index_dir / "bm25_counts.npz").tocsc()  # CSC = fast columns
    idf = np.load(index_dir / "bm25_idf.npy")
    doc_len = np.load(index_dir / "bm25_doclen.npy")
    with open(index_dir / "bm25_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    avgdl = meta["avgdl"] or 1.0
    denom = (K1 * (1.0 - B + B * doc_len / avgdl)).astype(np.float32)  # per doc
    return {"counts": counts, "idf": idf, "denom": denom,
            "vocab": meta["cv"].vocabulary_}


def bm25_search(query: str, index: dict, metas, top_k: int = 5):
    """Rank documents by BM25. Same return shape as search.search()."""
    from greek_text import tokenize  # same tokenizer as index time
    counts, idf, denom, vocab = (index["counts"], index["idf"],
                                 index["denom"], index["vocab"])
    scores = np.zeros(counts.shape[0], dtype=np.float32)
    for term in tokenize(query):
        j = vocab.get(term)
        if j is None:
            continue
        col = counts.getcol(j)              # sparse column: this term's tf per doc
        rows, tf = col.indices, col.data
        # BM25 term contribution: idf * tf*(k1+1) / (tf + k1*(1-b+b*|d|/avgdl))
        scores[rows] += idf[j] * (tf * (K1 + 1.0)) / (tf + denom[rows])

    # Recency boost: nudge recent docs up among similarly-relevant ones. Zero
    # scores stay zero (0 * mult = 0), so non-matching docs are still excluded.
    if RECENCY_LAMBDA > 0:
        scores *= _recency_mult(metas)

    k = min(top_k, scores.shape[0])
    if k == 0:
        return []
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return [(float(scores[i]), metas[int(i)]) for i in top if scores[i] > 0.0]
