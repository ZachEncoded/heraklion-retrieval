"""
Search the TF-IDF inverted index built by indexing/build_index.py.

Loads the saved artifacts, turns a user query into a TF-IDF vector using the
SAME Greek-aware tokenizer used at index time, and ranks documents by COSINE
SIMILARITY between the query vector and every document vector.

Because build_index.py stored the doc-term matrix L2-normalized (norm="l2")
and TfidfVectorizer.transform() L2-normalizes the query too, the cosine
similarity is simply the dot product:  scores = matrix @ query_vec.T

For each hit it returns the document TITLE and its URL (every corpus JSON has a
"url"; it was kept in doc_meta.json at index time).

This script lives in the project root. The index artifacts and the shared
tokenizer live under ./indexing, and paths below are anchored to this file's
location, so you can run it from anywhere:
    python search.py                       # interactive prompt (top 5)
    python search.py "παιδικοι σταθμοι"     # one-shot query (top 5)
    python search.py -k 10 "query here"     # top-10 instead of top-5
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# The tokenizer and the index live in the sibling ../indexing folder. Anchor
# paths to THIS file so the script works regardless of the current directory.
BASE_DIR = Path(__file__).resolve().parent
INDEXING_DIR = BASE_DIR.parent / "indexing"
INDEX_DIR = INDEXING_DIR / "index"

# Make greek_text importable from its subfolder (same tokenizer as index time).
sys.path.insert(0, str(INDEXING_DIR))
from greek_text import tokenize  # noqa: E402  (import after sys.path tweak)

DEFAULT_TOP_K = 5


def load_index(index_dir: Path):
    """Load the three artifacts search needs. Returns (matrix, vectorizer, metas)."""
    matrix_path = index_dir / "tfidf_matrix.npz"
    vec_path = index_dir / "vectorizer.pkl"
    meta_path = index_dir / "doc_meta.json"

    missing = [p.name for p in (matrix_path, vec_path, meta_path) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing index artifact(s): {', '.join(missing)}.\n"
            f"Run indexing/build_index.py first (looked in {index_dir.resolve()})."
        )

    matrix = sp.load_npz(matrix_path).tocsr()
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    metas = json.loads(meta_path.read_text(encoding="utf-8"))
    return matrix, vectorizer, metas


def search(query: str, matrix, vectorizer, metas, top_k: int = DEFAULT_TOP_K):
    """Return up to top_k (score, meta) pairs ranked by cosine similarity.

    The query is tokenized with the same pipeline used at index time, so Greek
    accents/final-sigma/stopwords are handled identically. Empty results come
    back as an empty list (e.g. all query terms were stopwords or out-of-vocab).
    """
    tokens = tokenize(query)
    if not tokens:
        return []

    # transform() uses the fitted vocabulary + idf and L2-normalizes the row,
    # so the dot product below is exactly cosine similarity.
    query_vec = vectorizer.transform([" ".join(tokens)])
    if query_vec.nnz == 0:
        return []  # every term was pruned out of the vocabulary (min_df/max_df)

    # scores[i] = cosine(query, doc_i). Dense 1-D array, one entry per document.
    scores = (matrix @ query_vec.T).toarray().ravel()

    # Take the top_k highest scores efficiently, then sort just those.
    k = min(top_k, scores.shape[0])
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    results = []
    for i in top_idx:
        s = float(scores[i])
        if s <= 0.0:
            continue  # no term overlap with the query; skip
        results.append((s, metas[int(i)]))
    return results


def format_results(query: str, results):
    """Render each hit as its title followed by the document URL."""
    if not results:
        return f'No results for "{query}".'
    lines = [f'Top {len(results)} results for "{query}":', ""]
    for rank, (score, meta) in enumerate(results, 1):
        title = meta.get("title") or "(no title)"
        url = meta.get("url") or "(no url)"
        lines.append(f"{rank:>2}. [{score:.4f}] {title}")
        lines.append(f"      {url}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Search the index (BM25 by default).")
    parser.add_argument("query", nargs="*", help="Query text. Omit for interactive mode.")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of results to return (default {DEFAULT_TOP_K}).")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25",
                        help="Ranking function (default bm25 — it measured best).")
    args = parser.parse_args()

    # BM25 is the default: it beat TF-IDF cosine on every metric of both evals
    # (see evaluate.py). tfidf is kept as a fallback for comparison.
    if args.scorer == "bm25":
        from bm25 import load_bm25, bm25_search
        metas = load_index(INDEX_DIR)[2]
        index = load_bm25(INDEX_DIR)
        do_search = lambda q: bm25_search(q, index, metas, args.top_k)
        print(f"Index loaded: {len(metas)} docs (scorer: BM25).", file=sys.stderr)
    else:
        matrix, vectorizer, metas = load_index(INDEX_DIR)
        do_search = lambda q: search(q, matrix, vectorizer, metas, args.top_k)
        print(f"Index loaded: {matrix.shape[0]} docs (scorer: TF-IDF).", file=sys.stderr)

    if args.query:
        query = " ".join(args.query)
        print(format_results(query, do_search(query)))
        return

    # Interactive loop.
    print("Enter a query (blank line or Ctrl+C to quit).", file=sys.stderr)
    try:
        while True:
            query = input("\nquery> ").strip()
            if not query:
                break
            print(format_results(query, do_search(query)))
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
