"""
Measure retrieval quality, so every change (stemming, BM25, ...) can be judged
objectively instead of by eyeballing a few queries.

Uses KNOWN-ITEM evaluation: each eval query is a short snippet taken from a real
document's body, and the "relevant" answer is that source document. A good
retriever should rank the source doc at (or near) the top. This needs no manual
labelling and is repeatable, so it's ideal for A/B-ing pipeline changes.

Metrics over the eval set (top-k, default k=10):
    Hit@1      fraction of queries whose #1 result is the source doc
    MRR@k      mean reciprocal rank of the source doc (1.0 = always rank 1)
    Recall@5   fraction of queries with the source doc in the top 5
    misses     queries where the source doc was not in the top-k at all

Usage:
    python evaluate.py --generate 300   # build eval/eval_set.json once (fixed
                                        # set, so results are comparable over time)
    python evaluate.py                  # run the eval set, print metrics
    python evaluate.py -k 20            # evaluate at a different cutoff

Keep the generated eval/eval_set.json checked in and REUSE it across experiments;
only regenerate if the corpus changes a lot. You can also hand-add real queries
to that file: {"query": "...", "relevant": ["<corpus filename>", ...]}.
"""

import argparse
import json
import random
from pathlib import Path

from search import load_index, search, INDEX_DIR
import sys

BASE_DIR = Path(__file__).resolve().parent    # search/
_ROOT = BASE_DIR.parent                        # project root
sys.path.insert(0, str(_ROOT / "indexing"))    # for greek_text / bm25 / build_chunks
from greek_text import tokenize  # noqa: E402

CORPUS_DIR = _ROOT / "corpus"
EVAL_PATH = _ROOT / "eval" / "eval_set.json"

SNIPPET_WORDS = 12       # length of the query snippet drawn from a doc body
MIN_BODY_WORDS = 60      # only sample from docs with a reasonably long body
SEED = 42                # fixed so the generated set is reproducible


def generate(n: int):
    """Build a fixed known-item eval set: snippet of a body -> its source file."""
    rng = random.Random(SEED)
    files = sorted(CORPUS_DIR.glob("*.json"))
    rng.shuffle(files)

    eval_set = []
    for path in files:
        if len(eval_set) >= n:
            break
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        words = (doc.get("body", "") or "").split()
        if len(words) < MIN_BODY_WORDS:
            continue
        # a random contiguous window from the body = a realistic "snippet" query
        start = rng.randint(0, len(words) - SNIPPET_WORDS)
        query = " ".join(words[start:start + SNIPPET_WORDS])
        # skip windows that are all stopwords/punctuation (no usable tokens)
        if len(tokenize(query)) < 3:
            continue
        eval_set.append({"query": query, "relevant": [path.name]})

    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(eval_set, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"Generated {len(eval_set)} known-item queries -> {EVAL_PATH}")


def run(top_k: int, set_path: Path, scorer: str):
    if not set_path.exists():
        raise SystemExit(f"No eval set at {set_path}. Run: python evaluate.py --generate 300")
    eval_set = json.loads(set_path.read_text(encoding="utf-8"))

    # Pick the scorer: TF-IDF cosine, whole-doc BM25, or passage-level BM25.
    if scorer == "bm25":
        from bm25 import load_bm25, bm25_search
        _, _, metas = load_index(INDEX_DIR)
        index = load_bm25(INDEX_DIR)
        do_search = lambda q: bm25_search(q, index, metas, top_k=top_k)
    elif scorer == "chunks":
        sys.path.insert(0, str(_ROOT / "indexing"))
        from build_chunks import load_chunk_index, chunk_search
        index, chunk_meta = load_chunk_index()
        do_search = lambda q: chunk_search(q, index, chunk_meta, top_k=top_k)
    else:
        matrix, vectorizer, metas = load_index(INDEX_DIR)
        do_search = lambda q: search(q, matrix, vectorizer, metas, top_k=top_k)

    hit1 = mrr = recall5 = 0.0
    misses = 0
    n = len(eval_set)
    for item in eval_set:
        relevant = set(item["relevant"])
        results = do_search(item["query"])
        ranked = [meta["file"] for _, meta in results]

        # first rank (1-indexed) at which a relevant doc appears
        rank = next((i + 1 for i, f in enumerate(ranked) if f in relevant), None)
        if rank is None:
            misses += 1
        else:
            mrr += 1.0 / rank
            if rank == 1:
                hit1 += 1
        recall5 += len(relevant & set(ranked[:5])) / len(relevant)

    print(f"Eval set: {n} known-item queries   (top_k={top_k})")
    print(f"  Hit@1     {hit1 / n:.3f}   (source doc ranked #1)")
    print(f"  MRR@{top_k:<3}  {mrr / n:.3f}   (1.0 = always rank 1)")
    print(f"  Recall@5  {recall5 / n:.3f}   (source doc in top 5)")
    print(f"  misses    {misses}/{n}   (source doc not in top {top_k})")


def main():
    ap = argparse.ArgumentParser(description="Measure retrieval quality (known-item).")
    ap.add_argument("--generate", type=int, metavar="N",
                    help="Build a fixed eval set of N known-item queries and exit.")
    ap.add_argument("-k", "--top-k", type=int, default=10,
                    help="Rank cutoff for the metrics (default 10).")
    ap.add_argument("--set", default=str(EVAL_PATH), metavar="FILE",
                    help="Eval set JSON to run (default: the generated known-item set).")
    ap.add_argument("--scorer", choices=["tfidf", "bm25", "chunks"],
                    default="tfidf", help="Ranking function to evaluate (default tfidf).")
    args = ap.parse_args()
    if args.generate:
        generate(args.generate)
    else:
        print(f"[scorer: {args.scorer}]")
        run(args.top_k, Path(args.set), args.scorer)


if __name__ == "__main__":
    main()
