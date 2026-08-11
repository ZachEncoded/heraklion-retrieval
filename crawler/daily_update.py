"""
daily_update.py — fast per-category incremental refresh.

Idea (yours): keep a per-category map of the item links currently listed on
that category's index page. On each run, re-fetch each index page (ONE request)
and compare its link set to what we stored. If the count/links changed, crawl
ONLY the new items — never the whole category, never the frozen archive.

Because we crawl exactly the difference, a category with 3 new items costs 3
page fetches and then moves on (your "stop after the new ones" optimization).
The listing order is preserved, so if the site lists newest-first we crawl the
newest first and can cap with --max-new-per-category.

One upgrade over a bare count: we store the actual LINK SET, not just a number.
Same cost (the index is one page), but it pinpoints WHICH links are new and
also catches the add+remove case where the count is unchanged but content isn't.

Reuse: the actual fetch/extract/dedup/PDF/manifest logic is crawler.py's. We
just run crawler.crawl() on the new URLs with depth 0 (crawl exactly those
pages, don't follow their links), so behaviour matches a normal crawl —
content dedup, PDF cache-first, manifest updates all apply.

State file:
    crawl_state/category_state.json   {category: {"index_url":..., "links":[...]}}

Usage:
    python daily_update.py --discover     # after a full crawl: suggest each
                                          # section's index page from the manifest
    python daily_update.py --bootstrap    # once: record current index links as
                                          # the baseline (no item crawling)
    python daily_update.py                # daily: crawl only NEW items per category
    python daily_update.py --rebuild-index  # also rebuild the search index after

Do NOT run this while a full crawl is in progress — both write the manifest.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import crawler  # noqa: E402  reuse its URL/extract/crawl logic
# Shared with crawler.py so the two never drift: the category->index map and the
# category_state.json read/write live in one place.
from category_map import CATEGORY_INDEXES, STATE_PATH, load_state, save_state  # noqa: E402


# --------------------------------------------------------------------------
# Index page -> ordered list of this category's item links
# --------------------------------------------------------------------------
def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = crawler.USER_AGENT
    return s


def fetch_index_links(index_url: str, category: str, session) -> list[str]:
    """Ordered, de-duplicated list of this category's item links on the index
    page (fetched live). Uses crawler.index_item_links so the filtering is
    identical to the crawl-time baseline. Order is preserved (newest-first
    listings stay newest-first)."""
    r = session.get(crawler.normalize_url(index_url), timeout=crawler.TIMEOUT)
    r.raise_for_status()
    page_links, _ = crawler.extract_links(r.text, index_url)
    return crawler.index_item_links(page_links, category, index_url)


# --------------------------------------------------------------------------
# Crawl a specific set of new URLs (reuse crawler, no link-following)
# --------------------------------------------------------------------------
def crawl_new(urls: list[str]):
    if not urls:
        return
    crawler.MAX_DEPTH = 0          # crawl exactly these pages, follow no links
    crawler.crawl(urls, crawler.OUTPUT_DIR)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def do_bootstrap():
    """Record the current index links as the baseline WITHOUT crawling items
    (they already exist in the corpus after a full crawl)."""
    state = load_state()
    session = _session()
    for category, index_url in CATEGORY_INDEXES.items():
        try:
            links = fetch_index_links(index_url, category, session)
        except requests.RequestException as e:
            print(f"  {category}: index fetch failed ({type(e).__name__}) — skipped")
            continue
        state[category] = {"index_url": index_url, "links": links}
        print(f"  {category}: baseline set to {len(links)} links")
    save_state(state)
    print(f"Baseline saved to {STATE_PATH.name}")


def do_run(max_new: int | None, rebuild: bool):
    state = load_state()
    session = _session()
    all_new: list[str] = []

    for category, index_url in CATEGORY_INDEXES.items():
        known = set(state.get(category, {}).get("links", []))
        try:
            current = fetch_index_links(index_url, category, session)
        except requests.RequestException as e:
            print(f"  {category}: index fetch failed ({type(e).__name__}) — skipped")
            continue

        if len(current) == len(known) and set(current) == known:
            print(f"  {category}: unchanged ({len(current)} links)")
            continue

        # New = listed now but not previously known, in listing order.
        new = [l for l in current if l not in known]
        to_crawl = new if max_new is None else new[:max_new]
        print(f"  {category}: {len(current)} links now, "
              f"{len(new)} new -> crawling {len(to_crawl)}")
        all_new.extend(to_crawl)

        # Baseline becomes: stable links + the ones we actually crawled.
        # (Uncrawled-new, if capped, stay out so they retry next run; removed
        #  links drop out automatically.)
        crawled = set(to_crawl)
        new_known = [l for l in current if l in known or l in crawled]
        state[category] = {"index_url": index_url, "links": new_known}

    if all_new:
        print(f"\nCrawling {len(all_new)} new item(s) total...")
        crawl_new(all_new)
    else:
        print("\nNothing new across all categories.")

    save_state(state)

    if rebuild:
        print("\nRebuilding search index...")
        subprocess.run([sys.executable, str(BASE_DIR.parent / "indexing" / "build_index.py")],
                       check=False)


def do_discover():
    """Suggest an index page per section from the manifest: the page that links
    to the most same-section HTML items is very likely that section's listing
    page. Run only after a full crawl (needs a complete manifest)."""
    manifest = json.loads(crawler.MANIFEST_PATH.read_text(encoding="utf-8"))
    pages = manifest.get("pages", {})
    # For each section, score candidate pages by how many same-section items
    # they link to.
    best: dict[str, tuple[int, str]] = {}
    for url, entry in pages.items():
        if not entry.get("file"):
            continue
        sec = crawler.section_of(url)
        links = entry.get("links", [])
        same = sum(1 for l in links if crawler.section_of(l) == sec)
        if same > best.get(sec, (0, ""))[0]:
            best[sec] = (same, url)
    print("Suggested CATEGORY_INDEXES (paste/adjust in daily_update.py):")
    print("CATEGORY_INDEXES = {")
    for sec, (count, url) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        print(f'    "{sec}": "{url}",   # links to ~{count} items')
    print("}")


def main():
    ap = argparse.ArgumentParser(description="Fast per-category incremental refresh.")
    ap.add_argument("--bootstrap", action="store_true",
                    help="Record current index links as baseline (no crawling).")
    ap.add_argument("--discover", action="store_true",
                    help="Suggest each section's index page from the manifest.")
    ap.add_argument("--max-new-per-category", type=int, default=None, metavar="N",
                    help="Cap new items crawled per category per run.")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Rebuild the search index after crawling new items.")
    args = ap.parse_args()

    if args.discover:
        do_discover()
    elif args.bootstrap:
        do_bootstrap()
    else:
        do_run(args.max_new_per_category, args.rebuild_index)


if __name__ == "__main__":
    main()
