"""
Shared category -> index-page map and its persisted per-category link state.

Both crawler.py (after a full crawl) and daily_update.py (after a refresh)
write crawl_state/category_state.json, so the daily diff always has a current
baseline of which item links each category lists. Keeping the config here means
the two scripts can never drift apart.

State file shape:
    { "<category>": { "index_url": "...", "links": ["...", ...] }, ... }

Complete CATEGORY_INDEXES with `python daily_update.py --discover` after a full
crawl (it suggests each section's index page from the manifest).
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# crawl_state lives at the project root (one level up from crawler/).
STATE_PATH = BASE_DIR.parent / "crawl_state" / "category_state.json"

# category (first URL path segment) -> its ROLLING index/listing page URL.
# These must be the "newest first" listings (not year-buckets), so new content
# shows up on them. Validated to return HTTP 200 with real item links.
# press + municipality are mirrors of the same releases (monitored both so a new
# item is caught under either URL form). Adjust/extend as the site changes.
CATEGORY_INDEXES: dict[str, str] = {
    "press":        "https://www.heraklion.gr/press/municipality-press-releases",
    "municipality": "https://www.heraklion.gr/municipality/municipality-press-releases",
    "agenda":       "https://www.heraklion.gr/agenda",
    "citizen":      "https://www.heraklion.gr/citizen/social-news",
    "culture":      "https://www.heraklion.gr/culture/syllogoi",
    "child":        "https://www.heraklion.gr/child/kids",
    "ourplace":     "https://www.heraklion.gr/ourplace/ourplace.html",
}


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass  # corrupted — treat as empty; callers rebuild it
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                          encoding="utf-8")
