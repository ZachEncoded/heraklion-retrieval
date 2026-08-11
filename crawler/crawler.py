"""
Polite website crawler for a RAG document store — Heraklion edition.

For each crawled page it writes one JSON file named after the page title,
with attached files as numbered components (for per-field index weights):

    {
        "title":   "...",              # page title
        "body":    "...",              # page text only (no file content)
        "url":     "...",
        "section": "deltia-typou",     # first URL path segment
        "component_file_1": {"title": "...", "body": "...", "url": "..."},
        "component_file_2": { ... },
        "timestamp": "2026-07-04T12:00:00+00:00"
    }

Key behaviours:
  - STRICT domain: only heraklion.gr / www.heraklion.gr. Subdomains and
    external sites are never visited. Redirects leaving the domain are dropped.
  - File components: PDFs linked from a page are downloaded, their text is
    extracted and stored as component_file_N (title = link anchor text).
  - Incremental re-crawls: manifest stores ETag/Last-Modified and a hash of
    the EXTRACTED TEXT (robust against rotating banners / dynamic HTML).
    Unchanged pages are logged, and wait only FAST_DELAY between requests.
  - Progress line every PROGRESS_EVERY pages: saved/unchanged/errors/frontier.
  - Dead links recorded in dead_links.json; dead documents deleted.
  - Respects robots.txt, rate-limits, normalizes + deduplicates URLs,
    strips boilerplate via trafilatura (BeautifulSoup fallback).

Dependencies:
    pip install requests trafilatura beautifulsoup4 pypdf
"""

import hashlib
import io
import json
import logging
import re
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode, unquote
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    trafilatura = None
    HAS_TRAFILATURA = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    PdfReader = None
    HAS_PYPDF = False

import category_map  # shared category->index map + category_state.json IO

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED_URL       = "https://www.heraklion.gr/"
# Data lives at the PROJECT ROOT (one level up from this crawler/ folder), so
# these work no matter what directory the crawler is launched from.
_ROOT          = Path(__file__).resolve().parent.parent
OUTPUT_DIR     = _ROOT / "corpus"            # JSON documents
STATE_DIR      = _ROOT / "crawl_state"       # manifest, pdf cache, reports
CRAWL_DELAY    = 1.5                         # seconds after full downloads
FAST_DELAY     = 0.3                         # seconds after unchanged pages
MAX_PAGES      = None                        # None = no limit
MAX_DEPTH      = None                        # None = no depth limit
TIMEOUT        = 20                          # per-request timeout (s)
MIN_BODY_CHARS = 200                         # min text for a page to be saved
MAX_TITLE_LEN  = 100                          # filename title part length
MAX_PDF_MB     = 100                          # skip PDFs bigger than this
PROGRESS_EVERY = 100                         # progress log frequency (pages)
USER_AGENT     = "MyRAGCrawler/0.1 (+contact: you@example.com)"

# PDF text is cached in pdf_cache/ and keyed by URL. Published PDFs are
# effectively immutable at a URL, so by default we TRUST the cache and never
# re-fetch a PDF we've already extracted (this is the big re-crawl speedup).
#   None -> trust cached PDF text forever (fastest).
#   <int> -> revalidate (conditional GET) only cached PDFs older than N days.
REVALIDATE_PDF_AFTER_DAYS = None

# Fetch+parse a page's PDFs in parallel. 1 = sequential (polite, default).
# >1 is faster on the COLD crawl but sends concurrent requests to the host,
# so raise it only for a first bulk crawl, then set back to 1 for updates.
PDF_WORKERS = 1

# Path prefixes we NEVER crawl. The /en/ mirror is broken: relative-link traps
# that spawn endless malformed URLs, and pages that are empty duplicates with no
# real content. Excluding it removes the main non-termination risk.
EXCLUDE_URL_PREFIXES: tuple[str, ...] = ("/en/",)

# Hosts we NEVER crawl. eservices.heraklion.gr is the e-services portal: the
# upstream chatbot handles e-service intents directly, so this retriever's corpus
# must never contain files from it. The strict-domain policy already blocks this
# subdomain; this is an explicit, documented second guarantee.
EXCLUDE_HOSTS: frozenset[str] = frozenset({"eservices.heraklion.gr"})

# --- Termination safeguards: the crawler ALWAYS halts, even on an unknown trap.
# 1) Productivity stop (adaptive): over the last PRODUCTIVITY_WINDOW processed
#    pages, if more than MAX_USELESS_FRACTION were "unproductive" (empty page or
#    error), we've wandered into a junk zone and stop. A healthy re-crawl is
#    mostly 'unchanged' (productive), so it never trips this.
# 2) Hard ceiling (guarantee): stop discovering new URLs once we've seen MAX_SEEN
#    of them — set well above the real site so it only ever catches runaways.
PRODUCTIVITY_WINDOW  = 1000
MAX_USELESS_FRACTION = 0.70
MAX_SEEN             = 80000

MANIFEST_PATH   = STATE_DIR / "manifest.json"
DEAD_LINKS_PATH = STATE_DIR / "dead_links.json"
PDF_CACHE_DIR   = STATE_DIR / "pdf_cache"    # extracted PDF text cache

TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign",
                   "utm_term", "utm_content", "fbclid", "gclid", "ref",
                   # Calendar/agenda date params generate endless URL variants
                   # of the SAME page — drop them so they collapse to one URL.
                   "day", "month", "year",
                   # Cache-buster: the site stamps a random ?rnd=<n> on links,
                   # so every link looks like a new URL — an infinite trap.
                   # Stripping it collapses all variants to the real URL.
                   "rnd"}

# URL paths that are infinite "crawler traps": the agenda calendar exposes a
# page per day (/agenda/YYYY/M/D), so the frontier never empties. These paths
# are never enqueued. Real event pages (/agenda/<slug>/<id>.html) still are.
TRAP_PATH_RE = re.compile(r"^/agenda/\d{4}/\d{1,2}(/\d{1,2})?/?$")

# Broken relative links (e.g. href="en/culture" instead of "/en/culture") make
# each hop append path segments — /en/.../en/... — an ever-deepening loop of
# empty pages. Real pages here are shallow with unique segments, so we treat a
# runaway-deep path, or a repeated segment in a deep path, as a trap.
MAX_URL_PATH_DEPTH = 8

PDF_EXTS  = (".pdf",)
SKIP_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
             ".zip", ".rar", ".7z", ".mp3", ".mp4", ".avi", ".css", ".js",
             ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
             ".xml", ".odt", ".ods", ".odp", ".rtf", ".txt", ".csv")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("crawler")


# --------------------------------------------------------------------------
# URL handling — STRICT domain policy
# --------------------------------------------------------------------------
def _host(url: str) -> str:
    return urlparse(url).netloc.lower()


def allowed_hosts_for(seed: str) -> set[str]:
    """Exactly the seed host, in bare and www. form. Nothing else."""
    bare = _host(seed).removeprefix("www.")
    return {bare, "www." + bare}


def is_internal(url: str, allowed: set[str]) -> bool:
    return _host(url) in allowed


def normalize_url(url: str) -> str:
    p = urlparse(url)
    query = urlencode(sorted(
        (k, v) for k, v in parse_qsl(p.query) if k not in TRACKING_PARAMS
    ))
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc.lower(), path, "", query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]


def path_ext(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def section_of(url: str) -> str:
    parts = [s for s in urlparse(url).path.split("/") if s]
    return parts[0] if parts else "root"


def is_trap(url: str) -> bool:
    """True for infinite/junk URLs we must never enqueue: the agenda calendar,
    runaway-deep paths, or a repeated segment in a deep path (the tell-tale sign
    of a relative-link accumulation loop, e.g. /en/.../en/...)."""
    path = urlparse(url).path
    if TRAP_PATH_RE.match(path):
        return True
    segs = [s for s in path.split("/") if s]
    if len(segs) > MAX_URL_PATH_DEPTH:
        return True
    if len(segs) >= 4 and len(segs) != len(set(segs)):
        return True
    return False


def url_excluded(url: str) -> bool:
    """True for hosts/path-prefixes we never crawl (the e-services portal, the
    broken /en/ mirror)."""
    p = urlparse(url)
    if p.netloc.lower() in EXCLUDE_HOSTS:
        return True
    return any(p.path == pref.rstrip("/") or p.path.startswith(pref)
               for pref in EXCLUDE_URL_PREFIXES)


# --------------------------------------------------------------------------
# Filenames
# --------------------------------------------------------------------------
_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

def filename_for(title: str, url: str) -> str:
    name = title.strip() or urlparse(url).path.strip("/") or "index"
    name = _INVALID_FS_CHARS.sub("", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    name = name[:MAX_TITLE_LEN] or "untitled"
    return f"{name}__{url_hash(url)}.json"


# --------------------------------------------------------------------------
# Manifest (crawl state for incremental re-crawls)
# --------------------------------------------------------------------------
def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Manifest corrupted — starting fresh")
    return {"pages": {}, "pdfs": {}}


def save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


def conditional_headers(entry: dict | None) -> dict:
    h = {}
    if entry:
        if entry.get("etag"):
            h["If-None-Match"] = entry["etag"]
        if entry.get("last_modified"):
            h["If-Modified-Since"] = entry["last_modified"]
    return h


def remember_validators(entry: dict, resp: requests.Response):
    entry["etag"] = resp.headers.get("ETag")
    entry["last_modified"] = resp.headers.get("Last-Modified")
    entry["last_fetch"] = datetime.now(timezone.utc).isoformat()


def text_hash(title: str, body: str) -> str:
    """Per-URL change-detection hash of the EXTRACTED text — stable across
    dynamic HTML noise (rotating banners, per-request tokens). Includes the
    title so a title/section change is noticed and the doc is re-saved."""
    return hashlib.sha1((title + "\n" + body).encode("utf-8")).hexdigest()


# Below this body length we don't trust body-only equality — fall back to exact
# title+body matching. Keep in sync with dedup_corpus.py.
DEDUP_MIN_BODY = 200


def dedup_key(title: str, body: str) -> str:
    """CROSS-URL content key. Body-only for substantial pages (so the same
    article under a different section label collapses), else exact title+body
    for short stubs. Must match dedup_corpus.py's dedup_key."""
    body = (body or "").strip()
    if len(body) >= DEDUP_MIN_BODY:
        return hashlib.sha1(body.encode("utf-8")).hexdigest()
    return hashlib.sha1(((title or "") + "\n" + body).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# robots.txt
# --------------------------------------------------------------------------
def load_robots(seed: str) -> RobotFileParser:
    rp = RobotFileParser()
    robots_url = urljoin(seed, "/robots.txt")
    try:
        rp.set_url(robots_url)
        rp.read()
        log.info("Loaded robots.txt from %s", robots_url)
    except Exception as e:
        log.warning("Could not read robots.txt (%s) — assuming allow-all", e)
        rp.parse([])
    return rp


# --------------------------------------------------------------------------
# Publish-date extraction
# --------------------------------------------------------------------------
# Press releases carry a Greek dateline like "Ηράκλειο 25-11-2022" (sometimes
# with spaces/commas); that's the most reliable source of the RELEASE date.
_DATELINE_RE = re.compile(
    r"(?:Ηρ[άα]κλειο|Ημερομην[ίι]α)[^\d]{0,15}"
    r"(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(20\d\d)")
_URLYEAR_RE = re.compile(r"(?:press-releases-|announcements-|/)(20[0-2]\d)(?:/|\b)")


def parse_publish_date(body: str, url: str, meta_date: str | None) -> str | None:
    """Best-effort article RELEASE date as 'YYYY-MM-DD' (or 'YYYY' if only a year
    is known), else None. Priority: the Greek dateline in the body ->
    trafilatura's metadata date -> a year in the URL."""
    from datetime import date
    m = _DATELINE_RE.search(body or "")
    if m:
        d, mo, y = (int(x) for x in m.groups())
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            pass
    if meta_date and re.match(r"20\d\d-\d\d-\d\d", meta_date):
        return meta_date[:10]
    m = _URLYEAR_RE.search(url)
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# Content extraction (boilerplate removal)
# --------------------------------------------------------------------------
def extract_content(html: str, url: str) -> tuple[str, str, str | None]:
    """Return (title, body, published). Body is main content only; published is
    the article's release date (YYYY-MM-DD / YYYY / None)."""
    title = ""
    body = ""
    meta_date = None

    if HAS_TRAFILATURA and trafilatura is not None:
        body = trafilatura.extract(
            html, url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        ) or ""
        meta = trafilatura.extract_metadata(html)
        if meta:
            if meta.title:
                title = meta.title
            meta_date = getattr(meta, "date", None)

    soup = BeautifulSoup(html, "html.parser")
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    if not body:
        for tag in soup(["script", "style", "nav", "header",
                         "footer", "aside", "form", "noscript"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body
        if main:
            body = main.get_text(separator="\n", strip=True)

    return title, body, parse_publish_date(body, url, meta_date)


def extract_links(html: str, base_url: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (page_links, file_links). file_links are (url, anchor) pairs."""
    soup = BeautifulSoup(html, "html.parser")
    page_links: list[str] = []
    file_links: list[tuple[str, str]] = []
    seen_files: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = str(a.get("href", "")).strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = urljoin(base_url, href)
        if urlparse(absolute).scheme not in ("http", "https"):
            continue
        absolute = normalize_url(absolute)
        ext = path_ext(absolute)

        if ext in PDF_EXTS:
            if absolute not in seen_files:
                seen_files.add(absolute)
                anchor = a.get_text(" ", strip=True)
                anchor = re.sub(r"\s*-\s*[\d.,]+\s*(KB|MB|GB)\s*$", "",
                                anchor, flags=re.IGNORECASE)
                file_links.append((absolute, anchor))
        else:
            page_links.append(absolute)

    return page_links, file_links


def file_title_fallback(url: str) -> str:
    name = unquote(Path(urlparse(url).path).stem)
    return name.replace("_", " ").replace("-", " ").strip() or url


# --------------------------------------------------------------------------
# PDF handling
# --------------------------------------------------------------------------
def extract_pdf_text(data: bytes) -> str:
    if not HAS_PYPDF or PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages).strip()
    except Exception as e:
        log.warning("PDF parse failed (%s)", type(e).__name__)
        return ""


# Per-thread requests.Session for parallel PDF workers — a Session isn't
# guaranteed thread-safe, so each worker thread gets its own.
_thread_local = threading.local()


def worker_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        _thread_local.session = s
    return s


def _pdf_cache_file(url: str) -> Path:
    return PDF_CACHE_DIR / f"{url_hash(url)}.txt"


def _pdf_cache_fresh(entry: dict) -> bool:
    """Can we trust the cached PDF text without re-fetching?
    Yes if we've extracted this URL before (entry has 'chars') and either we
    never revalidate, or the cache is younger than REVALIDATE_PDF_AFTER_DAYS."""
    if "chars" not in entry:
        return False                     # never successfully parsed before
    if REVALIDATE_PDF_AFTER_DAYS is None:
        return True                      # trust cache forever
    ts = entry.get("last_fetch")
    if not ts:
        return False
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(ts)
    except ValueError:
        return False
    return age.days < REVALIDATE_PDF_AFTER_DAYS


def get_pdf_text(url: str, session: requests.Session, manifest: dict) -> str:
    """Extracted text for a PDF URL. Cache-first: a PDF we've already parsed is
    reused from disk with NO network call (PDFs are immutable at a URL). Only
    brand-new PDF URLs (or stale ones, if REVALIDATE_PDF_AFTER_DAYS is set) are
    downloaded and parsed. Known-empty (scanned) PDFs return "" instantly."""
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = manifest["pdfs"].get(url, {})
    cache_file = _pdf_cache_file(url)

    # --- FAST PATH: trusted cache, no network, no politeness delay ----------
    if entry and cache_file.exists() and _pdf_cache_fresh(entry):
        try:
            return cache_file.read_text(encoding="utf-8")
        except OSError:
            pass  # unreadable cache — fall through and re-fetch

    try:
        resp = session.get(url, timeout=TIMEOUT,
                           headers=conditional_headers(entry))
    except requests.RequestException as e:
        log.warning("PDF fetch failed (%s): %s", type(e).__name__, url)
        time.sleep(CRAWL_DELAY)
        return cache_file.read_text(encoding="utf-8") if cache_file.exists() else ""

    if resp.status_code == 304 and cache_file.exists():
        time.sleep(FAST_DELAY)
        return cache_file.read_text(encoding="utf-8")

    time.sleep(CRAWL_DELAY)

    if resp.status_code != 200:
        log.info("PDF skipped (HTTP %d): %s", resp.status_code, url)
        return ""

    if len(resp.content) > MAX_PDF_MB * 1024 * 1024:
        log.info("PDF too large (>%d MB), skipped: %s", MAX_PDF_MB, url)
        return ""

    text = extract_pdf_text(resp.content)
    if not text:
        log.info("PDF yielded no text (scanned? needs OCR?): %s", url)

    cache_file.write_text(text, encoding="utf-8")
    remember_validators(entry, resp)
    entry["chars"] = len(text)
    manifest["pdfs"][url] = entry
    return text


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_document(url: str, title: str, body: str, components: list[dict],
                  published: str | None, out_dir: Path) -> Path:
    doc: dict = {
        "title": title,
        "body": body,
        "url": url,
        "section": section_of(url),
        "published": published,          # article RELEASE date (YYYY-MM-DD/YYYY)
    }
    for i, comp in enumerate(components, start=1):
        doc[f"component_file_{i}"] = comp
    doc["timestamp"] = datetime.now(timezone.utc).isoformat()  # crawl time

    path = out_dir / filename_for(title, url)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def delete_document(url: str, manifest: dict, out_dir: Path):
    entry = manifest["pages"].get(url)
    if entry and entry.get("file"):
        path = out_dir / entry["file"]
        if path.exists():
            path.unlink()
            log.info("Removed dead document: %s", entry["file"])
    manifest["pages"].pop(url, None)


# --------------------------------------------------------------------------
# Category map maintenance (baseline for daily_update.py's per-category diff)
# --------------------------------------------------------------------------
def index_item_links(page_links: list[str], category: str, index_url: str) -> list[str]:
    """From an index page's outgoing links, keep this category's item links:
    internal, same first-path-segment, not a trap, not a file. Order preserved."""
    allowed = allowed_hosts_for(index_url)
    idx = normalize_url(index_url)
    out: list[str] = []
    seen: set[str] = set()
    for link in page_links:
        if link in seen or link == idx:
            continue
        if not is_internal(link, allowed) or is_trap(link):
            continue
        if path_ext(link) in SKIP_EXTS:
            continue
        if section_of(link) != category:
            continue
        seen.add(link)
        out.append(link)
    return out


def update_category_map(manifest: dict):
    """Refresh category_state.json from the crawled index pages' stored links,
    so daily_update.py starts each day from a current baseline. Only categories
    whose index page has links in the manifest are touched; others are left
    as-is (daily_update refreshes those via a live fetch)."""
    state = category_map.load_state()
    updated = 0
    for category, index_url in category_map.CATEGORY_INDEXES.items():
        entry = manifest["pages"].get(normalize_url(index_url))
        if not entry or "links" not in entry:
            log.info("Category map: no crawled links for '%s' (%s) — left as-is",
                     category, index_url)
            continue
        links = index_item_links(entry["links"], category, index_url)
        state[category] = {"index_url": index_url, "links": links}
        updated += 1
        log.info("Category map: '%s' -> %d item links", category, len(links))
    if updated:
        category_map.save_state(state)
        log.info("Category map saved (%d categories) to %s",
                 updated, category_map.STATE_PATH)


# --------------------------------------------------------------------------
# Main crawl loop
# --------------------------------------------------------------------------
def crawl(seeds, out_dir: Path):
    """Breadth-first crawl of the site, modelled as a tree walk from the root.

    Mental model:
      * ROOT & BRANCHES: the seed (homepage) is the root. `frontier` is a FIFO
        queue, so popleft()/append() walk the site level by level, root outward
        — visiting branches breadth-first. `depth` is the level of each node.
      * VISIT ONCE: `seen` ensures every page is visited once. The web is really
        a graph (a page can have many parents / cross-links), so `seen` is what
        keeps this tree-walk finite and cycle-free — we never circle back to the
        root.
      * DEAD BRANCHES: if a subtree turns out to be junk (empty/error pages), the
        productivity check QUARANTINES that path prefix — adds it to
        `dynamic_excludes` and purges it from the frontier — and the walk simply
        continues with the sibling branches already waiting in the queue. No need
        to climb back up; the siblings are already queued.
      * TERMINATION: when every reachable branch has been visited, the frontier
        drains to empty and the loop ends — "back at the root, nothing left to
        visit". Three independent guarantees ensure it ALWAYS halts: the trap
        filters (is_trap / url_excluded), the productivity quarantine-or-stop,
        and the hard MAX_SEEN ceiling.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(seeds, str):
        seeds = [seeds]
    seeds = [normalize_url(s) for s in seeds]
    allowed = allowed_hosts_for(seeds[0])
    log.info("Allowed hosts (strict): %s", sorted(allowed))
    log.info("Seeds (%d): %s", len(seeds), seeds)
    if MAX_DEPTH is not None:
        log.info("Depth-limited crawl: MAX_DEPTH=%d (targeted refresh)", MAX_DEPTH)

    robots = load_robots(seeds[0])
    manifest = load_manifest()
    log.info("Manifest loaded: %d known pages, %d known PDFs",
             len(manifest["pages"]), len(manifest["pdfs"]))
    dead_links: dict[str, str] = {}

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    delay = CRAWL_DELAY
    rd = robots.crawl_delay(USER_AGENT)
    try:
        rd = float(rd) if rd is not None else None
    except (TypeError, ValueError):
        rd = None
    if rd is not None and rd > delay:
        delay = rd
        log.info("Using robots.txt crawl-delay: %.1fs", delay)

    # Cross-URL content dedup: dedup_key(text) -> first URL that had it.
    # Seeded from the "chash" field (written by saves and by dedup_corpus.py)
    # so a re-crawl won't recreate duplicates we already collapsed.
    content_index: dict[str, str] = {
        e["chash"]: (e.get("alias_of") or u)
        for u, e in manifest["pages"].items()
        if e.get("chash")
    }
    log.info("Content index seeded with %d known contents", len(content_index))

    frontier = deque((s, 0) for s in seeds)
    seen = set(seeds)
    saved = unchanged = denied = skipped = errors = processed = 0
    duplicates = 0
    # Termination bookkeeping.
    recent = deque(maxlen=PRODUCTIVITY_WINDOW)  # (unproductive_bit, url) per page
    dynamic_excludes: set[str] = set()          # trap subtrees quarantined at runtime
    should_stop = False
    ceiling_logged = False

    def enqueue_links(links: list[str], depth: int):
        nonlocal ceiling_logged
        for link in links:
            # Re-normalize here too: links can come from the manifest (stored by
            # older crawls) and may still carry cache-buster/tracking params
            # (?rnd=, day/month/year). Stripping them now collapses those
            # variants to the real URL and stops frontier explosions.
            link = normalize_url(link)
            if url_excluded(link):     # e.g. the broken /en/ mirror
                continue
            lpath = urlparse(link).path
            if any(lpath.startswith(p) for p in dynamic_excludes):
                continue               # subtree quarantined mid-crawl as a trap
            # Hard ceiling: once we've discovered MAX_SEEN URLs, stop queuing new
            # ones. The frontier then drains to empty — guaranteed termination.
            if MAX_SEEN is not None and len(seen) >= MAX_SEEN:
                if not ceiling_logged:
                    log.warning("Safety ceiling hit: seen=%d — no more new URLs "
                                "will be queued (draining frontier).", MAX_SEEN)
                    ceiling_logged = True
                return
            if link in seen or not is_internal(link, allowed):
                continue
            if path_ext(link) in SKIP_EXTS or path_ext(link) in PDF_EXTS:
                continue
            if is_trap(link):          # infinite calendar URLs — never crawl
                continue
            seen.add(link)
            frontier.append((link, depth + 1))

    while frontier and (MAX_PAGES is None or saved < MAX_PAGES):
        url, depth = frontier.popleft()
        useless = False   # set True on empty/error outcomes (productivity signal)

        if not robots.can_fetch(USER_AGENT, url):
            log.info("robots.txt disallows: %s", url)
            denied += 1
            continue

        page_entry = manifest["pages"].get(url, {})
        wait = delay                     # default politeness after this URL

        try:
            try:
                resp = session.get(url, timeout=TIMEOUT,
                                   headers=conditional_headers(page_entry))
            except requests.RequestException as e:
                log.warning("Fetch failed (%s): %s", type(e).__name__, url)
                errors += 1
                useless = True
                continue

            # --- redirect guard: never leave the domain -------------------
            final_url = normalize_url(resp.url)
            if not is_internal(final_url, allowed):
                log.info("Redirected outside domain, dropped: %s -> %s",
                         url, final_url)
                skipped += 1
                continue

            # --- unchanged since last crawl (HTTP 304) ---------------------
            if resp.status_code == 304:
                unchanged += 1
                wait = FAST_DELAY
                log.info("Unchanged [%d] (304) %s", unchanged, url)
                if MAX_DEPTH is None or depth < MAX_DEPTH:
                    enqueue_links(page_entry.get("links", []), depth)
                continue

            # --- permission / error handling -------------------------------
            if resp.status_code in (401, 403):
                log.info("Permission denied (%d): %s", resp.status_code, url)
                denied += 1
                continue
            if resp.status_code == 429:
                log.warning("Rate limited (429) — backing off 30s: %s", url)
                time.sleep(30)
                frontier.append((url, depth))
                continue
            if resp.status_code in (404, 410):
                log.info("Dead link (HTTP %d): %s", resp.status_code, url)
                dead_links[url] = f"HTTP {resp.status_code}"
                delete_document(url, manifest, out_dir)
                errors += 1
                useless = True
                continue
            if resp.status_code != 200:
                log.info("Skipping (HTTP %d): %s", resp.status_code, url)
                errors += 1
                useless = True
                continue

            ctype = resp.headers.get("Content-Type", "")
            if "text/html" not in ctype:
                skipped += 1
                useless = True
                continue

            # --- extract, then compare TEXT hash ---------------------------
            title, body, published = extract_content(resp.text, url)
            page_links, file_links = extract_links(resp.text, url)
            file_links = [(u, t) for (u, t) in file_links
                          if is_internal(u, allowed)]
            h = text_hash(title, body)
            current_pdfs = [u for u, _ in file_links]

            # Truly unchanged only if BOTH the article text AND the set of
            # attached PDFs match. A PDF added to (or removed from) an existing
            # page usually leaves the text untouched, so without the pdf-set
            # check the new attachment would be silently missed. If the PDFs
            # differ we fall through and re-save (get_pdf_text fetches just the
            # new PDF; cached ones are free).
            if (page_entry.get("hash") == h
                    and set(current_pdfs) == set(page_entry.get("pdfs", []))):
                unchanged += 1
                wait = FAST_DELAY
                log.info("Unchanged [%d] (text) %s", unchanged, url)
                remember_validators(page_entry, resp)
                page_entry["links"] = page_links
                manifest["pages"][url] = page_entry
                if MAX_DEPTH is None or depth < MAX_DEPTH:
                    enqueue_links(page_links, depth)
                continue

            # --- cross-URL content dedup -----------------------------------
            # Same content already saved under a different URL? Record an alias
            # and skip: no second file, no PDF re-download. Still follow links.
            dk = dedup_key(title, body)
            canonical = content_index.get(dk)
            if canonical is not None and canonical != url:
                duplicates += 1
                wait = FAST_DELAY          # no save/PDF work — use the fast delay
                log.info("Duplicate of %s — aliased, not saved: %s",
                         canonical, url)
                alias_entry = {"alias_of": canonical, "hash": h, "chash": dk}
                remember_validators(alias_entry, resp)
                manifest["pages"][url] = alias_entry
                if MAX_DEPTH is None or depth < MAX_DEPTH:
                    enqueue_links(page_links, depth)
                continue

            # --- build component_file_N entries -----------------------------
            # get_pdf_text is cache-first, so known PDFs cost only a file read;
            # only genuinely new PDF URLs hit the network + parser.
            def build_component(pair):
                file_url, anchor = pair
                sess = worker_session() if PDF_WORKERS > 1 else session
                return {
                    "title": anchor or file_title_fallback(file_url),
                    "body": get_pdf_text(file_url, sess, manifest),
                    "url": file_url,
                }

            if PDF_WORKERS > 1 and len(file_links) > 1:
                with ThreadPoolExecutor(max_workers=PDF_WORKERS) as ex:
                    components = list(ex.map(build_component, file_links))
            else:
                components = [build_component(p) for p in file_links]

            # --- keep only "valid" documents --------------------------------
            total_chars = len(body) + sum(len(c["body"]) for c in components)
            if total_chars < MIN_BODY_CHARS:
                log.info("Too little content, not saving: %s", url)
                skipped += 1
                useless = True
            else:
                path = save_document(url, title, body, components, published, out_dir)
                saved += 1
                page_entry.update({
                    "file": path.name,
                    "hash": h,
                    "chash": dk,
                    "links": page_links,
                    "pdfs": [u for u, _ in file_links],
                })
                remember_validators(page_entry, resp)
                manifest["pages"][url] = page_entry
                content_index[dk] = url       # remember for cross-URL dedup
                log.info("Saved [%d] %s (files: %d) -> %s",
                         saved, url, len(components), path.name)

            if MAX_DEPTH is None or depth < MAX_DEPTH:
                enqueue_links(page_links, depth)

        finally:
            time.sleep(wait)
            processed += 1
            recent.append((1 if useless else 0, url))
            if processed % PROGRESS_EVERY == 0:
                log.info(
                    "\n"
                    "+==================== PROGRESS ====================+\n"
                    "|  processed %-8d saved %-8d unchanged %-8d\n"
                    "|  duplicates %-7d errors %-8d\n"
                    "|  frontier %-9d seen %-9d\n"
                    "+=================================================+",
                    processed, saved, unchanged, duplicates, errors,
                    len(frontier), len(seen))
                save_manifest(manifest)

        # Productivity check: a full window dominated by unproductive pages
        # (empty/error) means we're in a junk zone. Rather than abandon the whole
        # crawl, QUARANTINE the offending path subtree (the common source of the
        # junk) and keep crawling everything else. Only halt if the junk has no
        # common source (scattered) — nothing a single quarantine could fix.
        if (len(recent) == PRODUCTIVITY_WINDOW
                and sum(b for b, _ in recent) >= MAX_USELESS_FRACTION * PRODUCTIVITY_WINDOW):
            junk = [u for b, u in recent if b]

            def _subtree(u: str) -> str:      # first two path segments
                segs = [s for s in urlparse(u).path.split("/") if s][:2]
                return "/" + "/".join(segs)

            prefix, hits = Counter(_subtree(u) for u in junk).most_common(1)[0]
            if prefix.count("/") >= 2 and hits >= 0.5 * len(junk):
                dynamic_excludes.add(prefix)
                kept = [(u, d) for (u, d) in frontier
                        if not urlparse(u).path.startswith(prefix)]
                purged = len(frontier) - len(kept)
                frontier.clear()
                frontier.extend(kept)
                recent.clear()            # fresh window after removing the source
                log.warning("Quarantined trap subtree %s (%d/%d recent junk; "
                            "purged %d queued URLs) — continuing crawl.",
                            prefix, hits, len(junk), purged)
            else:
                log.warning("Productivity stop: %d of the last %d pages were "
                            "unproductive with no common source — halting.",
                            len(junk), PRODUCTIVITY_WINDOW)
                should_stop = True

        if should_stop:
            break

    # --- final reports -------------------------------------------------------
    save_manifest(manifest)
    update_category_map(manifest)   # refresh daily_update.py's per-category baseline
    if dead_links:
        DEAD_LINKS_PATH.write_text(
            json.dumps(dead_links, ensure_ascii=False, indent=2),
            encoding="utf-8")
        log.info("Dead links report: %s (%d entries)",
                 DEAD_LINKS_PATH, len(dead_links))

    log.info("Done. saved=%d unchanged=%d duplicates=%d denied=%d skipped=%d "
             "errors=%d frontier_left=%d", saved, unchanged, duplicates,
             denied, skipped, errors, len(frontier))


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="Polite incremental crawler for the RAG corpus.",
        epilog="Daily targeted refresh example:\n"
               "  python crawler.py --seeds /press/municipality-press-releases "
               "/agenda --max-depth 2\n"
               "Full monthly crawl:\n"
               "  python crawler.py",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", metavar="URL",
                    help="Start URLs or site paths (e.g. /press). "
                         "Default: the site homepage (full crawl).")
    ap.add_argument("--max-depth", type=int, metavar="N",
                    help="Limit crawl to N link-hops from the seeds. Use a small "
                         "value (1-2) for a fast targeted refresh of live "
                         "sections. Default: unlimited (full crawl).")
    ap.add_argument("--max-pages", type=int, metavar="N",
                    help="Stop after saving N pages (safety cap).")
    ap.add_argument("--pdf-workers", type=int, metavar="N",
                    help="Fetch a page's PDFs with N parallel workers (cold "
                         "crawls). Default: 1 (sequential, polite).")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.max_depth is not None:
        MAX_DEPTH = args.max_depth
    if args.max_pages is not None:
        MAX_PAGES = args.max_pages
    if args.pdf_workers is not None:
        PDF_WORKERS = args.pdf_workers
    # Resolve bare paths (/press) to full URLs against the site homepage.
    if args.seeds:
        seeds = [s if "://" in s else urljoin(SEED_URL, s) for s in args.seeds]
    else:
        seeds = [SEED_URL]
    crawl(seeds, OUTPUT_DIR)