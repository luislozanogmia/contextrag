#!/usr/bin/env python3
"""
contextrag.py – enhanced web RAG utilities (standalone, stdlib-only)

Enhanced for exact fact retrieval with improved extraction and ranking.

Subcommands:
  fetch       Fetch a URL, respect robots.txt by default, return raw bytes.
  extract     Convert HTML to plain, readable text with infobox extraction.
  snippetize  Turn plain text into verbatim sentence snippets, saved as JSONL.
  search      Enhanced BM25 search with phrase/numeric boosting.
  compose     Compose evidence block from top-k snippets.
  ask         Pipe composed block to local LLM runner.

No databases. No external deps. Verbatim only. Enhanced for factual precision.
"""
from __future__ import annotations

import argparse
import subprocess
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import sys
import urllib.parse
import urllib.request
import urllib.error
import urllib.robotparser
from dataclasses import dataclass
from html.parser import HTMLParser
from html import unescape
from typing import Dict, Iterable, List, Optional, Tuple, Set
import math
import html as _py_html
from urllib.parse import urlparse, parse_qs, unquote, urljoin

# ----------------------------- constants -----------------------------
# Default to a local 'data' directory for portability
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_UA = "ContextRAG-Agent/0.2 (+https://github.com/yourusername/contextrag)"
REQUEST_TIMEOUT = 20

REF_TOKENS = {"retrieved", "doi", "isbn", "pmid", "s2cid", "bibcode", "arxiv", "cite", "citation"}
INFOBOX_INDICATORS = {
    "infobox", "vevent", "vcard", "geography", "planet", "biota",
    "geobox", "settlement", "biography", "taxobox", "speciesbox",
    "box", "summary", "keyfacts", "data", "info", "facts"
}


# Enhanced stopwords for better BM25 scoring
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these", "those", "i",
    "me", "my", "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "they", "them", "their"
}

IGNORE_CLASS_BITS = {
"navbox", "vertical-navbox", "sidebar", "toc", "hatnote",
"ambox", "tmbox", "ombox", "cmbox", "metadata", "sister",
"portal", "infobox-subbox", "reflist", "noprint"
}
def _has_ignored_class(attrs):
    for name, value in attrs:
        if name and name.lower() in ("class", "id") and value:
            low = value.lower()
            if any(bit in low for bit in IGNORE_CLASS_BITS):
                return True
    return False

# --- SERP helpers (Bing) -----------------------------------------------


_BING_HOSTS = {"bing.com", "www.bing.com"}

def _host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _is_bing_serp(url: str) -> bool:
    """True if URL looks like a Bing web search results page."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.netloc.lower() not in _BING_HOSTS:
        return False
    # Bing SERP is typically /search?q=...
    if p.path.strip("/") != "search":
        return False
    q = parse_qs(p.query)
    return "q" in q and len(q["q"]) > 0

def _extract_bing_result_urls(html: bytes, base_url: str) -> list[str]:
    """
    Extract destination (non-SERP) links from a Bing results page.
    Strategy:
      1) Preferred: /ck/a?...&u=<ENCODED_DEST>&...
      2) Fallback: visible organic anchors inside result blocks (<li class="b_algo">).
    We purposely avoid ads/sidebars and internal bing links.
    """
    text = html.decode("utf-8", errors="ignore")

    urls: list[str] = []

    # 1) Bing redirector pattern: href="/ck/a?...&u=<ENCODED>&..."
    for m in re.finditer(r'href="(/ck/a\?[^""]+)"', text):
        href = _py_html.unescape(m.group(1))
        abs_href = urljoin(base_url, href)
        try:
            qp = parse_qs(urlparse(abs_href).query)
            uvals = qp.get("u", []) or qp.get("UL", [])  # sometimes capitalized
            if uvals:
                dest = unquote(uvals[0])
                # Filter internal bing stuff
                if "bing.com" not in urlparse(dest).netloc.lower():
                    urls.append(dest)
        except Exception:
            continue

    # 2) Fallback: anchors inside organic results blocks
    # We look for <li class="b_algo"> ... <a href="http...">
    # This is intentionally broad but excludes most internal links.
    if not urls:
        for li in re.finditer(r'<li[^>]+class="[^""]*\bb_algo\b[^""]*"[^>]*>(.*?)</li>', text, re.DOTALL | re.IGNORECASE):
            chunk = li.group(1)
            for am in re.finditer(r'<a[^>]+href="([^""]+)"', chunk, re.IGNORECASE):
                ahref = _py_html.unescape(am.group(1))
                if not ahref.startswith("http"):
                    ahref = urljoin(base_url, ahref)
                try:
                    h = urlparse(ahref).netloc.lower()
                    if h and "bing.com" not in h:
                        urls.append(ahref)
                except Exception:
                    continue

    # De-dup while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _is_serp_url(url: str) -> bool:
    """Unified SERP gate. For now we only allow Bing (Google caused blocks)."""
    return _is_bing_serp(url)

def _resolve_serp_destinations(session_get, url: str, headers: dict[str, str] | None = None) -> list[str]:
    """
    Fetch the SERP HTML (with the provided session_get method) and extract destination URLs.
    session_get should be a callable like: lambda url, **kw: requests.get(url, **kw)
    """
    resp = session_get(url, headers=headers or {})
    resp.raise_for_status()
    html = resp.content
    if _is_bing_serp(url):
        return _extract_bing_result_urls(html, url)
    # Default: nothing
    return []
# ----------------------------- small utils -----------------------------

def iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters for better matching (ASCII-safe mapping)."""
    replacements = {
        "\u00D7": "x",      # × multiplication sign
        "\u2212": "-",      # − minus sign
        "\u2013": "-",      # – en dash
        "\u2014": "-",      # — em dash
        "\u201C": '"',      # " left double quote
        "\u201D": '"',      # " right double quote
        "\u2018": "'",      # ' left single quote
        "\u2019": "'",      # ' right single quote
        "\u2026": "...",    # … ellipsis
        # superscripts → caret notation
        "\u2070": "^0", "\u00B9": "^1", "\u00B2": "^2", "\u00B3": "^3",
        "\u2074": "^4", "\u2075": "^5", "\u2076": "^6", "\u2077": "^7",
        "\u2078": "^8", "\u2079": "^9",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # normalize spaced scientific notation: "1.2 × 10 ^ 6" -> "1.2 x 10^6"
    text = re.sub(r"(\d(?:\.\d+)?)\s*(?:x|\u00D7)\s*10\s*\^?\s*(\d+)", r"\1 x 10^\2", text)
    return text


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def domain_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()

# ----------------------------- Generic Search Engine Utilities -----------------------------

def _fetch_search_page(url: str, headers: Dict[str, str]) -> str:
    """Generic function to fetch any search engine page."""
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        raw = r.read()
        ce = (r.headers.get("Content-Encoding") or "").lower()
        if "gzip" in ce:
            import gzip
            with contextlib.suppress(Exception):
                raw = gzip.decompress(raw)
        return raw.decode("utf-8", "ignore")

def _save_debug_html(html: str, engine: str, query: str) -> None:
    """Save debug HTML for any search engine (logs to local debug directory)."""
    base_dir = os.path.join(os.getcwd(), "debug_logs")
    os.makedirs(base_dir, exist_ok=True)
    safe_query = re.sub(r"[^a-zA-Z0-9_\-]+", "_", query.strip())[:100]
    debug_file = os.path.join(base_dir, f"debug_serp_{engine}_{safe_query}.html")
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[ContextRAG] Debug SERP HTML saved → {debug_file}", file=sys.stderr)

def _dedupe_urls(urls: List[str], max_results: int) -> List[str]:
    """Generic URL deduplication and limiting."""
    seen = set()
    result = []
    for url in urls:
        if url not in seen and len(result) < max_results:
            seen.add(url)
            result.append(url)
    return result

# ----------------------------- Bing-Specific Functions -----------------------------

def _strip_bing_redirect(url: str) -> str:
    """Remove Bing's /ck/a redirect wrapper; extract true URL from 'u=' param if present."""
    if "/ck/a" in url or "/ck/a?!" in url:
        try:
            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            if 'u' in params and params['u']:
                return urllib.parse.unquote(params['u'][0])
        except Exception:
            pass
    return url

def _extract_bing_urls(html: str, max_results: int = 8) -> List[str]:
    """Extract organic result URLs from Bing SERP HTML."""
    from html import unescape
    
    links: List[str] = []
    
    # H2 anchor targets are typical organic links
    h2_link_pat = re.compile(
        r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>.*?</a>\s*</h2>',
        flags=re.I | re.S
    )
    # Fallback: any anchor in b_algo blocks
    algo_link_pat = re.compile(
        r'<li[^>]+class="[^""]*\bb_algo\b[^""]*"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>',
        flags=re.I | re.S
    )
    
    for pat in (h2_link_pat, algo_link_pat):
        for m in pat.finditer(html):
            links.append(unescape(m.group(1)))
    
    # Clean Bing-specific URLs
    cleaned: List[str] = []
    bing_hosts = ("bing.com", "r.bing.com", "go.microsoft.com")
    
    for url in links:
        url = _strip_bing_redirect(url)
        if not url.startswith(("http://", "https://")):
            continue
        host = urllib.parse.urlsplit(url).netloc.lower()
        if any(bh in host for bh in bing_hosts):
            continue
        cleaned.append(url)
    
    return _dedupe_urls(cleaned, max_results)

def _search_bing(query: str, top: int = 3) -> List[str]:
    """Search Bing and extract result URLs."""
    try:
        qs = {
            "q": query,
            "setlang": "en",
            "cc": "US",
            "ensearch": "1",
        }
        serp_url = "https://www.bing.com/search?" + urllib.parse.urlencode(qs)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        }
        
        html = _fetch_search_page(serp_url, headers)
        
        # Extract results section if present
        block = html
        m = re.search(r'<ol[^>]+id=[""]b_results[""][^>]*>(.*?)</ol>', html, flags=re.I | re.S)
        if m:
            block = m.group(1)
        
        _save_debug_html(block, "bing", query)
        urls = _extract_bing_urls(block, max_results=top)
        print(f"Debug: Found {len(urls)} URLs (Bing): {urls}", file=sys.stderr)
        return urls
        
    except Exception as e:
        print(f"Bing search failed: {e}", file=sys.stderr)
        return []

# ----------------------------- DuckDuckGo-Specific Functions -----------------------------

def _extract_ddg_urls(html: str, max_results: int = 8) -> List[str]:
    """Extract organic result URLs from DuckDuckGo HTML."""
    from html import unescape
    
    links: List[str] = []
    
    # Updated pattern to match DDG's actual structure
    ddg_link_pat = re.compile(
        r'href="[^""]*?/l/\?[^""]*?uddg=([^"]+)[^""]*?"',
        flags=re.I
    )
    
    for match in ddg_link_pat.finditer(html):
        try:
            encoded_url = match.group(1)
            decoded_url = urllib.parse.unquote(encoded_url)
            if decoded_url.startswith(("http://", "https://")):
                links.append(decoded_url)
        except Exception:
            continue
    
    # Alternative pattern for result__a class links
    if not links:
        result_link_pat = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="[^""]*?uddg=([^"]+)[^""]*?"[^>]*>',
            flags=re.I
        )
        for match in result_link_pat.finditer(html):
            try:
                encoded_url = match.group(1)
                decoded_url = urllib.parse.unquote(encoded_url)
                if decoded_url.startswith(("http://", "https://")):
                    links.append(decoded_url)
            except Exception:
                continue
    
    return _dedupe_urls(links, max_results)

def _search_duckduckgo(query: str, top: int = 3) -> List[str]:
    """Search DuckDuckGo and extract result URLs."""
    try:
        qs = urllib.parse.urlencode({"q": query})
        ddg_url = f"https://html.duckduckgo.com/html/?{qs}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        html = _fetch_search_page(ddg_url, headers)
        _save_debug_html(html, "ddg", query)
        
        urls = _extract_ddg_urls(html, max_results=top)
        print(f"Debug: Found {len(urls)} URLs (DDG): {urls}", file=sys.stderr)
        return urls
        
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}", file=sys.stderr)
        return []

# ----------------------------- Main Search Resolver -----------------------------

def resolve_query_to_urls(query: str, provider: str = "bing", top: int = 3) -> List[str]:
    """Resolve search query to destination URLs using DuckDuckGo only."""
    
    # Skip Bing entirely for now
    print(f"Searching with DuckDuckGo for: {query}", file=sys.stderr)
    urls = _search_duckduckgo(query, top)
    
    if not urls:
        print("No URLs resolved from DuckDuckGo. Cannot proceed with ingestion.", file=sys.stderr)
    
    return urls

# Backward compatibility - keep old function names
extract_bing_result_urls = _extract_bing_urls  # For any existing references
resolve_query_via_ddg = _search_duckduckgo     # For any existing references

# ----------------------------- robots + fetch -----------------------------

def robots_allows(url: str, user_agent: str = DEFAULT_UA) -> bool:
    """
    Return True if fetching is allowed.
    Special-case: Wikipedia article paths (/wiki/...) and search APIs are allowed.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path or "/"

    # Allowed path for Wikipedia articles (e.g., https://en.wikipedia.org/wiki/Sun)
    if host.endswith("wikipedia.org") and path.startswith("/wiki/"):
        return True

    # NEW: Allow search API endpoints
    search_hosts = {
        "www.bing.com", "bing.com", 
        "duckduckgo.com", "html.duckduckgo.com",
        "www.google.com", "google.com"
    }
    
    search_paths = {"/search", "/html/", "/l/"}
    
    if host in search_hosts and any(path.startswith(sp) for sp in search_paths):
        return True

    # Standard robots.txt check for everything else
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urllib.parse.urljoin(base, "/robots.txt"))
    with contextlib.suppress(Exception):
        rp.read()
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True

def fetch_raw(url: str, timeout: int = REQUEST_TIMEOUT, user_agent: str = DEFAULT_UA,
              ignore_robots: bool = False) -> Tuple[int, Dict[str, str], bytes]:
    if not ignore_robots and not robots_allows(url, user_agent):
        raise PermissionError(f"Blocked by robots.txt: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        headers = {k.lower(): v for k, v in resp.headers.items()}
        data = resp.read()
    return status, headers, data

def detect_charset(headers: Dict[str, str]) -> str:
    ctype = headers.get("content-type", "")
    if "charset=" in ctype:
        cs = ctype.split("charset=", 1)[1].split(";")[0].strip().strip("\"'")
        if cs:
            return cs
    return "utf-8"

# ----------------------------- Enhanced HTML -> text extractor -----------------------------

class EnhancedHTMLExtractor(HTMLParser):
    """
    Enhanced extractor for exact fact retrieval:
    - Keep text from: title, h1..h4, p, li, dt/dd, th/td
    - Extract infobox tables as key: value lines
    - Better section detection and cleanup
    - Positional tracking for ranking boosts
    """
    
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_ignored = False
        self.in_table = False
        self.in_superscript = False
        self.superscript_content = ""
        self.table_is_infobox = False
        self.table_rows: List[List[str]] = []
        self.current_row: List[str] = []
        self.current_cell = ""
        
        self.tag_stack: List[str] = []
        self.title = ""
        self.lines: List[str] = []
        self.infobox_lines: List[str] = []
        self.position_marker = 0  # Track position for ranking
        self.ignore_stack = []
        
    def handle_starttag(self, tag, attrs):
        # ignore non-content
        if tag in ("script", "style", "noscript"):
            self.in_ignored = True
            return

        # ignore common Wikipedia chrome
        if tag in ("div", "nav", "aside", "section", "ul", "table") and _has_ignored_class(attrs):
            self.ignore_stack.append(tag)
            self.in_ignored = True
            return

        # superscripts: start collecting
        if tag == "sup":
            self.in_superscript = True
            self.superscript_content = ""
            return
        if tag in ("title", "h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "th", "td"):
            self.tag_stack.append(tag)
           
    def handle_endtag(self, tag):
        # end script/style/noscript
        if tag in ("script", "style", "noscript"):
            self.in_ignored = False
            return

        # end ignored chrome blocks (div/nav/aside/section/ul/table with ignored classes)
        if self.ignore_stack and tag == self.ignore_stack[-1]:
            self.ignore_stack.pop()
            self.in_ignored = bool(self.ignore_stack)
            return

        if tag == "sup":
            if self.in_superscript and self.superscript_content.strip():
                caret = "^" + self.superscript_content.strip()
                if self.in_table:
                    self.current_cell += " " + caret
                else:
                    self.lines.append(caret)
            self.in_superscript = False
            self.superscript_content = ""
            return

        # Pop from tag_stack when closing content tags
        if tag in ("title", "h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "th", "td"):
            if self.tag_stack and self.tag_stack[-1] == tag:
                self.tag_stack.pop()
            return

        # --- table/infobox handling ---
        if tag in ("td", "th") and self.in_table:
            cell_text = normalize_ws(self.current_cell)
            if cell_text:
                self.current_row.append(cell_text)
            self.current_cell = ""
            return

        if tag == "tr" and self.in_table:
            # keep first two cells as key/value
            if len(self.current_row) >= 2:
                self.table_rows.append(self.current_row[:2])
            self.current_row = []
            return

        if tag == "table" and self.in_table:
            if self.table_is_infobox:
                self._process_infobox_table()
            self.in_table = False
            self.table_is_infobox = False
            self.table_rows = []
            self.current_row = []
            self.current_cell = ""
            return

    def handle_data(self, data):
        if self.in_ignored:
            return
            
        text = normalize_ws(data)
        if not text:
            return
            
        # Table cell data
        if self.in_table:
            self.current_cell += " " + text
            return
        
        # Regular content
        if not self.tag_stack:
            return
            
        current = self.tag_stack[-1]
        if current == "title":
            self.title = (self.title + " " + text).strip() if self.title else text
        else:
            self.lines.append(text)
            self.position_marker += len(text)
    
    def _is_infobox_table(self, attrs: List[Tuple[str, Optional[str]]]) -> bool:
        """Detect if table is likely an infobox."""
        for name, value in attrs:
            if name.lower() in ("class", "id") and value:
                value_lower = value.lower()
                if any(indicator in value_lower for indicator in INFOBOX_INDICATORS):
                    return True
        return False
    

    
    def _process_infobox_table(self):
        """Process infobox table into key: value lines."""
        for row in self.table_rows:
            if len(row) >= 2:
                key = normalize_ws(row[0]).rstrip(":")
                value = normalize_ws(row[1])
                if key and value and len(key) < 100 and len(value) < 300:
                    # Format as key: value for easy detection
                    self.infobox_lines.append(f"{key}: {value}")
    
    def extract(self, html: str) -> Tuple[str, str, List[str]]:
        """Extract title, body text, and infobox lines."""
        self.feed(html)
        body_text = normalize_ws(" ".join(self.lines))
        return self.title or "", body_text, self.infobox_lines

def html_bytes_to_enhanced_text(raw: bytes) -> Tuple[str, str, List[str]]:
    """Enhanced extraction with infobox support."""
    try:
        html = raw.decode("utf-8", errors="ignore")
    except Exception:
        html = raw.decode("latin-1", errors="ignore")
    parser = EnhancedHTMLExtractor()
    return parser.extract(html)

# ----------------------------- Enhanced snippetizer -----------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# Enhanced patterns for data-focused extraction
MEASUREMENT_PATTERNS = [
    r"\b\w+: \s*[\d.,]+\s*(?:km|kg|m|tons?|miles?|meters?|kilometers?|Â°C|Â°F|years?|days?)\b",
    r"\b(?:diameter|radius|mass|volume|temperature|distance|height|width|depth|area|density|speed|velocity|acceleration|pressure):\s*[\d.,]+",
    r"\b[\d.,]+\s*(?:Ã—|x|\*)\s*10\^?[ -]+\s*(?:kg|km|m|tons?|years?)",  # Scientific notation
    r"\b[\d.,]+\s*(?:million|billion|trillion)\s*(?:km|kg|tons?|years?|miles?)",
    r"\b[\d.,]+\s*(?:km|miles?|meters?)\s*(?:diameter|radius|wide|long|tall|high|deep)\b",
]


def extract_measurements(text: str) -> List[str]:
    """Extract measurement-like data from text for model reasoning."""
    measurements = []
    for pattern in MEASUREMENT_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            measurement = normalize_ws(match.group())
            if len(measurement) > 5:  # Avoid tiny fragments
                measurements.append(measurement)
    return list(set(measurements))  # Deduplicate

def is_data_rich_snippet(text: str) -> bool:
    """Check if snippet contains multiple data points for reasoning."""
    measurements = extract_measurements(text)
    numbers = re.findall(r"\b\d[\d.,]*\d\b", text)  # Multi-digit numbers
    return len(measurements) >= 1 or len(numbers) >= 2

def classify_time_scope(text: str) -> str:
    """Classify snippet as past, present, or future based on linguistic markers."""
    text_lower = text.lower()
    
    # Future indicators
    future_markers = [
        "will ", "models predict", "expected to", "by the year", "forecast", 
        "projected", "anticipated", "likely to", "should expand", "plans to",
        "estimates suggest", "simulations show"
    ]
    
    # Past indicators  
    past_markers = [
        "was ", "were ", "had ", "in 19", "in 20", "previously", 
        "formerly", "used to", "once was", "historically", "ago"
    ]
    
    # Check for future markers
    if any(marker in text_lower for marker in future_markers):
        return "future"
    
    # Check for past markers
    if any(marker in text_lower for marker in past_markers):
        return "past"
    
    return "present"

def enhanced_text_to_snippets(text: str, infobox_lines: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    """
    Data-focused snippetizer for model reasoning.
    Prioritizes measurement-rich content over comparison hunting.
    Returns list of (snippet_text, snippet_type, time_scope) tuples.
    """
    snippets: List[Tuple[str, str, str]] = []
    
    # First, add infobox lines as data-rich snippets (highest priority)
    if infobox_lines:
        for line in infobox_lines:
            if len(line.strip()) >= 15:  # Lower threshold for data
                measurements = extract_measurements(line)
                snippet_type = "data" if measurements else "infobox"
                time_scope = classify_time_scope(line)
                snippets.append((normalize_unicode(line.strip()), snippet_type, time_scope))
    
    # Process main text with focus on data density
    text = text.strip()
    if not text:
        return snippets
    
    # Turn bullet markers into sentence boundaries
    text = re.sub(r"\s*[â€¢\-\u2022]\s+", ". ", text)
    
    sentences: List[str] = []
    for sent in SENT_SPLIT_RE.split(text):
        s = normalize_ws(normalize_unicode(sent))
        if not s:
            continue

        # Lower threshold for data-rich content
        if len(s) < 20:
            continue

        # Drop obvious reference/bibliography lines
        low = s.lower()
        if any(tok in low for tok in REF_TOKENS):
            continue
        if low.startswith("retrieved ") or low.startswith("^ "):
            continue

        # Prioritize data-rich sentences
        if len(s) > 240:
            # Soft split long sentences on ; or : to keep chunks <= 240 chars
            parts = re.split(r";|:\s+", s)
            accum: List[str] = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if not accum:
                    accum.append(p)
                elif len("; ".join(accum + [p])) <= 240:
                    accum.append(p)
                else:
                    sentences.append("; ".join(accum))
                    accum = [p]
            if accum:
                sentences.append("; ".join(accum))
        else:
            sentences.append(s)

    # Add sentences with data-focused classification and time scope
    for sent in sentences:
        measurements = extract_measurements(sent)
        time_scope = classify_time_scope(sent)
        
        if measurements:
            snippet_type = "data"
        elif is_data_rich_snippet(sent):
            snippet_type = "numeric" 
        else:
            snippet_type = "text"
        snippets.append((sent, snippet_type, time_scope))
    
    return snippets

# ----------------------------- JSONL helpers -----------------------------
def append_enhanced_snippets_jsonl(snippets: List[Tuple[str, str, str]], source_url: str, source_title: str,
                                 to_file: Optional[str] = None) -> Tuple[int, str]:
    """Enhanced JSONL storage with snippet types and time scope."""
    dom = domain_of(source_url)
    if not to_file:
        to_file = os.path.join(DATA_DIR, f"{dom}.jsonl")
    ensure_parent_dir(to_file)
    
    # Dedupe-by-hash within this file
    existing = set()
    if os.path.exists(to_file):
        with open(to_file, "r", encoding="utf-8") as f:
            for line in f:
                with contextlib.suppress(Exception):
                    obj = json.loads(line)
                    existing.add(obj.get("quote_hash", ""))
    
    added = 0
    with open(to_file, "a", encoding="utf-8") as f:
        for snippet_text, snippet_type, time_scope in snippets:
            qh = sha1(snippet_text + "|" + source_url)
            if qh in existing:
                continue
            rec = {
                "text": snippet_text,
                "source_url": source_url,
                "source_title": source_title,
                "domain": dom,
                "snippet_type": snippet_type,
                "time_scope": time_scope,
                "fetched_at": iso_now(),
                "quote_hash": qh,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing.add(qh)
            added += 1
    
    return added, to_file

# ----------------------------- Enhanced BM25 search with boosting -----------------------------

def tokenize(text: str) -> List[str]:
    """Enhanced tokenization with stopword removal."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in STOPWORDS]
    return tokens

def extract_quoted_phrases(query: str) -> List[str]:
    """Extract quoted phrases from query for exact matching."""
    phrases = re.findall(r'"([^"]+)"', query)
    return [p.lower() for p in phrases if len(p.strip()) > 2]

def contains_phrase(text: str, phrase: str) -> bool:
    """Check if text contains the exact phrase (case-insensitive)."""
    return phrase in text.lower()

@dataclass
class EnhancedDoc:
    id: int
    text: str
    url: str
    title: str
    domain: str
    snippet_type: str = "text"
    time_scope: str = "present"

class EnhancedBM25:
    """Enhanced BM25 with phrase matching and numeric boosting."""
    
    def __init__(self, docs: List[EnhancedDoc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.N = len(docs)
        self.k1 = k1
        self.b = b
        self.df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []
        self.doclen: List[int] = []
        self.avgdl = 0.0
        self._build()

    def _build(self):
        total = 0
        for d in self.docs:
            ct = {}
            length = 0
            for t in tokenize(d.text):
                ct[t] = ct.get(t, 0) + 1
                length += 1
            self.tf.append(ct)
            self.doclen.append(length)
            total += length
            for t in ct:
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = (total / self.N) if self.N else 0.0

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return max(0.0, math.log((self.N - n + 0.5) / (n + 0.5) + 1.0))


    def base_score(self, q: List[str], i: int) -> float:
        """Standard BM25 score."""
        s = 0.0
        tf = self.tf[i]
        dl = self.doclen[i] or 1
        for term in q:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self.idf(term)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += idf * (f * (self.k1 + 1)) / denom
        return s
    
    def calculate_data_focus_boosts(self, doc: EnhancedDoc, query: str, query_entities: Set[str]) -> float:
        """Calculate ranking boosts focused on data richness for model reasoning."""
        boost = 1.0
        text_lower = doc.text.lower()
        
        # Entity co-occurrence boost (e.g., "sun" and "earth" in same snippet)
        entity_matches = sum(1 for entity in query_entities if entity in text_lower)
        if entity_matches >= 2:
            boost += 0.5  # Strong boost for multi-entity snippets
        elif entity_matches == 1:
            boost += 0.2  # Moderate boost for single entity
        
        # Data richness boost
        measurements = extract_measurements(doc.text)
        if len(measurements) >= 2:
            boost += 0.4  # High boost for measurement-rich snippets
        elif len(measurements) == 1:
            boost += 0.2  # Moderate boost for single measurements
        
        # Infobox and structured data priority
        if doc.snippet_type == "data":
            boost += 0.6  # Highest boost for extracted measurements
        elif doc.snippet_type == "infobox":
            boost += 0.3  # Good boost for infobox content
        elif doc.snippet_type == "numeric":
            boost += 0.2  # Some boost for numeric content
        
        # Scientific notation boost (common in measurements)
        if re.search(r"\d+(\.\d+)?\s*Ã—\s*10", doc.text):
            boost += 0.15
        
        return boost
    
    def extract_query_entities(self, query: str) -> Set[str]:
        """Extract key entities from query for co-occurrence boosting."""
        # Simple entity extraction - can be enhanced
        query_lower = query.lower()
        entities = set()
        
        # Common astronomical/physical entities
        entity_patterns = [
            r"\b(sun|solar|star)\b",
            r"\b(earth|terrestrial|planet)\b", 
            r"\b(moon|lunar|satellite)\b",
            r"\b(mars|venus|jupiter|saturn|mercury|neptune|uranus|pluto)\b",
            r"\b(galaxy|milky way|universe)\b"
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, query_lower)
            entities.update(matches)
        
        # Add explicit words from query as potential entities
        words = re.findall(r"\b[a-z]+\b", query_lower)
        for word in words:
            if len(word) > 3 and word not in STOPWORDS:
                entities.add(word)
        
        return entities

    def search(self, query: str, top_k: int = 6, domain_filter: Optional[Set[str]] = None,
              boost_data: bool = True) -> List[Tuple[float, EnhancedDoc]]:
        """Data-focused search optimized for model reasoning."""
        q_tokens = tokenize(query)
        query_entities = self.extract_query_entities(query) if boost_data else set()
        
        if not q_tokens or not self.docs:
            return []
        
        scored: List[Tuple[float, EnhancedDoc]] = []
        for i, d in enumerate(self.docs):
            if domain_filter and d.domain not in domain_filter:
                continue
            
            # Base BM25 score
            base_sc = self.base_score(q_tokens, i)
            if base_sc <= 0:
                continue
            
            # Apply data-focused boosts
            if boost_data:
                boost_multiplier = self.calculate_data_focus_boosts(d, query, query_entities)
                final_score = base_sc * boost_multiplier
            else:
                final_score = base_sc
            
            scored.append((final_score, d))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Dedupe identical texts but prioritize data-rich duplicates
        seen_texts = {}
        uniq: List[Tuple[float, EnhancedDoc]] = []
        
        for sc, d in scored:
            if d.text in seen_texts:
                # Keep higher scoring version of duplicate text
                existing_score = seen_texts[d.text]
                if sc > existing_score:
                    # Remove old version and add new one
                    uniq = [(s, doc) for s, doc in uniq if doc.text != d.text]
                    uniq.append((sc, d))
                    seen_texts[d.text] = sc
                continue
            
            seen_texts[d.text] = sc
            uniq.append((sc, d))
            
            if len(uniq) >= top_k:
                break
        
        return uniq

# ----------------------------- CLI handlers -----------------------------

def compose_data_focused_block(query: str, results: List[Tuple[float, EnhancedDoc]],
                             min_needed: int = 3, compact: bool = False, time_scope: str = "all") -> str:
    """Compose evidence block optimized for model reasoning with related data."""
    lines = []
    lines.append("[EVIDENCE - related data for reasoning]")
    lines.append(f"Query: {query}")
    lines.append("")

    # Filter out promotional/offer content
    # NOTE: Removed "best", "offer", "deal", "buy" to avoid filtering legitimate news
    # (e.g., "company acquired X", "best discoveries of 2024")
    ad_keywords = {
        "discount", "sale", "shop", "coupon", "black friday",
        "limited time", "save", "free shipping", "must have", "recommended",
        "top rated", "#1", "won\'t", "don\'t miss", "exclusive offer",
        "act now", "save now"
    }

    def is_promotional(text: str, title: str = "") -> bool:
        """Check if content is promotional/advertising."""
        check_text = (text + " " + title).lower()
        return any(keyword in check_text for keyword in ad_keywords)

    # Filter out promotional results
    results = [(s, d) for s, d in results if not is_promotional(d.text, d.title or "")]

    # Filter by time scope if specified
    if time_scope != "all":
        results = [(s, d) for s, d in results if d.time_scope == time_scope]
    
    if len(results) < min_needed:
        lines.append(f"insufficient_evidence: true  # found {len(results)} < {min_needed}")
        if time_scope != "all":
            lines.append(f"# Note: filtered to {time_scope} time scope")
        lines.append("")
        lines.append("Use: Request more specific entities or check if data exists.")
        return "\n".join(lines)

    lines.append("insufficient_evidence: false")
    if time_scope != "all":
        lines.append(f"# Time scope filter: {time_scope}")
    lines.append("")
    
    # Group by data richness for reasoning
    data_results = [(s, d) for s, d in results if d.snippet_type == "data"]
    infobox_results = [(s, d) for s, d in results if d.snippet_type == "infobox"]
    numeric_results = [(s, d) for s, d in results if d.snippet_type == "numeric"]
    text_results = [(s, d) for s, d in results if d.snippet_type == "text"]
    
    # Data-rich content first
    for idx, (score, d) in enumerate(data_results, 1):
        scope = getattr(d, "time_scope", "ALL").upper()
        lines.append(f"- [{scope}] [DATA] \"{d.text}\" - {d.title or d.domain} ({d.url})")

    # Numeric content
    for idx, (score, d) in enumerate(numeric_results, len(data_results) + 1):
        scope = getattr(d, "time_scope", "ALL").upper()
        lines.append(f"- [{scope}] [NUMERIC] \"{d.text}\" - {d.title or d.domain} ({d.url})")

    # Optional compact view
    if not compact:
        for idx, (score, d) in enumerate(infobox_results, len(data_results) + len(numeric_results) + 1):
            scope = getattr(d, "time_scope", "ALL").upper()
            lines.append(f"- [{scope}] [INFOBOX] \"{d.text}\" - {d.title or d.domain} ({d.url})")

        for idx, (score, d) in enumerate(text_results, len(data_results) + len(infobox_results) + len(numeric_results) + 1):
            scope = getattr(d, "time_scope", "ALL").upper()
            lines.append(f"- [{scope}] \"{d.text}\" - {d.title or d.domain} ({d.url})")

    lines.append("")
    mode_note = "# Compact mode: showing only [DATA] and [NUMERIC] facts for reasoning" if compact else "# Use the related data above to reason about relationships and calculations"
    lines.append(mode_note)
    if not compact:
        lines.append("# Prioritize [DATA] and [INFOBOX] facts for quantitative analysis")
    if time_scope == "present":
        lines.append("# Time scope: present-day facts only")
    return "\n".join(lines)

def cmd_fetch(a: argparse.Namespace) -> int:
# SERP detection and auto-resolution (Bing only)
    parsed_url = urllib.parse.urlparse(a.url)
    host = (parsed_url.netloc or "").lower()
    path = parsed_url.path or "/"

    is_bing_serp = host.endswith("bing.com") and path.startswith("/search")

    if is_bing_serp and not getattr(a, 'allow_serp', False):
        print("SERP detected (Bing). Resolving to destination...", file=sys.stderr)

        # Extract the original query from the SERP URL
        query_params = urllib.parse.parse_qs(parsed_url.query)
        q = query_params.get('q', [None])[0]
        if not q:
            print("SERPNotAllowed: Could not extract query from SERP URL.", file=sys.stderr)
            return 2

        # Resolve to top organic result(s) from Bing
        urls = resolve_query_to_urls(q, provider="bing", top=1)

        if urls:
            print(f"Resolved to: {urls[0]}", file=sys.stderr)
            a.url = urls[0]  # Replace SERP URL with destination URL
        else:
            print("SERPNotAllowed: Could not resolve any destination URLs from search results.", file=sys.stderr)
            return 2

    
    # Continue with normal fetch logic
    try:
        # When --allow-serp is used, ignore robots.txt for SERP fetching
        ignore_robots = a.ignore_robots or getattr(a, 'allow_serp', False)
        status, headers, data = fetch_raw(a.url, timeout=a.timeout, ignore_robots=ignore_robots)
    except PermissionError as e:
        print(str(e), file=sys.stderr)
        return 2
    except urllib.error.HTTPError as e:
        print(f"HTTP error {e.code}: {e.reason}", file=sys.stderr)
        return 3
    except urllib.error.URLError as e:
        print(f"URL error: {e.reason}", file=sys.stderr)
        return 4
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 5

    if a.show_headers:
        print(f"Status: {status}", file=sys.stderr)
        for k, v in headers.items():
            print(f"{k}: {v}", file=sys.stderr)
    
    if a.to_file:
        try:
            ensure_parent_dir(a.to_file)
            with open(a.to_file, "wb") as f:
                f.write(data)
            print(f"Wrote {len(data)} bytes to {a.to_file}")
            return 0
        except Exception as e:
            print(f"File write error: {e}", file=sys.stderr)
            return 6
    else:
        # print decoded with robust encoding handling
        encoding = detect_charset(headers) if a.decode == "auto" else a.decode
        if a.decode == "none":
            sys.stdout.buffer.write(data)
            return 0
        
        # Handle various encoding scenarios with fallback chain
        try:
            # First, check for gzip compression
            if headers.get('content-encoding') == 'gzip':
                import gzip
                data = gzip.decompress(data)
            
            # Try the detected/specified encoding first
            text = data.decode(encoding, errors="replace")
        except (UnicodeDecodeError, Exception):
            try:
                # Google often returns Latin-1 encoded content
                text = data.decode("latin-1", errors="replace")
            except Exception:
                # Final fallback to UTF-8 with error replacement
                text = data.decode("utf-8", errors="replace")
        
        print(text)
        return 0

def cmd_extract(a: argparse.Namespace) -> int:
    if not a.from_file and not a.from_url:
        print("Provide --from-file or --from-url", file=sys.stderr)
        return 2
    if a.from_file and a.from_url:
        print("Use only one of --from-file or --from-url", file=sys.stderr)
        return 2

    # Get raw html bytes
    if a.from_url:
        try:
            status, headers, data = fetch_raw(a.from_url, timeout=REQUEST_TIMEOUT, ignore_robots=a.ignore_robots)
        except Exception as e:
            print(f"Fetch error: {e}", file=sys.stderr)
            return 3
        raw = data
    else:
        try:
            with open(a.from_file, "rb") as f:
                raw = f.read()
        except Exception as e:
            print(f"Read error: {e}", file=sys.stderr)
            return 3

    # Enhanced HTML -> text with infobox extraction
    title, text, infobox_lines = html_bytes_to_enhanced_text(raw)

    # Strip numeric citation brackets like [ 148 ]
    text = re.sub(r"[[\]\s*\d+\s*[]]", "", text)

    # Enhanced section cutoff - more aggressive reference removal
    CUT_AFTER = (
        "References", "Notes", "Further reading", "Bibliography", "Citations", 
        "Sources", "External links", "See also", "Footnotes", "Works cited"
    )
    BOUNDARY = r"(?:^|[.!?]\s)"  # start of text or after sentence end
    for marker in CUT_AFTER:
        pattern = BOUNDARY + re.escape(marker) + r"\b"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            text = text[:m.start()]
            break

    # Prepare output - include infobox content at the top
    ensure_parent_dir(a.to_file)
    out_parts = []
    
    if title:
        out_parts.append(title)
        out_parts.append("")
    
    # Add infobox content first (highest priority facts)
    if infobox_lines:
        out_parts.append("[INFOBOX FACTS]")
        out_parts.extend(infobox_lines)
        out_parts.append("")
    
    out_parts.append(text)
    
    out_text = "\n".join(out_parts)
    
    with open(a.to_file, "w", encoding="utf-8") as f:
        f.write(out_text)
    
    print(f"Wrote enhanced text to {a.to_file} ({len(infobox_lines)} infobox facts)")
    return 0

def cmd_snippetize(a: argparse.Namespace) -> int:
    try:
        with open(a.from_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Read error: {e}", file=sys.stderr)
        return 2

    # Separate infobox content if present
    infobox_lines = []
    main_text = content
    
    if "[INFOBOX FACTS]" in content:
        _, after = content.split("[INFOBOX FACTS]", 1)
        infobox_section, sep, rest = after.lstrip().partition("\n\n")
        infobox_lines = [ln.strip() for ln in infobox_section.split("\n")
                        if ln.strip() and ":" in ln]
        main_text = rest if sep else ""   # <- exclude infobox from main text
    else:
        main_text = content

    snippets = enhanced_text_to_snippets(main_text, infobox_lines)
    added, path = append_enhanced_snippets_jsonl(
        snippets, a.source_url, a.source_title or "", to_file=a.to_file
    )
    
    snippet_stats = {
        "data":   sum(1 for _, t, _ in snippets if t == "data"),
        "infobox":sum(1 for _, t, _ in snippets if t == "infobox"),
        "numeric":sum(1 for _, t, _ in snippets if t == "numeric"),
        "text":   sum(1 for _, t, _ in snippets if t == "text"),
    }
    
    time_stats = {
        "present": sum(1 for _, _, ts in snippets if ts == "present"),
        "future":  sum(1 for _, _, ts in snippets if ts == "future"),
        "past":    sum(1 for _, _, ts in snippets if ts == "past"),
    }

    
    print(json.dumps({
        "ok": True,
        "snippets": len(snippets),
        "added": added,
        "jsonl": path,
        "types": snippet_stats,
        "time_scope": time_stats
    }, indent=2))
    return 0

def cmd_compose(a: argparse.Namespace) -> int:
    # Load docs
    dom_filter = set(a.domains.split(",")) if a.domains else None
    files = a.files.split(",") if a.files else None
    docs = load_enhanced_docs(files, dom_filter)
    if not docs:
        print("No snippets found. Ingest and snippetize first.", file=sys.stderr)
        return 2

    # Data-focused search
    bm = EnhancedBM25(docs)
    hits = bm.search(
        a.query, 
        top_k=a.top_k, 
        domain_filter=dom_filter,
        boost_data=getattr(a, 'boost_data', True)
    )

    # Build data-focused block
    block = compose_data_focused_block(a.query, hits, min_needed=getattr(a, 'min_needed', 3), 
                                     compact=getattr(a, 'compact', False), 
                                     time_scope=getattr(a, 'time_scope', 'all'))

    # Write or print
    if a.out:
        ensure_parent_dir(a.out)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(block)
        print(f"Wrote data-focused prompt block to {a.out}")
    else:
        print(block)
    return 0

def cmd_resolve(a: argparse.Namespace) -> int:
    """Resolve search query to destination URLs."""
    urls = resolve_query_to_urls(
        a.query, 
        provider=getattr(a, 'provider', 'google'), 
        top=getattr(a, 'top', 3)
    )
    
    if not urls:
        print("No URLs found for query.", file=sys.stderr)
        return 1
    
    # Print one URL per line
    for url in urls:
        print(url)
    
    return 0

def cmd_ingest(a: argparse.Namespace) -> int:
    """One-shot convenience: resolve → fetch → extract → snippetize."""
    print(f"Resolving query: '{a.query}'")
    
    # Resolve URLs
    urls = resolve_query_to_urls(
        a.query,
        provider=getattr(a, 'provider', 'google'),
        top=getattr(a, 'top', 3)
    )
    
    if not urls:
        print("No URLs resolved. Cannot proceed with ingestion.", file=sys.stderr)
        return 1
    
    print(f"Found {len(urls)} URLs to process:")
    for i, url in enumerate(urls, 1):
        print(f"{i}. {url}")
    print()
    
    total_snippets = 0
    
    # Process each URL through the pipeline
    for i, url in enumerate(urls):
        try:
            domain = urllib.parse.urlparse(url).netloc
            temp_html = f"temp_ingest_{i}.html"
            temp_text = f"temp_ingest_{i}.txt"
            
            print(f"Processing {domain}...")
            
            # Fetch
            status, headers, data = fetch_raw(url)
            with open(temp_html, "wb") as f:
                f.write(data)
            
            # Extract
            title, text, infobox_lines = html_bytes_to_enhanced_text(data)
            
            # Prepare output text
            out_parts = []
            if title:
                out_parts.append(title)
                out_parts.append("")
            if infobox_lines:
                out_parts.append("[INFOBOX FACTS]")
                out_parts.extend(infobox_lines)
                out_parts.append("")
            out_parts.append(text)
            
            with open(temp_text, "w", encoding="utf-8") as f:
                f.write("\n".join(out_parts))
            
            # Snippetize
            snippets = enhanced_text_to_snippets(text, infobox_lines)
            
            # Determine output file
            out_jsonl = getattr(a, 'out_jsonl', None)
            if not out_jsonl:
                out_jsonl = os.path.join(DATA_DIR, f"{domain}.jsonl")
            
            # Use extracted title or manual title
            source_title = title if getattr(a, 'title', 'auto') == 'auto' else getattr(a, 'title', f"Ingested Content {i+1}")
            
            added, path = append_enhanced_snippets_jsonl(
                snippets, url, source_title, to_file=out_jsonl
            )
            
            total_snippets += added
            print(f"  Added {added} snippets to {path}")
            
            # Cleanup temp files
            os.remove(temp_html)
            os.remove(temp_text)
            
        except Exception as e:
            print(f"  Error processing {url}: {e}")
            continue
    
    print(f"\nIngestion complete! Added {total_snippets} total snippets.")
    print(f"Run: python contextrag.py compose \"{a.query}\" --time-scope present")
    return 0
    # Load docs
    dom_filter = set(a.domains.split(",")) if a.domains else None
    files = a.files.split(",") if a.files else None
    docs = load_enhanced_docs(files, dom_filter)
    if not docs:
        print("No snippets found. Ingest and snippetize first.", file=sys.stderr)
        return 2

    # Data-focused search
    bm = EnhancedBM25(docs)
    hits = bm.search(
        a.query, 
        top_k=a.top_k, 
        domain_filter=dom_filter,
        boost_data=getattr(a, 'boost_data', True)
    )

    # Build data-focused block
    block = compose_data_focused_block(a.query, hits, min_needed=getattr(a, 'min_needed', 3), 
                                     compact=getattr(a, 'compact', False), 
                                     time_scope=getattr(a, 'time_scope', 'all'))

    # Write or print
    if a.out:
        ensure_parent_dir(a.out)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(block)
        print(f"Wrote data-focused prompt block to {a.out}")
    else:
        print(block)
    return 0

def cmd_ask(a: argparse.Namespace) -> int:
    # Read prompt file (or stdin later if you want)
    try:
        with open(a.infile, "r", encoding="utf-8") as f:
            payload = f.read()
    except Exception as e:
        print(f"Read error: {e}", file=sys.stderr)
        return 2

    # Run the provided command and pipe payload to stdin
    try:
        # Use shlex.split() to safely parse command without shell=True
        proc = subprocess.Popen(
            shlex.split(a.runner),
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = proc.communicate(payload)
    except Exception as e:
        print(f"Runner error: {e}", file=sys.stderr)
        return 3

    if stderr:
        # Don't fail on stderr; just surface it for visibility
        print(stderr, file=sys.stderr)

    if stdout:
        print(stdout, end="")

    return proc.returncode or 0

def load_enhanced_docs(files: Optional[List[str]] = None, 
                      domains_filter: Optional[Set[str]] = None) -> List[EnhancedDoc]:
    """Load docs with enhanced snippet type information."""
    docs: List[EnhancedDoc] = []
    did = 0
    file_list: List[str] = []
    
    if files:
        file_list = files
    else:
        if os.path.isdir(DATA_DIR):
            for fn in os.listdir(DATA_DIR):
                if fn.endswith(".jsonl"):
                    file_list.append(os.path.join(DATA_DIR, fn))
    
    for path in file_list:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    dom = obj.get("domain", "")
                    if domains_filter and dom not in domains_filter:
                        continue
                    docs.append(EnhancedDoc(
                        id=did,
                        text=obj.get("text", ""),
                        url=obj.get("source_url", ""),
                        title=obj.get("source_title", ""),
                        domain=dom,
                        snippet_type=obj.get("snippet_type", "text"),
                        time_scope=obj.get("time_scope", "present")
                    ))
                    did += 1
        except FileNotFoundError:
            continue
    return docs

def cmd_search(a: argparse.Namespace) -> int:
    """Data-focused search command."""
    dom_filter = set(a.domains.split(",")) if a.domains else None
    files = a.files.split(",") if a.files else None
    docs = load_enhanced_docs(files, dom_filter)
    
    bm = EnhancedBM25(docs)
    results = bm.search(
        a.query, 
        top_k=a.top_k, 
        domain_filter=dom_filter,
        boost_data=getattr(a, 'boost_data', True)
    )
    
    payload = [{
        "score": round(sc, 6),
        "text": d.text,
        "source_url": d.url,
        "source_title": d.title,
        "domain": d.domain,
        "snippet_type": d.snippet_type,
        "measurements": extract_measurements(d.text) if hasattr(d, 'text') else []
    } for sc, d in results]
    
    output_json = json.dumps({
        "insufficient_evidence": len(payload) < getattr(a, 'min_needed', 3),
        "count": len(payload),
        "results": payload,
        "search_info": {
            "data_focused": getattr(a, 'boost_data', True),
            "measurement_extraction": True
        }
    }, indent=2, ensure_ascii=True)
    print(output_json)
    return 0

# ----------------------------- main -----------------------------

def build_enhanced_parser() -> argparse.ArgumentParser:
    """Enhanced argument parser with new options."""
    p = argparse.ArgumentParser(
        description="Enhanced web RAG utilities - exact fact retrieval (fetch, extract, snippetize, search)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # fetch - enhanced with SERP detection
    pf = sub.add_parser("fetch", help="Fetch a URL and return raw content")
    pf.add_argument("url")
    pf.add_argument("--to-file", dest="to_file")
    pf.add_argument("--show-headers", action="store_true")
    pf.add_argument("--ignore-robots", action="store_true")
    pf.add_argument("--allow-serp", action="store_true", 
                   help="Allow fetching Google SERPs directly (not recommended)")
    pf.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT)
    pf.add_argument("--decode", choices=["auto", "utf-8", "latin-1", "none"], default="auto")
    pf.set_defaults(func=cmd_fetch)

    # extract - unchanged
    pe = sub.add_parser("extract", help="Extract readable text from HTML with infobox facts")
    pe.add_argument("--from-file")
    pe.add_argument("--from-url")
    pe.add_argument("--to-file", required=True)
    pe.add_argument("--ignore-robots", action="store_true")
    pe.set_defaults(func=cmd_extract)

    # snippetize - unchanged
    ps = sub.add_parser("snippetize", help="Convert text to JSONL snippets with factual priority")
    ps.add_argument("--from-file", required=True)
    ps.add_argument("--source-url", required=True)
    ps.add_argument("--source-title", default="")
    ps.add_argument("--to-file")
    ps.set_defaults(func=cmd_snippetize)

    # resolve - new URL discovery command
    presolve = sub.add_parser("resolve", help="Resolve search query to destination URLs")
    presolve.add_argument("query", help="Search query")
    presolve.add_argument("--provider", choices=["google"], default="google",
                         help="Search provider")
    presolve.add_argument("--top", type=int, default=3,
                         help="Number of URLs to return")
    presolve.set_defaults(func=cmd_resolve)

    # ingest - new convenience command
    pingest = sub.add_parser("ingest", help="One-shot: resolve → fetch → extract → snippetize")
    pingest.add_argument("query", help="Search query")
    pingest.add_argument("--provider", choices=["google"], default="google")
    pingest.add_argument("--top", type=int, default=1,
                        help="Number of URLs to process")
    pingest.add_argument("--title", choices=["auto", "manual"], default="auto",
                        help="Title extraction method")
    pingest.add_argument("--out-jsonl", help="Output JSONL file (default: data/domain.jsonl)")
    pingest.set_defaults(func=cmd_ingest)

    # search - enhanced with boosting options
    psearch = sub.add_parser("search", help="Enhanced BM25 search with phrase/numeric boosting")
    psearch.add_argument("query")
    psearch.add_argument("--top_k", type=int, default=6)
    psearch.add_argument("--domains", help="Comma-separated domain filter (e.g., wikipedia.org)")
    psearch.add_argument("--files", help="Comma-separated JSONL files to search; default ./data/*.jsonl")
    psearch.add_argument("--boost-phrases", action="store_true", default=True, 
                        help="Boost exact phrase matches (default: enabled)")
    psearch.add_argument("--no-boost-phrases", dest="boost_phrases", action="store_false",
                        help="Disable phrase boosting")
    psearch.add_argument("--boost-numeric", action="store_true", default=True,
                        help="Boost numeric/factual content (default: enabled)")
    psearch.add_argument("--no-boost-numeric", dest="boost_numeric", action="store_false",
                        help="Disable numeric boosting")
    psearch.add_argument("--min-needed", type=int, default=3,
                        help="Minimum snippets for sufficient evidence")
    psearch.set_defaults(func=cmd_search)

    # compose - enhanced with factual prioritization
    pc = sub.add_parser("compose", help="Compose enhanced evidence block prioritizing factual content")
    pc.add_argument("query")
    pc.add_argument("--top_k", type=int, default=6)
    pc.add_argument("--domains", help="Comma-separated domain filter")
    pc.add_argument("--files", help="Comma-separated JSONL files to search; default ./data/*.jsonl")
    pc.add_argument("--out", help="Write composed block to this file (otherwise prints to stdout)")
    pc.add_argument("--boost-phrases", action="store_true", default=True)
    pc.add_argument("--no-boost-phrases", dest="boost_phrases", action="store_false")
    pc.add_argument("--boost-numeric", action="store_true", default=True)
    pc.add_argument("--no-boost-numeric", dest="boost_numeric", action="store_false")
    pc.add_argument("--min-needed", type=int, default=3)
    pc.add_argument("--compact", action="store_true", help="Show only [DATA] and [NUMERIC] snippets")
    pc.add_argument("--time-scope", choices=["present", "future", "past", "all"], default="all", 
                   help="Filter snippets by time scope (default: all)")
    pc.set_defaults(func=cmd_compose)

    # ask - unchanged
    pa = sub.add_parser("ask", help="Pipe a prompt file to a local runner command")
    pa.add_argument("--runner", required=True, help="Command to execute, e.g., 'ollama run llama3'")
    pa.add_argument("--in", dest="infile", required=True, help="Path to prompt file (e.g., from compose)")
    pa.set_defaults(func=cmd_ask)

    return p

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_enhanced_parser()
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
