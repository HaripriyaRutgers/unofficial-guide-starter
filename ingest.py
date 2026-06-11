"""
ingest.py — Stage 1 of the RAG pipeline: Document Ingestion.

Scrapes 10 source URLs about non-SWE CS career pathways, strips out
boilerplate/navigation/ads, and saves clean plain text to raw_docs/,
one .txt file per URL (named by domain).

Forum sources (reddit, teamblind) split into individual posts/replies.
Career-map sources (screenskills, uchicago) split into per-role sections.

Run with:  python ingest.py
Libraries:  requests, beautifulsoup4, json, os, re  (no LangChain)
"""

import os
import re
import html  # standard library — used to decode HTML entities (&amp; -> &)
import functools
from urllib.parse import urljoin  # resolve relative role links to absolute URLs

import requests
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# CONFIG: the 10 sources, each tagged with a "kind" that drives how we split.
# kind = "article"  -> save as one document (chunking happens later in chunk.py)
# kind = "forum"    -> split into individual posts/replies here
# kind = "career_map" -> split into per-role sections here
# "name" becomes the raw_docs/<name>.txt filename, and the chunk source label.
# ---------------------------------------------------------------------------
SOURCES = [
    {"name": "reddit",        "kind": "forum",      "url": "https://www.reddit.com/r/cscareerquestions/"},
    {"name": "awesome_subreddits", "kind": "article", "url": "https://github.com/iCHAIT/awesome-subreddits"},
    {"name": "themuse",       "kind": "article",    "url": "https://www.themuse.com/advice/computer-science-degree-major-jobs"},
    {"name": "collegewise",   "kind": "article",    "url": "https://go.collegewise.com/alternative-pathways-to-a-career-in-computer-science"},
    # ScreenSkills VFX is a *browse directory*: the landing page only holds role
    # TITLES (each a "job-profile-card" link); the actual description + skills
    # live on per-role sub-pages. So we crawl each card link and treat one
    # sub-page = one role block. "link_class" tells the crawler which links to follow.
    {"name": "screenskills",  "kind": "career_map_crawl",
     "url": "https://www.screenskills.com/job-profiles/browse/visual-effects-vfx/",
     "link_class": "job-profile-card__link"},
    {"name": "awesome_cybersecurity", "kind": "article", "url": "https://github.com/d0midigi/awesome-cybersecurity-subreddits"},
    {"name": "careerexplorer", "kind": "career_map", "url": "https://www.careerexplorer.com/careers/?page=2"},
    {"name": "teamblind",     "kind": "forum",      "url": "https://www.teamblind.com/post/non-swe-career-opportunities-for-cs-major-bh4qscys"},
    {"name": "technicalwriting", "kind": "article", "url": "https://www.everythingtechnicalwriting.com/everything-you-need-to-know-about-technical-writing/"},
    {"name": "uchicago",      "kind": "career_map", "url": "https://careeradvancement.uchicago.edu/careers-in/gaming/"},
]

OUTPUT_DIR = "raw_docs"

# A browser-like User-Agent. Many sites (Reddit, Teamblind, CareerExplorer)
# return 403 to the default python-requests UA, so we impersonate a browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Class/id tokens that mark boilerplate (nav, ads, share, etc.). These are
# matched as WHOLE WORDS, not raw substrings — otherwise "ad" would nuke
# "heading"/"read"/"shadow" and "header" would nuke "subheading", deleting
# real content. We treat -, _, and whitespace as word boundaries.
JUNK_CLASS_PATTERNS = [
    "nav", "navbar", "menu", "footer", "header", "masthead", "cookie",
    "banner", "sidebar", "ad", "ads", "advert", "share", "promo",
    "social", "newsletter", "subscribe", "breadcrumb", "modal", "popup",
]

# Pre-compile one regex that matches any junk token as a delimited word.
# We normalize -/_ to spaces first, so \b reliably bounds each token.
_JUNK_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in JUNK_CLASS_PATTERNS) + r")\b"
)


@functools.lru_cache(maxsize=4096)
def _is_junk_attr(attr_blob):
    """True if a (space-normalized) class/id string names a boilerplate token."""
    return bool(_JUNK_RE.search(attr_blob))

# Tags that are structurally boilerplate regardless of class — removed wholesale.
JUNK_TAGS = ["nav", "header", "footer", "aside", "script", "style",
             "noscript", "form", "button", "svg", "iframe"]


def fetch(url):
    """GET a URL with a browser UA. Returns HTML text, or None on error.

    We never crash on a bad source: timeouts / non-200 / connection errors
    just print a warning and return None so the main loop can skip it.
    """
    try:
        # timeout=20 so a hanging server can't stall the whole run.
        resp = requests.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:  # DNS failure, timeout, conn reset...
        print(f"  [WARN] request failed for {url}: {e}")
        return None
    # Treat anything other than 200 OK as a skip (403 bot-block, 404, 5xx...).
    if resp.status_code != 200:
        print(f"  [WARN] non-200 status {resp.status_code} for {url} — skipping")
        return None
    return resp.text


def clean_text(s):
    """Decode HTML entities and normalize whitespace into clean prose."""
    # html.unescape handles &amp; -> &, &nbsp; -> \xa0, &#39; -> ', etc. in one call.
    s = html.unescape(s)
    # Convert non-breaking spaces (from &nbsp;) to real spaces.
    s = s.replace("\xa0", " ")
    # Collapse runs of spaces/tabs into a single space.
    s = re.sub(r"[ \t]+", " ", s)
    # Collapse 3+ newlines down to a paragraph break (keeps structure, drops gaps).
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def strip_boilerplate(soup):
    """Mutate the soup in place to delete navigation/ads/boilerplate nodes."""
    # 1) Remove HTML comments (often contain ad/tracking markup).
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    # 2) Remove structurally non-content tags (nav, footer, script, ...).
    for tag in soup(JUNK_TAGS):
        tag.decompose()  # decompose() removes the tag AND its contents from the tree.

    # 3) Remove any element whose class/id contains a junk substring.
    # Collect matches first, THEN decompose — decomposing mid-iteration would
    # detach child tags (their .attrs becomes None) and crash the loop.
    to_remove = []
    for el in soup.find_all(True):  # True matches every tag.
        if el.attrs is None:        # already detached by an earlier removal.
            continue
        classes = el.get("class") or []  # may be missing -> default to [].
        # Gather class tokens + id, normalize -/_ to spaces, lowercase, so the
        # whole-word regex can bound each token cleanly.
        attr_blob = " ".join(list(classes) + [el.get("id") or ""]).lower()
        attr_blob = re.sub(r"[-_]", " ", attr_blob)
        if attr_blob.strip() and _is_junk_attr(attr_blob):
            to_remove.append(el)
    for el in to_remove:
        el.decompose()  # safe now: we're not iterating the live tree.
    return soup


def extract_article(soup):
    """Pull the main body text from an article-style page.

    Prefer a <main>/<article> container if present; otherwise fall back to
    the whole <body>. Returns one cleaned string.
    """
    # Many article sites wrap content in <article> or <main> — use it if present.
    container = soup.find("article") or soup.find("main") or soup.body or soup
    # get_text with newline separator keeps paragraph boundaries readable.
    return clean_text(container.get_text(separator="\n"))


def clean_github(text):
    """Extra cleaning for GitHub 'awesome list' README pages.

    These pages are full of emoji section markers (❇️ 🥷 🛡️), smart symbols,
    and bare one-word category headings. The emoji/symbols sit flush against
    words ("...prem❇️thats...") and break mid-word when chunked, and the lone
    headings add noise. So for GitHub sources only we:
      1. strip every non-ASCII char (emoji + special symbols) to a space,
      2. drop lines that are only 1-3 words (bare section headers),
      3. drop lines with no letters/digits (pure bullets/symbols/whitespace).
    """
    # 1) Replace any run of non-ASCII bytes with a single space.
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    kept = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 3) Skip lines that contain no alphanumeric content at all
        #    (e.g. "---", "•", leftover "<div>" fragments became symbols).
        if not re.search(r"[A-Za-z0-9]", line):
            continue
        # 2) Skip bare section headers: 1-3 "words" carry no real content.
        if len(line.split()) <= 3:
            continue
        kept.append(line)

    # Re-join and run the normal whitespace normalizer over the result.
    return clean_text("\n".join(kept))


def extract_forum(soup, source_name):
    """Extract individual posts/replies as separate labeled text units.

    Forum HTML is fragile and changes often (Reddit/Teamblind are also heavy
    JS apps), so we use a tolerant heuristic: treat each reasonably long
    paragraph/comment-like block as one reply. Each unit is prefixed with a
    metadata header so chunk.py can recover source + type.
    """
    units = []
    # Candidate reply containers: comment divs, list items, and paragraphs.
    # We cast a wide net because forum markup varies a lot between sites.
    candidates = soup.find_all(["p", "li", "div"])
    seen = set()  # de-dupe: nested divs would otherwise repeat the same text.
    for el in candidates:
        text = clean_text(el.get_text(separator=" "))
        # A "reply" must be substantive: skip tiny UI fragments (buttons, votes).
        if len(text) < 80:
            continue
        if text in seen:
            continue
        seen.add(text)
        # Prefix each unit with a metadata header (parsed back out in chunk.py).
        header = f"[type=forum_reply source={source_name}]"
        units.append(f"{header}\n{text}")
    return units


def extract_career_map(soup, source_name):
    """Extract each role/section as a separate text unit.

    Career-map pages list many roles. We treat each heading (h1-h4) plus the
    text that follows it (until the next heading) as one role profile, and
    capture the heading as role_name in a metadata header.
    """
    units = []
    headings = soup.find_all(["h1", "h2", "h3", "h4"])
    for h in headings:
        role_name = clean_text(h.get_text(separator=" "))
        if not role_name:
            continue
        # Walk forward through siblings, collecting text until the next heading.
        body_parts = []
        for sib in h.find_all_next():
            # Stop when we reach the next heading — that begins a new role.
            if sib.name in ("h1", "h2", "h3", "h4"):
                break
            if sib.name in ("p", "li"):
                body_parts.append(clean_text(sib.get_text(separator=" ")))
        body = "\n".join(p for p in body_parts if p)
        # Skip headings that have no real description under them.
        if len(body) < 60:
            continue
        # role_name is escaped of brackets so it can't break the header format.
        safe_role = role_name.replace("[", "(").replace("]", ")")
        header = f"[type=career_profile source={source_name} role_name={safe_role}]"
        units.append(f"{header}\n{body}")
    return units


def extract_career_map_crawl(soup, source_name, base_url, link_class):
    """Crawl a career-map DIRECTORY page into one role block per sub-page.

    Some career maps (ScreenSkills) only list role TITLES as links on the
    landing page — the real description/skills text lives on each role's own
    sub-page. We follow every `link_class` link, fetch that page, and keep its
    full body as a single 'role block'. chunk.py then emits one chunk per role
    (sliding-windowed only if the block exceeds 500 tokens), so a role's title,
    description and skills stay grouped instead of fragmenting into one-liners.
    """
    units = []
    seen = set()  # de-dupe: the same role can be linked more than once.
    for a in soup.find_all("a", class_=link_class):
        href = a.get("href")
        if not href:
            continue
        full = urljoin(base_url, href)  # turn "/job-profiles/.../runner/" into a full URL.
        if full in seen:
            continue
        seen.add(full)

        # The link text (title under the card image) is the role name.
        role_name = clean_text(a.get_text(separator=" ")) or "(role)"

        sub_html = fetch(full)  # same tolerant fetch: warns + skips on error.
        if sub_html is None:
            continue
        sub = BeautifulSoup(sub_html, "html.parser")
        strip_boilerplate(sub)          # drop nav/footer/etc. on the sub-page too.
        body = extract_article(sub)     # the role's full description + skills text.

        # Skip pages that yielded no real content (e.g. a redirect/stub).
        if len(body) < 60:
            continue

        # Escape brackets so role_name can't corrupt the [ ... ] header format.
        safe_role = role_name.replace("[", "(").replace("]", ")")
        header = f"[type=career_profile source={source_name} role_name={safe_role}]"
        units.append(f"{header}\n{body}")
        print(f"    crawled role: {role_name} ({len(body)} chars)")
    return units


def main():
    # Create the output folder if it doesn't already exist.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for src in SOURCES:
        name, kind, url = src["name"], src["kind"], src["url"]
        print(f"\n=== {name} ({kind}) ===\n{url}")

        raw_html = fetch(url)
        if raw_html is None:
            continue  # fetch already warned; move on without crashing.

        soup = BeautifulSoup(raw_html, "html.parser")
        strip_boilerplate(soup)  # delete nav/ads/footers before extracting.

        # Route to the right extractor based on the source kind.
        if kind == "forum":
            units = extract_forum(soup, name)
            # Join units with a clear delimiter so chunk.py can split them apart.
            content = "\n\n----\n\n".join(units)
        elif kind == "career_map_crawl":
            # Directory page: follow each role link and grab its full sub-page.
            units = extract_career_map_crawl(soup, name, url, src.get("link_class"))
            content = "\n\n----\n\n".join(units)
        elif kind == "career_map":
            units = extract_career_map(soup, name)
            content = "\n\n----\n\n".join(units)
        else:  # article
            content = extract_article(soup)
            # GitHub READMEs need extra emoji/symbol/header stripping.
            if "github.com" in url:
                content = clean_github(content)

        # If extraction produced nothing useful, warn and skip writing an empty file.
        if not content or len(content.strip()) < 50:
            print(f"  [WARN] extracted almost no content for {name} — skipping save")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
        # Write UTF-8 so non-ASCII characters (smart quotes, etc.) survive.
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Print a preview so the user can eyeball that cleaning worked.
        print(f"  saved -> {out_path}  ({len(content)} chars)")
        print("  --- first 500 chars ---")
        print("  " + content[:500].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
