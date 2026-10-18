"""Step 3 — Clean: turn a page's raw Hugo Markdown into clean Markdown for chunking.

What we remove is noise for both the embedding model and the LLM: template tags,
HTML tags, and URLs. What we keep is everything a reader would see on the page.

We output *Markdown*, not plain text, on purpose:
- the chunker (step 4) splits on `##` headings and reads `{#anchor}` ids for deep links
- the LLM reads Markdown tables and lists well

Run it to see the effect:
    .venv/bin/python -m rag.clean                     # stats for config.DEFAULT_SECTIONS
    .venv/bin/python -m rag.clean --all               # stats for the whole handbook
    .venv/bin/python -m rag.clean --show handbook/people-group/acceptable-use-policy.md
"""

import argparse
import html
import re
import shlex
from collections import Counter
from dataclasses import replace

from rag import config
from rag.load import Page, load_page, load_pages

# --- Shortcode rules -------------------------------------------------------
# Every shortcode name in the corpus falls into one of these groups.
# Anything not listed is dropped and counted, so we can see what we lose.

# Wrappers: drop the tags, keep the text between them.
# If the opening tag has a label (e.g. details summary="..."), keep it as a bold line.
WRAPPERS = {"alert", "details", "panel", "note", "card", "cardpane", "tab", "tabpane"}
WRAPPER_LABEL_ARGS = ("summary", "title", "header")

# Data lookups: Hugo looks the argument up in a data file. We keep the argument itself,
# e.g. {{< member-by-name "Elyse Mallinger" >}} -> Elyse Mallinger
LOOKUPS = {"member-by-name", "member-by-gitlab", "member-and-role-by-gitlab", "manager-by-report-name", "label"}

# Icons that stand for a word.
ICONS = {"yes": "Yes", "no": "No"}

# Shortcodes we drop on purpose, without counting them as "unknown".
# `ref` builds a link URL (we drop URLs anyway); `a` is an anchor id; the rest are media.
KNOWN_DROPS = {"ref", "a", "external", "youtube", "vimeo", "figure"}

# {{< name args >}}, {{% name args %}}, and closing tags {{< /name >}}.
# The name must start with a letter, so escaped examples like {{</* ref "x" */>}} are left alone
# (Hugo shows those literally; they only appear in docs about shortcodes).
SHORTCODE_RE = re.compile(r"\{\{[<%]\s*(?P<closing>/?)\s*(?P<name>\w[\w./-]*)(?P<args>.*?)\s*/?\s*[>%]\}\}")


def parse_args(raw: str) -> tuple[list[str], dict[str, str]]:
    """Split shortcode args into positional and named: `"X" color="red"` -> (["X"], {"color": "red"})."""
    try:
        tokens = shlex.split(raw)
    except ValueError:  # unbalanced quotes, e.g. an apostrophe in an unquoted arg
        tokens = raw.split()
    positional = [t for t in tokens if "=" not in t]
    named = dict(t.split("=", 1) for t in tokens if "=" in t)
    return positional, named


def read_include(rel_path: str) -> str:
    """Raw text of an `include` file under assets/. Hugo inserts it as-is: no front matter,
    and a leading `---` is just a horizontal rule."""
    file = config.INCLUDES_DIR / rel_path
    return file.read_text(encoding="utf-8") if file.is_file() else ""


def clean_shortcodes(text: str, stats: Counter, depth: int = 0) -> str:
    def handle(match: re.Match) -> str:
        name, closing = match["name"], match["closing"]
        positional, named = parse_args(match["args"])

        if name in WRAPPERS:
            stats[f"wrapper   {name}"] += 0 if closing else 1
            label = next((named[k] for k in WRAPPER_LABEL_ARGS if k in named), None)
            return f"\n**{label}**\n" if label and not closing else ""

        if name == "include" and positional and depth < 3:  # depth guard against include loops
            stats[f"include   {name}"] += 1
            return clean_shortcodes(read_include(positional[0]), stats, depth + 1)

        if name in LOOKUPS:
            stats[f"lookup    {name}"] += 1
            values = positional or [named.get("name", "")]
            return ", ".join(values)

        if name in ICONS:
            stats[f"icon      {name}"] += 1
            return ICONS[name]

        if not closing:
            kind = "dropped  " if name in KNOWN_DROPS else "UNKNOWN  "
            stats[f"{kind} {name}"] += 1
        return ""

    return SHORTCODE_RE.sub(handle, text)


# --- HTML and Markdown noise -----------------------------------------------

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
# A tag must start with a letter, so `a < b`, `<https://...>` and `<me@x.com>` are left alone.
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][\w-]*(?:\s[^<>]*)?/?>")

# Link URLs may contain one level of parentheses, e.g. wikipedia/Foo_(bar).
URL_IN_PARENS = r"\((?:[^()\s]|\([^()]*\))*(?:\s+\"[^\"]*\")?\)"
IMAGE_RE = re.compile(r"!\[[^\]]*\]" + URL_IN_PARENS)
LINK_RE = re.compile(r"\[([^\]]*)\]" + URL_IN_PARENS)
REF_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
REF_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s+\S+.*$", re.MULTILINE)
AUTOLINK_RE = re.compile(r"<https?://[^>\s]+>")

CODE_FENCE_RE = re.compile(r"^ *(```|~~~).*?^ *\1", re.DOTALL | re.MULTILINE)


def clean_html_and_links(text: str) -> str:
    text = HTML_COMMENT_RE.sub("", text)
    text = SCRIPT_STYLE_RE.sub("", text)
    text = BR_RE.sub(" ", text)          # a newline would break a table row
    text = HTML_TAG_RE.sub("", text)
    text = IMAGE_RE.sub("", text)        # alt text is mostly "image" or a file name
    text = LINK_RE.sub(r"\1", text)      # [text](url) -> text
    text = REF_LINK_RE.sub(r"\1", text)  # [text][ref] -> text
    text = REF_DEF_RE.sub("", text)      # [ref]: url
    text = AUTOLINK_RE.sub("", text)
    return html.unescape(text)           # &amp; -> &, &nbsp; -> non-breaking space


def outside_code_blocks(text: str, fn) -> str:
    """Apply fn only to the parts of text outside ``` fenced code blocks."""
    parts, last = [], 0
    for m in CODE_FENCE_RE.finditer(text):
        parts.append(fn(text[last:m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(fn(text[last:]))
    return "".join(parts)


def tidy_whitespace(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)  # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                   # at most one blank line
    return text.strip() + "\n"


# --- Public API --------------------------------------------------------------

def clean_markdown(text: str, stats: Counter | None = None) -> str:
    stats = stats if stats is not None else Counter()
    text = clean_shortcodes(text, stats)  # first, so included files get cleaned too
    text = outside_code_blocks(text, clean_html_and_links)
    return tidy_whitespace(text)


def clean_page(page: Page, stats: Counter | None = None) -> Page:
    return replace(page, body=clean_markdown(page.body, stats))


def clean_pages(pages: list[Page]) -> tuple[list[Page], Counter]:
    stats = Counter()
    return [clean_page(p, stats) for p in pages], stats


# --- CLI ---------------------------------------------------------------------

def print_stats(pages: list[Page], cleaned: list[Page], stats: Counter) -> None:
    before = sum(len(p.body) for p in pages)
    after = sum(len(p.body) for p in cleaned)
    print(f"Pages: {len(pages):,}   Characters: {before:,} -> {after:,} ({1 - after / before:.0%} removed)\n")
    print("Shortcodes handled (count of opening tags):")
    for key, n in sorted(stats.items()):
        if n:
            print(f"  {key:<45}{n:>6}")

    leftovers = sum(len(SHORTCODE_RE.findall(p.body)) for p in cleaned)
    tags = sum(len(HTML_TAG_RE.findall(CODE_FENCE_RE.sub("", p.body))) for p in cleaned)
    print(f"\nLeft after cleaning: {leftovers} shortcodes, {tags} HTML tags outside code blocks")


def show_page(rel_path: str) -> None:
    page = load_page(config.CONTENT_DIR / rel_path)
    if page is None:
        raise SystemExit(f"{rel_path} is a draft")
    stats = Counter()
    cleaned = clean_page(page, stats)
    print(f"===== BEFORE ({len(page.body):,} chars) =====\n{page.body}")
    print(f"===== AFTER ({len(cleaned.body):,} chars) =====\n{cleaned.body}")
    print("===== SHORTCODES =====")
    for key, n in sorted(stats.items()):
        print(f"  {key:<45}{n:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean handbook pages and print stats.")
    parser.add_argument("sections", nargs="*", help="sections under content/handbook/ (default: config.DEFAULT_SECTIONS)")
    parser.add_argument("--all", action="store_true", help="clean the whole handbook")
    parser.add_argument("--show", metavar="PATH", help="show one page before/after, path relative to content/")
    args = parser.parse_args()

    if args.show:
        show_page(args.show)
        return
    sections = None if args.all else (args.sections or config.DEFAULT_SECTIONS)
    pages = load_pages(sections)
    cleaned, stats = clean_pages(pages)
    print_stats(pages, cleaned, stats)


if __name__ == "__main__":
    main()
