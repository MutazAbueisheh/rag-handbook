"""Step 4 — Chunk: split each cleaned page into small, self-describing pieces.

Why chunk at all? We embed each chunk as one vector. One vector for a 16k-word page
would average everything on it into mush; one vector per section stays on-topic.

How:
1. Split the page into sections at every heading. Sections are the author's own topic
   boundaries, so they make the best chunks.
2. Group them: a heading at CHUNK_SPLIT_LEVEL or higher (#, ##, ###) always starts a new
   chunk. Deeper sub-sections (####, #####) are packed together into one chunk while they
   fit in CHUNK_MAX_TOKENS, and a new chunk starts at a sub-heading only when needed.
3. A single section that is still too long is cut into pieces: first at blank lines
   (paragraphs), then at line breaks (table rows, list items), then at words.
   Consecutive pieces share CHUNK_OVERLAP_TOKENS so a sentence cut in half appears in both.
4. Prefix every chunk with "Page title > Heading > Subheading". After splitting, a chunk
   often doesn't name its own topic; the prefix puts it back for the embedding model.
5. Remember the heading's anchor, so citations can link to the exact section.

Run it:
    .venv/bin/python -m rag.chunk                 # stats for config.DEFAULT_SECTIONS
    .venv/bin/python -m rag.chunk --all           # stats for the whole handbook
    .venv/bin/python -m rag.chunk --show handbook/people-group/visas.md
"""

import argparse
import re
import statistics
from collections import Counter
from dataclasses import dataclass

from rag import config
from rag.clean import clean_page, clean_pages
from rag.load import Page, load_page, load_pages


@dataclass
class Chunk:
    id: str            # stable id, e.g. "handbook/values/_index.md#12"; lets us re-index in place
    text: str          # what gets embedded: heading-path prefix + section text
    url: str           # deep link: page URL + "#anchor"
    page_url: str
    title: str         # page title
    heading: str       # heading path inside the page, e.g. "Collaboration > Kindness"
    section: str
    path: str


# --- Token counting ------------------------------------------------------------
# The embedding model counts in WordPiece tokens: common words are 1 token, rare words,
# numbers and punctuation are more. We don't need exact counts to pick chunk sizes, so we
# estimate from words instead of adding a tokenizer dependency.
# Calibration (200 random chunks, real counts from Ollama's `prompt_eval_count`):
# median 1.32 tokens/word, p10 1.16, p90 1.70 (numbers, code, odd names). So a "500-token"
# chunk is really 450-650 tokens, well within the model's 2048-token window in Ollama.

def count_tokens(text: str) -> int:
    return round(len(text.split()) * config.TOKENS_PER_WORD)


# --- 1. Split a page into heading sections -----------------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{#([^}]+)\})?\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def heading_anchor(text: str, seen: Counter) -> str:
    """Hugo's default ("github") heading id: lowercase, spaces -> '-', drop other punctuation.
    A repeated heading on the same page gets -1, -2, ... like Hugo does."""
    slug = "".join(
        ch if ch.isalnum() or ch in "-_" else "-" if ch.isspace() else ""
        for ch in text.lower()
    )
    n = seen[slug]
    seen[slug] += 1
    return f"{slug}-{n}" if n else slug


def plain_heading(text: str) -> str:
    """Heading text without Markdown emphasis: '**Step 1:** `foo`' -> 'Step 1: foo'."""
    return re.sub(r"[*_`]", "", text).strip()


@dataclass
class Section:
    level: int         # heading level, 1-6; 0 for the text before the first heading
    path: list[str]    # heading texts from the top of the page down to this section
    anchor: str        # "" for the text before the first heading
    text: str          # the section's own text, without its heading line or sub-sections


def split_sections(body: str) -> list[Section]:
    sections = [Section(0, [], "", "")]
    stack: list[tuple[int, str]] = []  # (level, heading text) of the headings we're inside
    lines: list[str] = []
    seen = Counter()
    in_code = False

    for line in body.splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code  # a "# comment" inside a code block is not a heading
        m = None if in_code else HEADING_RE.match(line)
        if not m:
            lines.append(line)
            continue
        sections[-1].text = "\n".join(lines).strip()
        level, text, explicit_id = len(m[1]), plain_heading(m[2]), m[3]
        stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, text)]
        sections.append(Section(level, [h for _, h in stack], explicit_id or heading_anchor(text, seen), ""))
        lines = []
    sections[-1].text = "\n".join(lines).strip()
    return sections


def group_sections(sections: list[Section]) -> list[list[Section]]:
    """A heading at CHUNK_SPLIT_LEVEL or higher starts a new group; deeper ones join the current one."""
    groups: list[list[Section]] = []
    for s in sections:
        if not groups or s.level <= config.CHUNK_SPLIT_LEVEL:
            groups.append([])
        groups[-1].append(s)
    return groups


# --- 2. Cut long sections into overlapping pieces -----------------------------

def split_units(text: str, budget: int) -> list[str]:
    """Break text into the largest units that fit: paragraphs, then lines, then word runs."""
    units = []
    for para in re.split(r"\n\s*\n", text):
        if count_tokens(para) <= budget:
            units.append(para)
            continue
        for line in para.split("\n"):
            if count_tokens(line) <= budget:
                units.append(line)
                continue
            words = line.split()
            step = int(budget / config.TOKENS_PER_WORD)
            units += [" ".join(words[i:i + step]) for i in range(0, len(words), step)]
    return units


def tail_words(text: str, tokens: int) -> str:
    """The last `tokens` worth of words, used as overlap at the start of the next piece.
    Sliced from the original text so line breaks (lists, table rows) survive."""
    n = int(tokens / config.TOKENS_PER_WORD)
    starts = [m.start() for m in re.finditer(r"\S+", text)]
    return text[starts[max(len(starts) - n, 0)]:] if n and starts else ""


def split_long(text: str, budget: int) -> list[str]:
    """Pack units greedily into pieces of at most `budget` tokens, with overlap between them."""
    if count_tokens(text) <= budget:
        return [text]
    pieces, current = [], []
    for unit in split_units(text, budget):
        candidate = "\n\n".join(current + [unit])
        if current and count_tokens(candidate) > budget:
            pieces.append("\n\n".join(current))
            overlap = tail_words(pieces[-1], config.CHUNK_OVERLAP_TOKENS)
            fits = count_tokens(overlap) + count_tokens(unit) <= budget
            current = [overlap, unit] if overlap and fits else [unit]
        else:
            current.append(unit)
    pieces.append("\n\n".join(current))
    return pieces


# --- 3. Pack sections into chunks, with prefix and anchor -----------------------

def render(sections: list[Section]) -> str:
    """Join sections into one text. The first section's heading is in the prefix; later
    (deeper) headings stay inline so the reader and the embedding model still see them.
    A heading is joined to its text by a single newline, so when a long section is cut at
    blank lines, the heading stays with the paragraph below it."""
    parts = [sections[0].text] + [f"{'#' * s.level} {s.path[-1]}\n{s.text}" for s in sections[1:]]
    return "\n\n".join(p for p in parts if p.strip())


def chunk_page(page: Page) -> list[Chunk]:
    chunks = []

    def heading_path(section: Section) -> list[str]:
        return [h for h in section.path if h != page.title]  # a "# Title" heading would repeat the title

    def budget_for(section: Section) -> int:
        """Tokens left for text after the prefix of a chunk that starts at `section`."""
        return config.CHUNK_MAX_TOKENS - count_tokens(" > ".join([page.title] + heading_path(section)))

    def emit(sections: list[Section]):
        first = sections[0]
        path = heading_path(first)
        prefix = " > ".join([page.title] + path)
        for piece in split_long(render(sections), budget_for(first)):
            chunks.append(Chunk(
                id=f"{page.path}#{len(chunks)}",
                text=f"{prefix}\n\n{piece}",
                url=f"{page.url}#{first.anchor}" if first.anchor else page.url,
                page_url=page.url,
                title=page.title,
                heading=" > ".join(path),
                section=page.section,
                path=page.path,
            ))

    for group in group_sections(split_sections(page.body)):
        # A heading with no text of its own (directly followed by a sub-heading) is skipped;
        # its name still appears in the prefix of the chunks below it.
        group = [s for s in group if s.text]
        current: list[Section] = []
        for s in group:
            if current and count_tokens(render(current + [s])) > budget_for(current[0]):
                emit(current)
                current = []
            current.append(s)
        if current:
            emit(current)
    return chunks


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    return [chunk for page in pages for chunk in chunk_page(page)]


# --- CLI -------------------------------------------------------------------------

def print_stats(pages: list[Page], chunks: list[Chunk]) -> None:
    sizes = sorted(count_tokens(c.text) for c in chunks)
    print(f"Pages: {len(pages):,}   Chunks: {len(chunks):,}   ({len(chunks) / len(pages):.1f} per page)\n")
    print("Chunk size in estimated tokens (including the prefix):")
    print(f"  min {sizes[0]}   median {statistics.median(sizes):.0f}   "
          f"p90 {sizes[int(len(sizes) * 0.9)]}   max {sizes[-1]}")
    buckets = Counter(min(s // 100, 5) for s in sizes)
    for b in range(6):
        label = f"{b * 100}-{b * 100 + 99}" if b < 5 else "500"
        print(f"  {label:>8}: {buckets[b]:>6}  {'#' * round(60 * buckets[b] / len(sizes))}")

    tiny = [c for c in chunks if count_tokens(c.text.split("\n\n", 1)[1]) < 20]
    with_anchor = sum("#" in c.url for c in chunks)
    print(f"\nChunks with under 20 tokens of text: {len(tiny):,}")
    print(f"Chunks with a deep-link anchor: {with_anchor:,} of {len(chunks):,}")


def show_page(rel_path: str) -> None:
    page = load_page(config.CONTENT_DIR / rel_path)
    if page is None:
        raise SystemExit(f"{rel_path} is a draft")
    for c in chunk_page(clean_page(page)):
        print(f"===== {c.id}   ~{count_tokens(c.text)} tokens   {c.url}")
        print(c.text if len(c.text) < 1200 else c.text[:600] + "\n  [...]\n" + c.text[-400:])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk handbook pages and print stats.")
    parser.add_argument("sections", nargs="*", help="sections under content/handbook/ (default: config.DEFAULT_SECTIONS)")
    parser.add_argument("--all", action="store_true", help="chunk the whole handbook")
    parser.add_argument("--show", metavar="PATH", help="show one page's chunks, path relative to content/")
    args = parser.parse_args()

    if args.show:
        show_page(args.show)
        return
    sections = None if args.all else (args.sections or config.DEFAULT_SECTIONS)
    pages, _ = clean_pages(load_pages(sections))
    print_stats(pages, chunk_pages(pages))


if __name__ == "__main__":
    main()
