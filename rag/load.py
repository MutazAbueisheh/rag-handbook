"""Step 2 — Load: read handbook Markdown files into Page objects with metadata.

Run it to print corpus stats:
    .venv/bin/python -m rag.load                # sections in config.DEFAULT_SECTIONS
    .venv/bin/python -m rag.load --all          # the whole handbook
    .venv/bin/python -m rag.load values legal   # specific sections
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import frontmatter

from rag import config


@dataclass
class Page:
    path: str          # relative to content/, e.g. "handbook/values/_index.md"
    url: str           # public URL, used for citations
    title: str
    description: str
    section: str       # e.g. "values"; used later as a retrieval filter
    body: str          # raw Markdown, front matter removed (cleaned in step 3)


def path_to_url(rel_path: Path) -> str:
    """Map a content path to its public URL, the way Hugo does.

    handbook/values/_index.md -> https://handbook.gitlab.com/handbook/values/
    handbook/foo/bar.md       -> https://handbook.gitlab.com/handbook/foo/bar/
    Hugo lowercases URLs by default, so we do too.
    """
    parts = list(rel_path.with_suffix("").parts)
    if parts[-1] == "_index":  # a section's index page is the folder URL itself
        parts = parts[:-1]
    return f"{config.SITE_URL}/{'/'.join(parts).lower()}/"


def section_of(rel_path: Path) -> str:
    """The section a page belongs to.

    Most pages live under handbook/<section>/; a few top-level trees
    (teamops/, job-description-library/) are their own section.
    """
    parts = rel_path.parts
    if parts[0] == "handbook":
        return parts[1] if len(parts) > 2 else "handbook"  # handbook/_index.md -> "handbook"
    return parts[0]


def iter_markdown_files(sections: list[str] | None) -> list[Path]:
    """All .md files to load, sorted so runs are reproducible."""
    if sections is None:
        roots = [config.CONTENT_DIR]
    else:
        roots = [config.CONTENT_DIR / "handbook" / s for s in sections]
        missing = [r for r in roots if not r.is_dir()]
        if missing:
            raise SystemExit(f"Unknown section(s): {', '.join(r.name for r in missing)}")
    return sorted(p for root in roots for p in root.rglob("*.md"))


def load_page(file: Path) -> Page | None:
    """Parse one file. Returns None for drafts, which Hugo doesn't publish."""
    post = frontmatter.load(file)
    if post.get("draft") is True:
        return None
    rel = file.relative_to(config.CONTENT_DIR)
    # Fall back to the folder/file name if a page has no title.
    fallback_title = (rel.parent.name if rel.stem == "_index" else rel.stem).replace("-", " ").title()
    return Page(
        path=str(rel),
        url=path_to_url(rel),
        title=str(post.get("title") or fallback_title).strip(),
        description=str(post.get("description") or "").strip(),
        section=section_of(rel),
        body=post.content,
    )


def load_pages(sections: list[str] | None = config.DEFAULT_SECTIONS) -> list[Page]:
    """Load every published page in the given sections (None = whole handbook)."""
    pages = []
    for file in iter_markdown_files(sections):
        page = load_page(file)
        if page is not None:
            pages.append(page)
    return pages


def print_stats(pages: list[Page]) -> None:
    words = {p.path: len(p.body.split()) for p in pages}
    total = sum(words.values())
    print(f"Pages: {len(pages):,}   Words: {total:,}   (~{int(total * 1.3):,} tokens)\n")

    pages_per_section = Counter(p.section for p in pages)
    words_per_section = Counter()
    for p in pages:
        words_per_section[p.section] += words[p.path]
    print(f"{'section':<28}{'pages':>7}{'words':>11}")
    for section, n in pages_per_section.most_common():
        print(f"{section:<28}{n:>7}{words_per_section[section]:>11,}")

    no_title = [p for p in pages if not p.title]
    no_desc = [p for p in pages if not p.description]
    tiny = [p for p in pages if words[p.path] < 50]
    print(f"\nMissing description: {len(no_desc)}   Under 50 words: {len(tiny)}   Empty title: {len(no_title)}")

    print("\nLargest pages:")
    for p in sorted(pages, key=lambda p: words[p.path], reverse=True)[:5]:
        print(f"  {words[p.path]:>7,}  {p.url}")

    sample = pages[0]
    print("\nSample page:")
    for field in ("path", "url", "title", "description", "section"):
        print(f"  {field:<12} {getattr(sample, field)}")
    print(f"  {'body':<12} {sample.body[:200]!r}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load handbook pages and print stats.")
    parser.add_argument("sections", nargs="*", help="sections under content/handbook/ (default: config.DEFAULT_SECTIONS)")
    parser.add_argument("--all", action="store_true", help="load the whole handbook")
    args = parser.parse_args()

    sections = None if args.all else (args.sections or config.DEFAULT_SECTIONS)
    print(f"Loading: {'everything' if sections is None else ', '.join(sections)}\n")
    print_stats(load_pages(sections))


if __name__ == "__main__":
    main()
