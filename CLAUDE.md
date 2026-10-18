# rag-handbook

A from-scratch RAG (Retrieval-Augmented Generation) built as a **learning exercise**.
It is the user's first RAG, so explain choices and keep each step's code readable.
Work **step by step** and pause for review after each step.

Full plan, with the reasons behind each design option:
`/home/mabueisheh/.claude/plans/i-want-to-use-iridescent-falcon.md`

## Corpus

- The GitLab public handbook, a Hugo site, checked out at `../handbook`
  (`/home/mabueisheh/code/RAG/handbook`). **Read-only: never edit or commit there.**
- Pages are Markdown files under `../handbook/content/` (~4,600 files, ~6.3M words) with YAML front matter (`title`, `description`).
- Page path to public URL: `content/handbook/values/_index.md` → `https://handbook.gitlab.com/handbook/values/`.
  A non-index page `foo/bar.md` → `/foo/bar/`.
- Hugo shortcodes must be cleaned in four ways:
  - wrappers (`alert`, `details`, `panel`): keep the inner text
  - `{{% include "includes/x.md" %}}`: inline the file from `../handbook/assets/`
  - data lookups (`member-by-name "X"`, `label`): replace with the argument
  - embeds (`youtube`) and unknown shortcodes: drop, and log a count of what was dropped

## Chosen stack (approved)

From scratch (no LlamaIndex/LangChain) · Python 3.12.2 venv in `.venv` · Ollama
`nomic-embed-text` (768-dim) for embeddings · Chroma persisted to `chroma_db/` ·
Claude (`claude-haiku-5-5`, with `claude-sonnet-5-5` as an option) for answers · CLI first, Streamlit later.

All settings live in `rag/config.py`. Run code with `.venv/bin/python -m rag.<module>`.
Pin dependency versions in `pyproject.toml`.

## Pipeline steps and progress

1. [x] Setup: venv, deps, `rag/config.py`, README, git init. Pushed to github.com/MutazAbueisheh/rag-handbook (private), branch `main`.
2. [x] Load (`rag/load.py`): walk `.md` files, parse front matter, build metadata
   (`path`, `url`, `title`, `description`, `section`), and print corpus stats.
   Start with the sections in `config.DEFAULT_SECTIONS`.
3. [x] Clean (`rag/clean.py`): shortcodes as above, plus stripping HTML to text.
4. [ ] Chunk (`rag/chunk.py`): split on `##`/`###` headings and cap at ~500 tokens with ~50 overlap.
   Prefix each chunk with `title > heading path` and store the heading anchor for deep links.
5. [ ] Embed + store (`rag/index.py`): batch-embed with Ollama and upsert into Chroma.
6. [ ] Retrieve: embed the query, take the top-k, and allow an optional filter by section.
7. [ ] Generate (`rag/query.py`): answer only from the retrieved context, cite URLs,
   and reply "not in the handbook" otherwise. Needs `ANTHROPIC_API_KEY`, which is **not set yet**.
8. [ ] CLI REPL with a `--debug` flag that shows the retrieved chunks and their scores.
9. [ ] Eval: 20–30 hand-written Q/A pairs in `eval/questions.yaml`, then measure hit@k and MRR.
10. [ ] Iterate: chunk sizes, hybrid BM25 + vector search, a reranker, query rewriting. Optionally a Streamlit UI.

Update the checkboxes as steps finish.

## Notes

- Embedding speed measured at setup: a batch of 32 short texts took ~1.6 s on this machine.
- Load the `claude-api` skill before writing the Anthropic SDK code.
