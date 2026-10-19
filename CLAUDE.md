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
answers from Ollama `qwen2.5:3b` (local, offline) or the DeepSeek API `deepseek-flash` (switch with `--llm` or `config.LLM_PROVIDER`) · CLI first, Streamlit later.

All settings live in `rag/config.py`. Run code with `.venv/bin/python -m rag.<module>`.
Pin dependency versions in `pyproject.toml`.

## Pipeline steps and progress

1. [x] Setup: venv, deps, `rag/config.py`, README, git init. Pushed to github.com/MutazAbueisheh/rag-handbook (private), branch `main`.
2. [x] Load (`rag/load.py`): walk `.md` files, parse front matter, build metadata
   (`path`, `url`, `title`, `description`, `section`), and print corpus stats.
   Start with the sections in `config.DEFAULT_SECTIONS`.
3. [x] Clean (`rag/clean.py`): shortcodes as above, plus stripping HTML to text.
4. [x] Chunk (`rag/chunk.py`): split on `##`/`###` headings and cap at ~500 tokens with ~50 overlap.
   Prefix each chunk with `title > heading path` and store the heading anchor for deep links.
5. [x] Embed + store (`rag/index.py`): batch-embed with Ollama and upsert into Chroma.
6. [x] Retrieve (`rag/retrieve.py`): embed the query, take the top-k, and allow an optional filter by section.
7. [x] Generate (`rag/query.py`): answer only from the retrieved context, cite sources by number
   (mapped to URLs in code), and reply "Not in the handbook." otherwise. Uses local Ollama `qwen2.5:3b`.
   Prompt layout was tuned by testing; see the comment above `SYSTEM_PROMPT`.
8. [x] CLI REPL (`rag/cli.py`) with a `--debug` flag that shows the retrieved chunks and their scores.
9. [x] Eval (`rag/evaluate.py`): 23 hand-written Q/A pairs (easy/medium/hard) in `eval/questions.yaml`;
   measures hit@k and MRR (chunk and page level), and with `--answers` checks LLM answers by keyword.
   Runs are saved to `eval/results/` for comparison.
10. [x] Iterate (details and evidence in `docs/experiments.md`). Added: 10 "everyday"-phrased eval questions,
   an LLM judge (`--judge`), BM25 from scratch (`rag/bm25.py`), hybrid search with RRF (K=10, default),
   LLM query rewriting (`--rewrite`) and an LLM reranker (`--rerank`). Best retrieval = `--rewrite --rerank`
   (hit@1 0.85, MRR 0.90 vs 0.61 / 0.73 for hybrid alone). Not done: index the whole handbook (~5-9 h) and
   re-run the eval, chunk-size experiments, merging tiny chunks, Streamlit UI.

Update the checkboxes as steps finish.

## Notes

- Embedding speed (CPU only, no GPU): ~275-550 tokens/s on real chunks. Default sections (1,574 chunks) took ~14 min; the whole handbook (~48k chunks, ~9M tokens) would take ~5-9 h. `rag.index` is a resumable sync, so it can run overnight.
- DeepSeek: key is `DEEPSEEK_API_KEY` in `~/.bashrc`. Non-interactive shells don't load it; use
  `eval "$(grep DEEPSEEK_API_KEY ~/.bashrc)"` before running. `deepseek-flash` = DeepSeek-V4.1-Flash
  (check `GET https://api.deepseek.com/models`); ~2-5 s per answer; full eval ~75 s.
- Eval baseline (23 questions, my manual review): qwen2.5:3b 15/23 correct; deepseek-flash 22/23
  (m1 "vacation cap" still fails because retrieval misses the global Flexible PTO chunk).
- Step 10 retrieval (31 answerable questions, chunk hit@6 / MRR): vector 0.87/0.71, hybrid(k=10) 0.90/0.73,
  hybrid+rewrite 0.94-0.97/0.68-0.75 over 3 runs. Rewrites vary run to run even at temperature 0, so
  compare rewrite configs over several runs. Sweep script idea: cache rewrites so configs share them.
- Step 10 end-to-end (33 questions, deepseek-flash answers, DeepSeek judge, correct/partial/wrong):
  vector 28/3/2, hybrid 30/3/0 (best, now default), hybrid+rewrite 28/3/2 (that run's rewrites lost p4).
  Remaining partials: m1 (Flexible PTO chunk still not retrieved), h4, p9 (interim bonus amount missed).
- Step 10 reranking (retrieval only, 3 runs each): hybrid hit@1 0.61 / MRR 0.73; +rerank 0.81 / 0.87;
  +rewrite+rerank 0.85 / 0.90 and the only config that finds m1. End-to-end answers stayed at 30/33 correct
  (small eval is at its ceiling). All DeepSeek experiments together cost about $0.46.
- LLM speed (qwen2.5:3b on CPU): reads ~30 prompt tokens/s, so a 6-chunk prompt (~2k tokens) takes ~60-80 s per answer.
- Load the `claude-api` skill before writing the Anthropic SDK code (if switching generation to Claude).
