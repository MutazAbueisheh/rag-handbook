# rag-handbook

A from-scratch Retrieval-Augmented Generation (RAG) exercise over the
[GitLab public handbook](https://handbook.gitlab.com/), read from the sibling
checkout at `../handbook/content`.

## Setup

```shell
~/.pyenv/versions/3.12.2/bin/python -m venv .venv
.venv/bin/pip install -e .
ollama pull nomic-embed-text
export ANTHROPIC_API_KEY=...   # needed for answer generation only
```

## Pipeline

load → clean → chunk → embed → store → retrieve → generate. Settings live in `rag/config.py`.
