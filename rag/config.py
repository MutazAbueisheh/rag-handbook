"""Central settings for the RAG pipeline. Change values here, not in each module."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source corpus: the handbook checkout next to this project (read-only).
HANDBOOK_ROOT = PROJECT_ROOT.parent / "handbook"
CONTENT_DIR = HANDBOOK_ROOT / "content"
INCLUDES_DIR = HANDBOOK_ROOT / "assets"  # `{{% include "includes/x.md" %}}` is relative to assets/
SITE_URL = "https://handbook.gitlab.com"

# Sections under content/handbook/ to index. Start small; set to None for everything.
DEFAULT_SECTIONS = ["values", "people-group", "total-rewards"]

# Chunking
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# Embeddings (local, via Ollama)
EMBED_MODEL = "nomic-embed-text"

# Vector store (Chroma, persisted to a folder)
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "handbook"

# Retrieval + generation
TOP_K = 6
LLM_MODEL = "claude-haiku-5-5"
