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
CHUNK_SPLIT_LEVEL = 3          # #, ##, ### always start a new chunk; deeper headings only when the chunk is full
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50      # only used when one section is too long and must be cut into pieces
TOKENS_PER_WORD = 1.3          # rough estimate for nomic-embed-text's tokenizer; see rag/chunk.py

# Embeddings (local, via Ollama)
EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH_SIZE = 32          # texts per Ollama call
# nomic-embed-text was trained with task prefixes; leaving them out lowers retrieval quality.
EMBED_DOC_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "

# Vector store (Chroma, persisted to a folder)
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "handbook"

# Retrieval
TOP_K = 6
# Step 10 results (31 eval questions, chunk hit@6): vector 0.87, hybrid 0.90, hybrid+rewrite 0.94-0.97.
RETRIEVAL_MODE = "hybrid"      # "vector" (embeddings only) or "hybrid" (embeddings + BM25 keywords)
QUERY_REWRITE = False          # LLM rewrites the question into extra search queries (best, but needs the API)
RERANK = False                 # an LLM re-orders the top candidates by reading them with the question
RERANK_CANDIDATES = 20         # how many fused candidates the reranker sees
RERANK_LLM = "deepseek"
REWRITE_N = 3                  # how many extra queries
REWRITE_LLM = "deepseek"       # which provider rewrites (see LLM_MODELS below)
FUSION_CANDIDATES = 30         # in hybrid/rewrite modes, how deep each ranked list goes before merging
RRF_K = 10                     # Reciprocal Rank Fusion constant. 60 is the textbook value, but with it a
                               # chunk ranked ~8th in every list beat one ranked 2nd in a single list;
                               # 10 gives top ranks more weight (hybrid hit@3: 0.77 -> 0.87)

# Generation. Pick the provider here, or per run with --llm on query / cli / evaluate.
#   ollama    local and offline, ~60-80 s per answer on this CPU
#   deepseek  DeepSeek API (needs DEEPSEEK_API_KEY), ~2-5 s per answer, sends prompts to DeepSeek
LLM_PROVIDER = "ollama"
LLM_MODELS = {
    "ollama": "qwen2.5:3b",
    "deepseek": "deepseek-flash",   # DeepSeek-V4.1-Flash; list models: GET https://api.deepseek.com/models
}
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_TEMPERATURE = 0.0          # no creativity: stick to the sources, same answer every time
# Context window in tokens. Ollama's default is smaller than our prompt (6 chunks ≈ 4k tokens)
# and it silently drops the start of a prompt that doesn't fit, so set it explicitly.
LLM_NUM_CTX = 8192
NOT_FOUND = "Not in the handbook."
