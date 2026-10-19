"""Turn text into vectors with the local Ollama embedding model.

Used by step 5 (index the chunks) and step 6 (embed the user's question).

nomic-embed-text is an *asymmetric* model: it was trained on pairs of
"search_query: <question>" and "search_document: <passage>", so documents and
queries get different prefixes. Both sides must use them, or matches get worse.
"""

import ollama

from rag import config


def embed_documents(texts: list[str]) -> list[list[float]]:
    """One 768-dim vector per text. Call it with a batch: one HTTP request per batch."""
    response = ollama.embed(model=config.EMBED_MODEL, input=[config.EMBED_DOC_PREFIX + t for t in texts])
    return response["embeddings"]


def embed_query(text: str) -> list[float]:
    response = ollama.embed(model=config.EMBED_MODEL, input=config.EMBED_QUERY_PREFIX + text)
    return response["embeddings"][0]
