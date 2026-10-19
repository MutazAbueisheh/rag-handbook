"""Step 5 — Embed + store: put every chunk's vector and metadata into Chroma.

Chroma is a vector database stored as files in chroma_db/. For each chunk it keeps:
- id         the chunk id ("handbook/values/_index.md#3"), so re-running replaces, not duplicates
- embedding  the 768 numbers from nomic-embed-text
- document   the chunk text (returned at query time and sent to the LLM)
- metadata   url, title, heading, section, ... (for citations and filtering)

Indexing is a *sync*, so re-running is cheap:
- chunks whose text is unchanged (same hash) are skipped, not re-embedded
- chunks that are new or changed are embedded and upserted
- chunks that no longer exist (e.g. after a chunking change) are deleted
Each batch is saved as soon as it's embedded, so an interrupted run resumes where it stopped.

Run it:
    .venv/bin/python -m rag.index            # sections in config.DEFAULT_SECTIONS
    .venv/bin/python -m rag.index --all      # the whole handbook (~48k chunks, slow)
    .venv/bin/python -m rag.index --reset    # delete the collection first, then rebuild
"""

import argparse
import hashlib
import time

import chromadb

from rag import config
from rag.chunk import Chunk, chunk_pages
from rag.clean import clean_pages
from rag.embed import embed_documents
from rag.load import load_pages


def get_collection(reset: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    if reset and config.COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION_NAME)
    return client.get_or_create_collection(
        config.COLLECTION_NAME,
        # Cosine similarity: compares the direction of two vectors, ignoring their length.
        configuration={"hnsw": {"space": "cosine"}},
        # We embed with Ollama ourselves; without this Chroma would attach its own default model.
        embedding_function=None,
    )


def text_hash(chunk: Chunk) -> str:
    """Fingerprint of everything that goes into the vector (model + prefix + text)."""
    key = f"{config.EMBED_MODEL}|{config.EMBED_DOC_PREFIX}|{chunk.text}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def metadata(chunk: Chunk) -> dict:
    return {
        "url": chunk.url,
        "page_url": chunk.page_url,
        "title": chunk.title,
        "heading": chunk.heading,
        "section": chunk.section,
        "path": chunk.path,
        "hash": text_hash(chunk),
    }


def sync(collection: chromadb.Collection, chunks: list[Chunk], sections: list[str] | None) -> None:
    # What's stored already? Only look at the sections we're indexing, so indexing
    # "values" doesn't delete the "legal" chunks from an earlier run.
    where = {"section": {"$in": sections}} if sections else None
    stored = collection.get(where=where, include=["metadatas"])
    stored_hash = {id_: m["hash"] for id_, m in zip(stored["ids"], stored["metadatas"])}

    current_ids = {c.id for c in chunks}
    stale = [id_ for id_ in stored_hash if id_ not in current_ids]
    todo = [c for c in chunks if stored_hash.get(c.id) != text_hash(c)]
    print(f"Chunks: {len(chunks):,}   unchanged: {len(chunks) - len(todo):,}   "
          f"to embed: {len(todo):,}   stale to delete: {len(stale):,}")

    if stale:
        collection.delete(ids=stale)

    start = time.time()
    for i in range(0, len(todo), config.EMBED_BATCH_SIZE):
        batch = todo[i:i + config.EMBED_BATCH_SIZE]
        collection.upsert(
            ids=[c.id for c in batch],
            embeddings=embed_documents([c.text for c in batch]),
            documents=[c.text for c in batch],
            metadatas=[metadata(c) for c in batch],
        )
        done = i + len(batch)
        rate = done / (time.time() - start)
        eta = (len(todo) - done) / rate
        print(f"\r  embedded {done:,}/{len(todo):,}   {rate:.0f} chunks/s   ETA {eta / 60:.1f} min ", end="", flush=True)
    if todo:
        print(f"\n  done in {(time.time() - start) / 60:.1f} min")
    print(f"Collection '{collection.name}' now holds {collection.count():,} chunks in {config.CHROMA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and store them in Chroma.")
    parser.add_argument("sections", nargs="*", help="sections under content/handbook/ (default: config.DEFAULT_SECTIONS)")
    parser.add_argument("--all", action="store_true", help="index the whole handbook")
    parser.add_argument("--reset", action="store_true", help="delete the collection and rebuild from scratch")
    args = parser.parse_args()

    sections = None if args.all else (args.sections or config.DEFAULT_SECTIONS)
    print(f"Indexing: {'everything' if sections is None else ', '.join(sections)}")
    pages, _ = clean_pages(load_pages(sections))
    chunks = chunk_pages(pages)
    sync(get_collection(reset=args.reset), chunks, sections)


if __name__ == "__main__":
    main()
