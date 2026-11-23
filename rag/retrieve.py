"""Step 6 — Retrieve: find the chunks most similar to a question.

Basic mode ("vector"):
1. Embed the question (with the "search_query: " prefix, see rag/embed.py).
2. Ask Chroma for the k nearest chunk vectors, optionally only within some sections.

Step 10 adds two options, both chosen in config.py or per call:
- mode="hybrid": also run BM25 keyword search (rag/bm25.py) and merge the two rankings.
- rewrite=True:  an LLM rewrites the question into a few search queries in handbook wording;
                 we search with each and merge all the rankings.
Rankings are merged with Reciprocal Rank Fusion (RRF): a chunk scores sum(1 / (60 + rank))
over the lists it appears in. RRF only uses ranks, so it can merge cosine similarities and
BM25 scores, which are on completely different scales.

This is the part of a RAG that most often goes wrong: if the right chunk isn't in
the top k, the LLM can't answer, however good it is. That's why step 9 evaluates
this function on its own.

Run it:
    .venv/bin/python -m rag.retrieve "Can I use my personal laptop for work?"
    .venv/bin/python -m rag.retrieve "Can I work from a WeWork?" --mode hybrid --rewrite
"""

import argparse
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from rag import config
from rag.bm25 import BM25
from rag.embed import embed_query
from rag.index import get_collection


@dataclass
class Hit:
    score: float       # vector mode: cosine similarity (1 = same direction); fused modes: RRF score
    text: str          # chunk text, including its "Title > Heading" prefix
    url: str           # deep link for citations
    title: str
    heading: str
    section: str
    id: str


# --- The two search methods: each returns chunk ids, best first ----------------

def vector_search(question: str, n: int, where: dict | None) -> list[str]:
    res = get_collection().query(query_embeddings=[embed_query(question)], n_results=n, where=where, include=[])
    return res["ids"][0]


@lru_cache(maxsize=1)
def corpus() -> tuple[list[str], list[str], list[dict], BM25]:
    """All chunks from Chroma plus a BM25 index over them, built once per process (~1 s)."""
    # Read in pages: one get() of the whole collection fails on the full handbook (~48k chunks)
    # with SQLite's "too many SQL variables".
    collection, page = get_collection(), 5000
    ids, docs, metas = [], [], []
    for offset in range(0, collection.count(), page):
        data = collection.get(include=["documents", "metadatas"], limit=page, offset=offset)
        ids += data["ids"]
        docs += data["documents"]
        metas += data["metadatas"]
    return ids, docs, metas, BM25(docs)


def bm25_search(question: str, n: int, sections: list[str] | None) -> list[str]:
    ids, _, metas, bm25 = corpus()
    hits = bm25.search(question, n=n * 3 if sections else n)  # extra, since the filter drops some
    return [ids[i] for i, _ in hits if not sections or metas[i]["section"] in sections][:n]


def rrf(rankings: list[list[str]]) -> dict[str, float]:
    """Reciprocal Rank Fusion: merge several ranked lists of ids into one score per id."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, id_ in enumerate(ranking, 1):
            scores[id_] += 1 / (config.RRF_K + rank)
    return scores


# --- Query rewriting ----------------------------------------------------------

REWRITE_PROMPT = """You help search GitLab's company handbook (HR policies, benefits, time off, \
compensation, values). Rewrite the question below into {n} different short search queries that use \
the words a company handbook would likely use: official policy names, HR terms, synonyms. \
Reply with the {n} queries only, one per line, no numbering.

Question: {question}"""


def rewrite_queries(question: str) -> list[str]:
    from rag.llm import complete  # imported here: only needed when rewriting is on

    prompt = REWRITE_PROMPT.format(n=config.REWRITE_N, question=question)
    reply = complete([{"role": "user", "content": prompt}], llm=config.REWRITE_LLM)
    queries = [line.strip(" -•*\t") for line in reply.splitlines() if line.strip()]
    return queries[:config.REWRITE_N]


# --- Reranking ------------------------------------------------------------------
# Search compares the question with each chunk separately and cheaply (one vector, one keyword score).
# A reranker reads the question and many chunks together, so it can tell "how many vacation days
# can I take" apart from "how do I request vacation days". It is too slow to run on every chunk,
# so it only re-orders the few candidates that search already found.

RERANK_PROMPT = """Below is a question and {n} numbered passages from GitLab's company handbook. \
Pick the passages that best answer the question, most useful first. A passage that directly states the \
answer beats one that only mentions the same topic. Reply with the passage numbers only, comma-separated, \
at most {top}.

Question: {question}

{passages}"""


def rerank(question: str, hits: list[Hit], top: int) -> list[Hit]:
    """Re-order `hits`: the LLM's picks first, then the rest in their original order."""
    import re
    from rag.llm import complete  # imported here: only needed when reranking

    passages = "\n\n".join(f"[{i}] {h.text[:1500]}" for i, h in enumerate(hits, 1))
    prompt = RERANK_PROMPT.format(n=len(hits), top=top, question=question, passages=passages)
    reply = complete([{"role": "user", "content": prompt}], llm=config.RERANK_LLM)
    picked = []
    for n in map(int, re.findall(r"\d+", reply)):
        if 1 <= n <= len(hits) and n - 1 not in picked:
            picked.append(n - 1)
    rest = [i for i in range(len(hits)) if i not in picked]
    return [hits[i] for i in picked + rest]


# --- Putting it together ----------------------------------------------------------

def retrieve(question: str, k: int = config.TOP_K, section: str | list[str] | None = None,
             mode: str | None = None, rewrite: bool | None = None, use_rerank: bool | None = None) -> list[Hit]:
    # Read the config at call time (not as default values), so the eval can switch modes.
    mode = mode or config.RETRIEVAL_MODE
    rewrite = config.QUERY_REWRITE if rewrite is None else rewrite
    use_rerank = config.RERANK if use_rerank is None else use_rerank
    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("The index is empty. Run: .venv/bin/python -m rag.index")
    sections = [section] if isinstance(section, str) else section
    where = {"section": {"$in": sections}} if sections else None

    if mode == "vector" and not rewrite and not use_rerank:
        # The original step 6 path: one vector search, scores are cosine similarities.
        res = collection.query(query_embeddings=[embed_query(question)], n_results=k, where=where,
                               include=["documents", "metadatas", "distances"])
        return [
            # Chroma reports cosine *distance* (0 = identical); similarity is easier to read.
            Hit(1 - dist, doc, meta["url"], meta["title"], meta["heading"], meta["section"], id_)
            for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    queries = [question] + (rewrite_queries(question) if rewrite else [])
    rankings = []
    for q in queries:
        rankings.append(vector_search(q, config.FUSION_CANDIDATES, where))
        if mode == "hybrid":
            rankings.append(bm25_search(q, config.FUSION_CANDIDATES, sections))
    keep = max(k, config.RERANK_CANDIDATES) if use_rerank else k
    fused = sorted(rrf(rankings).items(), key=lambda x: x[1], reverse=True)[:keep]

    ids, docs, metas, _ = corpus()
    position = {id_: i for i, id_ in enumerate(ids)}
    hits = []
    for id_, score in fused:
        i = position[id_]
        m = metas[i]
        hits.append(Hit(score, docs[i], m["url"], m["title"], m["heading"], m["section"], id_))
    return rerank(question, hits, top=k)[:k] if use_rerank else hits


def print_hits(hits: list[Hit], chars: int = 300) -> None:
    for rank, h in enumerate(hits, 1):
        body = h.text.split("\n\n", 1)[-1]  # drop the prefix; the heading line shows it
        snippet = " ".join(body.split())[:chars]
        print(f"{rank}. [{h.score:.3f}]  {h.title}" + (f" > {h.heading}" if h.heading else ""))
        print(f"   {h.url}")
        print(textwrap.indent(textwrap.fill(snippet + " …", 100), "   ") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the chunks retrieved for a question.")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=config.TOP_K, help=f"number of chunks (default {config.TOP_K})")
    parser.add_argument("--section", action="append", help="only search this section; repeat for several")
    parser.add_argument("--mode", choices=("vector", "hybrid"), default=None, help=f"default: {config.RETRIEVAL_MODE}")
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=None,
                        help="let an LLM rewrite the question into extra search queries")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                        help="let an LLM re-order the top candidates")
    parser.add_argument("--chars", type=int, default=300, help="characters of each chunk to show")
    args = parser.parse_args()

    if args.rewrite or (args.rewrite is None and config.QUERY_REWRITE):
        print("Rewritten queries:", *rewrite_queries(args.question), sep="\n  ")
        print()
    print_hits(retrieve(args.question, k=args.k, section=args.section, mode=args.mode, rewrite=args.rewrite,
                          use_rerank=args.rerank),
               chars=args.chars)


if __name__ == "__main__":
    main()
