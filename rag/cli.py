"""Step 8 — CLI: a chat loop over the handbook.

    .venv/bin/python -m rag.cli                 # ask questions, get cited answers
    .venv/bin/python -m rag.cli --debug         # also show the retrieved chunks and their scores
    .venv/bin/python -m rag.cli --llm deepseek  # answer with the DeepSeek API instead of local Ollama
    .venv/bin/python -m rag.cli --section total-rewards --k 4

Inside the loop:
    :debug            turn debug output on/off
    :k 4              change how many chunks are retrieved
    :section NAME     only search one section (":section all" to search everything)
    :quit             leave (or Ctrl-D)
"""

import argparse
import time

from rag import config
from rag.query import PROVIDERS, Answer, print_footer, stream_answer
from rag.retrieve import print_hits, retrieve

HELP = "Commands: :debug  :k N  :section NAME|all  :quit"


def answer_one(question: str, k: int, section: list[str] | None, debug: bool, llm: str) -> None:
    start = time.time()
    hits = retrieve(question, k=k, section=section)
    if debug:
        print(f"\n--- retrieved {len(hits)} chunks (score = cosine similarity) ---")
        print_hits(hits, chars=250)
        print("--- answer ---")

    answer = Answer(question)
    for piece in stream_answer(question, answer=answer, hits=hits, llm=llm):
        print(piece, end="", flush=True)
    print("\n")
    print_footer(answer, time.time() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the GitLab handbook.")
    parser.add_argument("--debug", action="store_true", help="show retrieved chunks and scores")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--section", action="append", help="only search this section; repeat for several")
    parser.add_argument("--llm", choices=PROVIDERS, default=config.LLM_PROVIDER, help="which LLM writes the answers")
    parser.add_argument("--mode", choices=("vector", "hybrid"), default=config.RETRIEVAL_MODE, help="retrieval mode")
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=config.QUERY_REWRITE,
                        help="LLM query rewriting before search")
    args = parser.parse_args()
    config.RETRIEVAL_MODE, config.QUERY_REWRITE = args.mode, args.rewrite
    debug, k, section = args.debug, args.k, args.section

    print(f"GitLab handbook Q&A  ·  {args.llm}/{config.LLM_MODELS[args.llm]}")
    print(HELP)
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":quit", ":q", "exit", "quit"):
            break
        if line == ":debug":
            debug = not debug
            print(f"debug {'on' if debug else 'off'}")
        elif line.startswith(":k "):
            k = int(line.split()[1])
            print(f"k = {k}")
        elif line.startswith(":section "):
            name = line.split(maxsplit=1)[1]
            section = None if name == "all" else [name]
            print(f"section = {name}")
        elif line.startswith(":"):
            print(HELP)
        else:
            answer_one(line, k, section, debug, args.llm)


if __name__ == "__main__":
    main()
