"""Step 7 — Generate: answer a question from the retrieved chunks with an LLM.

1. Retrieve the top-k chunks for the question (step 6).
2. Put them in the prompt as numbered sources [1]..[k], with strict rules:
   answer only from the sources, cite them by number, or say "Not in the handbook."
3. The LLM writes the answer; we map its [n] citations back to URLs.

Two interchangeable LLM providers (rag/llm.py, config.LLM_PROVIDER), same prompt for both:
- ollama:   local qwen2.5:3b, offline, slow on CPU
- deepseek: DeepSeek API (deepseek-flash), fast, needs DEEPSEEK_API_KEY

Why cite by number and not by URL? Small models often mangle long URLs. A number is
easy to copy, and our code knows exactly which deep link each number stands for.

Run it:
    .venv/bin/python -m rag.query "What does CREDIT stand for?"
    .venv/bin/python -m rag.query "What does CREDIT stand for?" --llm deepseek
"""

import argparse
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from rag import config
from rag.llm import PROVIDERS
from rag.retrieve import Hit, retrieve

# Prompt layout matters a lot for a 3B model. What we found by testing (step 7 notes):
# - Rules placed *before* ~1,500 tokens of sources were mostly ignored: the model wrote just "[1]".
# - Whatever comes *last* dominates. Ending on the refusal rule made it refuse everything,
#   even questions whose answer was source [1].
# So the refusal rule lives in the system prompt, and the user message ends with how to answer.
SYSTEM_PROMPT = f"""You answer questions about GitLab's company handbook, using only the sources the user provides.
If the sources contain nothing relevant to the question, reply exactly: {config.NOT_FOUND}"""

ANSWER_INSTRUCTIONS = """Answer the question using the sources. If they answer it only partly or indirectly, \
say what they do say. Write full sentences and put the source number after each fact, like [2]."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    """The user message: numbered sources first, then the question, then how to answer.
    Sources come first so the model reads the question and instructions last, right before
    it writes (and so Ollama can reuse its cache when only the question changes)."""
    sources = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, 1))
    return f"Sources:\n\n{sources}\n\n---\n\nQuestion: {question}\n\n{ANSWER_INSTRUCTIONS}"


@dataclass
class Answer:
    question: str
    text: str = ""
    hits: list[Hit] = field(default_factory=list)
    model: str = ""
    stats: dict = field(default_factory=dict)     # token counts (and timings, for Ollama)

    @property
    def cited(self) -> list[tuple[int, Hit]]:
        """The sources the answer actually cites, as (number, hit), in number order."""
        numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", self.text)})
        return [(n, self.hits[n - 1]) for n in numbers if 1 <= n <= len(self.hits)]


# --- Answering -----------------------------------------------------------------

def stream_answer(question: str, k: int = config.TOP_K, section=None, answer: Answer | None = None,
                  hits: list[Hit] | None = None, llm: str = config.LLM_PROVIDER) -> Iterator[str]:
    """Yield the answer text piece by piece as the model writes it, and fill in `answer`.
    Pass `hits` if you already retrieved them (the CLI does, to show them first)."""
    answer = answer if answer is not None else Answer(question)
    answer.hits = hits if hits is not None else retrieve(question, k=k, section=section)
    answer.model = config.LLM_MODELS[llm]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, answer.hits)},
    ]
    for piece in PROVIDERS[llm](answer.model, messages, answer.stats):
        answer.text += piece
        yield piece


def ask(question: str, k: int = config.TOP_K, section=None, llm: str = config.LLM_PROVIDER,
        hits: list[Hit] | None = None) -> Answer:
    """Non-streaming version: returns the full Answer (used by the eval)."""
    answer = Answer(question)
    for _ in stream_answer(question, k=k, section=section, answer=answer, llm=llm, hits=hits):
        pass
    return answer


def print_footer(answer: Answer, seconds: float) -> None:
    """The cited sources with their URLs, then timing."""
    if answer.cited:
        print("Sources:")
        for n, hit in answer.cited:
            print(f"  [{n}] {hit.title}" + (f" > {hit.heading}" if hit.heading else ""))
            print(f"      {hit.url}")
    s = answer.stats
    detail = f"{s.get('prompt_tokens', '?')} prompt tokens in, {s.get('answer_tokens', '?')} out"
    if "prompt_seconds" in s:
        detail += f"; reading took {s['prompt_seconds']:.0f}s, writing {s['answer_seconds']:.0f}s"
    if s.get("reasoning_tokens"):
        detail += f" (incl. {s['reasoning_tokens']} hidden reasoning)"
    print(f"\n({answer.model}: {seconds:.1f}s total, {detail})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question from the handbook.")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--section", action="append", help="only search this section; repeat for several")
    parser.add_argument("--llm", choices=PROVIDERS, default=config.LLM_PROVIDER, help="which LLM writes the answer")
    parser.add_argument("--mode", choices=("vector", "hybrid"), default=config.RETRIEVAL_MODE, help="retrieval mode")
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=config.QUERY_REWRITE,
                        help="LLM query rewriting before search")
    args = parser.parse_args()
    config.RETRIEVAL_MODE, config.QUERY_REWRITE = args.mode, args.rewrite

    answer = Answer(args.question)
    start = time.time()
    for piece in stream_answer(args.question, k=args.k, section=args.section, answer=answer, llm=args.llm):
        print(piece, end="", flush=True)
    print("\n")
    print_footer(answer, time.time() - start)


if __name__ == "__main__":
    main()
