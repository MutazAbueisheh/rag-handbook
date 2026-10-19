"""Step 9 — Eval: measure how well the RAG works on hand-written questions.

Two separate measurements, because they fail for different reasons:

1. Retrieval: is the chunk with the answer in the top k?
   - hit@k  share of questions whose answer chunk(s) are all in the top k
   - MRR    mean reciprocal rank: 1/rank of the answer, averaged (1.0 = always first,
            0.5 = usually second, 0 = never found in the top 10)
   We score two ways: the exact chunk (section) and the right page (any chunk of it).

2. Answers (--answers): the LLM answers from the top-k chunks, then we grade it.
   - keywords: does the answer contain the expected words? Crude, and wrong in both directions
     in practice (it passed "Manager only" because "VP" appeared later in the sentence).
   - judge (--judge): DeepSeek compares the answer with the expected answer and says
     correct / partial / wrong, with a reason. Much closer to a human reading the answer.

Run it:
    .venv/bin/python -m rag.evaluate                                  # retrieval only, seconds
    .venv/bin/python -m rag.evaluate --mode hybrid --rewrite          # try retrieval options
    .venv/bin/python -m rag.evaluate --answers --llm deepseek --judge # answers + judge, ~2-3 min
    .venv/bin/python -m rag.evaluate --answers                        # local Ollama answers, ~30 min
Each run is saved to eval/results/ with the settings that produced it.
"""

import argparse
import json
import time
from datetime import datetime

import yaml

from rag import config
from rag.llm import PROVIDERS, complete
from rag.query import ask
from rag.retrieve import retrieve

QUESTIONS_FILE = config.PROJECT_ROOT / "eval" / "questions.yaml"
RESULTS_DIR = config.PROJECT_ROOT / "eval" / "results"
MAX_RANK = 10          # retrieve this many and record where the answer is
KS = (1, 3, config.TOP_K, MAX_RANK)
LEVELS = ("easy", "medium", "hard", "everyday")


# --- Retrieval scoring ----------------------------------------------------------

def page_of(url: str) -> str:
    return url.split("#")[0]


def rank_of(question: dict, urls: list[str], page_level: bool) -> int | None:
    """Rank (1-based) at which ALL facts of the question are in the results, or None."""
    if page_level:
        urls = [page_of(u) for u in urls]
    ranks = []
    for fact in question["sources"]:
        accepted = {page_of(u) for u in fact} if page_level else set(fact)
        rank = next((i for i, u in enumerate(urls, 1) if u in accepted), None)
        if rank is None:
            return None
        ranks.append(rank)
    return max(ranks)


def summarize(rows: list[dict], key: str) -> dict:
    """hit@k for each k and MRR, over rows that have a rank under `key`."""
    ranks = [r[key] for r in rows]
    n = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["MRR"] = sum(1 / r for r in ranks if r) / n
    return out


# --- Answer scoring -----------------------------------------------------------------

def check_keywords(question: dict, text: str) -> bool:
    refused = text.strip().startswith(config.NOT_FOUND)
    if question.get("not_found"):
        return config.NOT_FOUND.lower() in text.lower()
    return not refused and all(kw.lower() in text.lower() for kw in question["keywords"])


JUDGE_PROMPT = """You grade answers from a question-answering system over GitLab's company handbook.

Question: {question}
Expected answer (written by a human who checked the handbook): {expected}
System's answer: {answer}

Grade the system's answer against the expected answer:
- CORRECT: it gives the key facts of the expected answer and says nothing that contradicts them.
  Extra correct detail is fine. If the expected answer is "Not in the handbook", an answer that
  clearly says the information is not available counts as CORRECT.
- PARTIAL: it gets part of the key facts right but misses or garbles an important part.
- WRONG: it misses the key facts, contradicts them, or refuses when the answer was expected.

Reply on one line: the grade (CORRECT, PARTIAL or WRONG), a colon, then a one-sentence reason."""

GRADES = ("CORRECT", "PARTIAL", "WRONG")


def judge(question: dict, answer: str) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(question=question["question"], expected=question["answer"], answer=answer)
    reply = complete([{"role": "user", "content": prompt}], llm="deepseek").strip()
    grade, _, reason = reply.partition(":")
    grade = grade.strip().upper()
    return (grade if grade in GRADES else "WRONG"), reason.strip() or reply


# --- Main -------------------------------------------------------------------------------

def fmt_rank(row: dict, key: str) -> str:
    if key not in row:
        return ""  # nothing to find (a "not in the handbook" question)
    return str(row[key]) if row[key] else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval (and optionally answers).")
    parser.add_argument("--level", choices=LEVELS, help="only questions of this level")
    parser.add_argument("--mode", choices=("vector", "hybrid"), default=config.RETRIEVAL_MODE)
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=config.QUERY_REWRITE,
                        help="LLM query rewriting before search")
    parser.add_argument("--answers", action="store_true", help="also generate answers and check keywords")
    parser.add_argument("--llm", choices=PROVIDERS, default=config.LLM_PROVIDER, help="which LLM answers")
    parser.add_argument("--judge", action="store_true", help="grade answers with DeepSeek (needs --answers)")
    args = parser.parse_args()
    # Set the retrieval options globally, so every retrieve() call in this run uses them.
    config.RETRIEVAL_MODE, config.QUERY_REWRITE = args.mode, args.rewrite

    questions = yaml.safe_load(QUESTIONS_FILE.read_text())
    if args.level:
        questions = [q for q in questions if q["level"] == args.level]

    rows = []
    for q in questions:
        row = {"id": q["id"], "level": q["level"], "question": q["question"]}
        hits = retrieve(q["question"], k=MAX_RANK)
        urls = [h.url for h in hits]
        if q["sources"]:
            row["rank"] = rank_of(q, urls, page_level=False)
            row["page_rank"] = rank_of(q, urls, page_level=True)
        if args.answers:
            start = time.time()
            # Answer from exactly the chunks we just scored (the top TOP_K of them).
            answer = ask(q["question"], llm=args.llm, hits=hits[:config.TOP_K])
            row["answer"] = answer.text
            row["seconds"] = round(time.time() - start)
            row["keywords_ok"] = check_keywords(q, answer.text)
            if args.judge:
                row["judge"], row["judge_reason"] = judge(q, answer.text)
            print(f"  {q['id']}: keywords {'PASS' if row['keywords_ok'] else 'FAIL'}"
                  + (f", judge {row['judge']}" if args.judge else "") + f"  ({row['seconds']}s)", flush=True)
        rows.append(row)

    # --- Per-question table
    print(f"\n{'id':<4} {'level':<9} {'chunk':>5} {'page':>5}  {'keyw':<5} {'judge':<8} question")
    for r in rows:
        kw = ("PASS" if r["keywords_ok"] else "FAIL") if "keywords_ok" in r else ""
        print(f"{r['id']:<4} {r['level']:<9} {fmt_rank(r, 'rank'):>5} {fmt_rank(r, 'page_rank'):>5}  "
              f"{kw:<5} {r.get('judge', ''):<8} {r['question']}")
    print("(chunk/page = rank where the answer was found, '-' = not in top 10; blank = nothing to find)")

    # --- Summary per level
    print(f"\n{'Retrieval':<18}" + "".join(f"{k:>9}" for k in [f"hit@{k}" for k in KS] + ["MRR"]))
    summary = {}
    for level in (*LEVELS, "all"):
        scored = [r for r in rows if "rank" in r and (level == "all" or r["level"] == level)]
        if not scored:
            continue
        for key, label in (("rank", "chunk"), ("page_rank", "page")):
            s = summarize(scored, key)
            summary[f"{level}/{label}"] = s
            print(f"{level + ' ' + label:<18}" + "".join(f"{v:>9.2f}" for v in s.values()) + f"   (n={len(scored)})")

    if args.answers:
        print(f"\nAnswers{'':<11}{'keywords':>10}" + ("   judge: correct / partial / wrong" if args.judge else ""))
        for level in (*LEVELS, "all"):
            sel = [r for r in rows if level == "all" or r["level"] == level]
            if not sel:
                continue
            ok = sum(r["keywords_ok"] for r in sel)
            summary[f"{level}/keywords"] = ok / len(sel)
            line = f"  {level:<16}{ok:>5}/{len(sel):<4}"
            if args.judge:
                counts = {g: sum(r["judge"] == g for r in sel) for g in GRADES}
                summary[f"{level}/judge"] = counts
                line += f"   {counts['CORRECT']:>7} / {counts['PARTIAL']:>7} / {counts['WRONG']:>5}"
            print(line)

    # --- Save the run, with the settings that produced it
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}{'+rewrite' if args.rewrite else ''}" + (f"-{args.llm}" if args.answers else "")
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{tag}.json"
    settings = {k: getattr(config, k) for k in ("CHUNK_SPLIT_LEVEL", "CHUNK_MAX_TOKENS", "CHUNK_OVERLAP_TOKENS",
                                                 "EMBED_MODEL", "TOP_K", "RETRIEVAL_MODE", "QUERY_REWRITE")}
    if args.answers:
        settings["LLM"] = f"{args.llm}/{config.LLM_MODELS[args.llm]}"
    out.write_text(json.dumps({"settings": settings, "summary": summary, "rows": rows}, indent=2))
    print(f"\nSaved {out.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
