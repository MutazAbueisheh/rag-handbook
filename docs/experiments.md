# Experiments and decisions

A record of what was tried while building this RAG over the GitLab handbook, what each experiment
showed, and what was decided because of it. Written in build order. Numbers come from real runs on
this machine (12-core laptop CPU, no GPU, 15 GB RAM).

**Contents**

1. [Corpus and scope](#1-corpus-and-scope)
2. [Loading](#2-loading)
3. [Cleaning](#3-cleaning)
4. [Chunking](#4-chunking)
5. [Embedding and storage](#5-embedding-and-storage)
6. [Retrieval baseline](#6-retrieval-baseline)
7. [Generation with a local model](#7-generation-with-a-local-model)
8. [The evaluation set](#8-the-evaluation-set)
9. [Local model vs DeepSeek](#9-local-model-vs-deepseek)
10. [Step 10: improving retrieval](#10-step-10-improving-retrieval)
11. [Decision log](#11-decision-log)
12. [Open items](#12-open-items)

---

## 1. Corpus and scope

**Source:** the GitLab public handbook (a Hugo site), read-only at `../handbook`. 4,625 Markdown files,
4,621 after dropping 4 drafts, about 6.2M words (about 8M tokens).

**Decision: start with three sections** (`values`, `people-group`, `total-rewards`): 93 pages, 220k words.
The whole handbook is 30 times larger and engineering alone is 30% of it. A small index makes every
experiment cheap, and everything else can be added once the pipeline is settled.

**Decision: build from scratch**, with no LlamaIndex or LangChain. This is a learning project, and the
interesting problems here (shortcodes, heading-aware chunks, citations) would need custom code in any
framework.

**Stack:** Python 3.12 venv, Ollama `nomic-embed-text` (768 dimensions) for embeddings, Chroma for
vectors, and Ollama `qwen2.5:3b` or the DeepSeek API for answers.

## 2. Loading

[rag/load.py](../rag/load.py) turns each file into a `Page` with `path`, `url`, `title`, `description`,
`section` and the raw body.

Facts checked in the data before writing code:

| Check | Result | Decision |
|---|---|---|
| Front matter format | All 4,625 files use YAML `---` | Use `python-frontmatter` |
| Pages with a custom `url:` or `slug:` | 0 | Build the URL from the file path |
| Draft pages (`draft: true`) | 4 | Skip them: Hugo does not publish them, so citing one would give a dead link |
| Paths with capitals | 1 (`billing-ops-GPO`) | Lowercase URLs, as Hugo does |
| Pages without a description | 23 of 93 | Do not rely on the description in chunk prefixes |

`section` is the folder under `handbook/`; the top-level `teamops/` and `job-description-library/` count
as their own sections. Step 6 uses `section` as a search filter.

## 3. Cleaning

[rag/clean.py](../rag/clean.py) turns raw Hugo Markdown into clean Markdown.

**Experiment: list every shortcode name in the corpus** before writing any rules. About 75 names, in
five groups, which became the dispatch table in the code:

| Group | Examples | Count | Rule |
|---|---|---|---|
| Wrappers | `alert` 653, `details` 228, `panel` 91, `note`, `card`, `tab` | ~1,000 | Drop the tags, keep the text. If the opening tag has a `summary`, `title` or `header`, keep it as a bold line |
| Include | `include` | 294 | Paste in the file from `assets/` (78 distinct targets, all exist) |
| Lookups | `member-by-name` 241, `label` 101, `member-by-gitlab` 58 | ~420 | Keep the argument |
| Icons | `yes` / `no` | 81 | Replace with the word |
| Dropped | `youtube` 134, `vimeo`, `figure`, `ref` | ~145 | Remove, and count |
| Unknown | `engineering/*`, `product/*`, `team-by-*`, ... | ~550 | Remove, and count as `UNKNOWN` |

**Decisions**
- **Output stays Markdown**, not plain text. The chunker needs `##` headings and `{#anchor}` ids, and
  the LLM reads Markdown tables well.
- **Link URLs are removed and the link text kept.** URLs waste tokens and blur embeddings. About 1,500
  external links in the three sections go to Google Docs and `internal.gitlab.com`, which we cannot read
  anyway.
- **Images are removed**, because alt text is mostly "image" or a file name.
- **Code blocks are left untouched**, so HTML examples inside them survive.
- **Escaped shortcodes** (`{{</* ref */>}}`, which Hugo shows literally) are left alone.

**Bugs found by running on the whole corpus**
1. Some include files begin with `---` as a horizontal rule. Reading them as front matter crashed the
   YAML parser. Fix: read includes as raw text, which is how Hugo treats them.
2. Escaped shortcodes were half-matched by the regex and showed up as `UNKNOWN /`. Fix: a shortcode name
   must start with a letter.

**Result:** 12% of characters removed across the corpus (14% in the default sections). No shortcodes
remain.

**Known gap, accepted.** Unknown shortcodes build their content from data that is not in the checkout.
The important one in our sections is the probation-period table
(`people-group/entities-with-probation`), which Hugo fills from `hugo.Data.public.entity_mapper`. A
question about probation in Germany cannot be answered from the files. It became eval question h7, where
the correct reply is "Not in the handbook".

## 4. Chunking

[rag/chunk.py](../rag/chunk.py) splits each cleaned page into pieces, prefixes each with
`Page title > Heading > Subheading`, and records a deep link.

### 4.1 Splitting experiment: split level

| Version | Behaviour | Chunks (default sections) | Problem |
|---|---|---|---|
| v1 | New chunk at `#`, `##`, `###` only | 1,563 | On the values page, all of "Kindness", "Short toes" and the other `#####` sub-values ended up in big chunks labelled "Collaboration", with no deep link |
| **v2 (kept)** | `#`..`###` always start a chunk; deeper sub-sections (`####`, `#####`) are packed together while they fit in 500 tokens, and a new chunk starts at a sub-heading only when needed | 1,584 | none seen |

The handbook has about 14,000 headings at `####` or deeper, so v1 would have blurred a lot of content.
With v2, each sub-value chunk has its own prefix and its own `#anchor`.

### 4.2 Token counting: calibration experiment

**Question:** can we estimate tokens from words (`words × TOKENS_PER_WORD`) instead of adding a tokenizer?
**Experiment:** embedded 200 random chunks and compared with the token counts Ollama reports.
**Result:** median 1.32 tokens per word (p10 1.16, p90 1.70, highest for numbers and code).
**Decision:** `TOKENS_PER_WORD = 1.3`. A "500-token" chunk is really 450 to 650 tokens, well inside the
2,048-token window, so no tokenizer dependency is needed.

**What the value does.** It only sets chunk size, because the word limit is `500 / TOKENS_PER_WORD`.
Measured on the default sections:

| Value | Word limit | Chunks | Long sections that had to be cut |
|---|---|---|---|
| 1.0 | 500 | 1,480 | 81 |
| 1.3 | 384 | 1,584 | 149 |
| 1.7 | 294 | 1,718 | 250 |

Most chunks are short sections, so the median barely moves. To change chunk size, change
`CHUNK_MAX_TOKENS`, not this ratio.

### 4.3 Bugs found by reading real chunks

- **The size cap ignored the prefix.** Some chunks were 573 tokens. Fix: the prefix counts toward the cap
  (max is now exactly 500).
- **Overlap could push a chunk over the cap.** Fix: skip the overlap if it does not fit.
- **A sub-heading could be glued to the end of the previous paragraph** after a cut, for example
  `"... later. #### Communication Tools"`. Fix: join a heading to its text with a single newline, so a cut
  at a blank line keeps them together.
- **Overlap flattened line breaks** into one line, which damaged lists and tables. Fix: slice the
  original text instead of re-joining words. This briefly crashed on short pieces (`IndexError`) and was
  fixed by clamping the index.

### 4.4 Result and weak spots

| | Default sections | Whole handbook |
|---|---|---|
| Chunks | 1,574 | 48,420 |
| Median size | ~146 tokens | ~133 tokens |
| Under 20 tokens of text | 56 (4%) | 3,593 (7%) |
| With a deep-link anchor | 98% | 96% |

**Weak spots, accepted for now**
- **Tiny chunks** (a heading plus one line of link names). Merging them with neighbours is possible, but
  we decided to measure before changing.
- **Fake headings.** On the New Zealand parental-leave page, "Primary Carer Leave" and "Partner Leave"
  are plain lines, not headings, so the chunker can only cut at paragraphs. Chunking can only respect
  structure the author actually marked up.
- **Overlap** (50 tokens) is only used when one section is too long and gets cut.

## 5. Embedding and storage

[rag/embed.py](../rag/embed.py) and [rag/index.py](../rag/index.py).

**Decisions**
- **Task prefixes.** `nomic-embed-text` was trained with `search_document: ` on passages and
  `search_query: ` on questions. Both are added in one place (`embed.py`). Leaving them out silently
  lowers retrieval quality.
- **Cosine similarity** in Chroma. It compares direction and ignores length.
- **`embedding_function=None`**, so Chroma does not attach its own default embedding model behind our back.
- **Indexing is a sync, not a rebuild.** Each chunk stores a hash of (model + prefix + text):
  unchanged chunks are skipped, changed ones are re-embedded, and chunks that no longer exist are deleted.
  Each batch is saved immediately, so an interrupted run resumes. Verified: the second run embedded 0 of
  1,574 chunks.

**Experiment: how fast is embedding?**
The setup benchmark (32 short texts in 1.6 s) turned out to be far too optimistic. Real chunks run at
about 275 to 550 tokens per second on CPU.

| Scope | Chunks | Tokens | Time |
|---|---|---|---|
| Default sections | 1,574 | ~300k | **13.7 min** (measured) |
| Whole handbook | 48k | ~9M | **~5 to 9 h** (estimated) |

**Decision:** tune on the default sections, freeze the settings, and embed the rest once, overnight.
Only some changes force a re-embed:

| Needs re-embedding | Does not |
|---|---|
| cleaning rules, chunk size / overlap / split level, prefix format, embedding model | top-k, section filters, BM25 / hybrid, reranking, query rewriting, prompt, answer model |

## 6. Retrieval baseline

[rag/retrieve.py](../rag/retrieve.py) embeds the question, asks Chroma for the k nearest chunks and
returns `Hit` objects. With k = 6, a test search already found the right section for several questions.

**Early observations (before any eval set existed)**
- Scores of 0.68 to 0.83 for good matches, about 0.60 to 0.62 for irrelevant ones: the gap is small and
  varies by question, so **no minimum-score cutoff** was added.
- A section filter that excludes the answer still returns k results, just irrelevant ones. Search always
  returns something; deciding "not in the handbook" is the generator's job.
- "How many days of PTO do I get?" ranked the real answer 6th, behind chunks that merely mention PTO and
  days. Embeddings match topic, not the exact question asked.

## 7. Generation with a local model

[rag/query.py](../rag/query.py). Retrieved chunks go into the prompt as numbered sources `[1]..[k]`; the
model cites by number and our code maps each number back to a deep link.

**Decision: cite by number, not by URL.** Small models often mangle long URLs.

**Decision: `num_ctx = 8192`.** Six chunks (about 2,000 to 2,500 tokens) do not fit in Ollama's default
window, and Ollama silently drops the start of a prompt that does not fit.

### Prompt layout experiment (qwen2.5:3b)

Same retrieved chunks, four layouts:

| Layout | CREDIT | PTO | Germany (not in handbook) | Laptop |
|---|---|---|---|---|
| Rules first, then sources | wrote only "[1]" | refused | n/a | n/a |
| Rules after sources | correct | refused | correct | n/a |
| Ends with the refusal rule | n/a | refused | correct | refused |
| **Refusal rule in the system prompt; user message ends with "answer and cite"** | correct | correct | correct | hedges, then answers |

**Inference:** a 3B model mostly follows whatever it read last, and treats refusing as the safe option.
**Decision:** refusal rule in the system prompt, answer instructions last. The findings are recorded in a
comment in the code.

**Measured cost:** the CPU reads about 30 prompt tokens per second, so a 6-chunk prompt takes 60 to 80 s
per answer. Writing the answer takes only a few seconds.

## 8. The evaluation set

[eval/questions.yaml](../eval/questions.yaml) and [rag/evaluate.py](../rag/evaluate.py).

**Design**
- **Two separate measurements**, because they fail for different reasons:
  1. *Retrieval*: the rank of the chunk holding the answer. Reported as hit@k (share of questions with
     rank <= k) and MRR (average of 1/rank), at chunk level (strict) and page level (lenient).
  2. *Answers*: graded after the LLM writes them.
- **Every expected answer was verified against the source file** before being written down.
- **Labels list the acceptable chunk URLs for each fact**, as lists, because some facts live in several
  chunks (for example the laptop policy appears in three places).
- **Levels:** easy (one fact, handbook wording), medium (paraphrased, or an item in a list), hard (several
  facts, a calculation, a trap, or nothing to find), and later everyday (plain words).
- **Two questions have no answer on purpose** (h7 probation in Germany, h8 per diem). The correct reply
  is "Not in the handbook".
- Every run is saved to `eval/results/` together with the settings that produced it.

**Label fix.** Question h4 (stock options and laptop) first "missed" retrieval only because its expected
sources were too narrow: the laptop answer exists in three chunks. The label was fixed, not the system.
**Lesson:** before trusting a failure, check whether the label is wrong.

### Baseline (23 questions, vector search, k = 6)

| | Easy | Medium | Hard | All |
|---|---|---|---|---|
| Right chunk in top 6 | 6/6 | 8/9 | 6/6 | 20/21 (95%) |
| MRR (chunk) | 0.81 | 0.72 | 0.72 | 0.76 |

Only one real retrieval failure: m1 ("cap on vacation days"). The handbook says "Flexible PTO" and never
"vacation", while a dozen country pages say "vacation".

## 9. Local model vs DeepSeek

**Setup.** Both used the same retrieval and the same prompt, so only the model differs.

**Why DeepSeek, and which model.** Local DeepSeek models are not practical here: the Ollama distills are
reasoning models that write a long hidden chain of thought (minutes per answer on CPU), the 7B and 8B
versions need more RAM than is free, and the 1.5B one is weaker than the Qwen model we already use. The
hosted API was the fit. The model name was taken from the API's own `/models` list (`deepseek-flash`,
DeepSeek-V4.1-Flash), not from web search results, which gave a different name.

**Result, graded by reading every answer:**

| | qwen2.5:3b (local) | deepseek-flash (API) |
|---|---|---|
| Correct | 15 / 23 | **22 / 23** |
| Time for 23 questions | ~25 min | **76 s** |
| Cost | free | under 1 cent |

DeepSeek fixed six of the seven questions that were not fully right: the refusal despite having the
answer (e4), the arithmetic (h2: "you keep 70 shares"), a self-contradiction (m5), and a misquote (h5:
"Minimal Valuable Change").

**Inference.** With retrieval unchanged, upgrading the model fixed most failures. The model had been the
main bottleneck, not retrieval. The one question that stayed partly wrong (m1) was a retrieval miss.

**The keyword check was unreliable in both directions:**
- passed m5, where the answer said "Manager approval" and mentioned the VP tier later in the same sentence;
- failed e3, which correctly said "three month window" where the handbook says "90 days";
- failed h7, which was the best possible answer (it explained that the table is missing).

**Decision:** keep keywords as a quick signal, and add an LLM judge (section 10).

**Tradeoff to remember.** DeepSeek receives the questions and the retrieved text. That is fine for a
public handbook, not for private documents.

## 10. Step 10: improving retrieval

### 10.1 Make the eval able to show improvement

With 95% retrieval hit@6 on the original questions there was almost nothing left to improve, and the
evidence was thin. Two changes first:

1. **Ten "everyday" questions (p1 to p10)**, phrased the way a team member would ask, with the handbook
   term swapped for a plain one: "get a friend hired" for *referral bonus*, "WeWork" for *co-working
   space*, "jury letter" for *Legal Civic Obligations*, "having a baby" for *Parental Leave*. Each
   answer was verified in the indexed chunk.
   Effect on the baseline: the everyday group scores only **0.70** hit@6 (three misses: p4, p6, p9),
   against 1.00 or 0.89 for the other levels.
2. **An LLM judge (`--judge`).** DeepSeek compares each answer with the expected one and returns CORRECT,
   PARTIAL or WRONG with a reason. Verified by reading its reasons for every non-correct grade: they were
   specific and accurate (for example "miscalculates a one-week absence as 7/14 days rather than 5/10
   business days").

### 10.2 Three retrieval techniques

| Technique | What it does | Where |
|---|---|---|
| **BM25** | Keyword scoring written from scratch. Rewards rare exact words ("H-1B", "401(k)", "Tech-Caddy") that embeddings can blur | [rag/bm25.py](../rag/bm25.py) |
| **Hybrid + RRF** | Run vector and BM25, then merge the two rankings with Reciprocal Rank Fusion. RRF uses ranks only, so cosine similarities and BM25 scores (different scales) can be merged | [rag/retrieve.py](../rag/retrieve.py) |
| **Query rewriting** | DeepSeek rewrites the question into 3 handbook-style search queries; all are searched and fused | `--rewrite` |

### 10.3 Experiment: why does m1 still fail? (diagnosis)

Rank of the target chunk (Flexible PTO overview) for the original question and the three rewrites:

| Query | Vector rank | BM25 rank |
|---|---|---|
| "Is there a cap on how many vacation days I can take?" | 31 | 27 |
| "Paid Time Off vacation days maximum" | 17 | 26 |
| "vacation day cap time off policy" | 10 | 17 |
| **"PTO limit maximum vacation days"** | **2** | 6 |

**Inference.** The vocabulary mismatch is real, and rewriting can fix it. But with the textbook fusion
constant, a chunk at rank ~8 in every list beat a chunk at rank 2 in just one list, so even this good
rewrite did not help. That led to the next experiment.

### 10.4 Experiment: the RRF constant

RRF scores a chunk as the sum of `1 / (K + rank)` over the lists it appears in. K = 60 is the textbook
value. Swept K across all configurations, with the rewrites cached so every configuration saw the same
ones (31 answerable questions, chunk level):

| Config | hit@1 | hit@3 | hit@6 | MRR |
|---|---|---|---|---|
| vector | 0.61 | 0.81 | 0.87 | 0.71 |
| hybrid, K=60 | 0.61 | 0.77 | 0.90 | 0.72 |
| hybrid, K=20 | 0.61 | 0.77 | 0.90 | 0.72 |
| **hybrid, K=10** | 0.61 | **0.87** | 0.90 | **0.73** |
| vector + rewrite, K=60 | 0.55 | 0.81 | 0.81 | 0.66 |
| vector + rewrite, K=10 | 0.58 | 0.84 | 0.94 | 0.73 |
| hybrid + rewrite, K=60 | 0.48 | 0.87 | 0.94 | 0.67 |
| hybrid + rewrite, K=10 | 0.58 | 0.90 | 0.94 | 0.74 |

**Decision: `RRF_K = 10`.** A smaller constant gives top ranks more weight. Hybrid hit@3 rose from 0.77 to
0.87 for free.

### 10.5 Experiment: is query rewriting stable?

The same configuration (hybrid + rewrite, K=10) run three times:

| Run | hit@1 | hit@3 | hit@6 | MRR | m1 rank |
|---|---|---|---|---|---|
| 1 | 0.58 | 0.87 | 0.97 | 0.73 | 4 |
| 2 | 0.48 | 0.94 | 0.97 | 0.68 | none |
| 3 | 0.61 | 0.87 | 0.94 | 0.75 | 3 |

**Inference.** DeepSeek returns different rewrites on every call, even at temperature 0, so one run can
land well on either side. The earlier sweep (0.94) was a lucky draw rather than the true value. Any
rewrite comparison needs several runs.

### 10.6 End-to-end result (33 questions, DeepSeek answers, DeepSeek judge)

| Retrieval | Correct | Partial | Wrong | hit@6 |
|---|---|---|---|---|
| vector | 28 | 3 | 2 | 0.87 |
| **hybrid, K=10** | **30** | **3** | **0** | **0.90** |
| hybrid + rewrite | 28 | 3 | 2 | 0.94 |

**What hybrid fixed:**
- **p4 (WeWork)**: wrong to correct. BM25 found the "co-working space" FAQ that vector search missed.
- **p6 (having a baby)**: wrong to correct. Vector search returned country-specific leave pages; hybrid
  surfaced the global "16 weeks paid Parental Leave" chunk.

**Why rewriting lost despite the best retrieval score.** In that run the rewrites pushed the co-working
chunk out of the top 6 (p4 went wrong again). In p8 the model turned "a week off" into 7 days instead of 5
working days. And the extra call varies run to run.

**Decisions**
- **Default is hybrid with `RRF_K = 10`** (free, offline, deterministic).
- **Rewriting stays opt-in** via `--rewrite` (config `QUERY_REWRITE = False`), because it needs an API
  call per question and varies between runs.

**Still partly wrong with the new default:**

| Question | Gap |
|---|---|
| m1 vacation cap | The global Flexible PTO chunk is still not retrieved |
| p9 filling in for a manager | Explains the interim bonus but leaves out the amount (greater of $1,000 or 10% of salary) |
| h4 leaving: options and laptop | Leaves out "IT Ops wipes the laptop" |

## 11. Decision log

| # | Decision | Reason | Evidence |
|---|---|---|---|
| 1 | Build from scratch, no framework | Learning goal; custom problems anyway | n/a |
| 2 | Index 3 sections first | Cheap experiments; embed the rest once | Embedding speed (section 5) |
| 3 | Skip drafts, lowercase URLs | Match what Hugo publishes | Data checks (section 2) |
| 4 | Keep Markdown, drop link URLs and images | Chunker needs headings; URLs add noise | Section 3 |
| 5 | Chunk at `#`..`###`, pack `####`+ | Sub-values need their own prefix and link | v1 vs v2 (section 4.1) |
| 6 | Estimate tokens as words x 1.3 | Avoids a tokenizer dependency | 200-chunk calibration |
| 7 | Prefix counts toward the 500 cap | Chunks were reaching 573 | Chunk-size stats |
| 8 | Embed with `search_document:` / `search_query:` prefixes | The model was trained that way | Model documentation |
| 9 | Cosine similarity | Standard for text embeddings | n/a |
| 10 | Hash-based sync indexing | Cheap re-runs, resumable | Second run embedded 0 chunks |
| 11 | No minimum-score cutoff | Score gap too small and question-dependent | Section 6 |
| 12 | Cite by number, map to URLs in code | Small models mangle URLs | Section 7 |
| 13 | Refusal rule in system prompt, answer instruction last | Layout test with the 3B model | Four-layout table (section 7) |
| 14 | `num_ctx = 8192` | Prompt exceeds Ollama's default window | Section 7 |
| 15 | DeepSeek `deepseek-flash` as the strong model | Hosted, fast, cheap; local DeepSeek unsuitable | Section 9 |
| 16 | Keep keyword check, add LLM judge | Keywords wrong both ways | Section 9 |
| 17 | Add everyday-phrased questions | Original set was too easy for retrieval | 0.95 vs 0.70 hit@6 |
| 18 | BM25 from scratch + hybrid with RRF | Embeddings blur exact terms | End-to-end 28 to 30 correct |
| 19 | `RRF_K = 10` | Top ranks should count more | K sweep (section 10.4) |
| 20 | Rewriting opt-in, hybrid default | Unstable and API-dependent | Three-run variance (section 10.5) |

## 12. Open items

1. **A reranker.** Re-score the top ~20 chunks against the question. Targets m1 and the fact that the
   best chunk is first only about 60% of the time (hit@1 = 0.61).
2. **Index the whole handbook** (about 5 to 9 h on CPU), then rerun the eval. Expect scores to drop,
   since 30 times more chunks compete, and add questions from the other sections.
3. **Chunk-size experiments.** Each needs a ~14 min re-embed on the small index.
4. **Tiny chunks:** merge or leave, to be decided by the eval.
5. **Eval hygiene:** run rewrite comparisons several times, and re-label questions after seeing what
   actually failed (m1 was labelled medium but is the hardest retrieval case).
6. **A Streamlit UI.**

**Where the numbers come from:** `.venv/bin/python -m rag.evaluate [--mode hybrid] [--rewrite] [--answers
--llm deepseek --judge]`. Saved runs are in `eval/results/`.
