"""BM25 keyword search, written from scratch (step 10).

Vector search matches *meaning*; BM25 matches *words*. They fail differently:
- vector search can rank "Netherlands vacation" above "Flexible PTO" because both are about time off
- BM25 rewards rare, exact words: "WeWork" means nothing to it, but "H-1B", "401(k)", "MVC" or
  "Tech-Caddy" are strong signals, and an embedding can blur those
That's why "hybrid" search runs both and merges the two ranked lists.

The BM25 score of a document for a query is a sum over the query's words:

    idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))

- tf:   how often the word appears in the document (more is better, with diminishing returns: k1)
- idf:  how rare the word is across all documents (rare words count much more than "the")
- b:    long documents are penalised a bit, so they don't win just by containing more words
"""

import math
import re
from collections import Counter, defaultdict

# A small stop-word list: words so common they only add noise to keyword matching.
STOP_WORDS = set("""a an and are as at be but by can do does for from how i if in is it its
me my of on or so that the their them there this to was we what when where which who why
will with you your""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase words, keeping inner hyphens and dots: "H-1B", "401(k)" -> "401", "k", "co-working"."""
    words = re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOP_WORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = []
        # Inverted index: word -> list of (doc number, how often the word appears in it)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            counts = Counter(tokenize(doc))
            self.doc_len.append(sum(counts.values()))
            for word, tf in counts.items():
                self.postings[word].append((i, tf))
        n = len(docs)
        self.avg_len = sum(self.doc_len) / n
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()}

    def search(self, query: str, n: int) -> list[tuple[int, float]]:
        """The n best documents as (doc number, score), best first."""
        scores: dict[int, float] = defaultdict(float)
        for word in set(tokenize(query)):
            for i, tf in self.postings.get(word, []):
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
                scores[i] += self.idf[word] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
