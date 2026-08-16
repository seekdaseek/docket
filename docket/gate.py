"""Candidate selection: which claims are worth asking the graph about.

The gate does one job and it is deliberately small. It narrows 5,687 claims to
a handful, and then the GRAPH decides everything that matters -- what was true
as of the question date, which value superseded which, and what sentence each
claim came from. If the gate is wrong the graph still refuses to invent an
answer; if the gate is right the graph still has to prove it.

Why lexical rather than embeddings, stated plainly because it is a real
trade-off and not a preference:

  This project's claim is memory that can be cross-examined. A BM25 hit can
  show exactly which terms matched and how much each contributed. An embedding
  hit cannot -- it can only assert a distance. Every retrieval here therefore
  ships `terms`, and the answerer passes them through to the citation.

  The cost is paraphrase. A question asking about "footwear" will not surface a
  claim that says "sneakers". That is measurable in the harness (gate recall
  against `answer_session_ids`) and `EmbeddingGate` is the slot it plugs into
  if the number says so. Nothing else in the pipeline changes.

A note on stemming, so this is not confused with a question already settled:
predicate CANONICALISATION was measured and REJECTED for chain grouping -- it
merged `works_at` with `working_on` and produced fewer, worse chains. That was
about IDENTITY: whether two claims describe the same relation. This is about
RECALL: whether a claim is worth looking at. A wrong stem here costs a wasted
candidate that the graph then discards, not a fabricated contradiction.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Words that carry no discrimination in a corpus where 98% of claims are about
# the same subject. Kept short on purpose: an aggressive stoplist throws away
# the short words that make a question specific ("no", "not", "off").
STOPWORDS = frozenset("""
a about after again all also am an and any are as at be because been before
being between both but by can could did do does doing done during each else
ever every few for from further get got had has have having he her here hers
him his how i if in into is it its just know let like make many may me might
more most much must my no nor not now of off on once only or other our out
over own please remember said same say see should since so some still such
tell than that the their them then there these they thing things this those
through to told too type under until up us use used very want was way we were
what when where whether which while who whom whose why will with would you
your
""".split())

# Question words that are rare INSIDE the corpus and therefore score high if
# they survive: `did` matched `did_system_update_on` at 7.6 before this list
# existed. A word that carries no meaning in a question carries none in a
# claim either, so one list covers both sides.

TOKEN = re.compile(r"[a-z0-9]+")

# Suffixes stripped longest-first. Conservative on purpose -- this is not a
# linguistic stemmer, it is enough to make plans/planning/planned meet.
SUFFIXES = ("ational", "iveness", "fulness", "ousness", "ization", "ations",
            "ically", "ingly", "edly", "ment", "ness", "tion", "sion", "ing",
            "ers", "ies", "ed", "es", "s")
MIN_STEM = 4
# The plain plural is allowed one character shorter, because a floor of 4 means
# three-letter nouns never take one: kit/kits, job/jobs, car/cars, bag/bags.
# This is NOT the MIN_STEM=3 that was measured and rejected -- that applied to
# every suffix and turned `shoes` into `sho` via `es`, costing recall. Here `es`
# still stops at 4, so `shoes` -> `shoe` is unchanged and only `-s` relaxes.
MIN_STEM_S = 3


# Consonants that double before -ing/-ed and must be collapsed again, or
# `planning` stems to `plann` while `plans` stems to `plan` and the two never
# meet. l/s/z are excluded: `fall`, `pass`, `buzz` are not doubled inflections.
DOUBLES = frozenset("bdgmnprt")


def stem(word: str) -> str:
    """Crude suffix stripping. Never shortens below MIN_STEM characters."""
    for suffix in SUFFIXES:
        floor = MIN_STEM_S if suffix == "s" else MIN_STEM
        if len(word) - len(suffix) >= floor and word.endswith(suffix):
            base = word[:-len(suffix)]
            if suffix == "ies":
                return base + "y"
            if suffix in ("ing", "ed") and len(base) >= 4 \
                    and base[-1] == base[-2] and base[-1] in DOUBLES:
                base = base[:-1]
            return base
    return word


def tokenise(text: str, stemming: bool = True) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords dropped, optionally stemmed.

    Snake_case splits for free, which matters: predicates in this corpus are
    `planning_hiking_trip` and the useful signal is the three words inside.
    """
    words = TOKEN.findall((text or "").lower())
    out = []
    for word in words:
        if word in STOPWORDS:
            continue
        out.append(stem(word) if stemming else word)
    return out


def claim_document(claim: dict) -> str:
    """The text a claim is findable by.

    Predicate and object carry nearly all of it. Subject is included but is
    `user` for 5,585 of 5,687 claims, so it contributes almost no separation --
    that measurement is why the entity layer is not the retrieval index.
    """
    return " ".join(str(claim.get(k) or "") for k in ("subj", "pred", "obj", "kind"))


class Hit:
    """One candidate, with the evidence for why it surfaced."""

    __slots__ = ("key", "score", "terms", "claim")

    def __init__(self, key: str, score: float, terms: dict, claim: dict):
        self.key = key
        self.score = round(score, 6)
        self.terms = terms          # term -> contribution, largest first
        self.claim = claim

    def as_dict(self) -> dict:
        return {"key": self.key, "score": self.score, "terms": self.terms,
                "claim": self.claim}

    def __repr__(self) -> str:
        top = ", ".join(list(self.terms)[:3])
        return f"<Hit {self.score:.3f} [{top}] {self.claim.get('pred')}>"


class LexicalGate:
    """BM25 over claim text. Deterministic, stdlib, no model, no network."""

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, stemming: bool = True):
        self.k1 = k1
        self.b = b
        self.stemming = stemming
        self.docs: dict[str, Counter] = {}
        self.lengths: dict[str, int] = {}
        self.claims: dict[str, dict] = {}
        self.by_session: dict[str, list] = {}
        self.df: Counter = Counter()
        self.avg_len = 0.0

    # -- building -----------------------------------------------------------
    def add(self, key: str, claim: dict) -> None:
        tokens = tokenise(claim_document(claim), self.stemming)
        counts = Counter(tokens)
        if key in self.docs:                      # replace, keep df honest
            for term in self.docs[key]:
                self.df[term] -= 1
                if self.df[term] <= 0:
                    del self.df[term]
        self.docs[key] = counts
        self.lengths[key] = len(tokens)
        self.claims[key] = claim
        sid = claim.get("sid")
        if sid is not None:
            bucket = self.by_session.setdefault(sid, [])
            if key not in bucket:
                bucket.append(key)
        for term in counts:
            self.df[term] += 1

    def finalise(self) -> "LexicalGate":
        total = sum(self.lengths.values())
        self.avg_len = (total / len(self.lengths)) if self.lengths else 0.0
        return self

    @property
    def size(self) -> int:
        return len(self.docs)

    # -- searching ----------------------------------------------------------
    def idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        # BM25's probabilistic idf, floored so a term in most documents still
        # contributes nothing rather than a negative score.
        return max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    def search(self, question: str, limit: int = 20,
               min_score: float = 0.0) -> list[Hit]:
        query = tokenise(question, self.stemming)
        if not query:
            return []
        wanted = set(query)
        scored: list[Hit] = []
        for key, counts in self.docs.items():
            terms: dict = {}
            total = 0.0
            length = self.lengths[key] or 1
            for term in wanted:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = self.idf(term)
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                contrib = idf * (tf * (self.k1 + 1)) / denom
                if contrib > 0:
                    terms[term] = round(contrib, 4)
                    total += contrib
            if total > min_score:
                ordered = dict(sorted(terms.items(), key=lambda kv: -kv[1]))
                scored.append(Hit(key, total, ordered, self.claims[key]))
        scored.sort(key=lambda h: (-h.score, h.key))
        return scored[:limit]


    def search_sessions(self, question: str, sessions: int = 3,
                        pool: int = 200, per_session: int = 40) -> list[Hit]:
        """Rank SESSIONS, then return every claim in the best ones.

        Half the benchmark's misses are aggregation questions -- how many model
        kits, how many hours driving in total, which store did I spend the most
        at. The answer is not in one claim; it is spread across fourteen to
        thirty-six claims in three or four sessions. Ranking claims and cutting
        at twelve cannot serve that: measured, the best gold claim for those
        questions sat at rank 19, 36, 80, even 177, and raising the cut from 12
        to 30 moved recall by one point while density fell.

        A session's score is the sum of its matching claims' scores. That
        deliberately rewards a session with several weak matches over one with
        a single strong one, because a session that keeps mentioning the topic
        is what an aggregation question is about.

        Claims that did not match at all are returned too, with score 0 and no
        terms -- they are the other kits, the other drives. Excluding them
        would return the evidence and drop the thing being counted.
        """
        hits = self.search(question, limit=pool)
        if not hits:
            return []
        totals: dict[str, float] = {}
        for hit in hits:
            sid = hit.claim.get("sid")
            if sid is not None:
                totals[sid] = totals.get(sid, 0.0) + hit.score
        best = sorted(totals, key=lambda s: (-totals[s], s))[:sessions]
        scored = {h.key: h for h in hits}
        out: list[Hit] = []
        for sid in best:
            keys = self.by_session.get(sid, [])[:per_session]
            for key in keys:
                hit = scored.get(key)
                if hit is None:
                    hit = Hit(key, 0.0, {}, self.claims[key])
                out.append(hit)
        out.sort(key=lambda h: (-h.score, h.key))
        return out


class EmbeddingGate:
    """Placeholder for a vector gate. Deliberately refuses rather than degrades.

    An embedding gate would be built here (bge-small ONNX, vectors held outside
    HydraDB since it has no vector index). It is not installed, and a gate that
    silently falls back to something else is how you end up reporting a number
    for a component you never ran.
    """

    name = "embedding"

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "no embedding backend is installed. Install onnxruntime and a "
            "sentence model, or use LexicalGate. This class exists so the "
            "harness can name which gate produced a number.")


def build_gate(claims: dict[str, dict], **kw) -> LexicalGate:
    """Index a {claim_key: claim_props} mapping."""
    gate = LexicalGate(**kw)
    for key, claim in claims.items():
        gate.add(key, claim)
    return gate.finalise()
