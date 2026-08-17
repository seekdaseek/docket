"""Answering, and refusing.

The refusal is the part worth reading. A memory system that always produces a
sentence is untestable: you cannot tell a recalled fact from a fluent guess.
This one distinguishes three outcomes and reports which it reached:

  ANSWERED   evidence existed as of the question date and the model used it
  ABSENT     the graph was searched and holds nothing relevant -- 'I do not
             have this', which is the correct answer to 30 of the benchmark's
             500 questions and the only one where confidence is guaranteed wrong
  UNMEASURED something failed. The model, the transport, an unparseable reply.
             Not the same as ABSENT and never collapsed into it.

The gate is allowed to be wrong; the graph is allowed to be empty; neither is
allowed to produce a confident sentence with nothing behind it.
"""
from __future__ import annotations

import datetime
import json

from .llm import LLMError

ANSWERED = "answered"
ABSENT = "absent"
UNMEASURED = "unmeasured"

NOT_IN_MEMORY = "NOT_IN_MEMORY"

# How far past the question date a claim may be said and still count as
# knowable. Zero is the principled default and it is WRONG for this benchmark:
# measured over all 500 oracle instances, 43 (8.6%) have evidence dated after
# the question they answer -- 8 within an hour, 24 within six, 33 within a day,
# and NOTHING beyond 24 hours. A strict filter drops the answer to 8.6% of the
# questions and the retrieval looks broken when it is not.
#
# The tolerance is therefore 24 hours, chosen as the smallest window that keeps
# every instance's evidence. It is counted on every run (`admitted_by_tolerance`)
# so the cost is never invisible, and it must be re-measured on longmemeval_s:
# the oracle file contains ONLY evidence sessions, so it cannot show what a
# 24-hour window lets in when there are distractors.
DEFAULT_TOLERANCE = 86_400

# REVERTED Aug 17 to the exact text of Run A, because Run A beat the rewrite on
# an identical 25-question slice: answered 15 vs 7, accuracy 0.3333 vs 0.2381,
# false refusal 0.375 vs 0.6667, trailing prose 13/25 vs 21/25. The rewrite
# added a long block about mention-vs-event dates and softened the refusal rule;
# whatever it bought, it cost half the answers and made the model chattier.
#
# The ONE surviving change is the DATE FORMAT in the evidence block (epoch ->
# human), isolated so it can be measured on its own. That is the change with a
# real mechanism behind it: the model was caught doing arithmetic on 10-digit
# integers in its own leaked commentary.
SYSTEM = """You answer a question using ONLY the numbered evidence supplied.

The evidence is claims extracted from the user's own past conversations, each
with the sentence it came from and when it was said. Everything shown was said
on or before the date the question is asked.

Rules, in order of precedence:
1. If the evidence does not contain the answer, reply with exactly NOT_IN_MEMORY
   and nothing else. Do not guess, do not reason from general knowledge, and do
   not answer from what is merely plausible.
2. When several pieces of evidence conflict, the most recent one is the current
   truth. Older values were superseded, not wrong.
3. Answer in as few words as the question allows. A name, a date, a number or a
   short phrase. No preamble, no restatement of the question.
4. Cite the evidence numbers you used.

Reply as JSON only, no markdown fences:
{"answer": "<the answer, or NOT_IN_MEMORY>", "used": [<evidence numbers>]}"""


def fmt(ts, dates: str = "human"):
    """Timestamp for the prompt. `epoch` restores the pre-Aug-17 rendering
    exactly, so the date format can be reverted with a flag rather than a
    code change."""
    return human_date(ts) if dates == "human" else ts


def human_date(ts) -> str:
    """Epoch -> a date a model can actually reason with.

    Measured Aug 17: every timestamp in the evidence block was a raw Unix epoch,
    and 9 of 9 false refusals on the temporal-reasoning slice were ordering or
    duration questions whose gold claim WAS in the evidence. The model was seen
    attempting arithmetic on 10-digit integers in its own leaked commentary
    ("attended on 1685307840, while ... was attended on 16..."). Asking a
    language model to subtract epochs is asking it to do the one thing it is
    worst at, and rule 2 then told it to refuse.
    """
    if ts is None:
        return "unknown"
    try:
        stamp = datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc)
    except (ValueError, OSError, OverflowError):
        return str(ts)
    return stamp.strftime("%Y-%m-%d (%a)")


def render_evidence(items: list[dict], limit_chars: int = 700,
                    dates: str = "human") -> str:
    """The evidence block the model sees. Numbered so citations are checkable.

    `mentioned` rather than `said`: the label is doing real work. The timestamp
    is when the user TALKED about the thing, not when the thing happened, and
    two claims from one conversation share it even when they describe events
    months apart. Naming it honestly is what lets the system prompt tell the
    model not to order events by it.
    """
    lines = []
    for i, item in enumerate(items, 1):
        claim = item.get("claim") or {}
        when = item.get("when") or claim.get("ts")
        head = (f"[{i}] {claim.get('subj')} {claim.get('pred')} "
                f"{claim.get('obj')}  (kind={claim.get('kind')}, "
                f"said {fmt(when, dates)})")
        lines.append(head)
        text = item.get("text")
        if text:
            snippet = text if len(text) <= limit_chars else text[:limit_chars] + "..."
            lines.append(f"    said: {snippet}")
        for older in item.get("superseded") or []:
            lines.append(f"    replaced an earlier value: {older.get('obj')} "
                         f"(said {fmt(older.get('ts'), dates)})")
    return "\n".join(lines)


class Answerer:
    """Gate -> graph -> model, with the refusal paths kept separate."""

    def __init__(self, retriever, gate, llm=None, *, candidates: int = 12,
                 evidence: int = 6, min_score: float = 0.0,
                 as_of_tolerance: int = DEFAULT_TOLERANCE,
                 sessions: int = 0, per_session: int = 40,
                 dates: str = "human"):
        self.retriever = retriever
        self.gate = gate
        self.llm = llm
        self.candidates = candidates
        self.evidence = evidence
        self.min_score = min_score
        self.as_of_tolerance = int(as_of_tolerance)
        # sessions > 0 switches the gate from ranking claims to ranking
        # sessions and returning everything in the best ones.
        self.sessions = int(sessions)
        self.per_session = int(per_session)
        # "human" or "epoch". The only variable left between this answerer and
        # the measured Run A baseline.
        self.dates = dates

    # -- selection ----------------------------------------------------------
    def gather(self, question: str, asked_at: int) -> dict:
        """Candidates from the gate, re-read from the graph, filtered as-of."""
        if self.sessions > 0:
            hits = self.gate.search_sessions(question, sessions=self.sessions,
                                             per_session=self.per_session)
        else:
            hits = self.gate.search(question, limit=self.candidates,
                                    min_score=self.min_score)
        report = {
            "gate": getattr(self.gate, "name", "unknown"),
            "hits": len(hits),
            "terms": sorted({t for h in hits for t in h.terms}),
            "dropped_future": 0,
            "dropped_missing": 0,
            "admitted_by_tolerance": 0,
            "as_of_tolerance": self.as_of_tolerance,
            "candidates": self.candidates,
            "mode": "sessions" if self.sessions else "claims",
        }
        items: list[dict] = []
        for hit in hits:
            item = self.retriever.evidence_for(hit.key)
            if not item.get("found"):
                report["dropped_missing"] += 1
                continue
            claim = item["claim"]
            ts = claim.get("ts")
            if ts is not None:
                past_question = int(ts) - int(asked_at)
                if past_question > self.as_of_tolerance:
                    # Beyond anything the dataset itself does. Using it would be
                    # leakage, and on a benchmark built from dated sessions that
                    # is the easiest way to score well and mean nothing.
                    report["dropped_future"] += 1
                    continue
                if past_question > 0:
                    # Inside the tolerance: kept, and counted, so the cost of
                    # the window is on every report rather than hidden in it.
                    report["admitted_by_tolerance"] += 1
            item["when"] = ts
            item["score"] = hit.score
            item["matched"] = hit.terms
            items.append(item)
        report["kept"] = len(items)
        # SELECT by relevance, PRESENT by recency. These are different jobs and
        # conflating them cost 18 points of recall: the old code sorted the
        # whole pool newest-first and truncated, which was survivable at 12
        # candidates for 6 slots (everything in the pool was already
        # relevance-ranked) and destructive in session mode, where the pool is
        # ~120 claims including deliberate zero-score ones. Taking the newest 25
        # of those is close to random with respect to the question.
        items.sort(key=lambda i: (-i["score"], -(i.get("when") or 0)))
        chosen = items[:self.evidence]
        # Now order what survived by time, because the tip has to lead: the
        # model is told the most recent value is the current one.
        chosen.sort(key=lambda i: (-(i.get("when") or 0), -i["score"]))
        return {"items": chosen, "report": report}

    # -- answering ----------------------------------------------------------
    def answer(self, question: str, asked_at: int) -> dict:
        gathered = self.gather(question, asked_at)
        items, report = gathered["items"], gathered["report"]

        if not items:
            reason = ("nothing in the graph matched, as of that date"
                      if report["hits"] else "no claim matched the question")
            return self._result(ABSENT, None, items, report, reason=reason)

        if self.llm is None:
            return self._result(UNMEASURED, None, items, report,
                                reason="no model configured; evidence gathered "
                                       "but nothing answered it")
        block = render_evidence(items, dates=self.dates)
        user = (f"Question: {question}\n"
                f"Asked at: {fmt(asked_at, self.dates)}\n\n"
                f"Evidence:\n{block}")
        try:
            reply = self.llm.complete_json(SYSTEM, user)
        except (LLMError, ValueError) as exc:
            return self._result(UNMEASURED, None, items, report,
                                reason=f"model call failed: {exc}")
        if not isinstance(reply, dict) or "answer" not in reply:
            return self._result(UNMEASURED, None, items, report,
                                reason=f"model returned an unusable shape: "
                                       f"{json.dumps(reply)[:200]}")
        text = str(reply.get("answer") or "").strip()
        used = reply.get("used") or []
        if not text or text.upper().replace(" ", "_") == NOT_IN_MEMORY:
            return self._result(ABSENT, None, items, report,
                                reason="the model was given evidence and said "
                                       "it does not contain the answer")
        return self._result(ANSWERED, text, items, report, used=used)

    # -- shaping ------------------------------------------------------------
    def _result(self, status: str, answer, items, report, reason=None,
                used=None) -> dict:
        cited = []
        for i, item in enumerate(items, 1):
            claim = item.get("claim") or {}
            cited.append({
                "n": i,
                "used": (i in (used or [])) if used else None,
                "claim_key": item.get("key"),
                "session": claim.get("sid"),
                "turn": claim.get("turn"),
                "said_at": claim.get("ts"),
                "triple": [claim.get("subj"), claim.get("pred"), claim.get("obj")],
                "matched_terms": item.get("matched"),
                "superseded": [c.get("obj") for c in item.get("superseded") or []],
            })
        out = {"status": status, "answer": answer, "evidence": cited,
               "retrieval": report}
        if reason:
            out["reason"] = reason
        return out
