#!/usr/bin/env python3
"""Answer questions against the graph.

    python3 tools/answer_run.py --question "Where do I live?" --at "2023/05/01 (Mon) 09:00"
    python3 tools/answer_run.py --data data/longmemeval_oracle.json --limit 20
    python3 tools/answer_run.py --data ... --limit 20 --no-model    # gate+graph only, free

The model call is the only irreversible spend here, so `--no-model` runs the
whole retrieval path and writes the evidence it would have reasoned over. Use
it to look at what the gate surfaces before paying to have anything read.

Results append to ~/docket/state/answers.jsonl, one row per question, and a
question already answered at the current prompt is skipped -- an evening that
ends with a closed lid loses nothing.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docket import dataset, timeparse  # noqa: E402
from docket.answer import ABSENT, ANSWERED, UNMEASURED, Answerer  # noqa: E402
from docket.gate import build_gate  # noqa: E402
from docket.hydra import HydraClient  # noqa: E402
from docket.llm import Anthropic, load_env  # noqa: E402
from docket.retrieve import Retriever  # noqa: E402

STATE = Path(os.environ.get("DOCKET_STATE",
                            Path.home() / "docket" / "state"))
ANSWERS = STATE / "answers.jsonl"
ENV = Path(os.environ.get("DOCKET_ENV", Path.home() / "docket" / ".env"))


def connect(args) -> HydraClient:
    """Exactly how ingest_run and extract_run build a client.

    The token default matters: with no token the node answers 401 and the
    first preflight run reported every shape as unmeasured for a reason that
    had nothing to do with the shapes. Same env names, same fallbacks, one
    behaviour across every tool that talks to this node.
    """
    return HydraClient(
        base_url=args.base or os.environ.get("HYDRA_URL", "http://127.0.0.1:8443"),
        admin_url=os.environ.get("HYDRA_ADMIN", "http://127.0.0.1:9090"),
        token=args.token or os.environ.get("HYDRA_TOKEN",
                                           "local-development-token-32-bytes"),
        graph=os.environ.get("HYDRA_GRAPH", "default"),
        namespace=os.environ.get("HYDRA_NAMESPACE", "default"),
        cell=os.environ.get("HYDRA_CELL", "cell-0"),
        consistency="causal",
    )


def select(instances, sample: int | None, seed: int, min_abstention: int):
    """A slice that contains both kinds of question.

    `--limit` slices the file, it does not sample: the first 25 oracle
    instances are 100% temporal-reasoning with ZERO abstention questions, so a
    limit-slice can only ever measure one half of the calibration pair. A
    prompt change that lowers false refusals MUST be checked against the
    abstention set in the same run, or it is optimising one number blind.

    Proportional by question type so the mix matches the benchmark, with an
    optional floor on abstention questions. The floor breaks proportionality
    on purpose and the run says so.

    Allocation is largest-remainder: exact shares floored, then the leftover
    handed out by biggest fractional part, then a round-robin top-up for any
    shortfall caused by a stratum running dry. A naive loop that recomputes
    proportions against a shrinking remainder under-fills -- it returned 40
    for a requested 60, caught by test_sample_size_is_respected.
    """
    if not sample or sample >= len(instances):
        return list(instances), False
    rng = random.Random(seed)
    strata: dict = {}
    for inst in instances:
        key = "abstention" if inst.is_abstention else inst.question_type
        strata.setdefault(key, []).append(inst)
    for group in strata.values():
        rng.shuffle(group)

    picked = []
    forced = False
    if min_abstention and strata.get("abstention"):
        absts = strata["abstention"]
        take = min(min_abstention, len(absts), sample)
        proportional = sample * len(absts) / len(instances)
        forced = take > proportional
        picked.extend(absts[:take])
        strata["abstention"] = absts[take:]

    remaining = sample - len(picked)
    pool = {k: v for k, v in strata.items() if v}
    total = sum(len(v) for v in pool.values())
    if remaining > 0 and total:
        exact = {k: remaining * len(v) / total for k, v in pool.items()}
        take = {k: min(int(v), len(pool[k])) for k, v in exact.items()}
        short = remaining - sum(take.values())
        # leftover by largest fractional part
        order = sorted(exact, key=lambda k: -(exact[k] - int(exact[k])))
        for key in order:
            if short <= 0:
                break
            if take[key] < len(pool[key]):
                take[key] += 1
                short -= 1
        # round-robin top-up if a stratum ran dry
        while short > 0 and any(take[k] < len(pool[k]) for k in pool):
            for key in pool:
                if short <= 0:
                    break
                if take[key] < len(pool[key]):
                    take[key] += 1
                    short -= 1
        for key, n in take.items():
            picked.extend(pool[key][:n])

    rng.shuffle(picked)
    return picked[:sample], forced


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open() as fh:
        for line in fh:
            try:
                out.add(json.loads(line)["question_id"])
            except (ValueError, KeyError):
                continue
    return out


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def make_llm(model: str | None):
    env = load_env(str(ENV))
    key = env.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(f"no ANTHROPIC_API_KEY in {ENV}")
    return Anthropic(key, model=model) if model else Anthropic(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question")
    ap.add_argument("--at", help="question date, e.g. '2023/05/01 (Mon) 09:00'")
    ap.add_argument("--data", help="LongMemEval json to read questions from")
    ap.add_argument("--limit", type=int,
                    help="first N instances. NOT a sample: the first 25 are "
                         "all one category with no abstention questions.")
    ap.add_argument("--sample", type=int,
                    help="stratified sample of N, proportional by question "
                         "type. Use this, not --limit, for any calibration "
                         "check.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epoch-dates", action="store_true",
                    help="render evidence timestamps as raw epochs, exactly as "
                         "before Aug 17. The only variable left between this "
                         "and the Run A baseline.")
    ap.add_argument("--min-abstention", type=int, default=0,
                    help="force at least K abstention questions into the "
                         "sample. Breaks proportionality; the run says so.")
    ap.add_argument("--no-model", action="store_true",
                    help="retrieval only: no spend, evidence written anyway")
    ap.add_argument("--model")
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--evidence", type=int, default=6)
    ap.add_argument("--sessions", type=int, default=0,
                    help="rank SESSIONS not claims, and take every claim in "
                         "the top N. Built for aggregation questions where the "
                         "answer is spread across many claims.")
    ap.add_argument("--per-session", type=int, default=40)
    ap.add_argument("--as-of-tolerance", type=int, default=None,
                    help="seconds past the question date a claim may still be "
                         "used. Default 86400, measured: the oracle's own "
                         "evidence lands up to 24h late and never further.")
    ap.add_argument("--base", default="")
    ap.add_argument("--token", default="")
    ap.add_argument("--out", default=str(ANSWERS))
    a = ap.parse_args()

    client = connect(a)
    client.wait_ready(seconds=30)

    t0 = time.time()
    retriever = Retriever(client)
    claims = retriever.all_claims()
    if not claims:
        print("the graph holds no Claim nodes: load them before answering",
              file=sys.stderr)
        return 4
    gate = build_gate(claims)
    print(f"gate {gate.name}: {gate.size} claims, vocab {len(gate.df)}, "
          f"built in {time.time() - t0:.1f}s", flush=True)

    llm = None if a.no_model else make_llm(a.model)
    kw = {} if a.as_of_tolerance is None else {"as_of_tolerance": a.as_of_tolerance}
    answerer = Answerer(retriever, gate, llm, candidates=a.candidates,
                        evidence=a.evidence, sessions=a.sessions,
                        per_session=a.per_session,
                        dates="epoch" if a.epoch_dates else "human", **kw)

    # -- one question -------------------------------------------------------
    if a.question:
        if not a.at:
            raise SystemExit("--question needs --at (the as-of date)")
        asked = timeparse.to_epoch(a.at)
        out = answerer.answer(a.question, asked)
        print(json.dumps(out, indent=2, default=str))
        return 0

    if not a.data:
        raise SystemExit("give either --question/--at or --data")

    # -- a slice of the benchmark ------------------------------------------
    out_path = Path(a.out)
    already = done_ids(out_path)
    instances = dataset.load(a.data, limit=a.limit)
    instances, forced = select(instances, a.sample, a.seed, a.min_abstention)
    mix = collections.Counter(
        "abstention" if i.is_abstention else i.question_type for i in instances)
    print(f"slice {len(instances)}: {dict(sorted(mix.items()))}", flush=True)
    if forced:
        print("      abstention floor applied -- this slice is NOT "
              "proportional, do not quote its accuracy as a benchmark number",
              flush=True)
    tally = {ANSWERED: 0, ABSENT: 0, UNMEASURED: 0, "skipped": 0}
    started = time.time()

    for inst in instances:
        if inst.question_id in already:
            tally["skipped"] += 1
            continue
        asked = int(inst.asked_at.timestamp())
        result = answerer.answer(inst.question, asked)
        tally[result["status"]] = tally.get(result["status"], 0) + 1
        append(out_path, {
            "question_id": inst.question_id,
            "question_type": inst.question_type,
            "is_abstention": inst.is_abstention,
            "question": inst.question,
            "asked_at": asked,
            "gold": inst.answer,
            "gold_sessions": inst.evidence_session_ids,
            "status": result["status"],
            "answer": result.get("answer"),
            "reason": result.get("reason"),
            "evidence": result["evidence"],
            "retrieval": result["retrieval"],
            "at": time.time(),
        })
        done = sum(v for k, v in tally.items() if k != "skipped")
        if done % 10 == 0:
            rate = done / max(time.time() - started, 1e-9)
            print(f"  {done} answered  {rate:.2f}/s  {tally}", flush=True)

    elapsed = time.time() - started
    print(f"\n{tally}  in {elapsed:.1f}s")
    if llm is not None:
        # This is the expensive run, so it prices itself. judge_run and
        # baseline_run reported spend from the start and this one did not,
        # which meant the first paid run produced no cost number at all.
        print(f"spend            {llm.calls} calls, {llm.retries} retries, "
              f"{llm.input_tokens} in / {llm.output_tokens} out tokens")
        if llm.trailing_prose:
            print(f"trailing prose   {llm.trailing_prose}/{llm.calls} replies "
                  f"carried commentary after the JSON (parsed anyway)")
        if tally.get(UNMEASURED):
            print(f"⚠ unmeasured     {tally[UNMEASURED]} — read their `reason` "
                  f"field before trusting any score from this file")
    print(f"wrote {out_path}")
    print("scoring is a separate step: this run records what was answered and "
          "what was refused, not whether either was right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
