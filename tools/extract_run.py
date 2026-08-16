#!/usr/bin/env python3
"""Extract claims from every session, then load them and their chains.

    python3 tools/extract_run.py --limit 20            # a slice, model + graph
    python3 tools/extract_run.py --extract-only        # spend on the model only
    python3 tools/extract_run.py --load-only           # replay what is cached

Extraction and loading are separate on purpose. The model call is the only
irreversible cost in this project, so its output is written to disk the moment
it arrives; loading reads from that file and can be rerun as often as needed
without paying twice.

Key: ANTHROPIC_API_KEY in ~/docket/.env, which sits outside the repo.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docket.claims import load_chains, load_claims
from docket.dataset import load, unique_sessions
from docket.extract import PROMPT_VERSION, ClaimStore, extract_session
from docket.hydra import HydraClient, HydraError
from docket.ids import IdRegistry
from docket.llm import Anthropic, LLMError, load_env
from docket.schema import Writer

HOME = os.path.expanduser("~/docket")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get(
        "DOCKET_DATA", f"{HOME}/data/longmemeval_oracle.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--claims", default=f"{HOME}/state/claims.jsonl")
    ap.add_argument("--report", default=f"{HOME}/state/extract-report.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=os.environ.get("DOCKET_MODEL"))
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--load-only", action="store_true")
    ap.add_argument("--retry-unmeasured", action="store_true",
                    help="call the model again for sessions it failed on")
    args = ap.parse_args()

    instances = load(args.data, limit=args.limit)
    sessions, _ = unique_sessions(instances)
    store = ClaimStore(args.claims)
    stale = store.stale()
    print(f"data   {len(sessions)} unique sessions")
    print(f"cache  {len(store.measured)} measured, {len(store.unmeasured)} "
          f"unmeasured already on disk")
    if stale:
        print(f"stale  {len(stale)} rows from an older prompt "
              f"(now v{PROMPT_VERSION}) -- they will be extracted again")

    summary: dict = {}

    if not args.load_only:
        env = load_env(f"{HOME}/.env")
        key = env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        model_kw = {"model": args.model} if args.model else {}
        try:
            llm = Anthropic(key or "", **model_kw)
        except LLMError as e:
            print(f"STOP: {e}")
            return 1
        print(f"model  {llm.model}, {args.workers} workers")

        fresh = {sid for sid in store.measured if sid not in stale}
        todo = [sid for sid in sorted(sessions) if sid not in fresh]
        if not args.retry_unmeasured:
            todo = [sid for sid in todo
                    if sid not in store.unmeasured or sid in stale]
        print(f"todo   {len(todo)} sessions")

        lock = threading.Lock()
        counters = {"ok": 0, "failed": 0, "claims": 0, "empty": 0}
        drops_total: dict = {}
        started = time.time()

        def work(sid):
            session = sessions[sid]
            try:
                claims, drops = extract_session(llm, session)
            except (LLMError, ValueError) as e:
                with lock:
                    store.record_unmeasured(sid, str(e)[:300])
                    counters["failed"] += 1
                return
            with lock:
                store.record_measured(sid, claims, drops, len(session.turns))
                counters["ok"] += 1
                counters["claims"] += len(claims)
                if not claims:
                    counters["empty"] += 1
                for reason, n in drops.items():
                    drops_total[reason] = drops_total.get(reason, 0) + n
                done = counters["ok"] + counters["failed"]
                if done % 25 == 0 or done == len(todo):
                    rate = done / max(time.time() - started, 0.001)
                    left = (len(todo) - done) / rate if rate else 0
                    print(f"       {done}/{len(todo)} sessions, "
                          f"{counters['claims']} claims, "
                          f"{rate:.1f}/s, ~{left / 60:.1f} min left")

        if todo:
            with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
                list(pool.map(work, todo))

        summary["extract"] = {
            "sessions_measured": counters["ok"],
            "sessions_unmeasured": counters["failed"],
            "sessions_with_no_claims": counters["empty"],
            "claims_kept": counters["claims"],
            "claims_dropped_by_reason": drops_total,
            "llm_calls": llm.calls,
            "llm_retries": llm.retries,
            "input_tokens": llm.input_tokens,
            "output_tokens": llm.output_tokens,
            "seconds": round(time.time() - started, 1),
        }
        print("extract")
        for k, v in summary["extract"].items():
            print(f"       {k}: {v}")

    if not args.extract_only:
        client = HydraClient(
            base_url=os.environ.get("HYDRA_URL", "http://127.0.0.1:8443"),
            admin_url=os.environ.get("HYDRA_ADMIN", "http://127.0.0.1:9090"),
            token=os.environ.get("HYDRA_TOKEN",
                                 "local-development-token-32-bytes"),
            consistency="causal")
        try:
            client.wait_ready(seconds=60)
        except HydraError as e:
            print(f"node not answering: {e}")
            return 2

        store = ClaimStore(args.claims)  # re-read: extraction just appended
        rows = []
        for sid, row in sorted(store.measured.items()):
            if sid not in sessions or int(row.get("prompt_version", 1)) < PROMPT_VERSION:
                continue
            rows.append({"sid": sid, "claims": row["claims"],
                         "ts": int(sessions[sid].when.timestamp())})

        writer = Writer(client, IdRegistry(bits=52))
        started = time.time()
        loaded = load_claims(writer, rows)
        chained = load_chains(writer, rows)
        summary["load"] = {**loaded, **chained,
                           "writes": writer.writes,
                           "deletes": writer.deletes,
                           "seconds": round(time.time() - started, 1)}
        print("load")
        for k, v in summary["load"].items():
            print(f"       {k}: {v}")

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
