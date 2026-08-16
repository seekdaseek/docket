#!/usr/bin/env python3
"""Run the structural ingest against a live HydraDB node.

    python3 tools/ingest_run.py --limit 20        # a small slice first
    python3 tools/ingest_run.py                   # the whole oracle file

Preflight runs first and can stop the ingest before it writes anything real:
if a label restatement wipes properties, or the id width the client wants is
not accepted, continuing would corrupt the graph in a way that looks fine.

Environment: HYDRA_URL, HYDRA_ADMIN, HYDRA_TOKEN, HYDRA_GRAPH,
HYDRA_NAMESPACE, HYDRA_CELL, DOCKET_DATA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docket import preflight
from docket.dataset import load, unique_sessions
from docket.hydra import HydraClient, HydraError
from docket.ingest import ingest

DATA = os.environ.get(
    "DOCKET_DATA",
    os.path.expanduser("~/docket/data/longmemeval_oracle.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N instances")
    ap.add_argument("--checkpoint",
                    default=os.path.expanduser("~/docket/state/ingest.jsonl"))
    ap.add_argument("--skip-preflight", action="store_true",
                    help="only after preflight has passed once on this node")
    ap.add_argument("--report",
                    default=os.path.expanduser("~/docket/state/ingest-report.json"))
    args = ap.parse_args()

    client = HydraClient(
        base_url=os.environ.get("HYDRA_URL", "http://127.0.0.1:8443"),
        admin_url=os.environ.get("HYDRA_ADMIN", "http://127.0.0.1:9090"),
        token=os.environ.get("HYDRA_TOKEN", "local-development-token-32-bytes"),
        graph=os.environ.get("HYDRA_GRAPH", "default"),
        namespace=os.environ.get("HYDRA_NAMESPACE", "default"),
        cell=os.environ.get("HYDRA_CELL", "cell-0"),
        consistency="causal",
    )

    print(f"node   {client.base_url}")
    try:
        waited = client.wait_ready(seconds=60)
    except HydraError as e:
        print(f"node not answering: {e}")
        return 1
    print(f"ready  after {waited:.1f}s")

    print(f"data   {args.data}")
    instances = load(args.data, limit=args.limit)
    sessions, collisions = unique_sessions(instances)
    turns = sum(len(s.turns) for s in sessions.values())
    print(f"       {len(instances)} instances, {len(sessions)} unique sessions, "
          f"{turns} turns")
    if collisions:
        print(f"       WARNING {len(collisions)} session ids carry two different "
              f"texts: {collisions[:5]}")

    id_bits = 52
    if args.skip_preflight:
        print("preflight skipped by flag")
    else:
        print("preflight")
        result = preflight.run(client, statements_expected=turns)
        print(preflight.summarise(result))
        if not result["id_bits"]:
            print("STOP: no id width was accepted and read back. Nothing written.")
            return 2
        if not result["label_restatement_safe"]:
            print("STOP: a label+id restatement does not preserve properties on "
                  "this node, so ingesting would wipe session timestamps. "
                  "Nothing written.")
            return 3
        id_bits = min(52, result["id_bits"])
        print(f"       using {id_bits}-bit ids")

    def progress(position, total, sid, n):
        if position % 25 == 0 or position == total:
            print(f"       {position}/{total} sessions, last {sid} ({n} turns)")

    started = time.time()
    try:
        summary = ingest(client, sessions, checkpoint_path=args.checkpoint,
                         id_bits=id_bits, progress=progress)
    except HydraError as e:
        print(f"FAILED: {e}")
        return 4
    summary["wall_seconds"] = round(time.time() - started, 1)
    summary["queries"] = client.queries

    print("done")
    for key, value in summary.items():
        if key == "sessions_empty":
            print(f"       sessions_empty: {len(value)}"
                  + (f" {value[:5]}" if value else ""))
        else:
            print(f"       {key}: {value}")

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
