"""What is actually in the benchmark file, before a cent is spent on it.

The number that decides Day 1 is the reuse factor. Each instance carries a
haystack of sessions, and the same session id appears across many instances.
If sessions are ingested once per instance, extraction is paid for many times
over; if they are ingested once globally and scoped at query time, the bill
and the graph both shrink by that factor. Nobody knows the factor until it is
counted, so it is counted here rather than assumed.

    python3 tools/inspect_data.py ~/docket/data/longmemeval_s_cleaned.json
    python3 tools/inspect_data.py ~/docket/data/longmemeval_oracle.json --limit 50

Reads the file once. The 264MB file needs roughly a gigabyte of memory as
parsed Python objects, so on a small machine start with --limit.
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docket.dataset import load, stats, unique_sessions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--limit", type=int, default=None,
                    help="only read the first N instances")
    args = ap.parse_args()

    insts = load(args.path, limit=args.limit)
    s = stats(insts)

    print(f"file            {args.path}")
    print(f"instances       {s['instances']}")
    print(f"abstention      {s['abstention']}")
    print(f"question types  {s['types']}")
    print()
    print(f"session slots   {s['session_slots']}  (sessions counted per instance)")
    print(f"unique sessions {s['unique_sessions']}")
    print(f"reuse factor    {s['reuse_factor']}x  "
          f"(ingesting once instead of per instance divides the bill by this)")
    print(f"id collisions   {len(s['id_collisions'])} {s['id_collisions'][:5]}")
    print()
    print(f"turns           {s['turns_in_unique']} in unique sessions")
    print(f"characters      {s['chars_in_unique']:,} in unique sessions")
    approx_tokens = s["chars_in_unique"] // 4
    print(f"approx tokens   {approx_tokens:,} to read once "
          f"(rough, 4 chars per token)")

    uniq, _ = unique_sessions(insts)
    if uniq:
        sizes = sorted(len(x.turns) for x in uniq.values())
        mid = sizes[len(sizes) // 2]
        print(f"turns/session   min {sizes[0]}  median {mid}  max {sizes[-1]}")

    ordered_wrong = 0
    for inst in insts:
        whens = [x.when for x in inst.sessions]
        if whens != sorted(whens):
            ordered_wrong += 1
    print(f"\nsanity          {ordered_wrong} instances still out of order "
          f"after loading (must be 0)")

    unsorted_in_file = 0
    import json
    with open(args.path) as fh:
        raw = json.load(fh)
    if args.limit:
        raw = raw[:args.limit]
    from docket.timeparse import parse as tparse
    for r in raw:
        ds = [tparse(d) for d in r["haystack_dates"]]
        if ds != sorted(ds):
            unsorted_in_file += 1
    print(f"                {unsorted_in_file} of {len(raw)} instances are stored "
          f"out of chronological order in the file itself")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
