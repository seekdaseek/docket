"""Timestamps as LongMemEval actually writes them.

Every date in the dataset looks like this:

    2023/04/10 (Mon) 23:07

`datetime.fromisoformat` cannot read it and neither can any of the obvious
formats, because of the parenthesised weekday. That weekday is also redundant
information which can, in principle, disagree with the date it decorates, so it
is checked rather than skipped: a session claiming a Monday that fell on a
Thursday is a corrupt record, and this project does not build a chronology on
records it cannot verify.

Ordering matters more than parsing here. In the oracle file the sessions of an
instance arrive OUT of chronological order (the first instance runs 17:50,
14:47, 17:15), so anything that treats list position as time silently corrupts
every temporal question in the benchmark. `order_sessions` exists so no caller
ever has to remember that.
"""
from __future__ import annotations

from datetime import datetime, timezone

FORMAT = "%Y/%m/%d (%a) %H:%M"


class BadTimestamp(ValueError):
    pass


def parse(text: str) -> datetime:
    """Parse one LongMemEval timestamp into an aware UTC datetime.

    The dataset carries no timezone. UTC is attached so every comparison in
    the project is between aware datetimes, and the README says plainly that
    the benchmark's clock is treated as UTC.
    """
    if not isinstance(text, str):
        raise BadTimestamp(f"timestamp must be a string, got {type(text).__name__}")
    raw = text.strip()
    try:
        dt = datetime.strptime(raw, FORMAT)
    except ValueError:
        raise BadTimestamp(f"not a LongMemEval timestamp: {raw!r}") from None
    claimed = raw.split("(", 1)[1].split(")", 1)[0]
    actual = dt.strftime("%a")
    if claimed != actual:
        raise BadTimestamp(
            f"{raw!r} claims {claimed} but that date is a {actual}")
    return dt.replace(tzinfo=timezone.utc)


def to_epoch(text: str) -> int:
    return int(parse(text).timestamp())


def order_sessions(session_ids: list, dates: list) -> list[tuple[int, str, datetime]]:
    """Return (original_index, session_id, when) sorted oldest first.

    Raises when the two lists disagree in length, because a session paired
    with the wrong timestamp produces a chronology that is confidently wrong,
    which is worse than one that fails.
    """
    if len(session_ids) != len(dates):
        raise BadTimestamp(
            f"{len(session_ids)} sessions but {len(dates)} dates; "
            f"they are positionally paired and must match")
    rows = [(i, sid, parse(d)) for i, (sid, d) in enumerate(zip(session_ids, dates))]
    rows.sort(key=lambda r: (r[2], r[0]))
    return rows
