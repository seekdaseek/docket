"""Node ids.

HydraDB requires an integer `id` on both ends of every CREATE, and that id IS
the identity key: a repeat write to the same id updates the node rather than
making a second one. So the client has to assign ids, and two different natural
keys landing on the same integer would silently merge two unrelated things.

Two defences, and the second is the one that actually holds:

1. A wide hash. `stable_id` takes the first N bits of a BLAKE2b digest of
   "kind|natural_key". Width is a parameter because the server's accepted id
   range is a MEASURED quantity, not an assumption -- see tools/preflight.py.

2. Exact detection. `IdRegistry` remembers every id it has issued and the key
   it issued it for, and raises on a genuine collision. Hash width makes a
   collision unlikely; the registry makes an undetected one impossible for the
   run, which is a different and stronger claim.

Ids are positive. Whether this server accepts a negative id is unmeasured, so
nothing here produces one.
"""
from __future__ import annotations

import hashlib

DEFAULT_BITS = 52


class IdCollision(RuntimeError):
    """Two different natural keys hashed to the same id."""


def stable_id(kind: str, natural_key: str, bits: int = DEFAULT_BITS) -> int:
    """A deterministic positive integer id for (kind, natural_key).

    Deterministic across runs and machines: the same key always produces the
    same id, which is what makes ingest resumable and idempotent.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("kind must be a non-empty string")
    if not isinstance(natural_key, str) or not natural_key:
        raise ValueError("natural_key must be a non-empty string")
    if not 32 <= bits <= 62:
        raise ValueError(f"bits must be between 32 and 62, got {bits}")
    digest = hashlib.blake2b(f"{kind}|{natural_key}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big") >> (64 - bits)
    # 0 is avoided so a falsy id can never be mistaken for a missing one.
    return value or 1


class IdRegistry:
    """Issues ids and refuses to issue the same one for two different keys."""

    def __init__(self, bits: int = DEFAULT_BITS):
        self.bits = bits
        self._by_id: dict[int, str] = {}

    def issue(self, kind: str, natural_key: str) -> int:
        key = f"{kind}|{natural_key}"
        node_id = stable_id(kind, natural_key, self.bits)
        seen = self._by_id.get(node_id)
        if seen is None:
            self._by_id[node_id] = key
        elif seen != key:
            raise IdCollision(
                f"id {node_id} was issued for {seen!r} and is now wanted for "
                f"{key!r}. Two unrelated nodes would silently merge. Rerun "
                f"with a wider id space (bits > {self.bits}).")
        return node_id

    @property
    def issued(self) -> int:
        return len(self._by_id)

    # Deliberately NO __len__. Defining it made a fresh registry falsy, and
    # `registry or IdRegistry()` in Writer silently discarded a caller's
    # 44-bit registry for a default 52-bit one. The ids still looked fine.
