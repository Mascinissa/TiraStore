"""Deterministic key generation for TiraStore.

Record keys and program hashes are SHA-256 hex digests computed from
normalized, canonical representations of the input.  This ensures that
identical logical inputs always produce the same key regardless of
cosmetic differences (whitespace, comment style, quote style, dict ordering).
"""

from __future__ import annotations

import hashlib
import json

from tirastore._schedule import normalize_program, normalize_schedule


def canonical_json(obj: dict) -> str:
    """Return a deterministic JSON string for *obj*.

    Keys are sorted, no extra whitespace, ASCII-safe.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def make_program_hash(program_source_code: str) -> str:
    """Compute a SHA-256 hex hash from the normalized program source code.

    The source is normalized (comments, includes, whitespace removed) before
    hashing so that cosmetically different versions of the same program
    produce the same hash.
    """
    normalized = normalize_program(program_source_code)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_key(
    program_hash: str,
    tiralib_schedule_string: str,
) -> str:
    """Compute a SHA-256 hex record key from a program hash and schedule.

    The schedule string is normalized before hashing so that equivalent
    schedules (differing only in whitespace or quote style) produce the
    same key.
    """
    input_obj = {
        "program_hash": program_hash,
        "tiralib_schedule_string": normalize_schedule(tiralib_schedule_string),
    }
    blob = canonical_json(input_obj).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
