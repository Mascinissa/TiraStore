"""Deterministic key generation for TiraStore.

The key for each record is the SHA-256 hex digest of the canonical JSON
representation of the input.  Canonical means: sorted keys, no extra
whitespace, ensure_ascii=True — so the same logical input always produces
the same key regardless of dict ordering or platform.
"""

from __future__ import annotations

import hashlib
import json


def canonical_json(obj: dict) -> str:
    """Return a deterministic JSON string for *obj*.

    Keys are sorted, no extra whitespace, ASCII-safe.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def make_key(
    program_name: str,
    program_tiramisu_source_code: str,
    tiralib_schedule_string: str,
) -> str:
    """Compute a SHA-256 hex key from the three input fields."""
    input_obj = {
        "program_name": program_name,
        "program_tiramisu_source_code": program_tiramisu_source_code,
        "tiralib_schedule_string": tiralib_schedule_string,
    }
    blob = canonical_json(input_obj).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def make_input_json(
    program_name: str,
    program_tiramisu_source_code: str,
    tiralib_schedule_string: str,
) -> str:
    """Return the canonical JSON text for the input fields (stored in DB)."""
    input_obj = {
        "program_name": program_name,
        "program_tiramisu_source_code": program_tiramisu_source_code,
        "tiralib_schedule_string": tiralib_schedule_string,
    }
    return canonical_json(input_obj)
