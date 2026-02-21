"""Schedule and program normalization and validation for TiraStore.

The schedule string encodes a sequence of Tiramisu optimization
transformations separated by ``|``.  Before storing or looking up a record,
the schedule is:

1. **Normalized** — whitespace stripped, comp names consistently single-quoted.
2. **Validated** — checked against the grammar of known transformations.

Program source code is normalized (for hashing only) by stripping comments,
``#include`` directives, and all whitespace so that cosmetically different
versions of the same program produce the same hash.
"""

from __future__ import annotations

import re


def normalize_schedule(schedule: str) -> str:
    """Normalize a schedule string.

    - Removes all whitespace.
    - Ensures all comp names inside ``comps=[...]`` blocks are wrapped in
      single quotes.

    Returns an empty string for empty/blank input.
    """
    if not schedule:
        return ""

    # Remove all whitespace
    s = re.sub(r"\s+", "", schedule)

    # Normalize comp names: ensure single quotes around each comp name
    def _normalize_comps(m: re.Match) -> str:
        inner = m.group(1)
        items = inner.split(",")
        normalized = []
        for item in items:
            item = item.strip("\"'")
            normalized.append(f"'{item}'")
        return f"comps=[{','.join(normalized)}]"

    s = re.sub(r"comps=\[([^\]]*)\]", _normalize_comps, s)
    return s


def validate_schedule(schedule: str) -> tuple[bool, str]:
    """Validate a schedule string format.

    Returns ``(True, "")`` if valid, or ``(False, reason)`` if invalid.
    An empty string is considered valid.
    """
    if not schedule:
        return True, ""

    # Work on a whitespace-free copy
    s = re.sub(r"\s+", "", schedule)

    # Quoted comp name (single or double quotes)
    _Q = r"""(?:'[^']*'|"[^"]*")"""
    # Unquoted comp name
    _UQ = r"[A-Za-z_]\w*"
    # Any comp name token
    _CN = rf"(?:{_Q}|{_UQ})"
    # comps=[...] block with at least one entry
    _COMPS = rf"comps=\[{_CN}(?:,{_CN})*\]"
    _INT = r"\d+"  # positive integer
    _SINT = r"-?\d+" # positive or negative integer
    _LX = r"L\d+"  # loop level token

    # Per-transformation patterns
    patterns = {
        "S": rf"^S\({_LX},{_LX},{_SINT},{_SINT},{_COMPS}\)$",
        "I": rf"^I\({_LX},{_LX},{_COMPS}\)$",
        "R": rf"^R\({_LX},{_COMPS}\)$",
        "P": rf"^P\({_LX},{_COMPS}\)$",
        "T2": rf"^T2\({_LX},{_LX},{_INT},{_INT},{_COMPS}\)$",
        "T3": rf"^T3\({_LX},{_LX},{_LX},{_INT},{_INT},{_INT},{_COMPS}\)$",
        "U": rf"^U\({_LX},{_INT},{_COMPS}\)$",
        "F": rf"^F\({_LX},{_COMPS}\)$",
    }
    compiled = {k: re.compile(v) for k, v in patterns.items()}

    for token in s.split("|"):
        if not token:
            return False, "Empty segment in schedule (leading, trailing, or double '|')."

        name = re.match(r"^([A-Z][A-Z0-9]*)", token)
        if not name:
            return False, f"Unrecognized token (does not start with a transformation name): {token!r}"

        tname = name.group(1)
        rule = compiled.get(tname)
        if rule is None:
            return False, f"Unknown transformation: {tname!r} in {token!r}"

        if not rule.match(token):
            return False, f"Malformed {tname} transformation: {token!r}"

    return True, ""


def normalize_program(source: str) -> str:
    """Normalize a Tiramisu/C++ program source for hash computation.

    The goal is to produce a canonical string such that two programs
    differing only in whitespace, comments, or ``#include`` lines yield
    the same hash.  The result is **not** meant to be human-readable.

    Steps:

    1. Remove block comments (``/* ... */``)
    2. Remove single-line comments (``// ...``)
    3. Remove ``#include`` lines
    4. Remove all whitespace (spaces, tabs, newlines)
    """
    if not source:
        return ""
    # 1. Remove block comments (non-greedy, handles multiline)
    text = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    # 2. Remove single-line comments
    text = re.sub(r"//[^\n]*", "", text)
    # 3. Remove #include lines
    text = re.sub(
        r'^\s*#\s*include\s*[<"][^>"]*[>"]\s*$', "", text, flags=re.MULTILINE
    )
    # 4. Remove all whitespace characters
    text = re.sub(r"\s+", "", text)
    return text
