"""Entry-level rewriting of a committed coefficient set, byte-preserving elsewhere.

The sets are hand-ordered YAML with comments, so a writer that owns a few entries must
replace exactly those and pass every other byte through. An entry is identified by its
NAME and its SCOPE, because one name legitimately appears once per part (and, for a
model-scoped entry, once per model). Scopes are compared parsed, so a reordered or
re-spaced scope is still the same entry rather than a reason to insert a duplicate.
"""

from __future__ import annotations

import re

import yaml

ENTRY = re.compile(r"^  - ([a-z_0-9]+):$")
SCOPE = re.compile(r"^      scope: (\{.*\})$")


def split(text: str) -> list[list[str]]:
    """The file as blocks: each entry with its indented body, and every other line alone.

    A blank line inside an entry's body -- a paragraph break in a folded rationale -- stays
    with the entry when the next non-blank line is still indented as part of it, so
    replacing the entry cannot leave a stale paragraph behind under the new one.
    """
    lines = text.split("\n")
    out: list[list[str]] = []
    for i, line in enumerate(lines):
        in_entry = bool(out) and ENTRY.match(out[-1][0]) is not None
        if ENTRY.match(line):
            out.append([line])
        elif in_entry and line.startswith("    "):
            out[-1].append(line)
        elif in_entry and not line.strip():
            nxt = next((x for x in lines[i + 1:] if x.strip()), "")
            if nxt.startswith("    "):
                out[-1].append(line)
            else:
                out.append([line])
        else:
            out.append([line])
    return out


def join(blocks: list[list[str]]) -> str:
    return "\n".join(line for b in blocks for line in b)


def name_of(block: list[str]) -> str | None:
    m = ENTRY.match(block[0])
    return m.group(1) if m else None


def scope_of(block: list[str]) -> str | None:
    """The entry's scope exactly as written, e.g. '{hardware: [h200]}'."""
    for line in block:
        m = SCOPE.match(line)
        if m:
            return m.group(1)
    return None


def scope_key(scope: str) -> tuple:
    """A scope's identity, independent of key order and spacing."""
    parsed = yaml.safe_load(scope) or {}
    return tuple(sorted((k, tuple(sorted(v)) if isinstance(v, list) else (v,))
                        for k, v in parsed.items()))


def find(blocks: list[list[str]], name: str, scope: str) -> int | None:
    """The one block with this name and scope, or None. Two is an error: the kernel keeps
    the last in file order, so a writer maintaining the first would leave the live one stale."""
    key = scope_key(scope)
    hits = [j for j, b in enumerate(blocks)
            if name_of(b) == name and scope_of(b) is not None
            and scope_key(scope_of(b)) == key]
    if len(hits) > 1:
        raise SystemExit(f"{name} {scope}: {len(hits)} entries share this name and scope")
    return hits[0] if hits else None


def upsert(blocks: list[list[str]], entry: list[str], after) -> None:
    """Replace the entry with this (name, scope) in place, or insert it after the last
    block `after(block)` accepts. Refuses rather than appending somewhere arbitrary."""
    name, scope = name_of(entry), scope_of(entry)
    if name is None or scope is None:
        raise SystemExit(f"not a scoped entry: {entry[0]!r}")
    j = find(blocks, name, scope)
    if j is not None:
        blocks[j] = entry
        return
    anchors = [k for k, b in enumerate(blocks) if after(b)]
    if not anchors:
        raise SystemExit(f"{name} {scope}: no anchor entry to place it after")
    blocks.insert(anchors[-1] + 1, entry)
