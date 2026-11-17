"""Which commits a router build could be, and whether the files that decide its mode are the same at all of them.

A router-table entry (contracts/spec/routertable) covers a build that a recipe names by version, such
as the Dynamo wheel 1.3.0.dev20260601, only if the files that decide the mode are the same at every
commit the build could have been made from. This script finds those commits and compares the files.
Run it on a clone of the router's repository; it reads trees and the named files only:

    python3 -I catalog/scripts/router_normal_form.py check <git-dir> <version> [--path P ...]

For a nightly version (X.Y.Z.devYYYYMMDD) the commits are the first-parent commits of `main` from the
head at 00:00 UTC of that date to the head at 00:00 UTC of the next (a nightly is made from the head
of main when its run was triggered, and its hour is not in the version). For a release (X.Y.Z) it is
the commit of the tag vX.Y.Z. For each file it prints the distinct git blobs, the commits holding
each, and the sha256 of each blob's normal form, then a verdict: `identical` (one blob),
`equivalent` (several blobs, one normal form) or `differs`.

The normal form (rule set `rust-logging-only`) applies to Rust files only; any other file is its own
normal form, so a Python file that differs is `differs`. For a Rust file the rules run in this order,
each a mechanical rewrite that changes no value the function computes:

  1. strip-logging-statements: delete a statement that is a `tracing::<level>!(...)` or
     `<level>!(...)` macro call, trace, debug, info, warn or error, which starts its line and ends at
     the closing parenthesis and its semicolon. The arguments of the removed call are assumed to have
     no effect on routing; `check --show-removed` prints every removed statement so a reader can
     confirm that.
  2. inline-return-binding: `let x = E;` on one line followed by `return x;` becomes `return E;`
     when x is not read in E or afterwards in the block (the binding only carried the value to the
     logging that rule 1 removed).
  3. drop-unread-tuple-element: `let (a, b) = {` ... `(E, x)` `};` becomes `let a = {` ... `E` `};`
     when the block's last line is a pair whose second element is a plain identifier x and b is
     read nowhere in the enclosing block.
  4. ignore-blank-lines: delete lines that are empty or only whitespace.

A rule never fires on anything else, and a difference that remains after the rules is reported as
`differs`, so the build is not covered.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import re
import subprocess
import sys

RULE_SET = "rust-logging-only"
RULES = ["strip-logging-statements", "inline-return-binding", "drop-unread-tuple-element", "ignore-blank-lines"]

DYNAMO_PATHS = [
    "components/src/dynamo/common/configuration/groups/router_args.py",
    "components/src/dynamo/frontend/main.py",
    "lib/runtime/src/pipeline/network/egress/push_router.rs",
    "lib/runtime/src/component/client.rs",
    "lib/llm/src/kv_router/prefill_router/mod.rs",
]

_LOG = re.compile(r"(?:tracing::)?\b(?:trace|debug|info|warn|error)!\s*\(")


def strip_logging(src: str, removed: list[str] | None = None) -> str:
    """Rule 1. Delete each logging macro statement that starts its line."""
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        m = _LOG.search(src, i)
        if not m:
            out.append(src[i:])
            break
        line_start = src.rfind("\n", 0, m.start()) + 1
        if src[line_start:m.start()].strip() != "":  # not at the start of a statement
            out.append(src[i:m.end()])
            i = m.end()
            continue
        out.append(src[i:line_start])
        j, depth = m.end(), 1
        while j < n and depth:
            c = src[j]
            if c == '"':
                j += 1
                while j < n and src[j] != '"':
                    j += 2 if src[j] == "\\" else 1
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        if src[j:j + 1] == ";":
            j += 1
        if removed is not None:
            removed.append(" ".join(src[line_start:j].split()))
        nl = src.find("\n", j)
        i = nl + 1 if nl != -1 and src[j:nl].strip() == "" else j
    return "".join(out)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block_after(lines: list[str], start: int, indent: int) -> list[str]:
    """The lines after `start` up to the first non-blank line indented less than `indent`."""
    out = []
    for line in lines[start + 1:]:
        if line.strip() and _indent(line) < indent:
            break
        out.append(line)
    return out


def _reads(name: str, lines: list[str]) -> bool:
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    return any(pat.search(line) for line in lines)


_LET_RETURN = re.compile(r"^(\s*)let (\w+)(?:: [^=]+)? = (.+);$")


def inline_return_binding(src: str) -> str:
    """Rule 2."""
    lines = src.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _LET_RETURN.match(lines[i])
        if m and i + 1 < len(lines):
            ind, name, expr = m.groups()
            nxt = lines[i + 1]
            if nxt == f"{ind}return {name};" and not _reads(name, [expr]) \
                    and not _reads(name, _block_after(lines, i + 1, len(ind))):
                out.append(f"{ind}return {expr};")
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


_LET_PAIR = re.compile(r"^(\s*)let \((\w+), (\w+)\) = \{$")


def _split_top_comma(s: str) -> tuple[str, str] | None:
    depth = 0
    for k in range(len(s) - 1, -1, -1):
        c = s[k]
        if c in ")]}":
            depth += 1
        elif c in "([{":
            depth -= 1
        elif c == "," and depth == 0:
            return s[:k].strip(), s[k + 1:].strip()
    return None


def drop_unread_tuple_element(src: str) -> str:
    """Rule 3."""
    lines = src.split("\n")
    i = 0
    while i < len(lines):
        m = _LET_PAIR.match(lines[i])
        if m:
            ind, a, b = m.groups()
            end = next((k for k in range(i + 1, len(lines)) if lines[k] == f"{ind}}};"), None)
            if end is not None:
                last = end - 1
                while last > i and not lines[last].strip():
                    last -= 1
                inner = re.match(r"^(\s*)\((.*)\)$", lines[last])
                pair = _split_top_comma(inner.group(2)) if inner else None
                body = lines[i + 1:end]
                if pair and re.fullmatch(r"\w+", pair[1]) and not _reads(b, body[:last - i - 1]) \
                        and not _reads(b, _block_after(lines, end, len(ind))):
                    lines[i] = f"{ind}let {a} = {{"
                    lines[last] = f"{inner.group(1)}{pair[0]}"
        i += 1
    return "\n".join(lines)


def ignore_blank_lines(src: str) -> str:
    """Rule 4."""
    return "\n".join(line for line in src.split("\n") if line.strip())


def normal_form(path: str, text: str, removed: list[str] | None = None) -> str:
    """The text a file is compared by. Only Rust files are rewritten."""
    if not path.endswith(".rs"):
        return text
    text = strip_logging(text, removed)
    text = inline_return_binding(text)
    text = drop_unread_tuple_element(text)
    return ignore_blank_lines(text)


def digest(path: str, text: str) -> str:
    return hashlib.sha256(normal_form(path, text).encode()).hexdigest()


def _git(git_dir: str, *args: str) -> str:
    r = subprocess.run(["git", "--git-dir", git_dir, *args], capture_output=True, text=True)
    return r.stdout


def window(git_dir: str, version: str) -> list[str]:
    """The commits `version` could be built from, newest first."""
    m = re.fullmatch(r"(\d+\.\d+\.\d+)\.dev(\d{8})", version)
    if not m:
        return [_git(git_dir, "rev-parse", f"v{version}^{{commit}}").strip()]
    day = datetime.datetime.strptime(m.group(2), "%Y%m%d").replace(tzinfo=datetime.timezone.utc)
    lo, hi = int(day.timestamp()), int((day + datetime.timedelta(days=1)).timestamp())
    rows = [l.split() for l in _git(git_dir, "log", "--first-parent", "--format=%H %ct", "main").splitlines()]
    if any(int(a[1]) < int(b[1]) for a, b in zip(rows, rows[1:])):
        raise SystemExit("committer times are not monotone along main: the window cannot be cut by time")
    newest = next(i for i, r in enumerate(rows) if int(r[1]) < hi)
    oldest = next(i for i, r in enumerate(rows) if int(r[1]) < lo)
    return [r[0] for r in rows[newest:oldest + 1]]


def check(git_dir: str, commits: list[str], paths: list[str], show_removed: bool = False) -> dict[str, dict]:
    """Per path: the distinct blobs (commits holding each, normal-form digest) and the verdict."""
    result = {}
    for p in paths:
        blobs: dict[str, list[str]] = {}
        for c in commits:
            b = _git(git_dir, "rev-parse", "--verify", "-q", f"{c}:{p}").strip()
            blobs.setdefault(b or "missing", []).append(c)
        rows = {}
        for b, cs in blobs.items():
            text = _git(git_dir, "show", f"{cs[0]}:{p}") if b != "missing" else ""
            removed: list[str] = []
            rows[b] = {"commits": sorted(cs), "sha256": digest(p, text) if b != "missing" else None}
            if show_removed and p.endswith(".rs"):
                normal_form(p, text, removed)
                rows[b]["removed"] = removed
        digests = {r["sha256"] for r in rows.values()}
        verdict = ("differs" if "missing" in rows or len(digests) > 1
                   else "identical" if len(rows) == 1 else "equivalent")
        result[p] = {"blobs": rows, "verdict": verdict}
    return result


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="compare the deciding files across a build's commits")
    c.add_argument("git_dir")
    c.add_argument("version")
    c.add_argument("--path", action="append", help="a deciding file (default: Dynamo's five)")
    c.add_argument("--show-removed", action="store_true", help="print every logging statement the rules removed")
    args = ap.parse_args(argv)
    commits = window(args.git_dir, args.version)
    print(f"{args.version}: {len(commits)} commit(s), rule set {RULE_SET}: {', '.join(RULES)}")
    for p, r in check(args.git_dir, commits, args.path or DYNAMO_PATHS, args.show_removed).items():
        print(f"{r['verdict']:10} {p}")
        for b, row in r["blobs"].items():
            print(f"    blob {b} normal-form sha256 {row['sha256']} at {len(row['commits'])} commit(s): "
                  f"{', '.join(row['commits'])}")
            for s in row.get("removed", []):
                print(f"        removed: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
