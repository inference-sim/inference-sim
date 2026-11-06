"""Lint for Python code that restates a Go validation rule (X15).

Each validation rule exists once, in Go, in blis-substrate. A Python script that needs a
rule calls the substrate's validate command; a Python check that restates one must pass a
parity test that runs it and the Go rule on the same failing examples.

This lint reads every tracked Python file outside tests and reports two kinds of
schema-like rule code:

  set     a literal list, tuple, set or dict (or a constant expression that builds one:
          a | b, a + b, a.union(b), set(x), frozenset(x), names assigned once) whose string
          members include two or more
          members of one closed Go value vocabulary (units, methods, provenances, source
          kinds, source roles, licences). Restating a vocabulary is restating the rule
          that it is closed. Scope keys are left out: they are document keys, and a
          script that builds a scope names them without restating a rule.
  compare a comparison of a document field named for a vocabulary (method, unit,
          provenance, licence) with a member of that vocabulary, in a function whose name
          says it validates or checks (validate*, check*, *_valid, *_ok).

The vocabularies come from `go run ./tools/prcheck vocab`, so the lint compares against
the Go sets, not a copy. Each finding must be listed in tools/pyrules/allowed.txt with the
reason it is not a rule (a display order, a parity-tested restatement); the list only
shrinks, and an entry that no longer occurs fails.

Usage: python3 tools/pyrules/pyrules.py <vocab.json>   (from the repository root)
Exit 0 clean, 1 findings.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

FIELDS = {"method": "methods", "unit": "units", "units": "units", "provenance": "provenances",
          "licence": "licences", "license": "licences"}
CHECK_FN = re.compile(r"^(validate|check)|(_valid|_ok)$")
ALLOWED = Path(__file__).with_name("allowed.txt")


def tracked_python() -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "*.py"], text=True).split()
    return [f for f in out if "/tests/" not in f and not Path(f).name.startswith("test_")
            and "/testdata/" not in f]


def field_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "get" and node.args \
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
            and isinstance(node.slice.value, str):
        return node.slice.value
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def strings_of(node: ast.AST, env: dict[str, list[str]] | None = None) -> list[str]:
    """The string constants a container expression holds, folding the constant
    expressions that build one: a | b, a + b, a.union(b), set(x), frozenset(x),
    tuple(x), list(x), sorted(x), and names assigned once from such an expression."""
    env = env or {}
    if isinstance(node, ast.Name):
        return list(env.get(node.id, []))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitOr, ast.Add)):
        return strings_of(node.left, env) + strings_of(node.right, env)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name) and f.id in ("set", "frozenset", "tuple", "list", "sorted") \
                and len(node.args) == 1:
            return strings_of(node.args[0], env)
        if isinstance(f, ast.Attribute) and f.attr == "union":
            return strings_of(f.value, env) + [s for a in node.args for s in strings_of(a, env)]
        return []
    elts = node.keys if isinstance(node, ast.Dict) else getattr(node, "elts", [])
    return [e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]


def names_of(tree: ast.AST) -> dict[str, list[str]]:
    """Names assigned exactly once, to an expression that folds to string constants."""
    count: dict[str, int] = {}
    value: dict[str, ast.AST] = {}
    for n in ast.walk(tree):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) and n.value else []
        for t in targets:
            if isinstance(t, ast.Name):
                count[t.id] = count.get(t.id, 0) + 1
                value[t.id] = n.value
    env: dict[str, list[str]] = {}
    for _ in range(len(value)):  # a name may be built from names assigned later in the file
        for k, v in value.items():
            if count[k] == 1:
                got = strings_of(v, env)
                if got:
                    env[k] = got
    return env


def findings(path: str, src: str, vocab: dict[str, list[str]]) -> list[str]:
    tree = ast.parse(src, filename=path)
    sets = {k: set(v) for k, v in vocab.items() if k != "scope_keys"}
    env = names_of(tree)
    out = []
    fns: list[tuple[int, int, str]] = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fns.append((n.lineno, n.end_lineno or n.lineno, n.name))

    def fn_at(line: int) -> str:
        best = ""
        for a, b, name in fns:
            if a <= line <= b:
                best = name
        return best

    for n in ast.walk(tree):
        if isinstance(n, (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.BinOp, ast.Call)):
            members = strings_of(n, env)
            for name, words in sets.items():
                hit = sorted(set(members) & words)
                if len(hit) >= 2:
                    out.append(f"{path}: set {name} {','.join(hit)}")
        if isinstance(n, ast.Compare):
            f = field_of(n.left)
            if f in FIELDS and CHECK_FN.search(fn_at(n.lineno)):
                for c in n.comparators:
                    vals = [c.value] if isinstance(c, ast.Constant) else strings_of(c, env)
                    for v in vals:
                        if isinstance(v, str) and v in sets[FIELDS[f]]:
                            out.append(f"{path}: compare {f} {v} in {fn_at(n.lineno)}")
    return sorted(set(out))


def load_allowed() -> dict[str, str]:
    allowed = {}
    for line in ALLOWED.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        finding, sep, reason = line.partition(" # ")
        if not sep or not reason.strip():
            raise SystemExit(f"{ALLOWED}: {line!r} gives no reason after ' # '")
        allowed[finding.strip()] = reason.strip()
    return allowed


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    vocab = json.loads(Path(argv[1]).read_text())
    found = []
    for f in tracked_python():
        found += findings(f, Path(f).read_text(), vocab)
    allowed = load_allowed()
    bad = [f"{f}: restates a Go rule; call the validate command, or list it in {ALLOWED.name} "
           f"with the reason it is not a rule (X15)" for f in found if f not in allowed]
    bad += [f"{ALLOWED.name}: {a!r} no longer occurs; remove it, the list only shrinks (X15)"
            for a in allowed if a not in found]
    for b in bad:
        print(b)
    print(f"pyrules: {len(tracked_python())} files, {len(found)} finding(s), {len(bad)} not allowed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
