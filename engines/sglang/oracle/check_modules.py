# Original to blis-sglang.
"""Fail unless every sglang module the scheduler's own files import is classified: covered by a
decision prefix of oracle/harness.json, or listed with a reason in oracle/modules.json (FID-1, X16).

check_mocks.py refuses a replacement under a decision prefix, so a decision module the prefixes
miss could be replaced unnoticed. This check derives the modules from SGLang's code instead of a
hand list: it parses python/sglang/srt/managers/scheduler*.py and
managers/scheduler_components/*.py at the checkout, collects every absolute sglang import, at any
depth, and fails on one that no decision prefix and no modules.json entry covers. It also fails
on a prefix or entry that names no module at the checkout, so the lists cannot go stale.

A prefix or entry covers a module when the module is it (without a trailing '.') or lies under it.

Usage: python3 -I oracle/check_modules.py <sglang checkout> [oracle/harness.json] [oracle/modules.json]
"""
import ast
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEDULER_FILES = ("python/sglang/srt/managers/scheduler*.py", "python/sglang/srt/managers/scheduler_components/*.py")


def imported(root: str) -> dict:
    """{module: first file importing it} for every absolute sglang import of the scheduler's files."""
    out = {}
    files = sorted({f for pat in SCHEDULER_FILES for f in glob.glob(os.path.join(root, pat))})
    if not files:
        raise SystemExit(f"{root}: no scheduler files under python/sglang/srt/managers; is it an SGLang checkout?")
    for f in files:
        with open(f) as fh:
            tree = ast.parse(fh.read(), f)
        rel = os.path.relpath(f, root)
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                names = [n.module]
            elif isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            for m in names:
                if m == "sglang" or m.startswith("sglang."):
                    out.setdefault(m, rel)
    return out


def covers(entry: str, module: str) -> bool:
    e = entry.rstrip(".")
    return module == e or module.startswith(e + ".")


def exists(root: str, entry: str) -> bool:
    path = os.path.join(root, "python", *entry.rstrip(".").split("."))
    return os.path.isfile(path + ".py") or os.path.isdir(path)


def problems(root: str, prefixes: list, not_decision: dict) -> list:
    bad = []
    mods = imported(root)
    for m, where in sorted(mods.items()):
        if not any(covers(p, m) for p in prefixes) and not any(covers(k, m) for k in not_decision):
            bad.append(f"{m} (imported by {where}): neither a decision prefix of oracle/harness.json nor an "
                       "entry of oracle/modules.json covers it; classify it")
    for p in prefixes:
        if not exists(root, p):
            bad.append(f"decision prefix {p}: no such module at the checkout")
    for k, why in sorted(not_decision.items()):
        if not exists(root, k):
            bad.append(f"modules.json entry {k}: no such module at the checkout")
        if not str(why).strip():
            bad.append(f"modules.json entry {k}: gives no reason")
    return bad


def main(argv: list) -> int:
    if not 1 <= len(argv) <= 3:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    root = argv[0]
    harness = argv[1] if len(argv) > 1 else os.path.join(HERE, "harness.json")
    modules = argv[2] if len(argv) > 2 else os.path.join(HERE, "modules.json")
    with open(harness) as f:
        prefixes = json.load(f)["decision_prefixes"]
    with open(modules) as f:
        not_decision = json.load(f)["not_decision"]
    bad = problems(root, prefixes, not_decision)
    for b in bad:
        print(b)
    print(f"check_modules: {len(imported(root))} imported modules, {len(bad)} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
