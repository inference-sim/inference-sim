# Original to blis-sglang.
"""Fail unless core/cache/pool_classes.json lists every allocator, KV pool and request pool SGLang
can select at the pinned commit (SGL-6): every class under python/sglang/ (tests aside) that
subclasses BaseTokenToKVPoolAllocator, KVCache or ReqToTokenPool, at any depth, must be listed as
modelled or refused, and every listed class must be defined at the site it names. A class listed
that is no subclass is allowed only if it is a stand-in (STAND_INS) or a concrete root.

The scan is static (ast): SGLang is read, not imported. Standard library only.

Usage: python3 -I oracle/check_pool_classes.py <sglang checkout> [<pool_classes.json>]
"""
import ast
import functools
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
LIST = os.path.join(os.path.dirname(HERE), "core", "cache", "pool_classes.json")
ROOTS = {"BaseTokenToKVPoolAllocator": "allocator", "KVCache": "kv_pool", "ReqToTokenPool": "req_to_token"}
# Not subclasses, but selected in a base class's place: DecodeReqToTokenPool borrows
# ReqToTokenPool's methods (disaggregation/decode.py:171); MambaPool is the request-indexed state
# pool (mem_cache/memory_pool.py:505).
STAND_INS = {"DecodeReqToTokenPool", "MambaPool"}
# A root that is itself a concrete class, selected as it is.
CONCRETE_ROOTS = {"ReqToTokenPool"}


def base_name(b):
    if isinstance(b, ast.Name):
        return b.id
    if isinstance(b, ast.Attribute):
        return b.attr
    return None


@functools.lru_cache(maxsize=None)
def scan(sglang_root: str) -> dict:
    """{class: [(site, [bases])]} for every class under python/sglang/, tests aside."""
    top = os.path.join(sglang_root, "python", "sglang")
    out = {}
    for dirpath, dirs, files in os.walk(top):
        dirs[:] = sorted(d for d in dirs if d not in ("test", "tests", "__pycache__"))
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            with open(path, encoding="utf-8") as fh:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        tree = ast.parse(fh.read(), path)
                except SyntaxError:
                    continue
            rel = os.path.relpath(path, os.path.join(top, "srt")).replace(os.sep, "/")
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    out.setdefault(node.name, []).append(
                        (f"{rel}:{node.lineno}", [n for n in map(base_name, node.bases) if n]))
    return out


def subclasses(classes: dict) -> dict:
    """{class: kind} for every subclass of a root, by name, at any depth."""
    kind = dict(ROOTS)
    changed = True
    while changed:
        changed = False
        for name, defs in classes.items():
            if name in kind:
                continue
            for _, bases in defs:
                hit = next((kind[b] for b in bases if b in kind), None)
                if hit:
                    kind[name] = hit
                    changed = True
                    break
    return {n: k for n, k in kind.items() if n not in ROOTS}


def problems(sglang_root: str, listing: dict) -> list:
    classes = scan(sglang_root)
    want = subclasses(classes)
    listed = {c["class"]: c for c in listing["classes"]}
    bad = []
    for name in sorted(want):
        c = listed.get(name)
        if c is None:
            sites = ", ".join(s for s, _ in classes[name])
            bad.append(f"{name} ({want[name]}, {sites}) is not in core/cache/pool_classes.json")
        elif c["kind"] != want[name]:
            bad.append(f"{name} is listed as a {c['kind']}, but subclasses the {want[name]} base")
    for name, c in sorted(listed.items()):
        if name not in want and name not in STAND_INS | CONCRETE_ROOTS:
            bad.append(f"{name} is listed but is no subclass of {', '.join(ROOTS)} at this commit")
        if c.get("status") not in ("modelled", "refused"):
            bad.append(f"{name}: status {c.get('status')!r}")
        if c["site"] not in {s for s, _ in classes.get(name, [])}:
            bad.append(f"{name}: no class {name} at {c['site']}")
    return bad


def main(argv) -> int:
    if len(argv) not in (1, 2):
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    with open(argv[1] if len(argv) == 2 else LIST) as f:
        listing = json.load(f)
    bad = problems(argv[0], listing)
    for b in bad:
        print(b)
    print(f"check_pool_classes: {len(listing['classes'])} classes listed, {len(bad)} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
