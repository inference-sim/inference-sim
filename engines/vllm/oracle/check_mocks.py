# Original to blis-vllm.
"""Fail if an oracle runner replaces vLLM code that oracle/mocks.json does not list, or if
mocks.json lists decision code.

The oracle may replace the GPU, the clock, transports and external services, never a
scheduling, KV-cache or connector decision (#2021 B8). Two checks:

1. Every entry's kind is allowed, and no entry's symbol lies under a decision module (a clock
   inside one is allowed: the clock is the environment even where a decision reads it).
2. The runners are read, not trusted. Each oracle/run_*.py is parsed, and every place it
   writes into an object it did not define is found (Sites below): attribute and item stores,
   setattr, delattr and del, and objects built with object.__new__ instead of their
   constructor. Each such site must be named, as "<runner>:<target>", in the `sites` of a
   mocks.json entry, so a patch nobody listed fails CI, and an entry whose sites no runner
   makes fails too. A setting changed after a vLLM config was constructed is a site as well:
   the constructor's own checks do not see it.

Usage: python3 -I oracle/check_mocks.py [oracle/mocks.json]
"""
import ast
import json
import os
import sys

ALLOWED_KINDS = {"gpu", "clock", "transport", "environment", "external-service"}
DECISION_PREFIXES = (
    "vllm.v1.core.",
    "vllm.v1.engine.core",
    "vllm.distributed.kv_transfer.kv_connector.v1.",
    "vllm.v1.kv_offload.",
)


def _root(node):
    """The name at the root of an attribute chain (a.b.c -> a), or None."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _chain(node) -> str:
    return ast.unparse(node)


class Sites(ast.NodeVisitor):
    """Every place one runner writes into an object it did not define: an attribute or item
    store on anything but `self` (a runner's own helper class), a setattr or delattr, a del of
    an attribute, and an object built with object.__new__. The runners write only into vLLM's
    modules, classes and objects, so this is default-deny: a write the list does not name
    fails, whatever the object."""

    def __init__(self):
        self.sites: dict[str, int] = {}

    def add(self, target: str, line: int) -> None:
        self.sites.setdefault(target, line)

    def _store(self, t, line):
        if isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._store(e, line)
        elif isinstance(t, ast.Attribute) and _root(t) != "self":
            self.add(_chain(t), line)
        elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) and _root(t.value) != "self":
            self.add(_chain(t.value) + "[]", line)

    def visit_Assign(self, node):
        for t in node.targets:
            self._store(t, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._store(node.target, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._store(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Delete(self, node):
        for t in node.targets:
            self._store(t, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node):
        fn = _chain(node.func)
        if fn in ("setattr", "delattr") and node.args and _root(node.args[0]) != "self":
            name = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "?"
            self.add(f"{_chain(node.args[0])}.{name}", node.lineno)
        if fn == "object.__new__" and node.args:
            self.add(f"object.__new__({_chain(node.args[0])})", node.lineno)
        self.generic_visit(node)


def runner_sites(oracle_dir: str) -> dict[str, int]:
    out = {}
    for name in sorted(os.listdir(oracle_dir)):
        if not (name.startswith("run_") and name.endswith(".py")):
            continue
        with open(os.path.join(oracle_dir, name)) as f:
            tree = ast.parse(f.read(), name)
        v = Sites()
        v.visit(tree)
        for target, line in v.sites.items():
            out[f"{name}:{target}"] = line
    return out


def main(path: str) -> int:
    with open(path) as f:
        spec = json.load(f)
    bad = []
    listed: dict[str, str] = {}
    for m in spec["mocks"]:
        sym, kind = m["symbol"], m["kind"]
        if kind not in ALLOWED_KINDS:
            bad.append(f"{sym}: kind {kind!r} is not one of {sorted(ALLOWED_KINDS)}")
        if kind != "clock" and sym.startswith(DECISION_PREFIXES):
            bad.append(f"{sym}: replaces decision code ({kind})")
        for s in m.get("sites", []):
            listed[s] = sym
    found = runner_sites(os.path.dirname(os.path.abspath(path)))
    for site, line in sorted(found.items()):
        if site not in listed:
            runner = site.split(":", 1)[0]
            bad.append(f"{runner}:{line}: writes {site.split(':', 1)[1]}, which is not listed in mocks.json "
                       "(add it to the sites of the entry that justifies it, or remove the patch)")
    for site, sym in sorted(listed.items()):
        if site not in found:
            bad.append(f"{sym}: lists site {site}, which no runner makes (remove it)")
    for b in bad:
        print("mocks.json:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "oracle/mocks.json"))
