# Original to blis-vllm.
"""Fail if oracle/lower/extract.py patches something oracle/lower/replacements.json does not list.

extract.py replaces vLLM names only by assigning into its own namespace (`ns[...] = ...`,
`ns.update(...)`, `ns["..."].__globals__[...] = ...`) and by `sys.modules[...]`. This reads those
sites from the syntax tree and requires each name to appear in the `symbol` of some entry, so a
replacement nobody listed fails (FID-1), and requires every entry to be of an allowed kind.

Usage: python3 -I oracle/lower/check_replacements.py oracle/lower/extract.py oracle/lower/replacements.json
"""
import ast
import json
import sys

ALLOWED = {"gpu", "clock", "transport", "environment", "external-service"}
# Plumbing the harness uses to run the functions, not a replacement of anything vLLM defines.
PLUMBING = {"__builtins__", "dataclass", "replace", "enum", "product", "groupby", "defaultdict",
            "Any", "annotations", "torch", "nullcontext"}
# Names the harness gives its own copies of the functions it runs, so the cases can call them: the
# function bodies are vLLM's, taken from the source by `ast` and only renamed.
RENAMED = {"_get_shard_info", "_set_cudagraph_sizes", "sync_cudagraph_and_dp_padding", "Manager"}
# Type names a signature mentions; annotations are evaluated when the def runs.
ANNOTATION_ONLY = {"CudaGraphManager", "ParallelConfig"}


def _is_ns(node):
    """The namespace the definitions run in: `ns`, or `X.__globals__` of a function taken from it."""
    return (isinstance(node, ast.Name) and node.id == "ns") or (
        isinstance(node, ast.Attribute) and node.attr == "__globals__")


def sites(tree):
    """Names written into the namespace the extracted vLLM definitions run in, or into sys.modules."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                    if _is_ns(t.value) or (isinstance(t.value, ast.Attribute) and t.value.attr == "modules"):
                        out.add(t.slice.value)
                if isinstance(t, ast.Attribute) and t.attr == "current_platform":
                    out.add(t.attr)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "update":
            for kw in n.keywords:
                out.add(kw.arg)
    return out


def main(argv):
    src, lst = argv[1], argv[2]
    tree = ast.parse(open(src, encoding="utf-8").read())
    spec = json.load(open(lst, encoding="utf-8"))
    syms = " ".join(e["symbol"] for e in spec["replacements"])
    bad = []
    for e in spec["replacements"]:
        if e["kind"] not in ALLOWED:
            bad.append(f"{e['symbol']}: kind {e['kind']!r} is not one of {sorted(ALLOWED)}")
    for name in sorted(sites(tree) - PLUMBING - ANNOTATION_ONLY - RENAMED):
        if name not in syms and name not in {"vllm", "vllm.platforms"}:
            bad.append(f"extract.py patches {name!r}, which replacements.json does not list")
    if bad:
        raise SystemExit("\n".join(bad))
    print("ok: every patch in", src, "is listed")


if __name__ == "__main__":
    main(sys.argv)
