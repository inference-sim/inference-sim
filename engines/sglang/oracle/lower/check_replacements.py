# Original to blis-sglang.
"""Fail if oracle/lower/extract.py replaces something oracle/lower/replacements.json does not list.

extract.py can replace SGLang's behaviour in three ways: a class it defines and hands to SGLang in
place of SGLang's own object, a run_serverargs function it calls that patches SGLang, and an
assignment into an attribute of a module it imports. This reads each from the syntax tree and
requires its name to be the symbol of an entry, of an allowed kind, so a replacement nobody listed
fails (FID-1). An entry that names nothing extract.py does fails too, so the list stays exact.

Usage: python3 -I oracle/lower/check_replacements.py oracle/lower/extract.py oracle/lower/replacements.json
"""
import ast
import json
import sys

ALLOWED = {"gpu", "clock", "transport", "environment", "platform", "external-service"}


def sites(tree):
    """The replacements extract.py makes, by the names an entry lists them under."""
    out = set()
    modules = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                modules.add((a.asname or a.name).split(".")[0])
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
            out.add(n.name)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.value.id == "run_serverargs" and n.func.attr not in ("model_facts",):
            out.add("run_serverargs." + n.func.attr)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Attribute):
                    base = t
                    while isinstance(base, ast.Attribute):
                        base = base.value
                    if isinstance(base, ast.Name) and base.id in modules:
                        out.add(ast.unparse(t))
    return out


def problems(source, listed):
    errs = []
    entries = listed.get("replacements", [])
    symbols = {e.get("symbol") for e in entries}
    for e in entries:
        if e.get("kind") not in ALLOWED:
            errs.append("%s: kind %r is not one of %s" % (e.get("symbol"), e.get("kind"), sorted(ALLOWED)))
        if not e.get("how"):
            errs.append("%s: no how" % e.get("symbol"))
    found = sites(ast.parse(source))
    for s in sorted(found - symbols):
        errs.append("extract.py replaces %s, which oracle/lower/replacements.json does not list (FID-1)" % s)
    for s in sorted(symbols - found):
        errs.append("replacements.json lists %s, which extract.py does not replace" % s)
    return errs


def main(argv):
    with open(argv[0]) as f:
        source = f.read()
    with open(argv[1]) as f:
        listed = json.load(f)
    errs = problems(source, listed)
    for e in errs:
        print(e, file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
