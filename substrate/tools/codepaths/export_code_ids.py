# Original to blis-substrate.
"""Export the CodeIDs engines state, as the engine-neutral file the registry's keys are derived from.

    python3 -I export_code_ids.py --out registry/inputs/engine_code_ids.json \
        NAME=PATH/code_paths.json [NAME=PATH/code_paths.json ...]
    python3 -I export_code_ids.py --out ... --check NAME=PATH ...

The substrate reads no file of an engine module (X7), so the CodeIDs an engine-path coefficient
is keyed by reach the registry's writers (registry/scripts/code_keys.py) through this file, not
through the engines' tables. The CodeIDs are taken from each table as it stands; a table that
does not load, or that omits a path, is refused. Each engine's own Go test holds its entry here to
the table it embeds, so a stale file fails there.

--check exits 1, writing nothing, when --out is not what the tables give.
"""

import argparse
import json
import sys

PATHS = ("host", "graphs")


def export(tables):
    """tables: {name: parsed code_paths.json}. Returns the document."""
    engines = {}
    for name, t in sorted(tables.items()):
        if t.get("kind") != "CodePathTable":
            raise SystemExit(f"{name}: not a CodePathTable")
        by_path = {p["path"]: p["code_id"] for p in t["paths"]}
        if sorted(by_path) != sorted(PATHS):
            raise SystemExit(f"{name}: paths {sorted(by_path)}, want {sorted(PATHS)}")
        e = t["engine"]
        engines[name] = {"version": e["version"], "commit": e["commit"], **{p: by_path[p] for p in PATHS}}
    return {"kind": "EngineCodeIDs", "engines": engines}


def render(doc):
    return json.dumps(doc, indent=2) + "\n"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("tables", nargs="+", metavar="NAME=PATH")
    a = ap.parse_args(argv)
    tables = {}
    for spec in a.tables:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"{spec!r}: want NAME=PATH")
        with open(path, encoding="utf-8") as f:
            tables[name] = json.load(f)
    want = render(export(tables))
    try:
        have = open(a.out, encoding="utf-8").read()
    except FileNotFoundError:
        have = None
    if have == want:
        return 0
    if a.check:
        print(f"{a.out} is not what the code-path tables give; run export_code_ids.py", file=sys.stderr)
        return 1
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(want)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
