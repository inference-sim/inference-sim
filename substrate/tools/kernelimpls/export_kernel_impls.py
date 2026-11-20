# Original to blis-substrate.
"""Export each engine's kernel_impls.json, as the engine-neutral copy the registry's checks read.

    python3 -I export_kernel_impls.py --out-dir registry/inputs NAME=PATH/kernel_impls.json [...]
    python3 -I export_kernel_impls.py --out-dir ... --check NAME=PATH ...

The substrate reads no file of an engine module (X7), so the kernel-implementation tables that
VAL-8 reads (the kernel hash an entry is valid for appears in a table some supported release pins)
reach it as registry/inputs/kernel_impls.NAME.json, not through the engines' directories. A copy
is the table as the engine states it; a file that is not a KernelImplTable of that engine is
refused. Each engine's own Go test holds its copy to the table it embeds, so a stale copy fails there.

--check exits 1, writing nothing, when a copy in --out-dir is not what the tables give.
"""

import argparse
import json
import os
import sys

REQUIRED = ("engine", "primitives", "archs", "gaps", "libraries", "implementations")


def export(tables):
    """tables: {name: parsed kernel_impls.json}. Returns {file name: rendered copy}."""
    out = {}
    for name, t in sorted(tables.items()):
        if t.get("kind") != "KernelImplTable":
            raise SystemExit(f"{name}: not a KernelImplTable")
        missing = [k for k in REQUIRED if k not in t]
        if missing:
            raise SystemExit(f"{name}: lacks {missing}")
        if t["engine"].get("kind") != name:
            raise SystemExit(f"{name}: the table is for engine {t['engine'].get('kind')!r}")
        out[f"kernel_impls.{name}.json"] = json.dumps(t, indent=2) + "\n"
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
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
    stale = []
    for fname, want in export(tables).items():
        path = os.path.join(a.out_dir, fname)
        try:
            have = open(path, encoding="utf-8").read()
        except FileNotFoundError:
            have = None
        if have == want:
            continue
        if a.check:
            stale.append(path)
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(want)
    if stale:
        print(f"{', '.join(stale)}: not what the kernel-implementation tables give; run export_kernel_impls.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
