"""Print the rows of a downloaded configiq revision as JSON lines, for `ledger ingest-configiq`.

    python3 -I ledger/configiq/export.py DIR > rows.jsonl

DIR holds the revision's files as sources.yaml pins them (manifest.json and pairs/*/...parquet).
Each line is {"file": <path under DIR>, "row": <index in the file>, "columns": {...}}, files in
path order and rows in file order. A NaN or missing value is null; every other value is written
as Python's json module writes it, numbers as repr prints them, so the same file gives the same
bytes. It decides nothing: the identity, the role key and what is a measurement are the Go
ingest tool's. Reading Parquet needs pyarrow.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def cell(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    import pyarrow.parquet as pq

    root = Path(argv[0])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    out = sys.stdout
    for path in sorted(p["path"] for p in manifest["pairs"]):
        for i, rec in enumerate(pq.read_table(root / path).to_pylist()):
            line = {"file": path, "row": i, "columns": {k: cell(v) for k, v in rec.items()}}
            out.write(json.dumps(line, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
            out.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
