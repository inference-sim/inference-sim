#!/usr/bin/env python3
"""Keep the part-wide host-overhead entries' scope.hardware equal to the catalog's part list (SUB-4, #242).

    python scripts/emit_host_overheads_scope.py            # rewrite the scope lines
    python scripts/emit_host_overheads_scope.py --check    # exit 1 if a line would change

The seven part-wide entries of cost-model-host-overheads.yaml (host_admission_per_request,
host_admission_per_token, host_output_token, host_completion, host_launch_eager_per_layer,
host_replay_graph_per_step, host_launch_per_kernel) are CPU work with no per-part value, so each lists every
part the catalog holds; adding a part to the catalog is then a rerun of this script, not a hand edit of seven
lines (#241 had to make one). Their values and prose stay in the file: this script owns the scope only, and
touches no entry that is scoped to a model. An entry that another entry of the same name scopes to one chip
(host_launch_per_kernel on b200 and b300, which carry their own fitted values) lists the catalog's parts
without those chips, so no part is scoped twice (VAL-4).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET = HERE.parent / "coefficients" / "cost-model-host-overheads.yaml"
CATALOG = HERE.parents[1] / "catalog" / "hardware"
PART_WIDE = ("host_admission_per_request", "host_admission_per_token", "host_output_token", "host_completion",
             "host_launch_eager_per_layer", "host_replay_graph_per_step", "host_launch_per_kernel")
ENTRY = re.compile(r"^  - (\w+):\n(?:(?!  - ).*\n)*", re.M)
SCOPE = re.compile(r"^(      scope: \{hardware: \[)([^\]]*)(\])(\})$", re.M)


def parts(catalog: Path = CATALOG) -> list[str]:
    return sorted(p.stem for p in catalog.glob("*.yaml"))


def render(text: str, catalog_parts: list[str]) -> str:
    blocks = [(m.group(1), m.start(), m.end(), m.group(0)) for m in ENTRY.finditer(text)]
    own: dict[str, set[str]] = {}
    for name, _, _, block in blocks:
        s = SCOPE.search(block)
        if name in PART_WIDE and s and "model:" not in block[s.start():s.end() + 1]:
            chips = [c.strip() for c in s.group(2).split(",")]
            if len(chips) == 1:
                own.setdefault(name, set()).add(chips[0])
    out, last = [], 0
    for name, start, end, block in blocks:
        out.append(text[last:start])
        s = SCOPE.search(block)
        wide = name in PART_WIDE and s and "model:" not in block[s.start():s.end() + 1]
        if wide and len(s.group(2).split(",")) > 1:
            want = [p for p in catalog_parts if p not in own.get(name, set())]
            block = block[:s.start()] + s.group(1) + ", ".join(want) + s.group(3) + s.group(4) + block[s.end():]
        out.append(block)
        last = end
    out.append(text[last:])
    return "".join(out)


def main(argv: list[str]) -> int:
    text = SET.read_text()
    new = render(text, parts())
    if new == text:
        return 0
    if "--check" in argv:
        print(f"{SET}: a part-wide host entry's scope differs from the catalog's part list; "
              f"run scripts/emit_host_overheads_scope.py", file=sys.stderr)
        return 1
    SET.write_text(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
