#!/usr/bin/env python3
"""Replace the module-level MLA decode floor with this part's attention-kernel floor.

WHY THIS SCRIPT EXISTS -- THE DEFECT IT CORRECTS. `attention_decode_floor_mla` was
committed as `method: measured, fitted: true` from AISimulate's
`mla_generation_module_perf.parquet`. That table is a MODULE measurement: it covers the
whole MLA block, down-projections included. The coefficient it feeds covers the attention
KERNEL alone, because blis-catalog prices `qkv_proj` and `o_proj` as separate `GEMM` nodes
in the same layer (see `models/deepseek-v3/graph.yaml`). So the fitted floor charges the
projection weight read a second time, on top of the GEMM nodes that already charge it.

`scripts/fit_attention_mla.py` records exactly this as its reason for being REJECTED, with
the arithmetic: DeepSeek-V3's projections are 24576x7168 and 7168x16384, which at fp8 is
293.6 MB and 61.2 us to read at H200's 4.80 TB/s, against a measured module floor of
44.0 us minimum and 63.8 us median over rows at 64 total tokens or fewer. The floor that
table reports IS the projection read, to within the spread of the measurement. The fit
being clean is not evidence it measures the right quantity.

The magnitudes say the same thing without the arithmetic: the fitted MLA floors are
4.7x to 6.2x each part's own attention-kernel floor (51.5-89.5 us against 9.5-14.5 us),
while the fitted MLA RATES are 0.80x to 1.13x of the part-wide rates. A kernel reading a
latent cache instead of per-head KV blocks plausibly shifts a rate by 20%; it does not
multiply a launch-and-setup floor by six.

THE END-TO-END EVIDENCE. Scored on the InferenceX corpus (573 points, vLLM lane) with
cmd/metricscore, as a 2x2 over the two terms:

    floor        rate         overall TPOT mean|e|    kimi-k2.5 TPOT
    part-wide    fitted              14.98%                6.38%     <- this script
    part-wide    part-wide           14.90%                6.91%     (no MLA entries)
    fitted       part-wide           17.32%               15.38%
    fitted       fitted              17.43%               14.57%     (as committed)

The RATE is sound and worth keeping: it is the only arm that improves kimi-k2.5 on its
own (6.91% -> 6.38%). The FLOOR is the whole regression: either arm carrying it lands near
15% on kimi-k2.5 and 17.3% overall, flipping the kernel from beating AISimulate's 15.39%
to losing to it. No chip family regresses under this correction and b200 improves slightly
(13.14% -> 13.03%).

This is the standing rule in docs/methodology.md doing its job: "no per-primitive fit ships
without an end-to-end check", because "the kernel's accuracy rests on partially cancelling
errors". The MLA pair is the sixth entry in that list -- a correction that improved a
per-primitive measurement and made the end-to-end score worse.

WHAT THIS SCRIPT WRITES. Each part's `attention_decode_floor_mla` becomes that part's own
committed `attention_decode_floor` -- the measured attention-kernel floor for the same
silicon -- with `method: assumed, fitted: false`, because it is no longer a measurement OF
an MLA kernel. The rate entries are left untouched.

WHY NOT DELETE THE FLOOR ENTRIES. The kernel gates on both terms together
(`if floor <= 0 || rate <= 0 { continue }` in new.go), so an absent floor silently discards
the good rate as well and reverts MLA to the part-wide pair. Keeping the pair with an
honest floor is what retains the rate's gain.

THE VALUES ARE NOT HARDCODED. Each floor is read from the `attention_decode_floor` entry
scoped to the same hardware in this same file, so the correction re-derives from the
committed source rather than from a number transcribed here. `--check` asserts the
committed file still satisfies that relation.

TO REPLACE THIS WITH A MEASUREMENT: an attention-only MLA sweep -- a table measuring the
MLA attention kernel without its projections. AISimulate ships none; both its MLA tables
(`mla_context_module_perf`, `mla_generation_module_perf`) are module-level. Subtracting a
modelled projection cost from the module figure would make this coefficient contingent on
the GEMM envelope it is meant to be independent of, which is worse than anchoring on the
same part's measured attention-kernel floor.

Usage:
    python scripts/correct_mla_floor.py --check
    python scripts/correct_mla_floor.py --write
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-attention.yaml"

RATIONALE = (
    "ASSUMED, not measured: this is this part's own measured attention-kernel decode "
    "floor, reused for the MLA kind. It is NOT a measurement of an MLA kernel, and the "
    "entry it replaces was. AISimulate's only MLA tables "
    "(mla_context_module_perf, mla_generation_module_perf) are MODULE measurements "
    "covering the whole MLA block including its down-projections, whereas this "
    "coefficient feeds the attention KERNEL alone -- blis-catalog prices qkv_proj and "
    "o_proj as separate GEMM nodes in the same layer, so a module-derived floor charges "
    "the projection weight read twice. The arithmetic confirms the identification rather "
    "than the naming: DeepSeek-V3's projections are 293.6 MB at fp8, 61.2 us to read at "
    "H200's 4.80 TB/s, against a measured module floor of 44.0 us minimum and 63.8 us "
    "median at 64 or fewer total tokens -- so the module floor IS the projection read. "
    "DIMENSION (derived). This part's OWN measured attention_decode_floor, which is a "
    "measurement of an attention kernel's launch-and-setup cost on this exact silicon. "
    "The MLA rate is fitted and kept: a latent-cache read plausibly shifts a RATE by 20% "
    "(the fitted MLA rates are 0.80x-1.13x of part-wide) but does not multiply a setup "
    "floor by six (the rejected module floors were 4.7x-6.2x). MAGNITUDE (the caveat). An "
    "MLA kernel's true setup may differ from a GQA kernel's; this entry asserts only that "
    "it is closer to this part's attention floor than to a module floor containing a "
    "projection read the model charges elsewhere. VALIDATED END-TO-END, which is what "
    "settles it: on the InferenceX corpus (573 points, vLLM lane) a 2x2 over the two "
    "terms gives overall TPOT mean|e| of 14.98% here against 17.43% for the module-derived "
    "floor, and kimi-k2.5 TPOT of 6.38% against 14.57% -- the module floor alone flips the "
    "kernel from beating AISimulate's 15.39% to losing to it. No chip family regresses and "
    "b200 improves (13.14% -> 13.03%). Per docs/methodology.md, no per-primitive fit ships "
    "without an end-to-end check; this is that check rejecting the measured-looking value. "
    "TO REPLACE THIS WITH A MEASUREMENT: an attention-only MLA sweep, measuring the kernel "
    "without its projections. Subtracting a modelled projection cost from the module figure "
    "would make this contingent on the GEMM envelope it must be independent of."
)


def entries(text: str) -> list[tuple[str, int, int]]:
    """Every top-level coefficient entry as (name, start_line, end_line_exclusive)."""
    lines = text.split("\n")
    out, cur, name = [], None, None
    for i, line in enumerate(lines):
        m = re.match(r"^  - ([a-z0-9_]+):\s*$", line)
        if m:
            if cur is not None:
                out.append((name, cur, i))
            name, cur = m.group(1), i
    if cur is not None:
        out.append((name, cur, len(lines)))
    return out


def scope_hw(lines: list[str], lo: int, hi: int) -> list[str]:
    for line in lines[lo:hi]:
        m = re.search(r"hardware:\s*\[([^\]]+)\]", line)
        if m:
            return [h.strip() for h in m.group(1).split(",")]
    return []


def field(lines: list[str], lo: int, hi: int, key: str) -> tuple[int, str] | None:
    for i in range(lo, hi):
        m = re.match(rf"^(\s+){key}:\s*(.*)$", lines[i])
        if m:
            return i, m.group(2).strip()
    return None


def part_wide_floors(lines: list[str], ents) -> dict[str, str]:
    """hardware name -> committed attention_decode_floor value, as written."""
    out: dict[str, str] = {}
    for name, lo, hi in ents:
        if name != "attention_decode_floor":
            continue
        v = field(lines, lo, hi, "value")
        if not v:
            continue
        for hw in scope_hw(lines, lo, hi):
            out[hw] = v[1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true",
                   help="assert each MLA floor equals its part's attention_decode_floor")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()

    text = SET_PATH.read_text()
    lines = text.split("\n")
    ents = entries(text)
    floors = part_wide_floors(lines, ents)
    if not floors:
        print("no attention_decode_floor entries found", file=sys.stderr)
        return 2

    targets = [(n, lo, hi) for n, lo, hi in ents if n == "attention_decode_floor_mla"]
    if not targets:
        print("no attention_decode_floor_mla entries; nothing to correct")
        return 0

    bad, plan = [], []
    for name, lo, hi in targets:
        hws = scope_hw(lines, lo, hi)
        vi = field(lines, lo, hi, "value")
        if not hws or not vi:
            bad.append(f"{name} at line {lo+1}: no scope or no value")
            continue
        want = floors.get(hws[0])
        if want is None:
            bad.append(f"{name} scoped to {hws}: that part has no attention_decode_floor")
            continue
        if vi[1] != want:
            bad.append(f"{name} scoped to {hws}: value {vi[1]} != "
                       f"attention_decode_floor {want}")
        plan.append((hws[0], lo, hi, vi[0], want))

    if args.check:
        for b in bad:
            print(f"  MISMATCH {b}")
        if bad:
            print(f"\n{len(bad)} MLA floor(s) do not equal their part's attention floor. "
                  f"Run with --write.")
            return 1
        print(f"all {len(targets)} MLA floor(s) equal their part's attention_decode_floor")
        return 0

    # --write: rewrite value, method, fitted, rationale; drop the module-table source.
    for hw, lo, hi, vi, want in reversed(plan):
        lines[vi] = re.sub(r"(value:\s*).*", r"\g<1>" + want, lines[vi])
        if (m := field(lines, lo, hi, "method")) is not None:
            lines[m[0]] = re.sub(r"(method:\s*).*", r"\g<1>assumed", lines[m[0]])
        if (f := field(lines, lo, hi, "fitted")) is not None:
            lines[f[0]] = re.sub(r"(fitted:\s*).*", r"\g<1>false", lines[f[0]])
        # The module parquet is no longer this value's source: it is what the value was
        # corrected AWAY from. Leaving it as role: primary would assert the entry derives
        # from a table it contradicts. The schema's SOURCE_ROLES are primary, supporting
        # and upper_bound -- there is no "rejected" role -- so the rejected table is named
        # in the rationale (which discusses it at length) rather than listed as evidence,
        # and the primary source becomes the sibling coefficient the value is read from.
        if (s := field(lines, lo, hi, "sources")) is not None:
            j = s[0] + 1
            while j < hi and re.match(r"^\s+- \{", lines[j]):
                j += 1
            lines[s[0]:j] = [
                "      sources:",
                '        - {kind: model, cite: "blis-registry '
                "coefficients/cost-model-attention.yaml attention_decode_floor scoped to "
                f"{hw} ({want} us), the measured attention-kernel decode floor for this "
                'part", role: primary}',
            ]
        r = field(lines, lo, hi, "rationale")
        if r:
            # rationale is a folded block: replace it and every continuation line.
            j = r[0] + 1
            while j < hi and (lines[j].startswith("        ") or not lines[j].strip()):
                j += 1
            lead = re.match(r"^(\s+)", lines[r[0]])
            indent = lead.group(1) if lead else "      "
            lines[r[0]:j] = [f"{indent}rationale: >", f"{indent}  {RATIONALE}"]
        print(f"  {hw:<14} floor -> {want}  (assumed, from this part's attention_decode_floor)")

    SET_PATH.write_text("\n".join(lines))
    print(f"\nwrote {len(plan)} corrected MLA floor(s) to {SET_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
