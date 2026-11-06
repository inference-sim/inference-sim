#!/usr/bin/env python3
"""Relane the MoE routing-imbalance coefficients onto the vLLM expert kernels.

WHY. `moe_routing_imbalance_median` multiplies the routed-expert FLOPs in
blis-latency-kernel (`kernel.go:403`), charging for the fact that a grouped GEMM is gated
by its slowest expert when routing is uneven. The committed values were fitted on
AISimulate's TRT-LLM MoE sweeps, whose `kernel_source` is one of three
`moe_torch_flow*` paths. vLLM does not run those: it runs Triton, FlashInfer-Cutlass or
Marlin expert kernels, seven distinct ones in the sweep, and they tolerate skew
differently. On h200 the dominant TRT-LLM kernel shows a 1.066 median penalty where every
vLLM kernel sits between 0.989 and 1.058.

THE ESTIMATOR IS THE COMMITTED ONE and is dispatch-correct by construction: it pairs a
skewed measurement against the BALANCED measurement of the same shape AND the same
`kernel_source`, so a ratio never crosses kernels. `fit_gemm_envelope.report_moe` is the
reference; this script reproduces that pairing and writes the result.

WHICH SKEW. The committed median comes from `power_law_1.01`, the milder of the two skews
the sweep carries. That choice is kept: `power_law_1.2` is a harder distribution than a
production router with auxiliary-loss balancing produces, and switching the basis would
change what the coefficient means rather than where it was measured.

ONE ENTRY PER PART. The median is the only figure the kernel reads (`kernel.go`
multiplies the routed FLOPs by it). A `moe_routing_imbalance_p90` family was written beside
it and read by nothing -- the kernel prices a mean step, not a tail -- so it was retired
(inference-sim/blis-registry#36); its value is kept in the median's rationale.

A100 IS NOT RELANED. Its vLLM MoE sweep has no shape measured under both a skewed and the
balanced distribution, so no ratio is derivable; it keeps its SGLang borrow, which is the
only lane carrying that family for the part.

Usage:
    python scripts/relane_moe_imbalance.py
    python scripts/relane_moe_imbalance.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-primitives.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")

# The parts whose vLLM MoE sweep yields paired shapes. a100_sxm is absent by measurement,
# not by choice -- see the module docstring.
PARTS = [("h200_sxm", "h200"), ("h100_sxm", "h100"), ("b200_sxm", "b200"),
         ("b300_sxm", "b300"), ("gb200", "gb200-nvl72"), ("gb300", "gb300"),
         ("l40s", "l40s")]
SKEW = "power_law_1.01"


def shape(d: dict, i: int) -> tuple:
    """The committed estimator's pairing key: geometry AND kernel, never across kernels."""
    return (d["moe_dtype"][i], d["num_tokens"][i], d["hidden_size"][i],
            d["inter_size"][i], d["topk"][i], d["num_experts"][i],
            d["moe_tp_size"][i], d["moe_ep_size"][i], d["kernel_source"][i])


def ratios(path: Path) -> list[float]:
    import pyarrow.parquet as pq

    d = pq.read_table(path).to_pydict()
    n = len(d["latency"])
    balanced = {}
    for i in range(n):
        if d["distribution"][i] == "balanced" and d["latency"][i] > 0:
            balanced[shape(d, i)] = d["latency"][i]
    out = []
    for i in range(n):
        if d["distribution"][i] != SKEW or d["latency"][i] <= 0:
            continue
        s = shape(d, i)
        if s in balanced and balanced[s] > 0:
            out.append(d["latency"][i] / balanced[s])
    return sorted(out)


def measure(data: Path, sku: str) -> dict | None:
    base = data / sku / "moe" / "vllm"
    if not base.is_dir():
        return None
    coll = sorted(os.listdir(base))[-1]
    path = base / coll / "moe_perf.parquet"
    if not path.exists():
        return None
    rs = ratios(path)
    if not rs:
        return None
    return {"coll": f"vllm/{coll}", "n": len(rs),
            "median": round(statistics.median(rs), 3),
            "p90": round(rs[int(0.9 * len(rs))], 3), "sku": sku}


def entry_lines(kind: str, chip: str, f: dict) -> list[str]:
    """Render one imbalance entry.

    Shared by the rewrite path, which maintains the parts already in the file, and the
    insert path, which adds a part that is not there yet, so an inserted entry is
    byte-identical to what a later re-run would produce.
    """
    name = f"moe_routing_imbalance_{kind}"
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/moe/{f['coll']}/"
            f"moe_perf.parquet, {SKEW} against balanced at identical shape and "
            f"kernel_source, {f['n']:,} paired shapes")
    if kind != "median":
        raise ValueError(f"no {kind} entry is written; see the module docstring")
    rat = ("Median ratio of skewed to balanced latency on vLLM's own expert "
           "kernels -- Triton, FlashInfer-Cutlass and Marlin, which is what vLLM "
           "runs rather than the `moe_torch_flow*` paths the TRT-LLM sweep "
           "measures. Paired per shape AND per kernel, so no ratio crosses "
           "implementations. This multiplies the routed-expert FLOPs in the "
           f"kernel's step time. The 90th percentile of the same ratios is "
           f"{f['p90']:.3f}, stated here rather than as a coefficient because no "
           "consumer prices a tail from it. It is ONE rank's kernel slowdown under "
           "uneven load across its own experts, measured on one GPU; it does not "
           "carry which expert-parallel rank is busiest, which is the run's resolved "
           "routing profile (routing/), and the two compose.")
    return [
        f"  - {name}:",
        f"      value: {f[kind]}",
        "      units: dimensionless",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{chip}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def insert_part(text: str, chip: str, fits: dict[str, dict]) -> tuple[str, int]:
    """Append the imbalance pair for a part the file does not carry yet.

    Placed immediately after the last existing imbalance entry, so the family stays
    contiguous and `rewrite` maintains the new entries on every later run.
    """
    f = fits.get(chip)
    if not f:
        return text, 0
    lines = text.split("\n")
    last = None
    for idx, line in enumerate(lines):
        if re.match(r"  - moe_routing_imbalance_median:$", line):
            j = idx + 1
            while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
                j += 1
            last = j
    if last is None:
        return text, 0
    block: list[str] = []
    block += entry_lines("median", chip, f)
    out = lines[:last] + block + lines[last:]
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, 1


def rewrite(text: str, fits: dict[str, dict]) -> tuple[str, int]:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    changed = 0
    while i < len(lines):
        m = re.match(r"  - (moe_routing_imbalance_(median)):$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
            j += 1
        block = lines[i:j]
        chip = None
        for b in block:
            sm = re.search(r"scope: \{hardware: \[([^\]]+)\]\}", b)
            if sm:
                scoped = [x.strip() for x in sm.group(1).split(",")]
                chip = scoped[0] if len(scoped) == 1 else None
        if chip is None or chip not in fits:
            out.extend(block)
            i = j
            continue
        out += entry_lines(m.group(2), chip, fits[chip])
        changed += 1
        i = j
    return "\n".join(out), changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file would change; write nothing")
    ap.add_argument("--insert", metavar="CHIP",
                    help="append the imbalance pair for a part the set does not carry "
                         "yet (the rewrite path only maintains parts already present)")
    args = ap.parse_args(argv[1:])

    fits: dict[str, dict] = {}
    missing: list[str] = []
    for sku, chip in PARTS:
        f = measure(Path(args.data), sku)
        if f is None:
            print(f"{chip:14} no paired vLLM MoE shapes; its committed entry is unchecked",
                  file=sys.stderr)
            missing.append(chip)
            continue
        fits[chip] = f
        print(f"{chip:14} {f['coll']:18} pairs={f['n']:6} "
              f"median={f['median']:.3f} p90={f['p90']:.3f}")
    if not fits or missing:
        print(f"nothing measured for: {', '.join(missing) or 'every part'}", file=sys.stderr)
        return 1

    before = SET_PATH.read_text(encoding="utf-8")

    if args.insert:
        chip = args.insert
        if chip not in {c for _, c in PARTS}:
            print(f"{chip}: not in PARTS; add it there first", file=sys.stderr)
            return 1
        if re.search(r"moe_routing_imbalance_median:(?:.|\n)*?hardware: \[[^\]]*\b"
                     + re.escape(chip) + r"\b", before):
            print(f"{chip}: already present; use the default rewrite path",
                  file=sys.stderr)
            return 1
        after, n = insert_part(before, chip, fits)
        if not n:
            print(f"{chip}: nothing to insert", file=sys.stderr)
            return 1
        SET_PATH.write_text(after, encoding="utf-8")
        print(f"\n{SET_PATH.name}: inserted {n} {chip} entries")
        return 0

    after, changed = rewrite(before, fits)
    if args.check:
        if before != after:
            print(f"\n{SET_PATH.name} would change ({changed} entries)", file=sys.stderr)
            return 1
        print(f"\n{SET_PATH.name} already matches the vLLM-lane measurement")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"\n{SET_PATH.name}: {changed} entries rewritten on the vLLM lane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
