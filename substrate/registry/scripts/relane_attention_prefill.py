#!/usr/bin/env python3
"""Maintain the prefill-attention pair in place, from its own fitter.

WHY THIS SCRIPT EXISTS. `attention_prefill_floor` and `attention_prefill_work_scale`
were the only fitted family in this registry with NO writer: `fit_attention_prefill.py`
prints a table and nothing transcribed it back, so the committed values could only be
updated by hand. They drifted, and nothing could notice -- the re-derivation suite in
`validator/test_generated_from_aisimulate.py` covers GEMM and collectives only.

WHAT DRIFTED, AND WHY IT IS A DENOMINATOR BUG RATHER THAN A DATA CHANGE.
`work_scale` is a FRACTION OF THE bf16 GEMM EFFICIENCY RAMP, which the fitter reads
from this registry (`fit_attention_prefill.py:85`). Commit 5be809b relaned that ramp
onto the vLLM lane and raised every part's asymptote:

    bf16 eps_max   at 746dc18 (prefill's fit)  ->  after 5be809b
    b200                  0.787                        0.946
    b300                  0.854                        1.000
    gb200-nvl72           0.778                        0.903
    h100                  0.903                        1.000
    h200                  0.890                        0.992

The prefill pair was never refitted against the new denominator, so the committed
numerators describe a ramp the registry no longer carries. The ABSOLUTE efficiency is
unchanged, which is the check that identifies this as a denominator bug: the product
`work_scale * eps_max` is conserved to within 2.3% on every part (b200 0.2991 ->
0.3027, b300 0.3758 -> 0.3800, gb200 0.2956 -> 0.2890, h100 0.3973 -> 0.4000,
h200 0.3916 -> 0.3968).

THE REFIT IS BETTER, ON THE VALID COMPARISON. methodology.md's standing rule is that a
re-fit landing on a different number is not automatically a correction, and
relane_gemm_envelope.py records having once made the INVALID comparison -- scoring each
candidate against its own target, which measures nothing. Scored properly, both
parameter sets against identical rows under the CURRENT envelope, the refit wins on all
five parts and its bias moves toward 1.0 in every case:

    chip          committed geo-err / bias    refit geo-err / bias
    h200              1.5161x / 0.807            1.5108x / 0.848
    h100              1.5280x / 0.799            1.5221x / 0.840
    b200              1.7071x / 0.771            1.6842x / 0.840
    b300              1.6993x / 0.767            1.6877x / 0.837
    gb200-nvl72       1.6643x / 0.789            1.6489x / 0.858

Bias below 1.0 is under-prediction, so every part becomes less under-priced. That
direction matters: methodology.md section 9.2 reports the kernel under-predicting the
whole forward pass by 13% to 47%.

THE VALUES ARE NOT HARDCODED. Each is read from `fit_attention_prefill.py` run against
the collection given, so re-running re-derives them. A fit that moves moves the file; a
fit that does not leaves it byte-identical, which `--check` asserts.

Usage:
    python scripts/relane_attention_prefill.py --check
    python scripts/relane_attention_prefill.py
    python scripts/relane_attention_prefill.py --insert gb300 gb300:gb300
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-attention.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/private/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")

# Each part with a vLLM context-attention sweep, and the collection its entries cite.
# One collection per part: globbing mixes engine versions and silently changes the row
# set, so a difference between parts stays silicon rather than a software release.
DEFAULT_PARTS = [
    ("h200_sxm", "h200", "vllm/0.25.0"),
    ("h100_sxm", "h100", "vllm/0.25.0"),
    ("b200_sxm", "b200", "vllm/0.25.0"),
    ("b300_sxm", "b300", "vllm/0.25.0"),
    ("gb200", "gb200-nvl72", "vllm/0.25.0"),
    ("gb300", "gb300", "vllm/0.25.0"),
    ("l40s", "l40s", "vllm/0.24.0"),
    ("a100_sxm", "a100-sxm", "vllm/0.14.0"),
]

DEVICE = {
    "h200": "NVIDIA H200",
    "h100": "NVIDIA H100 80GB HBM3",
    "b200": "NVIDIA B200",
    "b300": "NVIDIA B300",
    "gb200-nvl72": "NVIDIA GB200",
    "gb300": "NVIDIA GB300",
    "l40s": "NVIDIA L40S",
    "a100-sxm": "NVIDIA A100-SXM4-80GB",
}

# `h200           n= 27693 floor= 18.5us work_scale=0.40 geo-err 1.511x  (FLOPs-only 25.3x)`
LINE = re.compile(
    r"^(\S+)\s+n=\s*(\d+)\s+floor=\s*([\d.]+)us\s+work_scale=([\d.]+)\s+"
    r"geo-err\s+([\d.]+)x\s+\(FLOPs-only\s+([\d.]+)x\)")


def fit(sku: str, chip: str, collection: str, data: str) -> dict | None:
    """Run the committed fitter and parse its line for one part.

    Shelled out rather than imported so the number in the file is the number the
    documented command prints. An import could diverge from the CLI path.
    """
    out = subprocess.run(
        [sys.executable, str(HERE / "fit_attention_prefill.py"),
         "--collection", collection, "--data", data,
         "--registry", str(HERE.parent), f"{sku}:{chip}"],
        capture_output=True, text=True,
    )
    for line in out.stdout.splitlines():
        m = LINE.match(line.strip())
        if m and m.group(1) == chip:
            return {
                "sku": sku, "chip": chip, "coll": collection,
                "n": int(m.group(2)), "floor": float(m.group(3)),
                "ws": float(m.group(4)), "err": float(m.group(5)),
                "naive": float(m.group(6)),
            }
    return None


def entry_lines(kind: str, chip: str, f: dict,
                scoped: list[str] | None = None) -> list[str]:
    """Render one prefill entry. Shared by the rewrite and insert paths so an inserted
    entry is byte-identical to what a later re-run produces.

    `scoped` preserves a multi-part scope (the A100 80GB pair shares one) while the fit
    is keyed on the part the sweep belongs to; it defaults to the single chip.
    """
    scope = scoped or [chip]
    name = f"attention_prefill_{kind}"
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/attention/{f['coll']}/"
            f"context_attention_perf.parquet ({DEVICE[chip]}), "
            f"{f['n']} full-attention points")
    if kind == "floor":
        value = f["floor"]
        units = "us_per_transfer"
        rat = (f"The minimum cost of a prefill attention kernel, independent of prompt "
               f"length. A context kernel sets up a causal mask and tiles over a "
               f"two-dimensional problem where a generation kernel walks one row, so this "
               f"sits above the decode floor on every part. Fitted jointly with "
               f"attention_prefill_work_scale over {f['n']} points on the vLLM lane "
               f"({f['coll']}); the pair takes the geometric error from {f['naive']:.1f}x "
               f"under a FLOPs-only model to {f['err']:.3f}x.")
    else:
        value = f["ws"]
        units = "dimensionless"
        rat = (f"The fraction of the dense-GEMM efficiency ramp a prefill attention kernel "
               f"reaches: {value:.2f}, so it achieves well under what a well-shaped matmul "
               f"does at the same token count, which is why reusing the ramp unmodified "
               f"under-predicts. This is a fraction OF THE bf16 ramp this registry carries, "
               f"so it is only valid against that ramp -- a relane of gemm_eps_max_bf16 "
               f"changes the denominator and this value must be refitted with it. Fitted "
               f"jointly with attention_prefill_floor over {f['n']} points "
               f"({f['coll']}); geometric error {f['err']:.3f}x.")
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{', '.join(scope)}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def _entry_spans(
    lines: list[str],
) -> list[tuple[int, int, str | None, list[str], list[str]]]:
    """Every prefill entry as (start, end, chip, tail-lines, scope-list)."""
    spans = []
    i = 0
    while i < len(lines):
        m = re.match(r"  - attention_prefill_(floor|work_scale):$", lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
            j += 1
        block = lines[i:j]
        tail: list[str] = []
        while block and (not block[-1].strip() or block[-1].startswith("#")):
            tail.insert(0, block.pop())
        chip = None
        scoped: list[str] = []
        for b in block:
            sm = re.search(r"scope: \{hardware: \[([^\]]+)\]\}", b)
            if sm:
                scoped = [x.strip() for x in sm.group(1).split(",")]
                if len(scoped) == 1:
                    chip = scoped[0]
                elif set(scoped) <= {"a100-80", "a100-sxm"}:
                    # The two A100 80GB parts share one scope because they are the same
                    # silicon, and the sweep exists only under a100_sxm. Keyed on that
                    # part, with the shared scope preserved, exactly as
                    # relane_gemm_envelope.py does. Without this branch the A100 pair is
                    # silently skipped and stays stale while every other part moves.
                    chip = "a100-sxm"
        spans.append((i, j, chip, tail, scoped))
        i = j
    return spans


def has_entry(text: str, chip: str) -> bool:
    """Whether a prefill entry scoped to `chip` already exists."""
    return any(chip in sc for _, _, _, _, sc in _entry_spans(text.split("\n")))


def rewrite(text: str, fits: dict[str, dict]) -> tuple[str, int]:
    """Replace the pair for every chip in `fits`; pass everything else through."""
    lines = text.split("\n")
    out: list[str] = []
    changed = 0
    prev = 0
    for start, end, chip, tail, scoped in _entry_spans(lines):
        out.extend(lines[prev:start])
        prev = end
        m = re.match(r"  - attention_prefill_(floor|work_scale):$", lines[start])
        assert m is not None  # _entry_spans only yields lines this matches
        kind = m.group(1)
        if chip is None or chip not in fits:
            out.extend(lines[start:end])
            continue
        out += entry_lines(kind, chip, fits[chip], scoped)
        out.extend(tail)
        changed += 1
    out.extend(lines[prev:])
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, changed


def insert_part(text: str, chip: str, fits: dict[str, dict]) -> tuple[str, int]:
    """Append the prefill pair for a part the set does not carry yet."""
    f = fits.get(chip)
    if not f:
        return text, 0
    lines = text.split("\n")
    spans = _entry_spans(lines)
    if not spans:
        return text, 0
    at = spans[-1][1]
    while at > 0 and (not lines[at - 1].strip() or lines[at - 1].startswith("#")):
        at -= 1
    block: list[str] = []
    for kind in ("floor", "work_scale"):
        block += entry_lines(kind, chip, f)
    out = lines[:at] + block + lines[at:]
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, 2


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file would change; write nothing")
    ap.add_argument("--insert", metavar="CHIP",
                    help="append the pair for a part the set does not carry yet")
    ap.add_argument("parts", nargs="*", metavar="sku:chip[:collection]",
                    help="default: every part with a vLLM context sweep")
    args = ap.parse_args(argv[1:])

    if args.parts:
        parts = []
        for p in args.parts:
            bits = p.split(":")
            sku, chip = bits[0], bits[1]
            coll = bits[2] if len(bits) > 2 else next(
                (c for s, ch, c in DEFAULT_PARTS if ch == chip), "vllm/0.25.0")
            parts.append((sku, chip, coll))
    else:
        parts = DEFAULT_PARTS

    fits: dict[str, dict] = {}
    for sku, chip, coll in parts:
        f = fit(sku, chip, coll, args.data)
        if f is None:
            print(f"{chip:14} no context-attention fit at {coll}", file=sys.stderr)
            continue
        fits[chip] = f
        print(f"{chip:14} {coll:14} n={f['n']:6d} floor={f['floor']:5.1f}us "
              f"work_scale={f['ws']:.2f} geo-err {f['err']:.3f}x")
    if not fits:
        print("no parts fitted; nothing to do", file=sys.stderr)
        return 1

    before = SET_PATH.read_text(encoding="utf-8")

    if args.insert:
        chip = args.insert
        if chip not in fits:
            print(f"{chip}: not fitted above; pass it as a sku:chip argument",
                  file=sys.stderr)
            return 1
        if has_entry(before, chip):
            print(f"{chip}: prefill entries already present; use the rewrite path",
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
            print(f"\n{SET_PATH.name} would change ({changed} entries rendered)",
                  file=sys.stderr)
            return 1
        print(f"\n{SET_PATH.name} already matches the fitter")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"\n{SET_PATH.name}: {changed} entries rewritten from the fitter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
