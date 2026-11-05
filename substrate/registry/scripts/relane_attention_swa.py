#!/usr/bin/env python3
"""Relane the sliding-window decode pair onto the vLLM collection, in place.

WHY THIS SCRIPT EXISTS. The twelve `attention_decode_*_swa` entries were the last
attention coefficients fitted on TRT-LLM while every GQA and prefill entry had moved to
the vLLM lane. A model with mixed attention kinds — gpt-oss-120b alternates `swa`
(window 128) and `gqa` across 18 repeats, minimax-m3 across 57 — then has ONE predicted
step time summing vLLM-lane GQA layers and TRT-LLM-lane SWA layers. That is an engine
mixture inside a single number, and no score justifies it.

It rewrites rather than regenerates because `cost-model-attention.yaml` is hand-authored
(unlike the collectives set): only the ten entries named on the command line change, and
the file's other thirty are passed through byte-for-byte.

THE VALUES ARE NOT HARDCODED. Each is read from `fit_attention_by_kind.py` run against
the collection given, so re-running this script re-derives them. A fit that moves moves
the file; a fit that does not leaves it byte-identical.

L40S IS ON THE vLLM LANE BY DECISION, AND THE COST IS RECORDED. Commit 5be809b moved it
to `vllm/0.24.0` (its newest vLLM attention collection; there is no 0.25.0 for this part)
for lane correctness, knowing its held-out folds favour TRT-LLM: median held-out error
2.073x against TRT-LLM's 1.681x, 0 of 2 shared folds won (methodology section 7.1). That
trades measured fit quality for the engine BLIS prices, and the trade is stated rather
than reverted, because the alternative mixes two engines into one step for every
windowed model on the part.

EVERY COMMITTED PART IS MAINTAINED. Each default part carries its own collection, so
`--check` guards the l40s and gb300 entries too; an earlier default list omitted both,
which left four committed values with no writer re-deriving them.

Usage:
    python scripts/relane_attention_swa.py
    python scripts/relane_attention_swa.py --check
    python scripts/relane_attention_swa.py h200_sxm:h200:vllm/0.25.0
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

# Every part the set carries a measured SWA pair for, each with the vLLM collection it is
# fitted on. l40s has no vllm/0.25.0 attention collection; 0.24.0 is its newest.
DEFAULT_COLLECTION = "vllm/0.25.0"
DEFAULT_PARTS = [
    ("h200_sxm", "h200", DEFAULT_COLLECTION),
    ("h100_sxm", "h100", DEFAULT_COLLECTION),
    ("b200_sxm", "b200", DEFAULT_COLLECTION),
    ("b300_sxm", "b300", DEFAULT_COLLECTION),
    ("gb200", "gb200-nvl72", DEFAULT_COLLECTION),
    ("gb300", "gb300", DEFAULT_COLLECTION),
    ("l40s", "l40s", "vllm/0.24.0"),
]

DEVICE = {
    "h200": "NVIDIA H200",
    "h100": "NVIDIA H100 80GB HBM3",
    "b200": "NVIDIA B200",
    "b300": "NVIDIA B300",
    "gb200-nvl72": "NVIDIA GB200",
    "gb300": "NVIDIA GB300",
    "l40s": "NVIDIA L40S",
}

# `  swa   floor  14.0us  rate   864,000 B/us (0.18 of 4.800 TB/s)  geo-err 1.390x
#  (naive 121.9x)  n=31,890 lane=...`
SWA_LINE = re.compile(
    r"^\s*swa\s+floor\s+([0-9.]+)us\s+rate\s+([0-9,]+) B/us\s+"
    r"\(([0-9.]+) of ([0-9.]+) TB/s\)\s+geo-err ([0-9.]+)x.*?n=([0-9,]+)"
)


def fit(sku: str, chip: str, collection: str) -> dict | None:
    """Run the committed per-kind fitter and parse its sliding-window row."""
    out = subprocess.run(
        [sys.executable, str(HERE / "fit_attention_by_kind.py"),
         "--sku", sku, "--chip", chip, "--collection", collection],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"{sku}:{chip}: fitter failed\n{out.stderr}")
    for line in out.stdout.splitlines():
        m = SWA_LINE.match(line)
        if m:
            return {
                "floor": float(m.group(1)),
                # The fitter prints the rate in B/us, which is the unit the registry
                # stores (`units: bytes_per_us`). Scaling here once produced a
                # 1000x error written to all ten entries.
                "rate": float(m.group(2).replace(",", "")),
                "frac": m.group(3),
                "peak": m.group(4),
                "err": m.group(5),
                "n": int(m.group(6).replace(",", "")),
            }
    return None


def rationale(kind: str, chip: str, collection: str, f: dict) -> str:
    if kind == "floor":
        return (
            f"The minimum cost of a windowed decode kernel, independent of context. Fitted on "
            f"the vLLM lane ({collection}) — the engine BLIS prices, and the lane every GQA and "
            f"prefill entry in this set already uses, so a model with both kinds no longer sums "
            f"two engines in one step. Paired with attention_decode_rate_swa over {f['n']:,} "
            f"windowed points; geometric error {f['err']}x."
        )
    return (
        f"{f['rate'] / 1e6:.2f} TB/s, {f['frac']} of this part's {f['peak']} TB/s datasheet "
        f"bandwidth. A windowed kernel reads a bounded slice of the context rather than all of "
        f"it and sustains materially less effective bandwidth than a full-attention kernel on "
        f"the same silicon, so pricing it with the full-attention rate makes every windowed "
        f"layer too cheap. Fitted on the vLLM lane ({collection}) over {f['n']:,} points."
    )


def entry_lines(kind: str, chip: str, collection: str, f: dict) -> list[str]:
    """Render one SWA entry.

    Shared by the rewrite path, which maintains the parts already present, and the
    insert path, which adds a part that is not there yet, so an inserted entry is
    byte-identical to what a later re-run produces and --check stays green.
    """
    name = f"attention_decode_{kind}_swa"
    value = f["floor"] if kind == "floor" else f["rate"]
    cite = (
        f"NVIDIA AISimulate systems/data/{f['sku']}/attention/{collection}/"
        f"generation_attention_perf.parquet ({DEVICE[chip]}), "
        f"{f['n']:,} sliding-window decode points"
    )
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {'us_per_transfer' if kind == 'floor' else 'bytes_per_us'}",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{chip}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rationale(kind, chip, collection, f)}",
    ]


def has_swa_entry(text: str, chip: str) -> bool:
    """Whether an attention_decode_*_swa entry scoped to `chip` already exists.

    Walked entry by entry rather than matched with one regex: a pattern spanning the
    entry name to a `hardware: [...]` line crosses intervening entries and reports a
    hit when some other family in the same set carries the chip.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if not re.match(r"  - attention_decode_(?:floor|rate)_swa:$", lines[i]):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
            sm = re.search(r"scope: \{hardware: \[([^\]]+)\]\}", lines[j])
            if sm and chip in [x.strip() for x in sm.group(1).split(",")]:
                return True
            j += 1
        i = j
    return False


def insert_part(
    text: str, chip: str, collection: str, fits: dict[str, dict]
) -> tuple[str, int]:
    """Append the SWA pair for a part the set does not carry yet.

    Placed after the last existing SWA entry so the family stays contiguous and
    `rewrite` maintains the new entries on every later run.
    """
    f = fits.get(chip)
    if not f:
        return text, 0
    lines = text.split("\n")
    last = None
    for idx, line in enumerate(lines):
        if re.match(r"  - attention_decode_(?:floor|rate)_swa:$", line):
            j = idx + 1
            while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
                j += 1
            last = j
    if last is None:
        return text, 0
    # Keep file-level trailing lines (blank lines, column-0 comments) after the block.
    while last > 0 and (not lines[last - 1].strip() or lines[last - 1].startswith("#")):
        last -= 1
    block: list[str] = []
    for kind in ("floor", "rate"):
        block += entry_lines(kind, chip, collection, f)
    out = lines[:last] + block + lines[last:]
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, 2


def rewrite(text: str, fits: dict[str, dict]) -> tuple[str, int]:
    """Replace the floor/rate entries for every chip in `fits`; pass everything else through."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    changed = 0
    while i < len(lines):
        m = re.match(r"  - (attention_decode_(floor|rate)_swa):$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
            j += 1
        block = lines[i:j]
        # File-level trailing lines belong to the file, not the entry, and must survive
        # the rewrite; see the same guard in relane_gemm_envelope.py.
        tail: list[str] = []
        while block and (not block[-1].strip() or block[-1].startswith("#")):
            tail.insert(0, block.pop())
        chip = None
        for b in block:
            sm = re.search(r"scope: \{hardware: \[([^\]]+)\]\}", b)
            if sm:
                scoped = [x.strip() for x in sm.group(1).split(",")]
                # A single-part scope is what the SWA entries carry; a shared scope would
                # make "which part was fitted" ambiguous, so leave it to a human.
                chip = scoped[0] if len(scoped) == 1 else None
        if chip is None or chip not in fits:
            out.extend(block)
            out.extend(tail)
            i = j
            continue
        out += entry_lines(m.group(2), chip, fits[chip]["collection"], fits[chip])
        out.extend(tail)
        changed += 1
        i = j
    return "\n".join(out), changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help="lane/version for a sku:chip argument that names none "
                         f"(default {DEFAULT_COLLECTION})")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file would change; write nothing")
    ap.add_argument("--insert", metavar="CHIP",
                    help="append the SWA pair for a part the set does not carry yet "
                         "(the rewrite path only maintains parts already present)")
    ap.add_argument("parts", nargs="*", metavar="sku:chip[:collection]",
                    help="default: " + " ".join(f"{s}:{c}:{v}" for s, c, v in DEFAULT_PARTS))
    args = ap.parse_args(argv[1:])

    parts = []
    for p in args.parts:
        bits = p.split(":", 2)
        parts.append((bits[0], bits[1], bits[2] if len(bits) == 3 else args.collection))
    parts = parts or DEFAULT_PARTS
    fits: dict[str, dict] = {}
    missing: list[str] = []
    for sku, chip, collection in parts:
        f = fit(sku, chip, collection)
        if f is None:
            print(f"{chip:14} no sliding-window rows at {collection}", file=sys.stderr)
            if not args.parts:
                # A committed part the default run cannot re-derive is not guarded; under
                # --check that must fail rather than pass its entries through unchecked.
                missing.append(chip)
            continue
        f["sku"] = sku
        f["collection"] = collection
        # Unit guard. The rate is a fraction of this part's datasheet bandwidth, so
        # frac * peak(TB/s) * 1e6 is the same number in B/us. A parse or scaling slip
        # shows up here rather than in the registry: a 1000x error once reached all ten
        # entries because nothing cross-checked the magnitude.
        implied = float(f["frac"]) * float(f["peak"]) * 1e6
        if not 0.5 * implied <= f["rate"] <= 2.0 * implied:
            raise SystemExit(
                f"{chip}: rate {f['rate']:,.0f} B/us contradicts the fitter's own "
                f"{f['frac']} of {f['peak']} TB/s (= {implied:,.0f} B/us); "
                "refusing to write a value whose unit cannot be confirmed"
            )
        fits[chip] = f
        print(f"{chip:14} floor={f['floor']:5.1f}us  rate={f['rate']:>12,.0f} B/us  "
              f"geo-err {f['err']}x  n={f['n']:,}")
    if not fits:
        print("no parts fitted; nothing to do", file=sys.stderr)
        return 1
    if missing:
        print(f"default parts that did not fit: {', '.join(missing)}; their committed "
              f"entries would go unchecked", file=sys.stderr)
        return 1

    before = SET_PATH.read_text(encoding="utf-8")

    if args.insert:
        chip = args.insert
        if chip not in fits:
            print(f"{chip}: not fitted above; pass it as a sku:chip argument",
                  file=sys.stderr)
            return 1
        if has_swa_entry(before, chip):
            print(f"{chip}: SWA entries already present; use the rewrite path",
                  file=sys.stderr)
            return 1
        after, n = insert_part(before, chip, fits[chip]["collection"], fits)
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
        print(f"\n{SET_PATH.name} already matches the fit")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"\n{SET_PATH.name}: {changed} entries rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
