#!/usr/bin/env python3
"""Maintain the full-attention decode pair in place, from its own fitter.

WHY THIS SCRIPT EXISTS. `attention_decode_floor` and `attention_decode_rate` had no
writer: `fit_attention.py` prints a table and nothing transcribed it back, so the
committed values could only be updated by hand. That is the same gap that let the
prefill family drift (see relane_attention_prefill.py) and that left a superseded KDA
recipe in the reproduction guide. A fitted value with no writer and no re-derivation
test cannot be kept honest.

WHAT IT FITS. latency = floor + kv_bytes / rate, over full-attention generation rows
only. Windowed rows read a number of bytes bounded by the window rather than by the
context, so fitting the two kinds together fits one curve to two byte counts --
`attention_decode_*_swa` is the separate family, maintained by
relane_attention_swa.py.

THE LANE. vLLM, because BLIS prices vLLM serving. On matched geometries vLLM's decode
attention is materially slower than TRT-LLM's, so the lanes are not interchangeable;
each entry's citation names the collection it was fitted on.

THE VALUES ARE NOT HARDCODED. Each is read from `fit_attention.py` run against the
collection given, so re-running re-derives them. `--check` asserts the committed file
still matches.

Usage:
    python scripts/relane_attention_decode.py --check
    python scripts/relane_attention_decode.py --insert gb300 gb300:gb300
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

# One collection per part, as every other family in this registry pins. Globbing mixes
# engine versions and silently changes the row set.
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

POINTS = re.compile(r"# (\d+) full-attention decode points")
FLOOR = re.compile(r"attention_decode_floor\s+([\d.]+) us")
FRAC = re.compile(
    r"attention_decode_bw_fraction\s+([\d.]+) of ([\d.]+) TB/s = ([\d.]+) TB/s")
ERR = re.compile(r"geometric error\s+([\d.]+)x\s+\(pure KV read at [\d.]+ derate: "
                 r"([\d.]+)x\)")


def fit(sku: str, chip: str, collection: str, data: str) -> dict | None:
    """Run the committed fitter and parse its output for one part."""
    out = subprocess.run(
        [sys.executable, str(HERE / "fit_attention.py"),
         str(Path(data) / sku / "attention" / collection), "--chip", chip],
        capture_output=True, text=True,
    )
    t = out.stdout
    mp, mf, mfr, me = (POINTS.search(t), FLOOR.search(t), FRAC.search(t),
                       ERR.search(t))
    if not (mp and mf and mfr and me):
        return None
    frac, peak_tbs = float(mfr.group(1)), float(mfr.group(2))
    return {
        "sku": sku, "chip": chip, "coll": collection,
        "n": int(mp.group(1)), "floor": float(mf.group(1)),
        "frac": frac, "peak_tbs": peak_tbs,
        # rate = fraction x datasheet peak, computed rather than read back from the
        # fitter's "= X.XX TB/s" line: that figure is rounded to two decimals, which
        # turns h200's 2,592,000 B/us into 2,590,000 and makes a byte-identical
        # re-derivation look like drift. The fraction and the peak are both exact.
        "rate": round(frac * peak_tbs * 1e6, 6),
        "err": float(me.group(1)), "naive": float(me.group(2)),
    }


def entry_lines(kind: str, chip: str, f: dict, scoped: list[str] | None = None
                ) -> list[str]:
    """Render one decode entry; shared by the rewrite and insert paths."""
    scope = scoped or [chip]
    name = f"attention_decode_{kind}"
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/attention/{f['coll']}/"
            f"generation_attention_perf.parquet ({DEVICE[chip]}), "
            f"{f['n']} full-attention points")
    if kind == "floor":
        value = f["floor"]
        units = "us_per_transfer"
        rat = (f"The minimum cost of a decode attention kernel, independent of context. "
               f"Fitted jointly with attention_decode_rate over {f['n']} points; the "
               f"pair takes the geometric error from {f['naive']:.1f}x under a "
               f"pure-KV-read model to {f['err']:.3f}x. Fitted on the vLLM collection "
               f"({f['coll']}) because BLIS prices vLLM serving.")
    else:
        value = f["rate"]
        units = "bytes_per_us"
        rat = (f"{f['rate'] / 1e6:.2f} TB/s, which is {f['frac']} of this part's "
               f"{f['peak_tbs']} TB/s datasheet bandwidth. Fitted jointly with "
               f"attention_decode_floor over {f['n']} points ({f['coll']}); geometric "
               f"error {f['err']:.3f}x. A decode attention kernel does not reach "
               f"datasheet bandwidth, which is why the cost model carries a measured "
               f"fraction rather than the peak.")
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


def _spans(lines: list[str]):
    """Every full-attention decode entry as (start, end, kind, chip, tail, scope)."""
    out = []
    i = 0
    while i < len(lines):
        m = re.match(r"  - attention_decode_(floor|rate):$", lines[i])
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
        chip, scoped = None, []
        for b in block:
            sm = re.search(r"scope: \{hardware: \[([^\]]+)\]\}", b)
            if sm:
                scoped = [x.strip() for x in sm.group(1).split(",")]
                if len(scoped) == 1:
                    chip = scoped[0]
                elif set(scoped) <= {"a100-80", "a100-sxm"}:
                    chip = "a100-sxm"
        out.append((i, j, m.group(1), chip, tail, scoped))
        i = j
    return out


def has_entry(text: str, chip: str) -> bool:
    return any(chip in sc for _, _, _, _, _, sc in _spans(text.split("\n")))


def rewrite(text: str, fits: dict[str, dict]) -> tuple[str, int]:
    lines = text.split("\n")
    out: list[str] = []
    changed, prev = 0, 0
    for start, end, kind, chip, tail, scoped in _spans(lines):
        out.extend(lines[prev:start])
        prev = end
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
    f = fits.get(chip)
    if not f:
        return text, 0
    lines = text.split("\n")
    spans = _spans(lines)
    if not spans:
        return text, 0
    at = spans[-1][1]
    while at > 0 and (not lines[at - 1].strip() or lines[at - 1].startswith("#")):
        at -= 1
    block: list[str] = []
    for kind in ("floor", "rate"):
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
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--insert", metavar="CHIP")
    ap.add_argument("parts", nargs="*", metavar="sku:chip[:collection]")
    args = ap.parse_args(argv[1:])

    if args.parts:
        parts = []
        for p in args.parts:
            bits = p.split(":")
            coll = bits[2] if len(bits) > 2 else next(
                (c for s, ch, c in DEFAULT_PARTS if ch == bits[1]), "vllm/0.25.0")
            parts.append((bits[0], bits[1], coll))
    else:
        parts = DEFAULT_PARTS

    fits: dict[str, dict] = {}
    for sku, chip, coll in parts:
        f = fit(sku, chip, coll, args.data)
        if f is None:
            print(f"{chip:14} no decode fit at {coll}", file=sys.stderr)
            continue
        fits[chip] = f
        print(f"{chip:14} {coll:14} n={f['n']:6d} floor={f['floor']:5.1f}us "
              f"rate={f['rate']:,.0f} B/us ({f['frac']} of {f['peak_tbs']} TB/s) "
              f"geo-err {f['err']:.3f}x")
    if not fits:
        print("no parts fitted", file=sys.stderr)
        return 1

    before = SET_PATH.read_text(encoding="utf-8")

    if args.insert:
        chip = args.insert
        if chip not in fits:
            print(f"{chip}: not fitted above", file=sys.stderr)
            return 1
        if has_entry(before, chip):
            print(f"{chip}: decode entries already present", file=sys.stderr)
            return 1
        after, n = insert_part(before, chip, fits)
        if not n:
            return 1
        SET_PATH.write_text(after, encoding="utf-8")
        print(f"\n{SET_PATH.name}: inserted {n} {chip} entries")
        return 0

    after, changed = rewrite(before, fits)
    if args.check:
        if before != after:
            print(f"\n{SET_PATH.name} would change ({changed} rendered)",
                  file=sys.stderr)
            return 1
        print(f"\n{SET_PATH.name} already matches the fitter")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"\n{SET_PATH.name}: {changed} entries rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
