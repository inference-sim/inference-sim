#!/usr/bin/env python3
"""Maintain the MLA decode pair in place, from fit_attention_mla.py.

WHAT THIS COMMITS, AND WHY IT IS NOW POSSIBLE. `docs/methodology.md` recorded MLA as
unfittable, and commit 587c1be kept that as a negative result:

    MLA and sparse MLA are reported as unfittable rather than fitted from a guess. Their
    parquets carry no KV-head or head-dimension column, because an MLA kernel's byte
    count is a property of the architecture -- one latent vector per token, of width
    kv_lora_rank + qk_rope_head_dim -- rather than of a head width in the sweep.

That was accurate about the collection examined and not about the data. The vLLM MLA
MODULE tables carry `model` and `architecture` columns: every row is
`deepseek-ai/DeepSeek-V3` / `DeepseekV3ForCausalLM`, and blis-catalog states that
checkpoint's geometry (`models/deepseek-v3/graph.yaml`: kv_lora_rank 512,
qk_rope_head_dim 64). So the latent width is 576 elements, RESOLVED FROM THE CATALOG
rather than assumed, and the missing column is no longer missing. 137,874
mla_generation_module rows exist across parts on the vLLM lane.

WHY IT MATTERS. Ten catalog models declare an MLA or sparse-MLA layer -- deepseek-v2-lite,
deepseek-v3, deepseek-v4-pro, kimi-k2.5, kimi-k3, glm-5, glm-5.2, glm-5.2-fp8, glm-5.3 --
and deepseek-v4-pro and kimi-k3 appear in BOTH the FPM dataset and the InferenceX corpus.
With no entry the kernel prices an MLA layer with the full-attention GQA rate, which is
the wrong byte count by construction: GQA reads num_kv_heads * head_dim per token where
MLA reads one 576-element latent.

THE FORM, identical to the committed GQA pair:

    latency = floor + kv_bytes / rate,  kv_bytes = batch * step * 576 * dtype_width

TWO GRID DEFECTS FIXED HERE, both the same class as the collective RATE_GRID clamp.
FLOOR_GRID_US ran to 60.0us and h100, h200 and l40s all landed exactly on it -- clamped,
not converged. An MLA floor is legitimately larger than a GQA one because the per-call
setup reads a latent cache, so the GQA floors (9.5-19.5us) were no guide. Extended to
200us, the three move to 89.5 / 79.5 / interior and their linear-tail errors IMPROVE
(h100 1.329x -> 1.224x, h200 1.302x -> 1.244x). MIN_POINTS was 200 and gb300's 0.27.0
collection has 199 usable rows, so an entire part was excluded for one row.

L40S IS NOT COMMITTED. Its fit clamps at the top of the widened grid with a linear-tail
error of 1.792x against the 1.20-1.24x the other six reach, and a clamp that survives a
grid widening means the form does not describe the curve rather than that the search was
narrow. Recorded rather than silently dropped.

Usage:
    python scripts/relane_attention_mla.py --check
    python scripts/relane_attention_mla.py --insert h200
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
DEFAULT_CATALOG = os.environ.get(
    "BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog"))

PARTS = [("h200_sxm", "h200"), ("h100_sxm", "h100"), ("b200_sxm", "b200"),
         ("b300_sxm", "b300"), ("gb200", "gb200-nvl72"), ("gb300", "gb300")]

DEVICE = {
    "h200": "NVIDIA H200", "h100": "NVIDIA H100 80GB HBM3",
    "b200": "NVIDIA B200", "b300": "NVIDIA B300",
    "gb200-nvl72": "NVIDIA GB200", "gb300": "NVIDIA GB300",
}

HEAD = re.compile(r"^# (\S+) \((\S+)\), vLLM (\S+), kind=mla")
LANE = re.compile(r"^#\s+lane (\S+) / (\S+) / kv (\S+), (\d+) points")
FLOOR = re.compile(r"^attention_decode_floor_mla\s+([\d.]+) us")
RATE = re.compile(r"^attention_decode_rate_mla\s+(\d+) bytes_per_us "
                  r"\(([\d.]+) of ([\d.]+) TB/s\)")
ERR = re.compile(r"^#\s+geometric error ([\d.]+)x overall, ([\d.]+)x in the linear "
                 r"tail \((\d+) points\)")


def fit(sku: str, chip: str, data: str, catalog: str) -> dict | None:
    """Run the committed fitter and parse its report for one part."""
    out = subprocess.run(
        [sys.executable, str(HERE / "fit_attention_mla.py"), "--data", data,
         "--catalog", catalog, "--sku", sku, "--chip", chip],
        capture_output=True, text=True)
    got: dict = {"sku": sku, "chip": chip}
    for line in out.stdout.splitlines():
        for pat, keys in ((HEAD, ("_c", "_s", "coll")), (LANE, ("lane", "dtype", "kv", "n")),
                          (FLOOR, ("floor",)), (RATE, ("rate", "frac", "peak")),
                          (ERR, ("err", "tail", "tailn"))):
            m = pat.match(line.strip())
            if m:
                got.update(dict(zip(keys, m.groups())))
    need = {"coll", "lane", "floor", "rate", "tail", "tailn", "n"}
    if not need <= set(got):
        return None
    got["floor"] = float(got["floor"])
    got["rate"] = int(got["rate"])
    return got


def entry_lines(kind: str, chip: str, f: dict) -> list[str]:
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/mla/vllm/{f['coll']}/"
            f"mla_generation_module_perf.parquet ({DEVICE[chip]}), model "
            f"deepseek-ai/DeepSeek-V3, latent width 576 elements "
            f"(kv_lora_rank 512 + qk_rope_head_dim 64, from blis-catalog "
            f"models/deepseek-v3/graph.yaml), lane {f['lane']}, {f['n']} points")
    common = (
        f"An MLA kernel reads ONE latent vector per token, of width "
        f"kv_lora_rank + qk_rope_head_dim = 576 elements, so its byte count is a "
        f"property of the checkpoint rather than of a KV-head count the sweep varies -- "
        f"which is why the module table has no head columns and why this family was "
        f"recorded as unfittable before. What makes the fit possible is that the vLLM "
        f"module tables carry `model` and `architecture`, so the geometry is resolved "
        f"from the catalog rather than assumed. Fitted jointly with the other member of "
        f"this pair over {f['tailn']} points in the linear regime; geometric error "
        f"{f['tail']}x there and {f['err']}x over the whole sweep, the difference being "
        f"the floor-dominated rows at small context.")
    if kind == "floor":
        value, units = f["floor"], "us_per_transfer"
        rat = (f"The minimum cost of an MLA decode kernel on {chip}, independent of "
               f"context. Materially larger than this part's GQA decode floor because "
               f"the per-call setup reads a latent cache rather than a per-head KV "
               f"block. {common}")
    else:
        value, units = f["rate"], "bytes_per_us"
        rat = (f"{int(value) / 1e6:.2f} TB/s, which is {f['frac']} of this part's "
               f"{f['peak']} TB/s datasheet bandwidth. {common}")
    return [
        f"  - attention_decode_{kind}_mla:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{chip}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def _spans(lines: list[str]):
    pat = re.compile(r"  - attention_decode_(floor|rate)_mla:$")
    out = []
    i = 0
    while i < len(lines):
        m = pat.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
            j += 1
        block = lines[i:j]
        tail: list[str] = []
        while block and (not block[-1].strip() or block[-1].lstrip().startswith("#")):
            tail.insert(0, block.pop())
        chip = None
        for b in block:
            sm = re.search(r"scope: \{hardware: \[([^\]]+)\]\}", b)
            if sm:
                sc = [x.strip() for x in sm.group(1).split(",")]
                chip = sc[0] if len(sc) == 1 else None
        out.append((i, j, m.group(1), chip, tail))
        i = j
    return out


def rewrite(text: str, fits: dict) -> tuple[str, int]:
    lines = text.split("\n")
    out: list[str] = []
    changed, prev = 0, 0
    for start, end, kind, chip, tail in _spans(lines):
        out.extend(lines[prev:start])
        prev = end
        if chip is None or chip not in fits:
            out.extend(lines[start:end])
            continue
        out += entry_lines(kind, chip, fits[chip])
        out.extend(tail)
        changed += 1
    out.extend(lines[prev:])
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, changed


def insert(text: str, chip: str, fits: dict) -> tuple[str, int]:
    if chip not in fits:
        return text, 0
    lines = text.split("\n")
    spans = _spans(lines)
    if spans:
        at = max(s[1] for s in spans)
    else:
        # First MLA entry in the file: anchor after the last windowed entry, so the
        # per-kind families stay together.
        pat = re.compile(r"  - attention_decode_(floor|rate)_swa:$")
        at = None
        for i, ln in enumerate(lines):
            if pat.match(ln):
                j = i + 1
                while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
                    j += 1
                at = j
        if at is None:
            return text, 0
    while at > 0 and (not lines[at - 1].strip()
                      or lines[at - 1].lstrip().startswith("#")):
        at -= 1
    block: list[str] = []
    for kind in ("floor", "rate"):
        block += entry_lines(kind, chip, fits[chip])
    out = lines[:at] + block + lines[at:]
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, 2


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true")
    # The FLOOR half of this pair is no longer committed from this fitter. The module
    # tables it reads include the down-projections blis-catalog prices as separate GEMM
    # nodes, so a module-derived floor charges them twice -- end-to-end it cost 2.45
    # points of overall TPOT and 8.2 on kimi-k2.5. scripts/correct_mla_floor.py owns the
    # floor and commits it as `assumed` from the same part's measured attention-kernel
    # floor. The RATE is unaffected by that argument (0.80x-1.13x of part-wide, the right
    # order for a latent-cache read) and is still committed from here.
    #
    # So plain --check is expected to disagree about the floor, and --check-rate is the
    # drift gate that still applies: it asserts every committed rate is what this fitter
    # produces, and says nothing about the floor.
    ap.add_argument("--check-rate", action="store_true",
                    help="check only the fitted RATE entries; the floor is owned by "
                         "scripts/correct_mla_floor.py")
    ap.add_argument("--insert", metavar="CHIP")
    args = ap.parse_args(argv[1:])

    fits = {}
    for sku, chip in PARTS:
        f = fit(sku, chip, args.data, args.catalog)
        if f is None:
            print(f"{chip:14} no MLA fit", file=sys.stderr)
            continue
        fits[chip] = f
        print(f"{chip:14} vllm/{f['coll']:7} floor={f['floor']:6.1f}us "
              f"rate={f['rate']:>9,} B/us ({f['frac']} of {f['peak']} TB/s) "
              f"tail-err {f['tail']}x")
    if not fits:
        print("no parts fitted", file=sys.stderr)
        return 1

    before = SET_PATH.read_text(encoding="utf-8")
    if args.insert:
        chip = args.insert
        if chip not in fits:
            print(f"{chip}: not fitted above", file=sys.stderr)
            return 1
        if any(c == chip for _, _, _, c, _ in _spans(before.split("\n"))):
            print(f"{chip}: MLA already present", file=sys.stderr)
            return 1
        after, n = insert(before, chip, fits)
        if not n:
            return 1
        SET_PATH.write_text(after, encoding="utf-8")
        print(f"\n{SET_PATH.name}: inserted {n} {chip} MLA entries")
        return 0

    if args.check_rate:
        lines = before.split("\n")
        bad, seen = [], 0
        for lo, hi, kind, chip, _ in _spans(lines):
            if kind != "rate" or chip not in fits:
                continue
            got = None
            for b in lines[lo:hi]:
                vm = re.match(r"\s+value:\s*([\d.]+)\s*$", b)
                if vm:
                    got = float(vm.group(1))
                    break
            want = float(fits[chip]["rate"])
            seen += 1
            if got is None or got != want:
                bad.append(f"{chip}: committed rate {got} != fitted {want:,.0f}")
        for b in bad:
            print(f"  MISMATCH {b}", file=sys.stderr)
        if bad:
            print(f"\n{len(bad)} MLA rate(s) differ from the fitter", file=sys.stderr)
            return 1
        if not seen:
            print("no committed MLA rate matched a fitted part", file=sys.stderr)
            return 1
        print(f"\nall {seen} committed MLA rate(s) match the fitter")
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
    print(f"\n{SET_PATH.name}: {changed} MLA entries rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
