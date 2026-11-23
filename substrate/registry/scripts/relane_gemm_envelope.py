#!/usr/bin/env python3
"""Relane the GEMM efficiency envelope onto vLLM's own linear kernels.

WHY. `gemm_eps_max_*` and `gemm_m_half_*` price every dense matmul in every layer, so
they have the widest reach of any coefficient family. They were fitted on AISimulate's
TRT-LLM sweeps, and an earlier revision of this registry justified that by the lanes
"agreeing to 0.4-3.6% on shared shapes, same cuBLAS/CUTLASS kernels". The
`kernel_source` column shows that is TRUE FOR BF16 ONLY:

  * TRT-LLM reports one path for every dtype: `torch_flow`.
  * vLLM reports distinct named kernels -- `torch.nn.functional.linear` for bf16 (which
    IS the same cuBLAS GEMM), `CutlassFP8ScaledMMLinearKernel` for fp8,
    `FlashInferCuteDslNvFp4LinearKernel` for nvfp4, and for fp8_block a dispatch across
    `FlashInferFp8BlockScaledMMKernel` and `DeepGemmFp8BlockScaledMMKernel`.

For every quantized dtype these are different implementations, and the gap is largest
exactly where TRT-LLM has no counterpart: nvfp4 improves by 3.6x.

THE COMPARISON THAT DECIDED IT. An earlier pass scored each lane's fit against THAT
lane's own envelope, which is not a comparison -- a two-parameter fit always looks good on
its own data. The valid test scores BOTH candidate parameter sets against the engine being
predicted. Re-run that way the vLLM fit wins on 17 of 18 (part, dtype) pairs, by 1.11x to
3.63x, with one tie (gb200 fp8_block). `scripts/holdout_gemm_ramp.py` carries the
held-out K and large-M folds.

DISPATCH AND THE ENVELOPE. vLLM's fp8_block path dispatches on batch size --
`vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py:156` states "Small batches
(M < 32): FlashInfer's swapAB trick ... Large batches (M >= 32): DeepGEMM for peak
throughput" -- and the sweep's coverage matches (FlashInfer at M 1-17, DeepGEMM at M>=32).
The envelope estimator takes the best efficiency at each M, which selects the same kernel
vLLM selects, because vLLM's priority list is declared "in priority/performance order"
(`kernels/linear/__init__.py:435`). Verified rather than assumed: a dispatch-FILTERED fit
and the pooled-envelope fit give identical parameters on h200 fp8_block. If vLLM ever
prefers a slower kernel for a correctness reason, that equivalence breaks and the filter
becomes necessary.

A100 IS RELANED TOO. Its vLLM gemm sweep is smaller than SGLang's (21 token counts
against 74) but it measures the engine being predicted, and on the valid test -- both
candidate parameter sets scored against the vLLM envelope -- it wins 0.0673 to 0.1002.
Its MoE imbalance is NOT relaned: the vLLM MoE sweep for this part carries no `balanced`
rows at all, only the two skewed distributions, so no paired ratio exists to measure.

Usage:
    python scripts/relane_gemm_envelope.py
    python scripts/relane_gemm_envelope.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-primitives.yaml"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


fge = _load("fit_gemm_envelope")

DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
DEFAULT_CATALOG = os.environ.get(
    "BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog"))

# a100_sxm IS included: its vLLM gemm sweep (0.14.0) covers M = 1..8192 over 21 token
# counts with a highest observed efficiency of 0.958, and scored against that envelope the
# vLLM fit beats the SGLang one (rms 0.0673 against 0.1002). The SGLang sweep is larger (74
# token counts) but it measures a different engine, and sweep size does not substitute for
# measuring the engine being predicted.
PARTS = [("h200_sxm", "h200"), ("h100_sxm", "h100"), ("b200_sxm", "b200"),
         ("b300_sxm", "b300"), ("gb200", "gb200-nvl72"), ("gb300", "gb300"),
         ("l40s", "l40s"), ("a100_sxm", "a100-sxm")]
SUFFIX = {"bfloat16": "bf16", "fp8": "fp8", "fp8_block": "fp8_block", "nvfp4": "nvfp4"}
DEVICE = {"h200": "NVIDIA H200", "h100": "NVIDIA H100 80GB HBM3",
          "a100-sxm": "NVIDIA A100-SXM4-80GB",
          "b200": "NVIDIA B200", "b300": "NVIDIA B300",
          "gb200-nvl72": "NVIDIA GB200", "gb300": "NVIDIA GB300",
          "l40s": "NVIDIA L40S"}
# The vLLM kernel each dtype's fit describes, for the citation. Read off the sweep's
# own kernel_source column and cross-checked against vLLM's dispatch lists.
KERNEL = {
    "bfloat16": "torch.nn.functional.linear",
    "fp8": "CutlassFP8ScaledMMLinearKernel",
    "fp8_block": "FlashInfer (M<32) / DeepGEMM (M>=32) block-scaled dispatch",
    "nvfp4": "FlashInferCuteDslNvFp4LinearKernel",
}


def measure(data: Path, catalog: Path, sku: str, chip: str) -> dict:
    base = data / sku / "gemm" / "vllm"
    if not base.is_dir():
        return {}
    coll = sorted(os.listdir(base))[-1]
    rows = fge.read_rows(base / coll, "gemm_perf")
    if not rows:
        return {}
    peaks = fge.peaks_from_catalog(catalog, chip)
    out = {}
    for dtype, suffix in SUFFIX.items():
        peak = peaks.get(dtype)
        if not peak:
            continue
        env = fge.envelope(rows, dtype, peak)
        if len(env) < 8:
            continue
        eps, m_half, rms = fge.fit_ramp(env)
        out[suffix] = {"eps": eps, "m_half": m_half, "rms": rms, "dtype": dtype,
                       "coll": f"vllm/{coll}", "sku": sku, "ms": len(env),
                       "rows": sum(1 for r in rows if r["gemm_dtype"] == dtype)}
    return out


def entry_lines(kind: str, suffix: str, chip: str, scoped: list[str], f: dict) -> list[str]:
    """Render one coefficient entry.

    Shared by the rewrite path, which maintains the parts already in the file, and the
    insert path, which adds a part that is not there yet. One renderer rather than two
    so an inserted entry is byte-identical to the one a later re-run would produce;
    two renderers would drift and the drift would show up as a spurious --check
    failure on the next data update.
    """
    value = f["eps"] if kind == "eps_max" else f["m_half"]
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/gemm/{f['coll']}/"
            f"gemm_perf.parquet ({DEVICE[chip]}), {f['rows']:,} {f['dtype']} rows "
            f"over {f['ms']} token counts, kernel {KERNEL[f['dtype']]}")
    if kind == "eps_max":
        rat = (f"The fraction of this part's {f['dtype']} peak a large, well-shaped "
               f"matmul asymptotically reaches, on vLLM's own linear kernel rather "
               f"than the generic `torch_flow` path the TRT-LLM sweep measures. "
               f"Envelope rather than mean: a cost model predicts what a well-shaped "
               f"GEMM achieves. Least-squares residual {f['rms']:.4f} over "
               f"{f['ms']} token counts.")
    else:
        rat = (f"The token count at which the ramp reaches half its asymptote, so it "
               f"sets how fast a widening batch approaches peak. Fitted jointly with "
               f"gemm_eps_max_{suffix} on the same envelope; residual "
               f"{f['rms']:.4f}.")
    return [
        f"  - gemm_{kind}_{suffix}:",
        f"      value: {value}",
        "      units: " + ("dimensionless" if kind == "eps_max" else "tokens"),
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{', '.join(scoped)}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def has_gemm_entry(text: str, chip: str) -> bool:
    """Whether a gemm_* entry scoped to `chip` already exists.

    Walked entry by entry rather than matched with one regex over the whole file: a
    pattern spanning `gemm_...:` to a `hardware: [...]` line will happily cross
    intervening entries, so it reports a hit when some OTHER family in this same set
    carries the chip. That is not a hypothetical -- MoE imbalance lives in
    cost-model-primitives.yaml too, and inserting it first made exactly this guard
    refuse a GEMM insert that had not happened yet.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if not re.match(r"  - gemm_(?:eps_max|m_half)_[a-z0-9_]+:$", lines[i]):
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


def insert_part(text: str, chip: str, sku: str, fits: dict[str, dict]) -> tuple[str, int]:
    """Append a per-chip GEMM block for a part the file does not carry yet.

    Inserted before the trailing entry-count comment, in the same
    `# --- <chip> (<sku>) ---` form the other parts use, so the file keeps one
    structure and `rewrite` maintains the new block on every later run.
    """
    f = fits.get(chip)
    if not f:
        return text, 0
    lines = text.split("\n")
    # Insert above the trailing generated-count comment, which must stay last, and
    # restate the count: a stale total is the kind of retyped figure this project
    # treats as a defect in its own right.
    at = len(lines)
    count_at = None
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].startswith("# ") and "entries generated" in lines[idx]:
            count_at = idx
            at = idx
            break
    if count_at is None:
        # No count comment: append after the last non-empty line.
        for idx in range(len(lines) - 1, -1, -1):
            if lines[idx].strip():
                at = idx + 1
                break
    block = [f"  # --- {chip} ({sku}) ---"]
    for suffix in ("bf16", "fp8", "fp8_block", "nvfp4"):
        if suffix not in f:
            continue
        for kind in ("eps_max", "m_half"):
            block += entry_lines(kind, suffix, chip, [chip], f[suffix])
    if len(block) == 1:
        return text, 0
    added = (len(block) - 1) // 10
    out = lines[:at] + block + lines[at:]
    if count_at is not None:
        # The comment moved down by len(block) lines.
        new_at = count_at + len(block)
        m = re.match(r"# (\d+) entries generated\.", out[new_at])
        if m:
            out[new_at] = f"# {int(m.group(1)) + added} entries generated."
    # split("\n") on a trailing-newline file yields a final "" element, and joining
    # restores it. Guard anyway: losing the trailing newline makes the next rewrite
    # report 52 spurious changes, which is a real failure this hit.
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, added


def rewrite(text: str, fits: dict[str, dict]) -> tuple[str, int]:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    changed = 0
    while i < len(lines):
        m = re.match(r"  - gemm_(eps_max|m_half)_([a-z0-9_]+):$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
            j += 1
        block = lines[i:j]
        # An entry's own lines end at its last indented field. Anything after that --
        # a blank line, the file's trailing "# N entries generated." comment -- belongs
        # to the FILE, not the entry, and must survive a rewrite. Without this the
        # rewritten entry emits its 10 canonical lines and drops the rest, which
        # silently deleted the count comment once the last entry in the file became a
        # gemm_* one.
        # Trailing lines that belong to the FILE rather than to this entry: blank
        # lines, column-0 comments (the "# N entries generated." footer) and indented
        # section headers ("  # --- gb300 descriptors ---"). All must survive the
        # rewrite. An entry's own content is a `      key: value` field, so the test is
        # "not an indented field" rather than "not indented" -- the rationale body sits
        # at eight spaces and would be eaten by the looser test.
        tail: list[str] = []
        while block and (not block[-1].strip()
                         or block[-1].lstrip().startswith("#")):
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
                    # silicon; the sweep exists only under a100_sxm. Keyed on that, with
                    # the shared scope preserved below, so the entry still covers both.
                    chip = "a100-sxm"
        kind, suffix = m.group(1), m.group(2)
        f = fits.get(chip, {}).get(suffix) if chip else None
        if f is None:
            out.extend(block)
            out.extend(tail)
            i = j
            continue
        out += entry_lines(kind, suffix, chip, scoped, f)
        out.extend(tail)
        changed += 1
        i = j
    return "\n".join(out), changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--insert", metavar="CHIP",
                    help="append a per-chip block for a part the set does not carry "
                         "yet (the rewrite path only maintains parts already present)")
    args = ap.parse_args(argv[1:])

    fits: dict[str, dict] = {}
    for sku, chip in PARTS:
        f = measure(Path(args.data), Path(args.catalog), sku, chip)
        if not f:
            print(f"{chip:14} no vLLM GEMM sweep", file=sys.stderr)
            continue
        fits[chip] = f
        for suffix, v in sorted(f.items()):
            print(f"{chip:14} {suffix:10} {v['coll']:14} eps_max={v['eps']:.3f} "
                  f"m_half={v['m_half']:3} rms={v['rms']:.4f} Ms={v['ms']}")
    if not fits:
        return 1

    before = SET_PATH.read_text(encoding="utf-8")

    if args.insert:
        chip = args.insert
        sku = next((s for s, c in PARTS if c == chip), None)
        if sku is None:
            print(f"{chip}: not in PARTS; add it there first", file=sys.stderr)
            return 1
        if has_gemm_entry(before, chip):
            print(f"{chip}: GEMM entries already present; use the rewrite path",
                  file=sys.stderr)
            return 1
        after, n = insert_part(before, chip, sku, fits)
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
        print(f"\n{SET_PATH.name} already matches the vLLM-lane fit")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"\n{SET_PATH.name}: {changed} entries rewritten on the vLLM lane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
