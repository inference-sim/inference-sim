#!/usr/bin/env python3
"""REJECTED. Fits an MLA decode law that would double-charge the MLA projections.

Kept for the negative result, as `probe_moe_roofline.py` is: the fit is clean, physical and
wrong for this cost model, and deleting it would leave the next person to re-derive that.

THE FINDING. The vLLM tables are `mla_generation_module_perf` -- a MODULE measurement
covering the whole MLA block, projections included. The kernel's `attention_decode_floor`
covers the attention kernel ALONE, because the catalog prices `qkv_proj` and `o_proj` as
separate `GEMM` nodes in the same layer kind (see `models/deepseek-v3/graph.yaml`).

The arithmetic settles it rather than the naming. DeepSeek-V3's projections are
24576x7168 and 7168x16384, which at fp8 is 293.6 MB and 61.2 us to read at H200's
4.8 TB/s. The module's measured floor over rows at 64 total tokens or fewer is 44.0 us
minimum and 63.8 us median. So the floor this table reports IS the projection weight read,
to within the spread of the measurement -- and the kernel already charges that through the
GEMM nodes. Fitting `attention_decode_floor_mla` from here would charge it twice.

That is also why the floor pinned at the grid ceiling: the committed grid tops out at
60 us because an attention-kernel floor is 9 to 19 us on every part fitted so far. A fit
that wants 60+ us is reporting a different quantity.

WHAT WOULD FIX IT. An attention-only MLA measurement. AISimulate ships none: the only MLA
tables are `mla_context_module_perf` and `mla_generation_module_perf`, both module-level.
Subtracting a modelled projection cost from the module figure would make the coefficient a
derived quantity contingent on the GEMM envelope it is meant to be independent of, which
is worse than leaving the kind on the part-wide pair.

CONSEQUENCE, stated so it is visible rather than absorbed. `kimi-k2.5` and `deepseek-v3`
carry `kind: mla` and are priced on the unsuffixed part-wide floor and rate, which were
fitted on full attention. The byte count is already right -- the catalog gives MLA
`n_kv: 1` and a 512+64 latent, so the kernel reads one latent vector per token -- so the
error is confined to a floor and a rate borrowed from a different kernel, not to the
geometry. On the InferenceX corpus `kimi-k2.5` scores 7.68% TPOT against AISimulate's
11.68%, which bounds how much that borrowing costs today.

Original docstring follows.

Fit the MLA decode law, whose byte count is an architecture constant.

    python scripts/fit_attention_mla.py --data <data> --sku h200_sxm --chip h200
    python scripts/fit_attention_mla.py --data <data> --all

`fit_attention_by_kind.py` declares `FITTABLE = ("gqa", "swa")` and reports MLA as not
fittable from its columns. That is accurate about those columns and not about the data: an
MLA kernel reads ONE latent vector per token, of width `kv_lora_rank + qk_rope_head_dim`,
so its byte count is a property of the checkpoint rather than of a KV-head count and head
width the sweep varies. The sweep has no such columns because there is nothing to vary.

What makes the fit possible is that the vLLM module tables carry `model` and
`architecture`. Every row in every part's table is `deepseek-ai/DeepSeek-V3` (gb300 at
vLLM 0.27.0 adds `moonshotai/Kimi-K3`), and the catalog states that checkpoint's geometry:
`kv_lora_rank: 512` and `qk_rope_head_dim: 64`. So the latent width is 576 elements,
resolved from the catalog rather than assumed, and the remaining free parameters are the
same two the committed attention fits carry -- a floor and a rate.

FORM, identical to the committed one:

    latency = floor + kv_bytes / rate,  kv_bytes = batch * step * 576 * dtype_width

LANES THAT MUST NOT BE POOLED. Two columns change the measurement at identical geometry
and are fitted separately:

  * `kernel_source`. Hopper ships FLASHMLA and FLASH_ATTN_MLA, Blackwell FLASHINFER_MLA,
    and l40s/rtx TRITON_MLA. These are different implementations, not spellings.
  * `gemm_type`. At (batch 1, heads 128, step 8) on h200, bfloat16 reads 0.1344 ms against
    fp8_block's 0.1093 ms -- a 1.23x difference from one column.

Pooling all four h200 lanes gives a 39.7% median error; the best single lane gives 25.4%.
The reported lane is the one a decode step runs in: the fastest kernel at the served
cache dtype, which is what an engine selects.

REGIME. The law is a floor plus a linear read, and the measured curve has three parts: flat
below roughly a thousand total tokens where the floor dominates, a knee, and a linear tail.
On h200 FLASH_ATTN_MLA the residual is 27.4% below 1k tokens, 31.5% through the knee and
6.94% in the linear tail. The tail is the regime a long-context decode step sits in, and
the floor absorbs the flat end, so the fit is reported with that split rather than as one
aggregate that hides it.

UNITS. The `latency` column is MILLISECONDS. Treating it as seconds yields a rate of
2,070 TB/s on a part whose peak is 4.8 -- a 430x error that looks like a number until it is
checked against the datasheet.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# The grids the committed attention fit searches, reused so a difference in output is a
# difference in the data.
#
# THE FLOOR CEILING WAS BINDING. This grid ran to 60.0us and h100, h200 and l40s all
# landed exactly there -- the search was CLAMPED, not converged, which is the same defect
# the collective RATE_GRID had at its lower bound. An MLA floor is legitimately large:
# the kernel reads one latent vector per token of width kv_lora_rank + qk_rope_head_dim,
# so the per-call setup is heavier than a GQA decode's, and the committed GQA floors
# (9.5-19.5us) are no guide to where this one sits. Extended to 200us so the optimum is
# interior; a value still landing at the ceiling means the form is wrong, not the grid.
FLOOR_GRID_US = [x / 2 for x in range(2, 401)]          # 1.0 to 200.0 us
RATE_FRACTIONS = [x / 100 for x in range(5, 101)]        # 0.05 to 1.00 of datasheet peak

# A two-parameter fit over fewer rows than this is not worth stating. 150 rather than 200
# because gb300's vLLM 0.27.0 collection has 199 usable points in the linear regime and
# excluding a part for one row is a worse error than fitting it on 199.
MIN_POINTS = 150

# Total KV tokens above which the measured curve is linear in tokens rather than
# floor-dominated. Read off the h200 sweep: the marginal cost per token is flat above this
# and the floor explains the rows below it.
LINEAR_REGIME_TOKENS = 2 ** 21


def latent_width(catalog: Path, model_id: str) -> int:
    """Return kv_lora_rank + qk_rope_head_dim for a checkpoint, from the catalog.

    Resolved rather than assumed: the whole point of this fitter is that the width is a
    stated property of the model, so taking it from anywhere else would defeat it.
    """
    short = {"deepseek-ai/DeepSeek-V3": "deepseek-v3",
             "moonshotai/Kimi-K3": "kimi-k3"}.get(model_id)
    if short is None:
        raise SystemExit(f"no catalog mapping for {model_id!r}; add it rather than guess")
    graph = yaml.safe_load((catalog / "models" / short / "graph.yaml").read_text())
    for lk in graph["layer_kinds"]:
        for n in lk.get("nodes") or []:
            if n.get("op") == "Attention" and n.get("kv_lora_rank"):
                return int(n["kv_lora_rank"]) + int(n.get("qk_rope_head_dim", 0))
    raise SystemExit(f"{short}: no Attention node states kv_lora_rank")


def log_error(points, predict) -> float:
    """Geometric mean of the per-point ratio error, as the committed fits report it."""
    total = 0.0
    for x, y in points:
        p = predict(x)
        if p <= 0 or y <= 0:
            continue
        r = p / y
        total += (r if r >= 1 else 1 / r) - 1
    return 1 + total / len(points)


def fit(points, peak_bytes_per_s):
    best = None
    for floor_us in FLOOR_GRID_US:
        floor = floor_us * 1e-6
        for frac in RATE_FRACTIONS:
            rate = peak_bytes_per_s * frac
            err = log_error(points, lambda b, f=floor, r=rate: f + b / r)
            if best is None or err < best[0]:
                best = (err, floor_us, frac)
    assert best is not None
    return best


def pick_lane(d: pd.DataFrame) -> tuple[str, str, str]:
    """Choose the lane a decode step runs in: fastest kernel at the served cache dtype.

    An engine selects its backend; a fit that averaged the available ones would describe a
    deployment nobody runs.
    """
    kv = "fp8" if "fp8" in set(d["kv_cache_dtype"]) else "bfloat16"
    sub = d[d["kv_cache_dtype"] == kv]
    med = sub.groupby(["kernel_source", "gemm_type"])["latency"].median()
    kernel, gemm = med.idxmin()
    return str(kernel), str(gemm), kv


def report(data: Path, sku: str, chip: str, catalog: Path, lane: str) -> None:
    paths = sorted((data / sku / "mla" / "vllm").glob(f"{lane}/mla_generation_module_perf.parquet")
                   if lane != "*" else
                   (data / sku / "mla" / "vllm").glob("*/mla_generation_module_perf.parquet"))
    if not paths:
        print(f"# {chip}: no vLLM mla_generation_module table under {sku}")
        return
    path = paths[-1]
    d = pd.read_parquet(path)
    models = sorted(d["model"].unique())
    if len(models) != 1:
        d = d[d["model"] == "deepseek-ai/DeepSeek-V3"]
        models = ["deepseek-ai/DeepSeek-V3"]
    width = latent_width(catalog, models[0])
    kernel, gemm, kv = pick_lane(d)
    sub = d[(d["kernel_source"] == kernel) & (d["gemm_type"] == gemm)
            & (d["kv_cache_dtype"] == kv)].copy()
    if len(sub) < MIN_POINTS:
        print(f"# {chip}: only {len(sub)} rows in lane {kernel}/{gemm}/{kv}, "
              f"need {MIN_POINTS}")
        return
    elem = 1.0 if kv == "fp8" else 2.0
    sub["tokens"] = sub["batch_size"] * sub["step"]
    sub["kvbytes"] = sub["tokens"] * width * elem
    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    peak = facts["BwPeakTBs"] * 1e12

    points = [(r.kvbytes, r.latency * 1e-3) for r in sub.itertuples() if r.latency > 0]
    err, floor_us, frac = fit(points, peak)
    rate = peak * frac

    tail = [(r.kvbytes, r.latency * 1e-3) for r in sub.itertuples()
            if r.latency > 0 and r.tokens >= LINEAR_REGIME_TOKENS]
    tail_err = (log_error(tail, lambda b: floor_us * 1e-6 + b / rate)
                if len(tail) >= 20 else None)

    print(f"# {chip} ({sku}), vLLM {path.parent.name}, kind=mla")
    print(f"#   model {models[0]}, latent width {width} elements at {elem:.0f} B")
    print(f"#   lane {kernel} / {gemm} / kv {kv}, {len(points)} points")
    print(f"attention_decode_floor_mla       {floor_us:6.1f} us")
    print(f"attention_decode_rate_mla    {rate / 1e6:10.0f} bytes_per_us "
          f"({frac:.2f} of {peak / 1e12:.2f} TB/s)")
    print(f"#   geometric error {err:.3f}x overall"
          + (f", {tail_err:.3f}x in the linear tail ({len(tail)} points)"
             if tail_err else ""))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--catalog", type=Path,
                    default=Path(os.environ.get(
                        "BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog"))))
    ap.add_argument("--sku")
    ap.add_argument("--chip")
    ap.add_argument("--lane", default="*", help="a vLLM version directory, or * for newest")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv[1:])

    # Every part with an MLA module table and a catalog chip. gb300 is included: it was
    # absent from this list while its data existed, which is why its MLA pair went
    # unfitted through two passes over this family.
    #
    # L40S IS EXCLUDED DELIBERATELY. Its fit clamps at the top of FLOOR_GRID_US with a
    # linear-tail error of 1.792x, and widening that grid made the tail error WORSE
    # (1.792x against 2.536x before, but still far above the 1.20-1.24x the other six
    # reach). A clamp that does not resolve when the grid grows means the two-parameter
    # form does not describe this part's curve, not that the search was too narrow -- an
    # L40S has no NVLink and 56 usable points, and an MLA decode there is dominated by
    # something this form does not carry. Pass it explicitly with --sku l40s --chip l40s
    # to see the figure; it is not committed.
    pairs = [("h200_sxm", "h200"), ("h100_sxm", "h100"), ("b200_sxm", "b200"),
             ("b300_sxm", "b300"), ("gb200", "gb200-nvl72"), ("gb300", "gb300")]
    if args.all:
        for sku, chip in pairs:
            report(args.data, sku, chip, args.catalog, args.lane)
            print()
    else:
        if not args.sku or not args.chip:
            raise SystemExit("--sku and --chip, or --all")
        report(args.data, args.sku, args.chip, args.catalog, args.lane)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
