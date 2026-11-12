#!/usr/bin/env python3
"""Write the trtllm-gen prefill lane: the prefill pair measured on FlashInfer's trtllm-gen kernel alone.

    python scripts/emit_attention_trtllm_gen.py [--data <dir>] [--catalog <dir>] [--check]

WHY A LANE. `attention_config.use_trtllm_attention: true` makes a FlashInfer attention layer's
prefill run the trtllm-gen kernel whatever the cache dtype (vllm/utils/flashinfer.py:697-789 at
vLLM 5df63a72, v0.32.0); left unset, an fp8 cache runs FlashInfer's native prefill instead.
The committed `attention_prefill_floor` / `attention_prefill_work_scale` pair of b200, b300,
gb200-nvl72 and gb300 is fitted on every full-attention row of
`attention/vllm/0.25.0/context_attention_perf.parquet`, and that table holds three kernels:

    vllm_flashinfer_trtllmprefill                  24,720 rows   the trtllm-gen prefill kernel
    vllm_flashinfer_flashinfertrtllmapidecode       2,038 rows   one-token "context" batches vLLM
                                                                 classified as decodes
    vllm_triton_attn                                  816 rows   the Triton backend, fp8 cache

(`collect_attn.py::_dense_kernel_source` in ai-dynamo/aisimulate at 900e46f9 names each row by
the kernel vLLM ran.) The lane here refits the same form on the first kernel's rows only, so a
pool that reports the trtllm-gen kernel is priced on the kernel it runs. Decode is not laned:
in v0.32.0 the FlashInfer backend decodes with trtllm-gen or XQA whenever the chip supports it
(vllm/v1/attention/backends/flashinfer.py:871-881), so decode is left where it is.

THE FORM is `fit_attention_prefill.py`'s, unchanged: latency = floor + causal_flops /
(peak * eff(tokens) * work_scale), over the registry's own bf16 GEMM ramp, both parameters
chosen by minimising the mean absolute log ratio on the same grids. This script reproduces the
committed pooled pair exactly when given every row (`--pooled`), which is how it is checked
against the fitter it replaces here.

PER CACHE DTYPE. The 24,720 rows are 12,360 with a bf16 cache and 12,360 with an fp8 cache per
part, and the two differ (b200: floor 18.5 us at bf16, 21.0 at fp8, against 20.0 pooled over both),
so the lane is fitted once per dtype and the kernel reads the engine's: an `auto` or bf16 cache the
bf16 pair, an fp8 cache the fp8 pair. A cache dtype the sweep does not hold (nvfp4, the packed MLA
records) has no lane.

THE METHOD IS `transferred`, NOT `measured`. D5/VAL-8 key a `measured` entry by the hash of
the kernel the data exercised, and no kernel hash exists for these rows: AISimulate records an
image digest, and the hashes are to come from an engine release's `kernel_impls.json`, which
does not exist yet. The ratchet on fitted entries without a hash may only fall, and a
`transferred` value is the form D5 gives to a fit whose kernel identity is not shown. The
schema allows `fitted: true` only on `measured` and `calibrated`, so the entries say
`fitted: false`; the rationale says the number is a fit. The rule
the entry states: measured on the vLLM 0.25.0 image's trtllm-gen kernel on this chip, used for
the v0.32.0 pin. Nothing is moved between chips.

INTERVALS (VAL-10). `ci95` is the 2.5 and 97.5 percentiles of 200 bootstrap refits, resampling
rows with replacement under a fixed seed, each refit searching the fitter's grids within six
steps of the point estimate, widened by half a grid step either side because the optimum is only
located to the grid: a bootstrap whose every refit lands on one grid point is not certainty. Rows
of one shape family are not independent, so the interval is narrower than the data would justify;
it is stated that way in each rationale.

FIT RANGE (VAL-10). Each entry records the tokens, requests and context the rows covered. Every
row has no cached prefix (`step` is 0 throughout), so a prefill with cached context is outside the
data whatever its token count.

`--check` renders the block and exits 1 if the file differs. The block sits between the
BEGIN and END marker lines at the end of `cost-model-attention.yaml`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_attention_prefill as pooled  # noqa: E402  (the fitter this lane refits)

SET_PATH = HERE.parent / "coefficients" / "cost-model-attention.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA", "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")
DEFAULT_CATALOG = os.environ.get("BLIS_CATALOG", str(HERE.parents[1] / "catalog"))

COLLECTION = "vllm/0.25.0"
KERNEL_SOURCE = "vllm_flashinfer_trtllmprefill"

# Data-center Blackwell parts: the chips with both a trtllm-gen prefill kernel (SM100/SM103,
# vllm/utils/flashinfer.py:669-671) and a vLLM 0.25.0 context-attention sweep.
PARTS = [
    ("b200_sxm", "b200", "NVIDIA B200"),
    ("b300_sxm", "b300", "NVIDIA B300"),
    ("gb200", "gb200-nvl72", "NVIDIA GB200"),
    ("gb300", "gb300", "NVIDIA GB300"),
]

MIN_ROWS = 1000
# The cache dtypes of the sweep, by the name the lane carries and the name the column holds.
KV_COLUMN = {"bf16": "bfloat16", "fp8": "fp8"}

BOOTSTRAPS = 200
SEED = 20261110
WINDOW = 6  # grid steps searched either side of the point estimate in a bootstrap refit

BEGIN = "  # BEGIN trtllm-gen prefill lane (written by scripts/emit_attention_trtllm_gen.py; do not edit)"
END = "  # END trtllm-gen prefill lane"

FLOOR = np.array(pooled.FLOOR_GRID)
SCALE = np.array([round(s, 2) for s in pooled.SCALE_GRID])


def load(data: Path, sku: str, only_kernel: str | None, kv: str | None = None):
    """The full-attention rows of the chip's context sweep, and the work term of each."""
    path = data / sku / "attention" / COLLECTION / "context_attention_perf.parquet"
    d = pq.read_table(path).to_pydict()
    n = len(d["latency"])
    keep = [i for i in range(n)
            if d.get("window_size", [0] * n)[i] == 0 and d["latency"][i] > 0
            and (only_kernel is None or d["kernel_source"][i] == only_kernel)
            and (kv is None or d["kv_cache_dtype"][i] == KV_COLUMN[kv])]
    if only_kernel is not None and len(keep) < MIN_ROWS:
        raise SystemExit(f"{path}: {len(keep)} full-attention rows of kernel {only_kernel}, "
                         f"fewer than the {MIN_ROWS} a two-parameter fit needs; this sweep did "
                         f"not exercise the kernel, so no lane is written")
    col = {k: np.array([d[k][i] for i in keep], dtype=float)
           for k in ("batch_size", "isl", "num_heads", "head_dim", "latency")}
    col["step"] = np.array([d.get("step", [0] * n)[i] for i in keep], dtype=float)
    return col, len(keep)


def work_terms(col, peak: float, eps: float, m_half: float):
    """Seconds of ideal work per row on the ramp, and the measured seconds."""
    tok = col["batch_size"] * col["isl"]
    flops = 4.0 * col["batch_size"] * col["num_heads"] * col["head_dim"] * col["isl"] ** 2 / 2.0
    eff = eps * tok / (tok + m_half)
    return flops / (peak * eff), col["latency"] * 1e-3


def best(w, lat, floors, scales):
    """Floor and scale minimising the mean absolute log ratio, and that error."""
    best_err, best_f, best_s = math.inf, None, None
    for f in floors:
        pred = f * 1e-6 + w[None, :] / scales[:, None]
        err = np.mean(np.abs(np.log(pred / lat[None, :])), axis=1)
        i = int(err.argmin())
        if err[i] < best_err:
            best_err, best_f, best_s = float(err[i]), float(f), float(scales[i])
    return best_f, best_s, math.exp(best_err)


def fit_part(data: Path, catalog: Path, registry: Path, sku: str, chip: str,
             only_kernel: str | None, kv: str | None = None):
    eps, m_half = pooled.envelope(registry)[chip]
    peak = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())["TFlopsPeak"] * 1e12
    col, n = load(data, sku, only_kernel, kv)
    w, lat = work_terms(col, peak, eps, m_half)
    f, s, err = best(w, lat, FLOOR, SCALE)
    naive = math.exp(float(np.mean(np.abs(np.log(np.maximum(w, 1e-12) / lat)))))
    out = {"sku": sku, "chip": chip, "n": n, "floor": f, "scale": round(s, 2), "err": err,
           "naive": naive,
           "range": {"batch": (col["batch_size"].min(), col["batch_size"].max()),
                     "isl": (col["isl"].min(), col["isl"].max()),
                     "heads": (col["num_heads"].min(), col["num_heads"].max()),
                     "tokens": ((col["batch_size"] * col["isl"]).min(),
                                (col["batch_size"] * col["isl"]).max()),
                     "context": ((col["isl"] + col["step"]).min(), (col["isl"] + col["step"]).max()),
                     "step_max": col["step"].max()}}
    if only_kernel is None:
        return out
    rng = np.random.default_rng(SEED)
    fi, si = int(np.argmin(np.abs(FLOOR - f))), int(np.argmin(np.abs(SCALE - s)))
    fl = FLOOR[max(0, fi - WINDOW): fi + WINDOW + 1]
    sc = SCALE[max(0, si - WINDOW): si + WINDOW + 1]
    floors, scales = [], []
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        bf, bs, _ = best(w[idx], lat[idx], fl, sc)
        floors.append(bf)
        scales.append(bs)
    # Widened by half a grid step either side: the optimum is only located to the grid.
    fh, sh = float(FLOOR[1] - FLOOR[0]) / 2, float(SCALE[1] - SCALE[0]) / 2
    out["floor_ci"] = [max(float(np.percentile(floors, 2.5)) - fh, 0.0),
                       float(np.percentile(floors, 97.5)) + fh]
    out["scale_ci"] = [round(max(float(np.percentile(scales, 2.5)) - sh, 0.0), 4),
                       round(float(np.percentile(scales, 97.5)) + sh, 4)]
    return out


def fmt(x: float) -> str:
    x = float(x)
    return repr(round(x, 4)) if x != int(x) else repr(x)


def entry(kind: str, kv: str, f: dict, device: str, pooled_fit: dict) -> list[str]:
    chip = f["chip"]
    name = f"attention_prefill_{kind}_trtllm_gen_kv_{kv}"
    r = f["range"]
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/attention/{COLLECTION}/"
            f"context_attention_perf.parquet ({device}), {f['n']} full-attention rows of kernel "
            f"{KERNEL_SOURCE} at a {KV_COLUMN[kv]} KV cache, of the {pooled_fit['n']} the pooled "
            f"pair is fitted on")
    rule = ("measured on the vLLM 0.25.0 image's FlashInfer trtllm-gen prefill kernel on this "
            "chip and used for the v0.32.0 engine pin; no kernel hash shows the two builds run "
            "the same kernel, and nothing is moved between chips")
    shape = (f"Fitted rows span batch {int(r['batch'][0])} to {int(r['batch'][1])}, prompt "
             f"length {int(r['isl'][0])} to {int(r['isl'][1])} and {int(r['heads'][0])} to "
             f"{int(r['heads'][1])} query heads, all with no cached prefix (the sweep's step is "
             f"{int(r['step_max'])} throughout), so a prefill with cached context is outside "
             f"the data whatever its size; the fit_range records the span.")
    ci_note = ("The interval is the 2.5 and 97.5 percentiles of 200 bootstrap refits that "
               "resample rows, read on the fitter's grid (floor step 0.5 us, scale step 0.02) "
               "and widened by half a step either side, because the optimum is only located to "
               "the grid. Rows of one shape family are correlated, so it is narrower than the "
               "data would justify.")
    lane_note = (f"The pair of the {KV_COLUMN[kv]} KV cache sweep. It is read only where the caller "
                 f"reports, as the implementation of attention.prefill, one that this entry lists in "
                 f"kernel_impls (VAL-8); it lists none yet, so it is not read.")
    if kind == "floor":
        value, units, ci = f["floor"], "us_per_transfer", f["floor_ci"]
        text = (f"The minimum cost of the trtllm-gen prefill attention kernel at a {KV_COLUMN[kv]} "
                f"KV cache, independent of prompt length, on {chip}: {fmt(value)} us against the "
                f"pooled attention_prefill_floor of {fmt(pooled_fit['floor'])} us, which also "
                f"averages in the other cache dtype, the Triton rows and the one-token "
                f"decode-classified rows. Fitted jointly with attention_prefill_work_scale_"
                f"trtllm_gen_kv_{kv}; geometric error {f['err']:.3f}x against the pooled "
                f"{pooled_fit['err']:.3f}x. {lane_note} {shape} {ci_note}")
    else:
        value, units, ci = f["scale"], "dimensionless", f["scale_ci"]
        text = (f"The fraction of the bf16 GEMM efficiency ramp the trtllm-gen prefill kernel "
                f"reaches on {chip} at a {KV_COLUMN[kv]} KV cache: {value:.2f} against the pooled "
                f"{pooled_fit['scale']:.2f}. Valid only against the ramp this registry carries "
                f"(gemm_eps_max_bf16, gemm_m_half_bf16), and refitted with it. Fitted jointly "
                f"with attention_prefill_floor_trtllm_gen_kv_{kv}. {lane_note} {shape} {ci_note}")
    rng = [("tokens", r["tokens"]), ("requests", r["batch"]), ("context_tokens", r["context"])]
    return [
        f"  - {name}:",
        f"      value: {fmt(value)}",
        f"      units: {units}",
        "      method: transferred",
        "      fitted: false",
        f"      scope: {{hardware: [{chip}]}}",
        f'      transferred_from: "hardware={chip}"',
        f'      transfer_rule: "{rule}"',
        f"      ci95: [{fmt(ci[0])}, {fmt(ci[1])}]",
        "      fit_range:",
        *[f"        - {{dim: {d}, low: {fmt(lo)}, high: {fmt(hi)}}}" for d, (lo, hi) in rng],
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {text}",
    ]


def render(data: Path, catalog: Path, registry: Path) -> str:
    lines = [BEGIN]
    for sku, chip, device in PARTS:
        pooled_fit = fit_part(data, catalog, registry, sku, chip, None)
        for kv in KV_COLUMN:
            lane = fit_part(data, catalog, registry, sku, chip, KERNEL_SOURCE, kv)
            print(f"{chip:12} {kv:4} lane n={lane['n']:6d} floor={lane['floor']:5.1f}us "
                  f"scale={lane['scale']:.2f} err {lane['err']:.3f}x "
                  f"ci floor {lane['floor_ci']} scale {lane['scale_ci']} | "
                  f"pooled n={pooled_fit['n']} floor={pooled_fit['floor']:.1f} "
                  f"scale={pooled_fit['scale']:.2f} err {pooled_fit['err']:.3f}x")
            for kind in ("floor", "work_scale"):
                lines += entry(kind, kv, lane, device, pooled_fit)
    lines.append(END)
    return "\n".join(lines)


def splice(text: str, block: str) -> str:
    """Replace the marked block, or append it after the last entry."""
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        return head.rstrip("\n") + "\n" + block + tail
    return text.rstrip("\n") + "\n" + block + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file would change; write nothing")
    ap.add_argument("--pooled", action="store_true",
                    help="refit the committed pooled pair on every row and exit 1 if it differs")
    args = ap.parse_args(argv[1:])
    data, catalog, registry = Path(args.data), Path(args.catalog), HERE.parent

    if args.pooled:
        committed = yaml.safe_load(SET_PATH.read_text())["coefficients"]
        want = {}
        for e in committed:
            (n, b), = e.items()
            if n in ("attention_prefill_floor", "attention_prefill_work_scale"):
                for chip in b["scope"]["hardware"]:
                    want[(chip, n)] = b["value"]
        bad = 0
        for sku, chip, _ in PARTS:
            f = fit_part(data, catalog, registry, sku, chip, None)
            got = {"attention_prefill_floor": f["floor"], "attention_prefill_work_scale": f["scale"]}
            for n, v in got.items():
                ok = abs(want[(chip, n)] - v) < 1e-9
                bad += not ok
                print(f"{chip:12} {n:30} committed {want[(chip, n)]} refit {v} {'ok' if ok else 'DIFFERS'}")
        return 1 if bad else 0

    block = render(data, catalog, registry)
    before = SET_PATH.read_text(encoding="utf-8")
    after = splice(before, block)
    if args.check:
        if before != after:
            print(f"{SET_PATH.name} would change", file=sys.stderr)
            return 1
        print(f"{SET_PATH.name} already matches the fitter")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"{SET_PATH.name}: trtllm-gen lane written for {len(PARTS)} parts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
