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

KEYED BY THE BUILD THE DATA EXERCISED (D5, VAL-8). The kernel hashes come from the engine
release's kernel-implementation table (blis-vllm's kernel_impls.json, --kernel-impls), which
lists the pinned FlashInfer and each version runs were seen with, each citing its run record. A
part's run is matched to a table version by its image digest: the collection's
collection_meta.yaml records the image, and the table's run record for that version names the
same digest. For a matched part each coefficient has two entries:
  - `measured`, `fitted: true`, keyed (kernel_impls) to the trtllm-gen prefill build of the run's
    FlashInfer on the chip's architecture: a pool that runs that build reads it exactly. It
    states what the number includes (measures, VAL-9): the collector times the backend's forward
    pass by CUDA events over CUDA-graph replays, so device time of captured execution.
  - `transferred`, keyed to the pinned FlashInfer's build and recording the run's build in
    measured_on, when the two hashes differ: a pool on the pin reads the value as moved, at the
    transferred grade, with the rule stated. When the hashes are equal there is nothing to move.
A part whose run matches no table version (the B200 run's image is a local build that records
no FlashInfer version) keeps one unkeyed `transferred` entry per coefficient, which no lookup
reports as exact. The schema allows `fitted: true` only on `measured` and `calibrated`, so a
transferred entry says `fitted: false`; the rationale says the number is a fit. Nothing is moved
between chips.

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
import json
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
DEFAULT_KERNEL_IMPLS = os.environ.get("BLIS_KERNEL_IMPLS", "")
LANE_PRIMITIVE, LANE_SELECTION = "attention.prefill", "attention backend FLASHINFER"
LANE_LIBRARY, LANE_VARIANT = "flashinfer", "trtllm_batch_context_with_kv_cache"

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


def run_meta(data: Path, sku: str) -> dict:
    """The collection's record of the run: its image and the collector that timed it."""
    meta = yaml.safe_load((data / sku / "attention" / COLLECTION / "collection_meta.yaml").read_text())
    refs = sorted({c["collector_ref"] for c in meta["tables"]["context_attention_perf"]["collections"]})
    if len(refs) != 1:
        raise SystemExit(f"{sku}: the context sweep was collected by {refs}; one collector expected")
    return {"digest": meta["runtime"].get("image_digest", ""), "image": meta["runtime"].get("image", ""),
            "collector": refs[0]}


def arch_of(catalog: Path, chip: str) -> str:
    """The device architecture a chip's builds run on, as the table names it (10.3 is sm_103a)."""
    major, minor = str(yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())["ComputeCapability"]).split(".")
    return f"sm_{major}{minor}a"


def lane_builds(table: dict, digest: str, arch: str):
    """The trtllm-gen prefill builds for a part: the pinned FlashInfer's, and the one of the version
    whose run record names the run's image digest (None when no record does)."""
    def build(version):
        for r in table["implementations"]:
            if (r["primitive"], r["selection"], r["library"], r["library_version"], r["variant"], r["target_arch"]) == (
                    LANE_PRIMITIVE, LANE_SELECTION, LANE_LIBRARY, version, LANE_VARIANT, arch):
                return {"library": r["library"], "variant": r["variant"], "code_hash": r["code_hash"],
                        "version": version}
        raise SystemExit(f"kernel_impls.json lists no {LANE_LIBRARY} {version} {LANE_VARIANT} build for {arch}")
    libs = [l for l in table["libraries"] if l["name"] == LANE_LIBRARY]
    pinned = build(next(l["version"] for l in libs if l["pinned"]))
    seen = [l["version"] for l in libs if not l["pinned"] and digest and digest in l["pin"]]
    if len(seen) > 1:
        raise SystemExit(f"two {LANE_LIBRARY} versions cite the run image {digest}: {seen}")
    return pinned, (build(seen[0]) if seen else None)


def impl_yaml(b: dict) -> str:
    return f"{{library: {b['library']}, variant: {b['variant']}, code_hash: {b['code_hash']}}}"


def fmt(x: float) -> str:
    x = float(x)
    return repr(round(x, 4)) if x != int(x) else repr(x)


def entry(kind: str, kv: str, f: dict, device: str, pooled_fit: dict, keyed: dict | None = None) -> list[str]:
    """One entry of the lane. keyed is None for an unkeyed transfer, else {"form": "measured" or
    "transferred", "run": the run's build, "pinned": the pin's build, "meta": the run record,
    "table": the table's engine block}."""
    chip = f["chip"]
    name = f"attention_prefill_{kind}_trtllm_gen_kv_{kv}"
    r = f["range"]
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/attention/{COLLECTION}/"
            f"context_attention_perf.parquet ({device}), {f['n']} full-attention rows of kernel "
            f"{KERNEL_SOURCE} at a {KV_COLUMN[kv]} KV cache, of the {pooled_fit['n']} the pooled "
            f"pair is fitted on")
    if keyed is None:
        rule = ("measured on the vLLM 0.25.0 image's FlashInfer trtllm-gen prefill kernel on this "
                "chip and used for the engine's pin; the run records no FlashInfer version, so no "
                "kernel hash shows which build it ran, and nothing is moved between chips")
    else:
        run, pin = keyed["run"], keyed["pinned"]
        rule = (f"measured on FlashInfer {run['version']}'s trtllm-gen prefill build "
                f"({run['code_hash'][:12]}) on this chip and used for the pinned FlashInfer "
                f"{pin['version']}'s build ({pin['code_hash'][:12]}) on the same chip; the two "
                f"builds' precompiled kernel packages differ, so the value is moved, not shown, "
                f"and nothing is moved between chips")
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
    if keyed is None:
        lane_note = (f"The pair of the {KV_COLUMN[kv]} KV cache sweep. It is read only where the caller "
                     f"reports, as the implementation of attention.prefill, one that this entry lists in "
                     f"kernel_impls (VAL-8); it lists none, because the run's image records no FlashInfer "
                     f"version, so it is not read.")
    elif keyed["form"] == "measured":
        lane_note = (f"The pair of the {KV_COLUMN[kv]} KV cache sweep, keyed to the build the data "
                     f"exercised: FlashInfer {keyed['run']['version']}'s trtllm-gen prefill kernel on "
                     f"this chip, which the run's image ({keyed['meta']['image']}@{keyed['meta']['digest']}) "
                     f"installs. A pool that reports that build as the implementation of "
                     f"attention.prefill reads it exactly (VAL-8).")
    else:
        lane_note = (f"The pair of the {KV_COLUMN[kv]} KV cache sweep, moved to the pinned FlashInfer "
                     f"{keyed['pinned']['version']}'s build from the FlashInfer {keyed['run']['version']} "
                     f"build it was measured on (measured_on). A pool that reports the pinned build reads "
                     f"it at the transferred grade (VAL-8).")
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
    measured = keyed is not None and keyed["form"] == "measured"
    head = [f"  - {name}:", f"      value: {fmt(value)}", f"      units: {units}"]
    if measured:
        meta = keyed["meta"]
        harness = (f"AISimulate collector/vllm/collect_attn.py at ai-dynamo/aisimulate {meta['collector']}: "
                   f"the backend's forward pass, with its KV-cache write, timed by CUDA events over "
                   f"CUDA-graph replays")
        head += ["      method: measured", "      fitted: true", f"      scope: {{hardware: [{chip}]}}",
                 f"      kernel_impls: [{impl_yaml(keyed['run'])}]",
                 "      measures:", "        time: device", "        execution: captured",
                 f'        harness: "{harness}"']
    else:
        head += ["      method: transferred", "      fitted: false", f"      scope: {{hardware: [{chip}]}}"]
        if keyed is not None:
            head += [f"      kernel_impls: [{impl_yaml(keyed['pinned'])}]",
                     f"      measured_on: [{impl_yaml(keyed['run'])}]"]
        head += [f'      transferred_from: "hardware={chip}"', f'      transfer_rule: "{rule}"']
    sources = [f'        - {{kind: model, cite: "{cite}", role: primary}}']
    if keyed is not None:
        t = keyed["table"]
        sources.append(f'        - {{kind: model, cite: "inference-sim/blis-vllm kernel_impls.json for vLLM '
                       f'{t["version"]} ({t["commit"][:12]}): the {LANE_VARIANT} builds of FlashInfer '
                       f'{keyed["run"]["version"]} and {keyed["pinned"]["version"]} and the run record that '
                       f'names the image", role: supporting}}')
    return [
        *head,
        f"      ci95: [{fmt(ci[0])}, {fmt(ci[1])}]",
        "      fit_range:",
        *[f"        - {{dim: {d}, low: {fmt(lo)}, high: {fmt(hi)}}}" for d, (lo, hi) in rng],
        "      sources:",
        *sources,
        "      rationale: >",
        f"        {text}",
    ]


def render(data: Path, catalog: Path, registry: Path, table: dict) -> str:
    lines = [BEGIN]
    for sku, chip, device in PARTS:
        pooled_fit = fit_part(data, catalog, registry, sku, chip, None)
        # Fit first: a sweep that did not exercise the kernel is refused before anything else.
        lanes = {kv: fit_part(data, catalog, registry, sku, chip, KERNEL_SOURCE, kv) for kv in KV_COLUMN}
        meta = run_meta(data, sku)
        pinned, run = lane_builds(table, meta["digest"], arch_of(catalog, chip))
        forms = [None]
        if run is not None:
            base = {"run": run, "pinned": pinned, "meta": meta, "table": table["engine"]}
            forms = [dict(base, form="measured")]
            if run["code_hash"] != pinned["code_hash"]:
                forms.append(dict(base, form="transferred"))
        for kv, lane in lanes.items():
            print(f"{chip:12} {kv:4} lane n={lane['n']:6d} floor={lane['floor']:5.1f}us "
                  f"scale={lane['scale']:.2f} err {lane['err']:.3f}x "
                  f"ci floor {lane['floor_ci']} scale {lane['scale_ci']} | "
                  f"pooled n={pooled_fit['n']} floor={pooled_fit['floor']:.1f} "
                  f"scale={pooled_fit['scale']:.2f} err {pooled_fit['err']:.3f}x"
                  f" | {'keyed to FlashInfer ' + run['version'] if run else 'unkeyed: the run records no FlashInfer version'}")
            for kind in ("floor", "work_scale"):
                for keyed in forms:
                    lines += entry(kind, kv, lane, device, pooled_fit, keyed)
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
    ap.add_argument("--kernel-impls", default=DEFAULT_KERNEL_IMPLS,
                    help="the engine release's kernel_impls.json (default $BLIS_KERNEL_IMPLS)")
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

    if not args.kernel_impls:
        print("the engine release's kernel_impls.json is required (--kernel-impls or $BLIS_KERNEL_IMPLS): "
              "the lane is keyed by the builds it lists", file=sys.stderr)
        return 2
    table = json.loads(Path(args.kernel_impls).read_text(encoding="utf-8"))
    block = render(data, catalog, registry, table)
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
