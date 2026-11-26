#!/usr/bin/env python3
"""Fit the grouped-GEMM ramp and floor of the NVFP4 and BF16 routed experts on Blackwell (FlashInfer TRT-LLM).

    python scripts/fit_blackwell_moe_grouped_ramp.py            # print the fits, rewrite the file
    python scripts/fit_blackwell_moe_grouped_ramp.py --check    # exit 1 if the file would change

WHY THIS FAMILY EXISTS. The routed experts of an NVFP4 or BF16 MoE on a Blackwell part were priced on the
dense GEMM ramp of their format (gemm_*_nvfp4, gemm_*_bf16: cuBLAS and FlashInfer's CuTe-DSL linear kernels),
with no floor. Under the kernel's routed-expert composition (the routed FLOPs on the ramp SUMMED with the
distinct experts' stored bytes at BwPeak x hbm_derate) that prices a one-token MoE layer at a few
microseconds where AISimulate's tables of vLLM's fused-MoE kernel read 25-60 us: the dense ramp is 2.4x (NVFP4)
and 1.7x (BF16) off in geometric error on geometries it did not see. scripts/fit_nvfp4_grouped_ramp.py fitted a
token ramp alone, found it no better than the dense ramp, and parked it. The part that was missing is the
floor, which fit_blackwell_fp8_grouped_ramp.py already carries for block-FP8: with it the grouped fit is the
kernel's own ramp, floor and all. The kernel reads the entries below where they exist (kernel/new.go addFit),
and falls back to the dense ramp, recorded as an assumption, where they do not.

THE QUANTITY, the same composition as fit_blackwell_fp8_grouped_ramp.py, at the format's own peak and bytes:

    t = F + T*top_k*6*h*(i/tp) / (peak * eps_max * T / (T + M_half))
          + touched(T) * 3*h*(i/tp)*B / (BwPeak * hbm_derate),     touched(T) = E * (1 - ((E - top_k)/E)^T)

peak is the catalog's TFlopsNVFP4 or the dense BF16 TFlopsPeak. B is the stored bytes per weight: NVFP4 0.5 + 1/16
(one FP8 E4M3 scale per 16 values), BF16 2. Fitted at moe_ep 1 on balanced routing, since the kernel applies
the routing imbalance (moe_routing_imbalance_median) on top of the ramp.

THE DATA. AISimulate `moe/vllm/0.30.0/moe_perf.parquet` of each part: for NVFP4 the kernel_source
..._flashinfer_trtllm_trtllmnvfp4expertsmonolithic (vLLM's first choice, nvfp4.py:194-203, monolithic form
preferred), moe_dtype nvfp4; for BF16 ..._flashinfer_trtllm_trtllmbf16experts, moe_dtype bfloat16 (the 0.30.0
collection names it so). 1 to 16,384 tokens, every moe_tp the sweep carries, 13 to 15 expert geometries per dtype.
`latency` is MILLISECONDS.

EXCLUSION (physical), as fit_nvfp4_grouped_ramp.py declared it: a geometry with any row faster than reading its
touched experts' stored weights at the datasheet BwPeak with no derate is impossible as labelled and is excluded
whole (NVFP4: hidden 7168, inter 2048, top-8 of 256; BF16: 2048/5120/top-22 of 512 and 2688/1856/top-6 of 128).

VALIDATION, declared before the fit was written into the registry: k-fold by SHAPE, K = 4. The geometries are
sorted by (hidden, inter, top_k, experts) and geometry j goes to fold j mod 4, so every row of a geometry is
held out together (a random row split would train on a token count and test on its neighbour). Each fold is
fitted on the other three and scored on its own rows; the figure reported is the geometric error over all
held-out rows. Two further hold-outs never fitted: the 0.24.0 and 0.25.0 collections of the same kernel (earlier
builds) and, for NVFP4, the 0.31.0 collection (one geometry), where a part carries it.
The landing rule: a part's entries are written only if its k-fold held-out geometric error is below the
dense ramp's under the same composition on the same rows, and the fit is interior (eps_max below 1, floor
below its bound, M_half below its bound).

THE FIT. Least squares on the log ratio of predicted to measured time over (F, 1/eps, M_half/eps), which is
linear in the model's three terms, by Levenberg-Marquardt from several starts, bounded (F in [0, 300] us,
eps_max in [1e-3, 1], M_half in [1, 1e4] tokens). Values are rounded to the digits the entry states and the
errors reported are those of the rounded values.

INTERVALS (VAL-10). `ci95` is the 2.5 and 97.5 percentiles of 200 refits that resample GEOMETRIES with
replacement (rows of one geometry are not independent), each warm-started at the fit.

GRADED ASSUMED, STATED. kernel_impls.json of the engine release lists no sm_100a build for any MoE backend, so
these entries cannot be keyed to a code hash (VAL-8) and a fit to a kernel's table not keyed to it is not
`measured`: they are `assumed`, with the interval recorded, and are regraded `measured` and keyed when the engine
release lists the build. LANE. vLLM. Units: `latency` is MILLISECONDS in every AISimulate parquet.
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
import coefficient_blocks as cb  # noqa: E402

SET_PATH = HERE.parent / "coefficients" / "cost-model-primitives.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA", "/tmp/aisim/python/aisim/src/aisim_core/systems/data")
DEFAULT_CATALOG = os.environ.get("BLIS_CATALOG") or str(HERE.parents[1] / "catalog")

COLLECTION = "vllm/0.30.0"
OTHER_COLLECTIONS = ("vllm/0.24.0", "vllm/0.25.0", "vllm/0.31.0")
MAX_TOKENS = 16384
FOLDS = 4
PARTS = [("b200_sxm", "b200"), ("b300_sxm", "b300"), ("gb200", "gb200-nvl72"), ("gb300", "gb300")]

# dtype word -> the registry suffix, AISimulate's moe_dtype and kernel substring, the catalog peak,
# stored bytes per weight, the dense ramp's name and the experts kernel the rows measure.
FORMATS = {
    "nvfp4": {"suffix": "nvfp4_grouped", "moe_dtype": "nvfp4", "kernel": "flashinfer_trtllm_trtllmnvfp4expertsmonolithic",
              "peak": "TFlopsNVFP4", "bpw": 0.5 + 1 / 16, "dense": "nvfp4",
              "words": "NVFP4 (one FP8 E4M3 scale per 16 values, 0.5625 B/weight)",
              "kernel_words": "vLLM's FlashInfer TRT-LLM monolithic NVFP4 fused MoE"},
    "bf16": {"suffix": "bf16_grouped", "moe_dtype": "bfloat16", "kernel": "flashinfer_trtllm_trtllmbf16experts",
             "peak": "TFlopsPeak", "bpw": 2.0, "dense": "bf16",
             "words": "BF16 (2 B/weight)",
             "kernel_words": "vLLM's FlashInfer TRT-LLM BF16 fused MoE"},
}
NAMES = tuple(f"gemm_{p}_{f['suffix']}" for f in FORMATS.values() for p in ("eps_max", "m_half", "floor"))

FLOOR_MAX = 300e-6
A_MAX, B_MAX = 1e3, 1e7


def chip_facts(catalog: Path, chip: str, peak_key: str) -> tuple[float, float, float]:
    """(dense peak of the format in FLOP/s, datasheet HBM bandwidth in B/s, hbm_derate), from the catalog and the set."""
    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    return float(facts[peak_key]) * 1e12, float(facts["BwPeakTBs"]) * 1e12, committed("hbm_derate", chip)


def committed(name: str, chip: str) -> float:
    doc = yaml.safe_load(SET_PATH.read_text())
    hits = [body["value"] for e in doc["coefficients"] for n, body in e.items()
            if n == name and body["scope"] == {"hardware": [chip]}]
    if len(hits) != 1:
        raise SystemExit(f"{chip}: expected one part-wide {name}, found {len(hits)}")
    return float(hits[0])


def load(data: Path, sku: str, collection: str, fmt: dict) -> dict[str, np.ndarray]:
    """Balanced, moe_ep 1 rows of the format's kernel, 1..MAX_TOKENS tokens. Empty arrays where the collection lacks it."""
    path = data / sku / "moe" / collection / "moe_perf.parquet"
    if not path.is_file():
        raise SystemExit(f"{sku}: {path} is absent; the pinned collection is required")
    t = pq.read_table(path).to_pydict()
    idx = [i for i in range(len(t["latency"]))
           if fmt["kernel"] in t["kernel_source"][i] and t["moe_dtype"][i] == fmt["moe_dtype"]
           and t["distribution"][i] == "balanced" and t["moe_ep_size"][i] == 1
           and t["num_tokens"][i] <= MAX_TOKENS]
    for i in idx:
        if not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0):
            raise SystemExit(f"{sku}: row {i} has latency {t['latency'][i]}")
    cols = ("num_tokens", "hidden_size", "inter_size", "topk", "num_experts", "moe_tp_size")
    out = {c: np.asarray([t[c][i] for i in idx], dtype=float) for c in cols}
    out["s"] = np.asarray([t["latency"][i] * 1e-3 for i in idx], dtype=float)
    return out


def geometry(r: dict) -> list[tuple]:
    return list(zip(r["hidden_size"], r["inter_size"], r["topk"], r["num_experts"]))


def sub(r: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in r.items()}


def terms(r: dict, bw: float, bpw: float) -> tuple[np.ndarray, ...]:
    """Per row: tokens, the rank's routed FLOPs, the kernel's memory-half seconds, measured seconds."""
    T, h, i = r["num_tokens"], r["hidden_size"], r["inter_size"] / r["moe_tp_size"]
    k, e = r["topk"], r["num_experts"]
    F = T * k * 6 * h * i
    touched = e * (1 - ((e - k) / e) ** T)
    return T, F, touched * 3 * h * i * bpw / bw, r["s"]


def exclude_impossible(r: dict, bw_peak: float, bpw: float) -> tuple[dict, list[tuple]]:
    """Drop every geometry with a row faster than reading its touched experts at the datasheet bandwidth."""
    _, _, Hds, S = terms(r, bw_peak, bpw)
    geos = geometry(r)
    bad = {g for g, h, s in zip(geos, Hds, S) if s < h}
    keep = np.array([g not in bad for g in geos], dtype=bool)
    return sub(r, keep), sorted(bad)


def predict(p: np.ndarray, T, F, H, peak) -> np.ndarray:
    fl, a, b = p
    return fl + (a + b / T) * F / peak + H


def lm(T, F, H, S, peak, starts) -> np.ndarray:
    """Bounded Levenberg-Marquardt on the log residual, in (floor, a = 1/eps, b = M_half/eps)."""
    lo = np.array([0.0, 1.0, 0.0])
    hi = np.array([FLOOR_MAX, A_MAX, B_MAX])
    ls = np.log(S)
    best, best_cost = None, math.inf
    for p0 in starts:
        p = np.clip(np.asarray(p0, float), lo, hi)
        lam = 1e-2
        cost = float(np.sum((np.log(predict(p, T, F, H, peak)) - ls) ** 2))
        for _ in range(200):
            pr = predict(p, T, F, H, peak)
            r = np.log(pr) - ls
            J = np.stack([1.0 / pr, F / peak / pr, F / (peak * T) / pr], axis=1)
            scale = np.maximum(np.abs(J).max(axis=0), 1e-30)
            Jn = J / scale
            A = Jn.T @ Jn
            g = Jn.T @ r
            moved = False
            for _ in range(30):
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g) * (1.0 / scale)
                q = np.clip(p + step, lo, hi)
                c = float(np.sum((np.log(predict(q, T, F, H, peak)) - ls) ** 2))
                if c < cost:
                    done = cost - c < 1e-12 * max(cost, 1e-12)
                    p, cost, lam, moved = q, c, max(lam / 3, 1e-9), True
                    break
                lam *= 4
            if not moved or done:
                break
        if cost < best_cost:
            best, best_cost = p, cost
    return best


STARTS = [(f, a, b) for f in (5e-6, 30e-6) for a in (2.0, 8.0) for b in (20.0, 400.0)]


def fit(T, F, H, S, peak, starts=STARTS) -> np.ndarray:
    return lm(T, F, H, S, peak, starts)


def to_entry(p: np.ndarray) -> tuple[float, int, float]:
    """The (eps_max, M_half, floor in seconds) the entries state, rounded to their digits."""
    eps = round(1.0 / p[1], 3)
    return eps, max(1, int(round(p[2] / p[1]))), round(p[0] * 1e6, 1) * 1e-6


def from_entry(eps: float, mh: float, floor: float) -> np.ndarray:
    return np.array([floor, 1.0 / eps, mh / eps])


def errors(p: np.ndarray, T, F, H, S, peak) -> tuple[float, float]:
    """(geometric error, signed geometric ratio) of the prediction against the measurement."""
    r = np.log(predict(p, T, F, H, peak)) - np.log(S)
    return math.exp(float(np.mean(np.abs(r)))), math.exp(float(np.mean(r)))


def fold_of(geos: list[tuple]) -> np.ndarray:
    index = {g: j for j, g in enumerate(sorted(set(geos)))}
    return np.array([index[g] % FOLDS for g in geos])


def kfold(T, F, H, S, peak, geos, dense: np.ndarray) -> dict:
    """Held-out error of the grouped fit and of the dense ramp, pooled over the folds' held-out rows."""
    fold = fold_of(geos)
    pred_g = np.zeros_like(S)
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        p = from_entry(*to_entry(fit(T[tr], F[tr], H[tr], S[tr], peak)))
        pred_g[te] = predict(p, T[te], F[te], H[te], peak)
    pred_d = predict(dense, T, F, H, peak)
    out = {}
    for nm, pr in (("grouped", pred_g), ("dense", pred_d)):
        r = np.log(pr) - np.log(S)
        out[nm] = (math.exp(float(np.mean(np.abs(r)))), math.exp(float(np.mean(r))),
                   float(np.median(np.abs(np.expm1(r)))))
    return out


def bootstrap(T, F, H, S, peak, geos, p, n=200, seed=150) -> dict[str, tuple[float, float]]:
    """2.5 and 97.5 percentiles of refits on geometries resampled with replacement, warm-started at the fit."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(geos))
    rows = {g: np.array([j for j, x in enumerate(geos) if x == g]) for g in uniq}
    got = {"eps": [], "mh": [], "floor": []}
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        i = np.concatenate([rows[uniq[j]] for j in pick])
        q = fit(T[i], F[i], H[i], S[i], peak, starts=[p, (p[0] * 0.5, p[1], p[2])])
        eps, mh, fl = to_entry(q)
        got["eps"].append(eps)
        got["mh"].append(mh)
        got["floor"].append(fl * 1e6)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in got.items()}


def fit_part(data: Path, catalog: Path, sku: str, chip: str, word: str) -> dict:
    fmt = FORMATS[word]
    peak, bw_peak, derate = chip_facts(catalog, chip, fmt["peak"])
    bw = bw_peak * derate
    allrows = load(data, sku, COLLECTION, fmt)
    rows, excluded = exclude_impossible(allrows, bw_peak, fmt["bpw"])
    if len(rows["s"]) < 1000:
        raise SystemExit(f"{chip} {word}: only {len(rows['s'])} rows")
    geos = geometry(rows)
    T, F, H, S = terms(rows, bw, fmt["bpw"])
    dense = from_entry(committed(f"gemm_eps_max_{fmt['dense']}", chip), committed(f"gemm_m_half_{fmt['dense']}", chip), 0.0)
    cv = kfold(T, F, H, S, peak, geos, dense)
    p_raw = fit(T, F, H, S, peak)
    eps, mh, floor = to_entry(p_raw)
    p = from_entry(eps, mh, floor)
    train = errors(p, T, F, H, S, peak)
    ci = bootstrap(T, F, H, S, peak, geos, p_raw)
    other = {}
    for coll in OTHER_COLLECTIONS:
        try:
            r = load(data, sku, coll, fmt)
        except SystemExit:
            continue
        r, _ = exclude_impossible(r, bw_peak, fmt["bpw"])
        if len(r["s"]) < 50:
            continue
        t = terms(r, bw, fmt["bpw"])
        other[coll] = (len(t[3]), errors(p, *t, peak), errors(dense, *t, peak))
    interior = 0.0 < eps < 1.0 and floor < FLOOR_MAX and mh < B_MAX / A_MAX
    return {"word": word, "sku": sku, "chip": chip, "eps": eps, "mh": mh, "floor_us": round(floor * 1e6, 1),
            "train": train, "n": len(S), "geoms": len(set(geos)), "excluded": excluded, "ci": ci,
            "peak_tf": peak / 1e12, "cv": cv, "cv_n": len(S), "other": other,
            "ok": cv["grouped"][0] < cv["dense"][0] and interior}


def entry_lines(name: str, f: dict) -> list[str]:
    fmt = FORMATS[f["word"]]
    suffix = fmt["suffix"]
    cite = (f"NVIDIA AISimulate systems/data/{f['sku']}/moe/{COLLECTION}/moe_perf.parquet, "
            f"*{fmt['kernel']}*, moe_dtype {fmt['moe_dtype']}, balanced, moe_ep 1, 1-{MAX_TOKENS} tokens, "
            f"{f['n']} rows over {f['geoms']} expert geometries")
    excl = ("; excluded as physically impossible (a row faster than the datasheet weight read): "
            + ", ".join("hidden %d inter %d top-%d of %d" % g for g in f["excluded"]) if f["excluded"] else "")
    cv = f["cv"]
    common = (f"{fmt['kernel_words']} on SM100, which AISimulate's collector measures. Fitted jointly "
              f"(gemm_eps_max_{suffix}, gemm_m_half_{suffix}, gemm_floor_{suffix}) under the kernel's routed-expert "
              f"composition: the floor plus the rank's OWN routed FLOPs (tokens x top_k x 6 x hidden x inter / tp) at "
              f"the dense {'NVFP4' if f['word'] == 'nvfp4' else 'BF16'} peak ({f['peak_tf']:g} TFLOPS) times the ramp, "
              f"summed with the distinct experts' stored bytes, {fmt['words']}, at BwPeak x hbm_derate. "
              f"{FOLDS}-fold by geometry (all rows of a geometry held out together): held-out geometric error "
              f"{cv['grouped'][0]:.3f}x (signed {cv['grouped'][1]:.3f}) against the dense {fmt['dense']} ramp's "
              f"{cv['dense'][0]:.3f}x (signed {cv['dense'][1]:.3f}) under the same composition on the same rows "
              f"({f['cv_n']} rows{excl}). "
              + "".join(f"The {c} collection ({n} rows of the same kernel, not fitted): {e[0]:.3f}x against the dense "
                        f"ramp's {d[0]:.3f}x. " for c, (n, e, d) in f["other"].items())
              + "GRADED ASSUMED, NOT MEASURED: the engine release's kernel_impls.json lists no sm_100a build for the MoE "
                "backends, so the entry cannot name a code hash (VAL-8); it is regraded measured and keyed when the build "
                "is listed. It applies to every implementation without having been shown to match. A change to the "
                "composition -- the floor's place, sum versus max, the ramp's argument, the tensor-parallel division -- "
                "needs a refit. Reproduce with scripts/fit_blackwell_moe_grouped_ramp.py.")
    key = {"gemm_eps_max": "eps", "gemm_m_half": "mh", "gemm_floor": "floor"}[name.rsplit("_" + suffix, 1)[0]]
    if key == "eps":
        value, units = f"{f['eps']:.3f}", "dimensionless"
        rat = f"Asymptotic fraction of the part's dense {'NVFP4' if f['word'] == 'nvfp4' else 'BF16'} peak that {common}"
    elif key == "mh":
        value, units = str(f["mh"]), "tokens"
        rat = f"The step token count at which the ramp reaches half its asymptote. {common}"
    else:
        value, units = f"{f['floor_us']:g}", "us_per_transfer"
        rat = (f"The per-layer device time of the grouped experts however few tokens reach them, which the ramp's work "
               f"term alone cannot carry. {common}")
    lo, hi = f["ci"][key]
    fmtv = (lambda v: f"{v:.3f}") if key == "eps" else (lambda v: f"{v:g}")
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: assumed",
        "      fitted: false",
        f"      scope: {{hardware: [{f['chip']}]}}",
        f"      ci95: [{fmtv(min(lo, float(value)))}, {fmtv(max(hi, float(value)))}]",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def render(text: str, fits: list[dict]) -> str:
    blocks = cb.split(text)
    owned = {(f"gemm_{p}_{FORMATS[f['word']]['suffix']}", cb.scope_key(f"{{hardware: [{f['chip']}]}}"))
             for f in fits for p in ("eps_max", "m_half", "floor")}
    for blk in blocks:
        n = cb.name_of(blk)
        if n in NAMES and (n, cb.scope_key(cb.scope_of(blk) or "{}")) not in owned:
            raise SystemExit(f"{n} {cb.scope_of(blk)} is not an entry this writer maintains")
    for f in fits:
        scope = f"{{hardware: [{f['chip']}]}}"
        for p in ("eps_max", "m_half", "floor"):
            cb.upsert(blocks, entry_lines(f"gemm_{p}_{FORMATS[f['word']]['suffix']}", f),
                      after=lambda b: (cb.name_of(b) or "").startswith("gemm_")
                      and cb.scope_key(cb.scope_of(b) or "{}") == cb.scope_key(scope))
    return cb.join(blocks)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true", help="exit 1 if the file would change; write nothing")
    args = ap.parse_args(argv[1:])
    fits = []
    for word in FORMATS:
        for sku, chip in PARTS:
            f = fit_part(Path(args.data), Path(args.catalog), sku, chip, word)
            print(f"{word:5} {chip:12} eps_max={f['eps']:.3f} m_half={f['mh']:4d} floor={f['floor_us']:6.1f}us "
                  f"train {f['train'][0]:.3f}x  {FOLDS}-fold held-out {f['cv']['grouped'][0]:.3f}x "
                  f"(median |rel| {f['cv']['grouped'][2]:.3f}) vs dense {f['cv']['dense'][0]:.3f}x "
                  f"(median |rel| {f['cv']['dense'][2]:.3f}), n={f['cv_n']} geoms={f['geoms']}  "
                  + " ".join(f"{c}: {e[0]:.3f}x (dense {d[0]:.3f}x, n={n})" for c, (n, e, d) in f["other"].items())
                  + ("" if f["ok"] else "  REFUSED: the landing rule is not met"))
            if not f["ok"]:
                return 1
            fits.append(f)
    old = SET_PATH.read_text()
    new = render(old, fits)
    if args.check:
        if new != old:
            print(f"{SET_PATH.name} is out of date; run without --check")
            return 1
        return 0
    SET_PATH.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
