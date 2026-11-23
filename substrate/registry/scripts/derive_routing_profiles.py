#!/usr/bin/env python3
"""Derive expert-parallel routing imbalance from a per-token routed-expert capture.

Reads a ServingStudio capture (routes.u16 = uint16[tokens][layers][top_k], accepted generated
tokens in request order; trace.csv gives each request's output_len, and request i owns
output_len_i - 1 tokens). Never executes anything from the capture; run with `python -I`.

Definition (vLLM v0.31.0's own, vllm/distributed/eplb/eplb_state.py:66-71 and :601-636): for a
step, T[l, r] = routed (token, expert) assignments landing on rank r in layer l under the
default linear placement (vllm/config/parallel.py:192: expert e on rank e // (E / ep)).
vLLM logs balancedness = sum_l mean_r T / sum_l max_r T. The imbalance here is its
reciprocal, I_tok = sum_l max_r T / sum_l mean_r T. Every layer has the same mean (B * top_k
/ ep), so this is also the layer mean of max/mean.

I_exp is the busiest rank's DISTINCT experts touched (weights read, which times a
weight-bandwidth-bound decode grouped GEMM), layer-averaged, over the per-rank count expected
under uniform routing, per * (1 - ((E - k) / E) ** B), which is what blis-latency-kernel's
ExpertsTouched charges today. So it multiplies that term directly.

Split (pre-registered in the brief's ROUTING.md section 0): FIT = requests 0-39, HELD =
requests 40-99; a step's tokens are drawn uniformly at random, without replacement, from one
split. Also reported: the same statistic for uniformly random routing (the finite-sample
floor any router has).

Usage: python -I derive_routing_profiles.py <routes.u16> <manifest.json> <trace.csv> <out.json>
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

EPS = (2, 4, 8, 16)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
FIT_REQUESTS = range(0, 40)
SEED = 20261016


def samples_for(b: int) -> int:
    return int(min(400, max(40, 40000 // b)))


def stats(routes: np.ndarray, pool: np.ndarray, experts: int, rng,
          windows: list | None = None) -> dict:
    """Per (ep, B): token and distinct-expert imbalance over sampled steps.

    With windows (a list of (start, length) request segments) a step is B CONSECUTIVE
    tokens of one request, the prefill proxy; otherwise B tokens drawn at random from pool.
    """
    out = {}
    for ep in EPS:
        per = experts // ep
        for b in BATCHES:
            if b > len(pool):
                continue
            if windows is not None:
                fits = [(a, n) for a, n in windows if n >= b]
                if not fits:
                    continue
            n = samples_for(b)
            it, ie = np.empty(n), np.empty(n)
            for s in range(n):
                if windows is None:
                    toks = rng.choice(pool, size=b, replace=False)
                else:
                    a, ln = fits[rng.integers(len(fits))]
                    o = a + rng.integers(ln - b + 1)
                    toks = np.arange(o, o + b)
                sel = routes[toks].astype(np.int64)  # [b, L, K]
                k = sel.shape[2]
                L = sel.shape[1]
                lidx = np.arange(L)[None, :, None]
                loads = np.bincount((lidx * ep + sel // per).ravel(),
                                    minlength=L * ep).reshape(L, ep)
                pres = np.zeros((L, experts), dtype=bool)
                pres[np.broadcast_to(lidx, sel.shape), sel] = True
                touched = pres.reshape(L, ep, per).sum(2)
                it[s] = loads.max(1).sum() / loads.mean(1).sum()
                # Against the per-rank expectation under uniform routing, which is what
                # blis-latency-kernel's ExpertsTouched already charges (internal/price/
                # memory.go), so the ratio multiplies that term without double counting.
                uniform = per * (1 - ((experts - k) / experts) ** b)
                ie[s] = touched.max(1).mean() / uniform
            out[f"ep{ep}_b{b}"] = {
                "ep": ep, "batch_tokens": b, "samples": n,
                "tok_median": round(float(np.median(it)), 4),
                "tok_p10": round(float(np.quantile(it, 0.1)), 4),
                "tok_p90": round(float(np.quantile(it, 0.9)), 4),
                "exp_median": round(float(np.median(ie)), 4),
                "exp_p10": round(float(np.quantile(ie, 0.1)), 4),
                "exp_p90": round(float(np.quantile(ie, 0.9)), 4),
            }
    return out


def main() -> None:
    routes_p, man_p, trace_p, out_p = sys.argv[1:5]
    m = json.load(open(man_p))
    T, L, K, E = m["num_tokens"], m["num_layers"], m["top_k"], m["num_experts"]
    routes = np.fromfile(routes_p, dtype="<u2").reshape(T, L, K)
    rows = list(csv.DictReader(open(trace_p)))
    seg = np.array([int(r["output_len"]) - 1 for r in rows])
    if seg.sum() != T:
        raise SystemExit(f"segments sum to {seg.sum()}, corpus has {T} tokens")
    starts = np.concatenate([[0], np.cumsum(seg)[:-1]])
    owner = np.repeat(np.arange(len(seg)), seg)
    fit_mask = np.isin(owner, list(FIT_REQUESTS))
    res = {"tokens": T, "layers": L, "top_k": K, "experts": E,
           "fit_tokens": int(fit_mask.sum()), "held_tokens": int((~fit_mask).sum()),
           "seed": SEED, "segments_start": int(starts[0])}
    rng = np.random.default_rng(SEED)
    res["FIT"] = stats(routes, np.flatnonzero(fit_mask), E, rng)
    res["HELD"] = stats(routes, np.flatnonzero(~fit_mask), E, rng)
    segs = list(zip(starts.tolist(), seg.tolist()))
    res["FIT_PREFILL_PROXY"] = stats(routes, np.arange(T), E, rng,
                                     [w for i, w in enumerate(segs) if i in FIT_REQUESTS])
    res["HELD_PREFILL_PROXY"] = stats(routes, np.arange(T), E, rng,
                                      [w for i, w in enumerate(segs) if i not in FIT_REQUESTS])
    # Uniform routing, top_k distinct experts per (token, layer), as a top-k router produces.
    uni = np.concatenate([np.argsort(rng.random((512, L, E), dtype=np.float32), axis=2)
                          [:, :, :K].astype(np.uint16) for _ in range(20)])
    res["UNIFORM"] = stats(uni, np.arange(uni.shape[0]), E, rng)
    json.dump(res, open(out_p, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
