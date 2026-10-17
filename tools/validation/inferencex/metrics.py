"""InferenceX metric definitions, applied identically to BLIS's per-request rows.

Every metric carries the WINDOW it is computed over. The scorer only pairs a measured and a
predicted value whose definition id and window are equal (assert_same), so unlike quantities are
never compared; a metric one side cannot produce is returned as a named mismatch instead.

Agentic (aiperf) definitions: InferenceX inferencex-e2e/infx/results/agentic/request_metrics.py and
common.py at 9fa76586 (identical at the run commits, compare_ix.py):
  per request: ttft, e2el = aiperf request_latency (to the last content chunk), itl = tpot =
  (latency - ttft) / (OSL - 1) for OSL >= 2 (aiperf inter_token_latency_metric.py:48-55).
  stats_for: mean, p50, p75, p90, p95 (linear percentile, common.py:11-20), std = pstdev.
  intvty_{mean,p50,p75,p90,p95} = 1 / {stat}_itl (request_metrics.py:65-85).
  e2e_norm_intvty = 1 / stat(e2el / OSL) (request_metrics.py:88-110).
  throughput = sum(tokens) / (max end - min start) over profiled records (request_metrics.py:214-247).
Single-turn (vllm bench serving, InferenceX utils/bench_serving/benchmark_serving.py at the run commit,
utils/process_result.py): tpot per request as above; itl = pooled per-chunk gaps; e2el; ttft;
np.mean / np.median / np.percentile (linear) / np.std (population); intvty = 1000 / tpot_ms per stat;
tput_per_gpu = total_token_throughput / TP; output_tput_per_gpu = output_throughput / TP.

B2 (self-audit 2026-10-16): BLIS's per-request e2e_ms includes the kernel's host_completion term,
which follows the last streamed chunk (IS sim/metrics_measured.go:179-187). aiperf's and vllm bench's
latency end at the last chunk, so latency = e2e_ms - host_completion. blis_rows(..., b2=True) applies
it and the caller asserts the row latencies reproduce the aggregate's e2e_p50_ms.
"""

from __future__ import annotations

import math
import statistics

STATS = ("mean", "p50", "p75", "p90", "p95", "std")


def pct(xs, p):
    """numpy 'linear' percentile, as InferenceX common.percentile and np.percentile."""
    s = sorted(xs)
    if not s:
        return None
    k = (len(s) - 1) * p / 100
    f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (k - f) * (s[f + 1] - s[f])


def stats(xs, which=STATS):
    if not xs:
        return {}
    out = {}
    for w in which:
        if w == "mean":
            out[w] = statistics.mean(xs)
        elif w == "std":
            out[w] = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        elif w == "median":
            out[w] = pct(xs, 50)
        else:
            out[w] = pct(xs, float(w[1:]))
    return out


def blis_rows(m: dict, host_completion_ms: float, b2: bool, phase: str = "profile"):
    """BLIS per-request rows in aiperf terms: (start_ms, latency_ms, ttft_ms, isl, osl).
    Only measured-phase completed requests (BLIS keeps phase 'profile' on grace completions)."""
    rows = []
    for r in m["requests"]:
        a = r.get("agentic")
        if a is not None:
            if r.get("phase") != phase or a.get("outcome") != "completed":
                continue
        elif phase == "warmup" or r.get("e2e_ms", 0) <= 0:
            continue
        lat = r["e2e_ms"] - (host_completion_ms if b2 else 0.0)
        rows.append({"start_ms": r["arrived_at"] * 1e3, "lat_ms": lat, "ttft_ms": r["ttft_ms"],
                     "isl": r["num_prefill_tokens"], "osl": r["output_tokens"]})
    return rows


def request_metrics(rows: list[dict]) -> dict:
    """InferenceX agentic request metrics from per-request rows (ms, tokens). Profile window."""
    itls = [(r["lat_ms"] - r["ttft_ms"]) / (r["osl"] - 1) for r in rows if r["osl"] >= 2]
    out = {}
    for name, xs in (("ttft_ms", [r["ttft_ms"] for r in rows]), ("e2el_ms", [r["lat_ms"] for r in rows]),
                     ("itl_ms", itls), ("tpot_ms", itls)):
        for k, v in stats(xs).items():
            out[f"{k}_{name}"] = v
    for k in ("mean", "p50", "p75", "p90", "p95"):
        v = out.get(f"{k}_itl_ms")
        if v and v > 0:
            out[f"{k}_intvty"] = 1e3 / v
    norm = [r["lat_ms"] / 1e3 / r["osl"] for r in rows if r["osl"] > 0 and r["lat_ms"] > 0]
    for k in ("p75", "p90"):
        v = pct(norm, float(k[1:])) if norm else None
        if v:
            out[f"{k}_e2e_norm_intvty"] = 1.0 / v
    start = min(r["start_ms"] for r in rows)
    end = max(r["start_ms"] + r["lat_ms"] for r in rows)
    dur = (end - start) / 1e3
    ti, to = sum(r["isl"] for r in rows), sum(r["osl"] for r in rows)
    out.update({"output_tput_tps": to / dur, "input_tput_tps": ti / dur, "total_tput_tps": (ti + to) / dur,
                "duration_s": dur, "requests": len(rows), "mean_input_tokens": statistics.mean(r["isl"] for r in rows),
                "mean_output_tokens": statistics.mean(r["osl"] for r in rows), "total_output_tokens": to})
    return out


# InferenceX DB (agentic) key -> (harness key, scale to harness unit)
IX_AGENTIC = {f"{s}_{n}": (f"{h}_{n2}", 1e3) for s, h in
              (("mean", "mean"), ("median", "p50"), ("p75", "p75"), ("p90", "p90"), ("p95", "p95"), ("std", "std"))
              for n, n2 in (("ttft", "ttft_ms"), ("e2el", "e2el_ms"), ("itl", "itl_ms"), ("tpot", "tpot_ms"))}
IX_AGENTIC.update({f"{s}_intvty": (f"{h}_intvty", 1.0) for s, h in
                   (("mean", "mean"), ("median", "p50"), ("p75", "p75"), ("p90", "p90"), ("p95", "p95"))})
IX_AGENTIC.update({"output_tput_tps": ("output_tput_tps", 1.0), "input_tput_tps": ("input_tput_tps", 1.0),
                   "total_tput_tps": ("total_tput_tps", 1.0), "duration_seconds": ("duration_s", 1.0),
                   "total_requests_completed": ("requests", 1.0), "mean_input_tokens": ("mean_input_tokens", 1.0),
                   "mean_output_tokens_actual": ("mean_output_tokens", 1.0),
                   "total_generation_tokens": ("total_output_tokens", 1.0)})
# Agentic metrics InferenceX reports that BLIS cannot produce, with the reason.
AGENTIC_UNEXPRESSED = {
    "full_response_itl / full_response_intvty": "aiperf full_response ITL splits reasoning/tool content; BLIS has no content",
    "gpu_kv_cache_usage_pct": "max over the run of vllm:kv_cache_usage_perc; BLIS exports no KV-usage time series",
    "qps percentiles": "1 s completion-window distribution; BLIS exports the mean only (computed here from rows)",
    "power / energy": "not modelled",
    "theoretical_cache_hit_rate": "aiperf infinite-cache bound of the workload; reported as a workload check only",
}


def single_turn_blis(m: dict, rows: list[dict], tp_div: int) -> dict:
    """vllm bench serving metrics from BLIS rows (per request) and BLIS's pooled step series."""
    tpots = [(r["lat_ms"] - r["ttft_ms"]) / (r["osl"] - 1) for r in rows if r["osl"] > 1]
    out = {}
    for name, xs in (("ttft_ms", [r["ttft_ms"] for r in rows]), ("tpot_ms", tpots), ("e2el_ms", [r["lat_ms"] for r in rows])):
        out[f"mean_{name}"] = statistics.mean(xs)
        out[f"median_{name}"] = pct(xs, 50)
        out[f"p90_{name}"] = pct(xs, 90)
        out[f"p99_{name}"] = pct(xs, 99)
        out[f"p99.9_{name}"] = pct(xs, 99.9)
        out[f"std_{name}"] = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    # pooled per-step intervals (BLIS itl_* keys, measured window): vllm bench's per-chunk itl
    out["mean_chunk_itl_ms"], out["p90_chunk_itl_ms"], out["p99_chunk_itl_ms"] = m["itl_mean_ms"], m["itl_p90_ms"], m["itl_p99_ms"]
    for k in ("mean", "median", "p90", "p99", "p99.9"):
        v = out[f"{k}_tpot_ms"]
        out[f"{k}_intvty"] = 1e3 / v if v else None
    start = min(r["start_ms"] for r in rows)
    end = max(r["start_ms"] + r["lat_ms"] for r in rows)
    dur = (end - start) / 1e3
    ti, to = sum(r["isl"] for r in rows), sum(r["osl"] for r in rows)
    out.update({"tput_per_gpu": (ti + to) / dur / tp_div, "output_tput_per_gpu": to / dur / tp_div,
                "input_tput_per_gpu": ti / dur / tp_div, "requests": len(rows), "duration_s": dur})
    return out


IX_SINGLE = {f"{s}_{n}": (f"{s}_{n}_ms", 1e3) for s in ("mean", "median", "p90", "p99", "p99.9", "std")
             for n in ("ttft", "tpot", "e2el")}
IX_SINGLE.update({f"{s}_itl": (f"{s}_chunk_itl_ms", 1e3) for s in ("mean", "p90", "p99")})  # pooled per-chunk: never the agentic per-request *_itl_ms
IX_SINGLE.update({f"{s}_intvty": (f"{s}_intvty", 1.0) for s in ("mean", "median", "p90", "p99", "p99.9")})
IX_SINGLE.update({k: (k, 1.0) for k in ("tput_per_gpu", "output_tput_per_gpu", "input_tput_per_gpu")})
SINGLE_UNEXPRESSED = {
    "median_itl / std_itl / p99.9_itl": "pooled per-chunk ITL distribution: BLIS exports mean/p90/p95/p99 of its step series only",
    "std_intvty": "vllm bench std of per-chunk reciprocals; not derivable from BLIS outputs",
    "power / energy": "not modelled",
}


CHUNK_ITL_KEYS = ("mean_itl", "p90_itl", "p99_itl")


def drop_chunk_itl(keymap: dict, unexp: dict, stream_interval: int | None) -> tuple[dict, dict]:
    """With stream_interval N > 1 the engine streams one chunk per N tokens, so vllm bench's pooled per-chunk ITL
    measures N-token gaps while BLIS's step series is per token: not the same quantity, so not scored."""
    if not stream_interval or stream_interval <= 1:
        return keymap, unexp
    keymap = {k: v for k, v in keymap.items() if k not in CHUNK_ITL_KEYS}
    unexp = {**unexp, "mean_itl / p90_itl / p99_itl": f"stream_interval={stream_interval}: vllm bench pools per-chunk "
             f"gaps of ~{stream_interval} tokens; BLIS's step ITL is per token"}
    return keymap, unexp


def signed_pct(pred, meas):
    if pred is None or meas in (None, 0) or not all(map(math.isfinite, (pred, meas))):
        return None
    return (pred / meas - 1) * 100


def assert_same(meas: dict, pred: dict):
    """Each pair must name the same definition and window on both sides."""
    for k in meas:
        if k in pred:
            assert meas[k]["def"] == pred[k]["def"] and meas[k]["window"] == pred[k]["window"], \
                (k, meas[k]["def"], meas[k]["window"], pred[k]["def"], pred[k]["window"])
