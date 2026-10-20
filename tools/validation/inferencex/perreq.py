"""One-to-one request join of InferenceX request timelines with BLIS per-request rows, and
distribution distances on the joined request set. See README.md.

Measured side: the InferenceX Data API request timeline (rt_<row_id>.body, JSON; keys startNs,
endNs, durationS, version, requests[]). A request carries cid (the conversation: the trace id for a
main-agent turn, "<trace>::sa:<name>" for a subagent, "<trace>::fa:/aux:<n>" for a flattened agent),
ti (turn index within cid), isl, osl, phase (warmup|profiling), start/ack/end/credit (ns since startNs),
ttftMs, tpotMs, cancelled.
BLIS side: metrics.json requests[] with agentic.stream (== cid), agentic.turn (== ti), phase
(warmup|profile), agentic.outcome (completed|cancelled), arrived_at (s), ttft_ms, e2e_ms,
scheduling_delay_ms, num_prefill_tokens, output_tokens.

Identity: (cid, ti, k) where k is the k-th occurrence in start order (a recycled trace replays the
same stream again). Only the profile phase pairs. A request cancelled on EITHER side is excluded from
the pair set on both sides (same request set). A pair is VALID when OSL is equal and ISL agrees within
1% (aiperf counts the server's prompt tokens: chat template and cache-bust marker included).

Latency: measured end-start (aiperf request_latency, to the last chunk); BLIS e2e_ms - host_completion
(B2). ITL per request = (latency - ttft) / (OSL - 1), OSL >= 2. "front" = start->ack (measured: the
HTTP response headers, which vLLM sends after render/tokenize; BLIS has no such event, so it is
reported, never scored). BLIS scheduling_delay_ms is arrival->first scheduled (admission + queue).
"""

from __future__ import annotations

import bisect
import collections
import json
import math
from pathlib import Path

ISL_TOL = 0.01


def measured_rows(rt: dict) -> dict:
    out = {"profile": [], "warmup": [], "cancelled": 0}
    for r in rt["requests"]:
        ph = "profile" if r.get("phase") == "profiling" else "warmup"
        if r.get("cancelled"):
            out["cancelled"] += ph == "profile"
            row = {"cid": r["cid"], "ti": r["ti"], "start_ms": r["start"] / 1e6, "cancelled": True}
            out[ph].append(row)
            continue
        if r.get("ttftMs") is None or r.get("isl") is None or r.get("end") is None:
            out[ph].append({"cid": r["cid"], "ti": r["ti"], "start_ms": r["start"] / 1e6, "cancelled": True,
                            "incomplete": True})
            continue
        s, e = r["start"] / 1e6, r["end"] / 1e6
        lat = e - s
        osl = r["osl"]
        out[ph].append({"cid": r["cid"], "ti": r["ti"], "kind": r.get("srcKind"), "start_ms": s, "end_ms": e,
                        "lat_ms": lat, "ttft_ms": r["ttftMs"], "isl": r["isl"], "osl": osl,
                        "front_ms": (r["ack"] - r["start"]) / 1e6 if r.get("ack") else None,
                        "itl_ms": (lat - r["ttftMs"]) / (osl - 1) if osl and osl >= 2 else None,
                        "cancelled": False})
    return out


def blis_rows(m: dict, host_completion_ms: float) -> dict:
    out = {"profile": [], "warmup": [], "cancelled": 0}
    for r in m["requests"]:
        a = r.get("agentic") or {}
        ph = "warmup" if r.get("phase") == "warmup" else "profile"
        s = r["arrived_at"] * 1e3
        if a.get("outcome") != "completed":
            out["cancelled"] += ph == "profile"
            out[ph].append({"cid": a.get("stream"), "ti": a.get("turn"), "start_ms": s, "cancelled": True})
            continue
        lat = r["e2e_ms"] - host_completion_ms
        osl = r["output_tokens"]
        out[ph].append({"cid": a["stream"], "ti": a["turn"], "kind": a.get("kind"), "start_ms": s, "end_ms": s + lat,
                        "lat_ms": lat, "ttft_ms": r["ttft_ms"], "isl": r["num_prefill_tokens"], "osl": osl,
                        "sched_ms": r.get("scheduling_delay_ms"),
                        "itl_ms": (lat - r["ttft_ms"]) / (osl - 1) if osl >= 2 else None, "cancelled": False})
    return out


def keyed(rows):
    cnt = collections.Counter()
    d = {}
    for r in sorted(rows, key=lambda r: r["start_ms"]):
        k = (r["cid"], r["ti"])
        d[(k, cnt[k])] = r
        cnt[k] += 1
    return d


def annotate(rows, others=()):
    """In-flight at arrival on `rows` (others started and not yet ended), counting every completed
    request of the side (warmup included: a warmup request still running at the start of the profile
    window loads the server)."""
    done = [r for r in list(rows) + list(others) if not r["cancelled"] and "end_ms" in r]
    starts = sorted(r["start_ms"] for r in done)
    ends = sorted(r["end_ms"] for r in done)
    for r in rows:
        if r["cancelled"] or "end_ms" not in r:
            continue
        r["inflight"] = bisect.bisect_right(starts, r["start_ms"]) - bisect.bisect_right(ends, r["start_ms"]) - 1


def prior_context(rows):
    """Prefix proxy: the previous turn of the same conversation's ISL + OSL (tokens a prefix cache can hold
    for this turn; an upper bound on the cached prefix, identical on both sides by construction)."""
    by = collections.defaultdict(list)
    for r in rows:
        if not r["cancelled"]:
            by[r["cid"]].append(r)
    for rs in by.values():
        rs.sort(key=lambda r: (r["ti"], r["start_ms"]))
        prev = None
        for r in rs:
            r["prev_ctx"] = min(prev["isl"] + prev["osl"], r["isl"]) if prev else 0
            r["first_turn"] = prev is None
            prev = r


def join(meas: dict, blis: dict) -> dict:
    """Pair the profile phase. Returns pairs and the verification counts."""
    for side in (meas, blis):
        annotate(side["profile"], side["warmup"])
        prior_context(side["profile"] + side["warmup"])
    km, kb = keyed(meas["profile"]), keyed(blis["profile"])
    common = km.keys() & kb.keys()
    pairs, cancelled_either, mismatch = [], 0, 0
    for k in common:
        m, b = km[k], kb[k]
        if m["cancelled"] or b["cancelled"]:
            cancelled_either += 1
            continue
        if m["osl"] != b["osl"] or not m["isl"] or abs(b["isl"] / m["isl"] - 1) > ISL_TOL:
            mismatch += 1
            continue
        pairs.append((m, b))
    wk = keyed(meas["warmup"]).keys() & keyed(blis["warmup"]).keys()
    cross = len(km.keys() & keyed(blis["warmup"]).keys()) + len(kb.keys() & keyed(meas["warmup"]).keys())
    eq = [(m, b) for m, b in pairs]
    ver = {"measured_profile": len(km), "blis_profile": len(kb), "common_keys": len(common),
           "measured_only": len(km.keys() - kb.keys()), "blis_only": len(kb.keys() - km.keys()),
           "cancelled_measured": meas["cancelled"], "cancelled_blis": blis["cancelled"],
           "cancelled_either_excluded": cancelled_either, "isl_osl_mismatch": mismatch, "valid_pairs": len(pairs),
           "osl_equal_rate": (sum(1 for k in common if not km[k]["cancelled"] and not kb[k]["cancelled"]
                                  and km[k]["osl"] == kb[k]["osl"]) / max(1, len(common) - cancelled_either)),
           "isl_exact_rate": sum(1 for m, b in eq if m["isl"] == b["isl"]) / max(1, len(eq)),
           "isl_ratio_median": _median([b["isl"] / m["isl"] for m, b in eq]),
           "warmup_common_keys": len(wk), "phase_crossed_keys": cross}
    return {"pairs": pairs, "verify": ver}


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def quantile(xs, p):
    s = sorted(xs)
    if not s:
        return None
    k = (len(s) - 1) * p
    f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (k - f) * (s[f + 1] - s[f])


def wasserstein1(a, b):
    """W1 between two empirical distributions (exact, via the merged CDF)."""
    a, b = sorted(a), sorted(b)
    if not a or not b:
        return None
    if len(a) == len(b):  # equal-size samples: W1 is the mean gap between order statistics
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)
    pts = sorted(set(a) | set(b))
    ia = ib = 0
    w = 0.0
    for i in range(len(pts) - 1):
        x = pts[i]
        while ia < len(a) and a[ia] <= x:
            ia += 1
        while ib < len(b) and b[ib] <= x:
            ib += 1
        w += abs(ia / len(a) - ib / len(b)) * (pts[i + 1] - x)
    return w


def ks(a, b):
    a, b = sorted(a), sorted(b)
    if not a or not b:
        return None
    d = 0.0
    for x in sorted(set(a) | set(b)):
        d = max(d, abs(bisect.bisect_right(a, x) / len(a) - bisect.bisect_right(b, x) / len(b)))
    return d


def distances(meas_vals, blis_vals):
    """Median-normalised W1, KS and signed quantile errors (%) at p10/p50/p90/p99."""
    if len(meas_vals) < 2 or len(blis_vals) < 2:
        return None
    med = quantile(meas_vals, 0.5)
    out = {"n_meas": len(meas_vals), "n_blis": len(blis_vals),
           "w1_norm": wasserstein1(meas_vals, blis_vals) / med if med else None, "ks": ks(meas_vals, blis_vals)}
    for p in (0.1, 0.5, 0.9, 0.99):
        qm, qb = quantile(meas_vals, p), quantile(blis_vals, p)
        out[f"q{int(p * 100)}_err_pct"] = (qb / qm - 1) * 100 if qm else None
    return out


METRICS = ("ttft_ms", "itl_ms", "lat_ms", "osl")


def pair_distances(pairs, sel=None):
    """Distances over the joined request set (same requests on both sides)."""
    ps = [p for p in pairs if sel is None or sel(p)]
    out = {}
    for k in METRICS:
        mv = [m[k] for m, b in ps if m.get(k) is not None and b.get(k) is not None]
        bv = [b[k] for m, b in ps if m.get(k) is not None and b.get(k) is not None]
        out[k] = distances(mv, bv)
    return out


def log_ratio(m, b, k):
    x, y = m.get(k), b.get(k)
    if x is None or y is None or x <= 0 or y <= 0:
        return None
    return math.log(y / x)


def load_point(result: dict, cov_root: Path, rt_dir: Path):
    rid = result["source"]["db_rows"][0]
    rt = json.loads((rt_dir / f"rt_{rid}.body").read_text())
    m = json.loads((cov_root / result["metrics_path"]).read_text())
    return measured_rows(rt), blis_rows(m, result.get("b2_host_completion_ms", 0.777)), m
