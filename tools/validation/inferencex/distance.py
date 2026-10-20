"""Distribution distance between BLIS and InferenceX, per point and metric (RELEASE_V1 items 29-30).

Distances, all on the two MARGINAL distributions over the same window and request-set definition (no join):
  w1_norm  Wasserstein-1 / median(measured): the area between the two CDFs in units of the measured median.
  ks       sup |F_blis - F_meas|.
  q_err    signed % error of the BLIS quantile at p10/p50/p90/p99 (numpy-linear percentiles).
Request metrics (agentic: InferenceX request timeline, phase "profiling"; BLIS rows, phase "profile"; both
completed and not cancelled, warmups excluded on both sides): ttft_ms, tpot_ms ((lat - ttft)/(osl - 1),
osl >= 2; aiperf inter_token_latency, also InferenceX's tpotMs), e2el_ms, osl.
Server series (InferenceX trace-server-metrics, 1 s): num_running and num_waiting over the profile window
(first profiled start to last profiled end on each side); BLIS's are rebuilt from its per-request rows
(waiting = arrived and not yet first scheduled, running = scheduled and not finished; every phase, as the
server counts warmup tails too).
A pair is computed only when both sides' window and request-set definitions are identical (assert_same_population);
otherwise a named MISMATCH is recorded instead of a number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import metrics as M
import paths as PX

IXDIAG = PX.API
QS = (10, 50, 90, 99)
REQ_METRICS = ("ttft_ms", "tpot_ms", "e2el_ms", "osl")
SERIES_METRICS = ("num_running", "num_waiting")
REQ_SET = {"def": "completed, not cancelled, warmup excluded", "window": "profiling phase"}
SERIES_SET = {"def": "server queue depth, 1 s grid, every request in the engine",
              "window": "first profiled start .. last profiled end"}
# Named, never computed (the fields do not support a like-for-like pair).
UNSUPPORTED = {
    "queue_wait_ms": "InferenceX timeline has no engine queue time: ack - start is the HTTP ack and start - credit the "
                     "client credit wait; BLIS scheduling_delay_ms is engine arrival to first schedule",
    "kv_usage": "BLIS exports no KV-usage time series and no per-request block count",
}


def w1(a: list[float], b: list[float]) -> float:
    """Wasserstein-1 between two empirical distributions: integral of |F_a - F_b|."""
    a, b = sorted(a), sorted(b)
    na, nb = len(a), len(b)
    i = j = 0
    fa = fb = 0.0
    prev = min(a[0], b[0])
    tot = 0.0
    while i < na or j < nb:
        x = a[i] if j >= nb or (i < na and a[i] <= b[j]) else b[j]
        tot += abs(fa - fb) * (x - prev)
        prev = x
        while i < na and a[i] == x:
            i += 1
        while j < nb and b[j] == x:
            j += 1
        fa, fb = i / na, j / nb
    return tot


def ks(a: list[float], b: list[float]) -> float:
    a, b = sorted(a), sorted(b)
    na, nb = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < na and j < nb:
        x = min(a[i], b[j])
        while i < na and a[i] == x:
            i += 1
        while j < nb and b[j] == x:
            j += 1
        d = max(d, abs(i / na - j / nb))
    return d


def distances(pred: list[float], meas: list[float]) -> dict | None:
    pred = [x for x in pred if x is not None and math.isfinite(x)]
    meas = [x for x in meas if x is not None and math.isfinite(x)]
    if len(pred) < 2 or len(meas) < 2:
        return None
    med = M.pct(meas, 50)
    # Normaliser: the measured median; a series whose median is 0 (e.g. num_waiting on an idle engine) uses the
    # measured mean instead, and says so.
    basis, den = ("median", abs(med)) if med else ("mean", abs(sum(meas) / len(meas)))
    d1 = w1(pred, meas)
    out = {"n_blis": len(pred), "n_meas": len(meas), "w1": d1, "ks": ks(pred, meas), "median_meas": med,
           "w1_norm": d1 / den if den else None, "w1_norm_basis": basis}
    for q in QS:
        out[f"q{q}_err_pct"] = M.signed_pct(M.pct(pred, q), M.pct(meas, q))
    return out


def assert_same_population(meas: dict, pred: dict):
    assert meas["def"] == pred["def"] and meas["window"] == pred["window"], (meas, pred)


# ---------------------------------------------------------------- measured side
def load_timeline(row_id: int, base: Path = IXDIAG) -> dict | None:
    f = base / "request-timeline" / f"rt_{row_id}.body"
    return json.loads(f.read_text()) if f.exists() else None


def load_series(row_id: int, base: Path = IXDIAG) -> dict | None:
    f = base / "trace-server-metrics" / f"tsm_{row_id}.body"
    return json.loads(f.read_text()) if f.exists() else None


def ix_requests(tl: dict) -> tuple[list[dict], dict]:
    """Profiling, completed (ttft and osl stated), not cancelled. Returns rows and the exclusion census."""
    rows, census = [], {"warmup": 0, "cancelled": 0, "incomplete": 0}
    for r in tl["requests"]:
        if r.get("phase") != "profiling":
            census["warmup"] += 1
        elif r.get("cancelled"):
            census["cancelled"] += 1
        elif r.get("ttftMs") is None or r.get("osl") is None or r.get("end") is None:
            census["incomplete"] += 1
        else:
            lat = (r["end"] - r["start"]) / 1e6
            rows.append({"start_s": (tl["startNs"] + r["start"]) / 1e9, "end_s": (tl["startNs"] + r["end"]) / 1e9,
                         "ttft_ms": r["ttftMs"], "lat_ms": lat, "osl": r["osl"]})
    return rows, census


def req_values(rows: list[dict]) -> dict:
    return {"ttft_ms": [r["ttft_ms"] for r in rows], "e2el_ms": [r["lat_ms"] for r in rows],
            "tpot_ms": [(r["lat_ms"] - r["ttft_ms"]) / (r["osl"] - 1) for r in rows if r["osl"] >= 2],
            "osl": [float(r["osl"]) for r in rows]}


def ix_series(ts: dict, t0: float, t1: float) -> dict:
    """InferenceX queue depth samples (t relative to ts.startNs) inside [t0, t1] (absolute s)."""
    s0 = ts["startNs"] / 1e9
    qd = [q for q in ts.get("queueDepth") or [] if t0 <= s0 + q["t"] <= t1]
    return {"num_running": [q["running"] for q in qd], "num_waiting": [q["waiting"] for q in qd]}


# ---------------------------------------------------------------- BLIS side
def blis_series(m: dict, t0: float, t1: float) -> dict:
    """Queue depth on a 1 s grid over [t0, t1] (BLIS seconds) from per-request rows (all phases)."""
    import bisect
    arr, sch, end = [], [], []
    for r in m["requests"]:
        if r.get("e2e_ms", 0) <= 0:
            continue
        a = r["arrived_at"]
        arr.append(a)
        sch.append(a + r.get("scheduling_delay_ms", 0.0) / 1e3)
        end.append(a + r["e2e_ms"] / 1e3)
    arr.sort(), sch.sort(), end.sort()
    run, wait = [], []
    t = t0
    while t <= t1:
        ns = bisect.bisect_right(sch, t)
        run.append(float(ns - bisect.bisect_right(end, t)))
        wait.append(float(bisect.bisect_right(arr, t) - ns))
        t += 1.0
    return {"num_running": run, "num_waiting": wait}


def blis_req_rows(m: dict, hc_ms: float) -> list[dict]:
    rows = M.blis_rows(m, hc_ms, True)
    return [{"start_s": r["start_ms"] / 1e3, "end_s": (r["start_ms"] + r["lat_ms"]) / 1e3, "ttft_ms": r["ttft_ms"],
             "lat_ms": r["lat_ms"], "osl": r["osl"]} for r in rows]


# ---------------------------------------------------------------- one point
def point_distances(m: dict, hc_ms: float, row_id: int, db_metrics: dict | None, dp: int,
                    base: Path = IXDIAG) -> dict:
    """Distances for one agentic point. db_metrics: the DB row's metrics (population identity check)."""
    out = {"req_set": REQ_SET, "series_set": SERIES_SET, "metrics": {}, "mismatches": [],
           "unsupported": dict(UNSUPPORTED), "source": {"timeline": f"rt_{row_id}", "series": f"tsm_{row_id}"}}
    tl = load_timeline(row_id, base)
    if tl is None:
        out["mismatches"].append({"item": "request timeline", "detail": "no InferenceX timeline stored for this row"})
        return out
    ixr, census = ix_requests(tl)
    out["ix_census"] = census
    # Population identity: the timeline's profiled completed set must be the set the DB aggregate counts.
    want = (db_metrics or {}).get("total_requests_completed")
    if want is not None and int(want) != len(ixr):
        out["mismatches"].append({"item": "request set", "detail": f"timeline profiling completed {len(ixr)} != DB "
                                  f"total_requests_completed {int(want)}"})
        return out
    br = blis_req_rows(m, hc_ms)
    if not br or not ixr:
        out["mismatches"].append({"item": "request set", "detail": "empty profiled set on one side"})
        return out
    assert_same_population(REQ_SET, REQ_SET)
    pv, mv = req_values(br), req_values(ixr)
    for k in REQ_METRICS:
        d = distances(pv[k], mv[k])
        if d:
            out["metrics"][k] = d
    ts = load_series(row_id, base)
    if ts is None or not ts.get("queueDepth"):
        out["mismatches"].append({"item": "server series", "detail": "no InferenceX queueDepth series stored"})
        return out
    if dp > 1:
        out["mismatches"].append({"item": "server series", "detail": f"dp={dp}: InferenceX queueDepth is summed over "
                                  "engines; BLIS per-instance rows are rebuilt per pool, not compared"})
        return out
    mser = ix_series(ts, min(r["start_s"] for r in ixr), max(r["end_s"] for r in ixr))
    bser = blis_series(m, min(r["start_s"] for r in br), max(r["end_s"] for r in br))
    for k in SERIES_METRICS:
        d = distances(bser[k], mser[k])
        if d:
            out["metrics"][k] = d
    return out


# ---------------------------------------------------------------- single-turn: quantiles only
SINGLE_Q = {"ttft_ms": {50: "median_ttft_ms", 90: "p90_ttft_ms", 99: "p99_ttft_ms"},
            "tpot_ms": {50: "median_tpot_ms", 90: "p90_tpot_ms", 99: "p99_tpot_ms"},
            "e2el_ms": {50: "median_e2el_ms", 90: "p90_e2el_ms", 99: "p99_e2el_ms"}}


def single_turn_quantiles(result: dict) -> dict:
    """vllm bench rows carry no per-request data: only the p50/p90/p99 quantile errors exist (W1/KS named)."""
    e = result["measured_rows"][0]["signed_err_pct"]
    out = {"req_set": {"def": "all benchmark requests, warmups excluded", "window": "benchmark"}, "metrics": {},
           "mismatches": [{"item": "distribution", "detail": "vllm bench reports aggregates only: W1/KS not computable, "
                           "quantile errors at p50/p90/p99 only (no p10)"}], "unsupported": {}}
    for k, qm in SINGLE_Q.items():
        d = {f"q{q}_err_pct": e.get(name) for q, name in qm.items() if e.get(name) is not None}
        if d:
            out["metrics"][k] = d
    return out


DIST_KEYS = ("w1_norm", "ks", "q10_err_pct", "q50_err_pct", "q90_err_pct", "q99_err_pct")


def flat(dist: dict | None) -> dict:
    """{'ttft_ms.w1_norm': v, ...} for aggregation; quantile errors as absolute values are taken by the caller."""
    out = {}
    for k, d in ((dist or {}).get("metrics") or {}).items():
        for s in DIST_KEYS:
            if d.get(s) is not None:
                out[f"{k}.{s}"] = d[s]
    return out


def of_result(r: dict, pdir: Path | None = None) -> dict | None:
    """The point's distances: result.json 'distances', else the backfilled sibling distances.json."""
    if r.get("distances"):
        return r["distances"]
    if pdir is not None and (pdir / "distances.json").exists():
        return json.loads((pdir / "distances.json").read_text())
    return None
