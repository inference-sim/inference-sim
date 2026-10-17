"""Measured server state per scored point (KV_METRICS.md sec 1).

Sources, each read as UNTRUSTED data (run with python -I):
- single_turn points: the run's vLLM server log (restored DB `server_logs`). vLLM v0.31.0 LoggingStatLogger
  prints every 10 s per engine (v1/metrics/loggers.py:259 line format at the run's version):
  "Engine NNN: Avg prompt throughput: X tokens/s, Avg generation throughput: Y tokens/s, Running: R reqs,
  Waiting: W reqs, GPU KV cache usage: U%, Prefix cache hit rate: H%" (+ "Preemptions: P" when nonzero).
  R/W/U are the scheduler gauges at the last step of the interval (point samples), H is the interval's
  hits/queries; the throughputs are the interval's token counts / interval length.
- agentic points: InferenceX /api/v1/trace-server-metrics (chart_series v19, InferenceX-app
  packages/db/src/etl/compute-chart-series.ts @75a644b7): 1 s grid, kvCacheUsage = mean over logical engines of
  the 1 s time-mean of vllm:kv_cache_usage_perc; queueDepth = sum over engines of the 1 s time-mean of
  vllm:num_requests_{running,waiting}; prefixCacheHitRate = sum hits rate / sum queries rate; prefillTps =
  vllm:prompt_tokens rate; decodeTps = vllm:generation_tokens rate; promptTokensBySource = vllm:prompt_tokens_by_source
  rate per source. t=0 is the first scrape (warmup included, v12). The profiling window comes from the
  request timeline (min start .. max end of phase=='profiling' records), relative to the series' startNs.
Writes kvm/measured.jsonl (one line per point; derived, never committed).
Usage: uv run --no-project python -I kvmeas.py
"""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
from pathlib import Path
import sys as _s
_s.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402

HERE = Path(__file__).resolve().parent
POINTS = PX.ROOT / "work-r7" / "points"
OUT = PX.ROOT / "kvm" / "measured.jsonl"
API = PX.API
STAT = re.compile(r"Engine (\d+): Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) "
                  r"tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs,(?: Deferred: \d+ reqs,)?(?: Preemptions: (\d+),)?"
                  r" GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%")
PREEMPT = re.compile(r"(?i)preempt")


def body(p: Path):
    b = p.read_bytes()
    try:
        return json.loads(gzip.decompress(b))
    except OSError:
        return json.loads(b)


def log_stats(slid: int) -> dict | None:
    t = subprocess.run(["psql", "-d", PX.PGDATABASE, "-At", "-c", f"select server_log from server_logs where id={int(slid)}"],
                       capture_output=True, text=True, check=True).stdout
    return parse_stats(t)


def parse_stats(t: str) -> dict:
    rows = [m.groups() for m in STAT.finditer(t)]
    if not rows:
        return {"stat_lines": 0}
    eng = sorted({int(r[0]) for r in rows})
    # keep only intervals with activity (the server idles before/after the benchmark)
    act = [r for r in rows if float(r[1]) + float(r[2]) > 0 or int(r[3]) + int(r[4]) > 0]
    return {"stat_lines": len(rows), "engines": len(eng),
            "kv": [float(r[6]) / 100 for r in act], "running": [int(r[3]) for r in act],
            "waiting": [int(r[4]) for r in act], "hit": [float(r[7]) / 100 for r in act],
            "prompt_tps": [float(r[1]) for r in act], "gen_tps": [float(r[2]) for r in act],
            "preemptions": sum(int(r[5] or 0) for r in rows),
            "preempt_mentions": len(PREEMPT.findall(t))}


def profile_window(rid: int, start_ns: int):
    p = API / "request-timeline" / f"rt_{rid}.body"
    if not p.exists():
        return None
    d = body(p)
    s0 = d["startNs"]
    pr = [r for r in d["requests"] if r.get("phase") == "profiling"]
    if not pr:
        return None
    a = min(r["start"] for r in pr) + s0
    b = max(r["end"] for r in pr if r.get("end") is not None) + s0
    return [(a - start_ns) / 1e9, (b - start_ns) / 1e9]


def series(rid: int) -> dict | None:
    p = API / "trace-server-metrics" / f"tsm_{rid}.body"
    if not p.exists():
        return None
    d = body(p)
    win = profile_window(rid, d["startNs"])
    pick = lambda xs, f="value": [[x["t"], x[f]] for x in xs]  # noqa: E731
    return {"window_s": win, "pool_tokens": d.get("kvCachePoolTokens"), "meta": d.get("meta"),
            "kv": pick(d.get("kvCacheUsage") or []),
            "kv_by_engine": [[e["engineLabel"], pick(e["points"])] for e in d.get("kvCacheUsageByEngine") or []],
            "running": pick(d.get("queueDepth") or [], "running"), "waiting": pick(d.get("queueDepth") or [], "waiting"),
            "hit": pick(d.get("prefixCacheHitRate") or []), "prompt_tps": pick(d.get("prefillTps") or []),
            "gen_tps": pick(d.get("decodeTps") or []), "hits_tps": pick(d.get("prefixCacheHitsTps") or []),
            "by_source": {k: pick(v) for k, v in (d.get("promptTokensBySource") or {}).items()},
            "host_kv": pick(d.get("hostKvCacheUsage") or [])}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w") as fo:
        for f in sorted(POINTS.glob("*/*/result.json")):
            r = json.loads(f.read_text())
            if r.get("status") != "scored":
                continue
            src = r.get("source") or {}
            rec = {"point_id": r["point_id"], "run_id": r["run_id"], "scenario": r["scenario"], "set": r["set"],
                   "dir": str(f.parent.relative_to(PX.ROOT))}
            if src.get("server_log_id"):
                rec["log"] = log_stats(src["server_log_id"])
            if r["scenario"] == "agentic_traces":
                for rid in src.get("db_rows", []):
                    s = series(rid)
                    if s:
                        rec["series"] = s
                        rec["series_row"] = rid
                        break
            fo.write(json.dumps(rec) + "\n")
            n += 1
    print(n, "points ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
