"""Re-run scored r7 points with the kv-metrics BLIS binary to obtain `server_series` (KV_METRICS.md sec 3).
Same command as work-r7/<point>/command.txt; only the binary, --metrics-path and the series flag differ.
Selects points with a measured server-state signal (kvm/measured.jsonl: series or log stat lines).
Each simulation runs under $IX_HEAVY when set; width 2.
Usage: uv run --no-project python -I kvrun.py --blis BIN [--limit N] [--only agentic|single]"""
import argparse
import concurrent.futures as cf
import json
import shlex
import subprocess
from pathlib import Path
import sys as _s
_s.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402

HERE = Path(__file__).resolve().parent
C = PX.ROOT
FLAG = "--server-series-interval-s"


def job(r, blis):
    pdir = C / r["dir"]
    out = C / "kvm" / "runs" / r["point_id"]
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "metrics.json"
    if mp.exists():
        return r["point_id"], "cached"
    cmd = shlex.split((pdir / "command.txt").read_text().splitlines()[0])
    cmd[0] = blis
    i = cmd.index("--metrics-path")
    cmd[i + 1] = str(mp)
    cmd += [FLAG, "1" if r["scenario"] == "agentic_traces" else "10"]
    p = subprocess.run(PX.heavy(cmd), capture_output=True, text=True)
    (out / "stderr.log").write_text(p.stderr[-20000:])
    (out / "command.txt").write_text(" ".join(cmd) + "\n")
    return r["point_id"], "ok" if p.returncode == 0 and mp.exists() else f"rc={p.returncode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blis", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only")
    a = ap.parse_args()
    rs = []
    for l in (C / "kvm" / "measured.jsonl").open():
        r = json.loads(l)
        if not (r.get("series") or (r.get("log") or {}).get("stat_lines")):
            continue
        if a.only == "agentic" and r["scenario"] != "agentic_traces":
            continue
        if a.only == "single" and r["scenario"] == "agentic_traces":
            continue
        rs.append(r)
    rs.sort(key=lambda r: r["scenario"] != "agentic_traces")
    rs = rs[: a.limit] if a.limit else rs
    with cf.ThreadPoolExecutor(2) as ex:
        for pid, st in ex.map(lambda r: job(r, a.blis), rs):
            print(pid, st, flush=True)


if __name__ == "__main__":
    main()
