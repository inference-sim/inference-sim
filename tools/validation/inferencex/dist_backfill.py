"""Backfill distribution distances for already-scored points from their stored BLIS metrics.json (per-request rows)
and the InferenceX diagnostics: writes distances.json next to each result.json. Points whose BLIS per-request
output is missing are listed (need a re-score) in <work>/distances_backfill.json.

Usage: uv run --no-project python -I dist_backfill.py ../work ../work-r7
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import distance as DIST  # noqa: E402


def backfill(work: Path) -> dict:
    stat = {"work": str(work), "scored": 0, "written": 0, "agentic": 0, "needs_rescore": [], "no_timeline": []}
    for f in sorted((work / "points").glob("*/*/result.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "scored":
            continue
        stat["scored"] += 1
        if r["scenario"] == "single_turn":
            d = DIST.single_turn_quantiles(r)
        else:
            stat["agentic"] += 1
            mp = f.parent / "metrics.json"
            if not mp.exists():
                stat["needs_rescore"].append(r["point_id"])
                continue
            m = json.loads(mp.read_text())
            if not m.get("requests"):
                stat["needs_rescore"].append(r["point_id"])
                continue
            mr = r["measured_rows"][0]
            want = (mr["measured"].get("requests") or {}).get("value")
            d = DIST.point_distances(m, r["b2_host_completion_ms"], mr["row_id"],
                                     {"total_requests_completed": want}, int((r.get("resolved") or {}).get("dp") or 1))
            if any(x["item"] == "request timeline" for x in d["mismatches"]):
                stat["no_timeline"].append(r["point_id"])
        d["backfilled_from"] = {"metrics_sha256": r.get("metrics_sha256")}
        (f.parent / "distances.json").write_text(json.dumps(d))
        stat["written"] += 1
    (work / "distances_backfill.json").write_text(json.dumps(stat, indent=1))
    return stat


if __name__ == "__main__":
    for w in sys.argv[1:]:
        s = backfill(Path(w).resolve())
        print(s["work"], {k: (len(v) if isinstance(v, list) else v) for k, v in s.items() if k != "work"})
