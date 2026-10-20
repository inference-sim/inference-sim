"""Replay the FIT agentic points with the r7 build into work-perreq/ (cov.py's pipeline, separate work dir).

The r7 plan holds HELD points only; this driver takes the FIT agentic lines of the round-1 plan
(work/plan.jsonl, same partition) and runs them through cov.cmd_run unchanged, so FIT and HELD
are scored by the same build. Usage: uv run --no-project python -I run_fit_agentic.py [cov.py run flags]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cov  # noqa: E402

cov.WORK = cov.PX.ROOT / "work-perreq"
cov.WORK.mkdir(parents=True, exist_ok=True)
src = cov.PX.ROOT / "work" / "plan.jsonl"
rows = [json.loads(ln) for ln in src.open()]
rows = [x for x in rows if x["set"] == "FIT" and x["scenario"] == "agentic_traces"]
with (cov.WORK / "plan.jsonl").open("w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
sys.argv = [sys.argv[0], "run", "--set", "FIT", "--width", "1", "--batch", "20"] + sys.argv[1:]
cov.main()
