"""Re-score named points with the current spec.py into a separate work dir (block-size parity fix A/B).

Usage: uv run --no-project python -I run_blockfix.py <out work dir name> <src plan.jsonl> <point_id>... [-- cov.py run flags]
The source work dir is untouched: a fresh work dir gets a plan holding only the named points.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cov  # noqa: E402

args = sys.argv[1:]
extra = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
out, src, ids = args[0], Path(args[1]), set(args[2:])
cov.WORK = cov.PX.ROOT / out
cov.WORK.mkdir(parents=True, exist_ok=True)
rows = [json.loads(ln) for ln in src.open()]
rows = [x for x in rows if x["point_id"] in ids]
with (cov.WORK / "plan.jsonl").open("w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
print(f"{len(rows)} points -> {cov.WORK}", flush=True)
sys.argv = [sys.argv[0], "run", "--width", "1", "--batch", "50"] + extra
cov.main()
