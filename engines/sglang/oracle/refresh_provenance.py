# Original to blis-sglang.
"""Refresh committed golden traces' provenance from fresh traces SGLang produced in CI.

A golden's decisions are SGLang's output for its scenario; its `provenance` block records where that
output came from. When the environment a golden records is replaced (a new manifest version, a new
committed lock, CI's platform), the `oracle` job of .github/workflows/ci.yml runs every scenario
again and uploads the traces (artifact `oracle-traces`). This copies each fresh trace's provenance
into its golden, and only that. It refuses, and leaves the golden unchanged:

- a fresh trace whose decisions differ from the golden's in any key (that needs a re-recorded
  golden and a golden-reviewed trailer, FID-3);
- a fresh trace of another scenario file, or of another seed;
- a fresh trace or golden that is not of the trace format (oracle/traces.py).

It rewrites goldens, so it runs only with BLIS_SGLANG_ALLOW_GOLDEN_UPDATE=1 (FID-3).
oracle/check_goldens.py accepts a commit that changes only provenance blocks without a trailer;
oracle/check_provenance.py then checks the refreshed blocks against the tree.

Usage: BLIS_SGLANG_ALLOW_GOLDEN_UPDATE=1 python3 -I oracle/refresh_provenance.py <golden dir> <fresh dir>
  e.g. oracle/testdata/golden oracle-traces
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import traces  # noqa: E402  (oracle/traces.py; -I leaves the script directory off sys.path)

SUFFIX = ".trace.json"


def main(golden_dir: str, fresh_dir: str) -> int:
    if os.environ.get(traces.UPDATE_FLAG) != "1":
        print(traces.REFUSAL, file=sys.stderr)
        return 2
    names = sorted(f for f in os.listdir(golden_dir) if f.endswith(SUFFIX)) if os.path.isdir(golden_dir) else []
    if not names:
        print(f"{golden_dir}: no goldens")
        return 1
    bad = refreshed = 0
    for name in names:
        fresh_path = os.path.join(fresh_dir, name)
        if not os.path.exists(fresh_path):
            print(f"{name}: no fresh trace at {fresh_path}")
            bad += 1
            continue
        gpath = os.path.join(golden_dir, name)
        with open(gpath) as f:
            golden = json.load(f)
        with open(fresh_path) as f:
            fresh = json.load(f)
        malformed = [f"the golden: {p}" for p in traces.trace_problems(golden)]
        malformed += [f"the fresh trace: {p}" for p in traces.trace_problems(fresh)]
        if malformed:
            print(f"{name}: " + "; ".join(malformed))
            bad += 1
            continue
        g = {k: v for k, v in golden.items() if k != "provenance"}
        n = {k: v for k, v in fresh.items() if k != "provenance"}
        if g != n:
            keys = sorted(k for k in set(g) | set(n) if g.get(k) != n.get(k))
            print(f"{name}: the fresh trace's decisions differ ({', '.join(keys)}); provenance not refreshed")
            bad += 1
            continue
        gp, fp = golden["provenance"], fresh["provenance"]
        held = [k for k in ("scenario_sha256", "seed") if gp[k] != fp[k]]
        if held:
            print(f"{name}: the fresh trace ran another scenario ({', '.join(held)} differ); provenance not refreshed")
            bad += 1
            continue
        golden["provenance"] = fp
        with open(gpath, "w") as f:
            f.write(traces.dumps(golden))
        refreshed += 1
    print(f"{golden_dir}: {refreshed} provenance block(s) refreshed, {bad} refused")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
