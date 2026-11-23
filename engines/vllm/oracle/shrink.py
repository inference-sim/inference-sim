# Original to blis-vllm.
"""Shrink a scenario on which blis-vllm diverges from vLLM to a smallest one that still does.

Greedy delta debugging (#2021 B8: "each failure is shrunk and kept as a permanent test"):
drop requests one at a time, then shorten prompts and outputs, keeping each change only if the
Go side still diverges from vLLM's trace for the changed scenario. Each candidate runs through
vLLM (run_scheduler.py) and through the Go scheduler (go test -run TestScenarioFromEnv).

Usage: python -I oracle/shrink.py <vllm checkout> <failing scenario.json> <out.json>
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def diverges(vllm_root: str, sc: dict) -> bool:
    with tempfile.TemporaryDirectory() as d:
        sp, tp = os.path.join(d, "s.json"), os.path.join(d, "t.json")
        with open(sp, "w") as f:
            json.dump(sc, f)
        with open(tp, "w") as f:
            r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_scheduler.py"), vllm_root, sp],
                               stdout=f, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            return False  # vLLM refused the candidate (e.g. an invalid config): not a reproducer
        env = dict(os.environ, BLIS_VLLM_SCENARIO=sp, BLIS_VLLM_TRACE=tp)
        r = subprocess.run(["go", "test", "./core/sched/", "-run", "TestScenarioFromEnv$", "-count=1"],
                           cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode != 0


def main() -> None:
    vllm_root, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(src) as f:
        sc = json.load(f)
    if not diverges(vllm_root, sc):
        sys.exit("the scenario does not diverge; nothing to shrink")
    changed = True
    while changed:
        changed = False
        for i in range(len(sc["requests"]) - 1, -1, -1):
            cand = copy.deepcopy(sc)
            del cand["requests"][i]
            if cand["requests"] and diverges(vllm_root, cand):
                sc, changed = cand, True
        for i in range(len(sc["requests"])):
            for key in ("output", "prompt"):
                while len(sc["requests"][i][key]) > 1:
                    cand = copy.deepcopy(sc)
                    cand["requests"][i][key] = cand["requests"][i][key][:-1]
                    if not diverges(vllm_root, cand):
                        break
                    sc, changed = cand, True
    sc["name"] = os.path.splitext(os.path.basename(out))[0]
    sc["description"] = f"Shrunk from {os.path.basename(src)}: a permanent regression test."
    with open(out, "w") as f:
        json.dump(sc, f, indent=1)
    print(f"shrunk to {len(sc['requests'])} requests -> {out}")


if __name__ == "__main__":
    main()
