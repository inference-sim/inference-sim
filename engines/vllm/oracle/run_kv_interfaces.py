# Original to blis-vllm.
"""Recorded traces at the KV offload interfaces (design FID-5, §7.6): the calls a real feature makes.

Each offloading and multi-connector scenario runs exactly as oracle/run_offloading.py and
oracle/run_multiconnector.py run it, while oracle/kv_interfaces.py records every call the
OffloadingConnector's scheduler side makes at OffloadingManager, every call the manager makes at
its CachePolicy, and each spec's get_manager. Nothing is replaced beyond what those runners
replace (oracle/mocks.json); the recording only observes. core/connector/offloading/fid5_test.go
replays each trace against the port and compares every answer.

Usage:
    python -I oracle/run_kv_interfaces.py <vllm checkout> <out dir> <scenario dir>...
    python -I oracle/run_kv_interfaces.py <vllm checkout> <out dir> <scenario dir>... --check <golden dir>
With --check, each trace written must equal the committed one of the same name (its spec, manager
and policy calls), and the golden directory must hold no trace the run did not write.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "recorded-"


def run_scenario(vllm_root: str, path: str) -> dict:
    sys.path.insert(0, HERE)
    import run_multiconnector  # noqa: E402
    import run_offloading  # noqa: E402
    from kv_interfaces import Monitor  # noqa: E402
    from run_scheduler import provenance  # noqa: E402

    with open(path) as f:
        name = json.load(f)["config"]["connector"]["name"]
    runner = {"OffloadingConnector": run_offloading, "MultiConnector": run_multiconnector}[name]
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    with Monitor() as mon:
        runner.run_one(vllm_root, path)
    prov = provenance(vllm_root, path)
    prov["kind"] = "recorded"
    return {"provenance": prov, **mon.trace}


def pending(golden_dir: str) -> str | None:
    """The golden directory's <dir>.pending.json beside it, when the traces wait for an issue: its reason, or None.
    It stands only for the vLLM release it names, and only while no trace is committed beside it,
    as core/connector/offloading/fid5_test.go also checks."""
    path = os.path.normpath(golden_dir) + ".pending.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        p = json.load(f)
    import vllm  # noqa: E402

    have = [f for f in os.listdir(golden_dir) if f.endswith(".fid5.json")] if os.path.isdir(golden_dir) else []
    if have:
        raise SystemExit("%s: %d trace(s) committed beside %s; remove it" % (golden_dir, len(have), path))
    if not vllm.__version__.startswith(p["release"].lstrip("v")):
        raise SystemExit(path + " defers the traces for %s, and the oracle runs vLLM %s: record them (%s)"
                         % (p["release"], vllm.__version__, p["waits_for"]))
    return "%s (%s)" % (p["waits_for"], p["why"])


def compare(out_dir: str, golden_dir: str, names: list, prefix: str = PREFIX) -> int:
    """Each trace written equals its golden, and every golden of the prefix was written. While the
    goldens are pending, the traces are written (to the run's artifact) and nothing is compared."""
    why = pending(golden_dir)
    if why is not None:
        print("%d %s trace(s) written to %s; not compared: the goldens are pending until %s"
              % (len(names), prefix.rstrip("-"), out_dir, why))
        return 0
    bad = 0
    for n in names:
        with open(os.path.join(out_dir, n)) as f:
            got = json.load(f)
        gp = os.path.join(golden_dir, n)
        if not os.path.exists(gp):
            print("missing golden %s" % gp)
            bad += 1
            continue
        with open(gp) as f:
            want = json.load(f)
        for part in ("spec", "manager", "policy"):
            if got[part] != want[part]:
                print("%s: the %s calls differ from vLLM's own" % (gp, part))
                bad += 1
    have = os.listdir(golden_dir) if os.path.isdir(golden_dir) else []
    extra = sorted(set(f for f in have if f.startswith(prefix)) - set(names))
    for e in extra:
        print("golden %s was not produced by this run" % e)
        bad += 1
    return bad


def main(argv) -> int:
    check = None
    if "--check" in argv:
        i = argv.index("--check")
        check, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    vllm_root, out_dir, dirs = argv[0], argv[1], argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for d in dirs:
        for f in sorted(x for x in os.listdir(d) if x.endswith(".json") and not x.endswith(".trace.json")):
            trace = run_scenario(vllm_root, os.path.join(d, f))
            name = PREFIX + f[:-5] + ".fid5.json"
            with open(os.path.join(out_dir, name), "w") as fh:
                json.dump(trace, fh, indent=1, sort_keys=True)
            names.append(name)
    if check:
        bad = compare(out_dir, check, names)
        if bad:
            print("%d recorded interface trace(s) differ" % bad)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
