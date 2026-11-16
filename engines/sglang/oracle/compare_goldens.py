# Original to blis-sglang.
"""Fail unless every committed golden trace is SGLang's own output: each golden under
oracle/testdata/golden/ equals, outside its provenance block, the trace the oracle runner wrote
for the same scenario in this CI run; it names the pinned SGLang commit; and it records the same
mock version, model, model revision, committed lock and scenario as that run. A scenario with no
golden, a golden with no scenario, and no scenario at all fail (X16).

Usage: python3 -I oracle/compare_goldens.py <dir of fresh traces> <pinned sglang commit> [<scenarios dir> <golden dir>]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "testdata", "golden")
SCENARIOS = os.path.join(HERE, "scenarios")


def first_difference(a, b, path="") -> str:
    if type(a) is not type(b):
        return f"{path or '/'}: {a!r} != {b!r}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: present on one side only"
            d = first_difference(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return ""
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} entries != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}/{i}")
            if d:
                return d
        return ""
    return "" if a == b else f"{path}: {a!r} != {b!r}"


# The provenance a golden must share with a fresh run of its scenario.
PROVENANCE_HELD = ("mock_version", "model", "model_revision", "requirements_lock_sha256", "scenario_sha256")


def main(fresh_dir: str, commit: str, scenarios: str = SCENARIOS, golden_dir: str = GOLDEN) -> int:
    global GOLDEN
    GOLDEN = golden_dir
    bad = []
    names = sorted(n[:-5] for n in os.listdir(scenarios) if n.endswith(".json")) if os.path.isdir(scenarios) else []
    if not names:
        print(f"compare_goldens: no scenario in {scenarios}")
        return 1
    for name in names:
        golden_path = os.path.join(GOLDEN, name + ".trace.json")
        fresh_path = os.path.join(fresh_dir, name + ".trace.json")
        if not os.path.exists(golden_path):
            bad.append(f"{name}: no golden at oracle/testdata/golden/{name}.trace.json")
            continue
        if not os.path.exists(fresh_path):
            bad.append(f"{name}: the runner wrote no trace")
            continue
        with open(golden_path) as f:
            golden = json.load(f)
        with open(fresh_path) as f:
            fresh = json.load(f)
        gp, fp = golden.get("provenance", {}), fresh.get("provenance", {})
        if gp.get("sglang_commit") != commit:
            bad.append(f"{name}: the golden records SGLang {gp.get('sglang_commit')!r}, not {commit}")
        for key in PROVENANCE_HELD:
            if gp.get(key) != fp.get(key):
                bad.append(f"{name}: the golden records {key} {gp.get(key)!r}, but this run's is {fp.get(key)!r}")
        golden.pop("provenance", None)
        fresh.pop("provenance", None)
        d = first_difference(golden, fresh)
        if d:
            bad.append(f"{name}: the golden differs from SGLang's output at {d}")
    for g in sorted(os.listdir(GOLDEN)) if os.path.isdir(GOLDEN) else []:
        if g.endswith(".trace.json") and g[: -len(".trace.json")] not in names:
            bad.append(f"oracle/testdata/golden/{g}: no scenario of that name")
    for b in bad:
        print(b)
    print(f"compare_goldens: {len(names)} scenarios, {len(bad)} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
