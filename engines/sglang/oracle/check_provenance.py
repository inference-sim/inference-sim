# Original to blis-sglang.
"""The trace provenance check (FID-2, SGL-1): fail unless every committed golden trace records the
environment this tree would reproduce it in.

For each golden in the golden directories and golden files of oracle/harness.json (each case of a
golden file is one golden), the provenance block must be of the trace format (oracle/traces.py),
and:

- sglang_commit is the commit provenance.go pins;
- requirements_lock_sha256 is the sha256 of oracle/requirements.lock;
- mock_version is oracle/mocks.json's version;
- the scenario is a file of this tree: the one oracle_inputs entry whose sha256 is the recorded
  scenario_sha256, and that file still has that sha256;
- seed is that scenario's seed.

A golden that fails was recorded in an environment the tree no longer has. Run the scenarios in CI
and refresh the goldens' provenance from the fresh traces (oracle/refresh_provenance.py), or, if
their decisions changed, re-record them with a golden-reviewed trailer (FID-3). No golden at all
fails too (X16). The check reads files only; it needs neither SGLang nor its environment.

Usage: python3 -I oracle/check_provenance.py [<repository root>]
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import traces  # noqa: E402  (oracle/traces.py; -I leaves the script directory off sys.path)


def pinned_commit(root: str) -> str:
    with open(os.path.join(root, "provenance.go")) as f:
        m = re.search(r'^\tCommit:\s*"([0-9a-f]{40})",$', f.read(), re.M)
    if not m:
        raise SystemExit("provenance.go: no Upstream Commit of 40 hex digits")
    return m.group(1)


def file_sha256(path: str):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return None


def goldens(root: str) -> list:
    """(label, trace) for every golden oracle/harness.json names."""
    with open(os.path.join(root, "oracle", "harness.json")) as f:
        cfg = json.load(f)
    out = []
    for d in cfg["golden_dirs"]:
        top = os.path.join(root, d)
        for dirpath, _, names in sorted(os.walk(top)):
            for n in sorted(names):
                if n.endswith(".json"):
                    path = os.path.join(dirpath, n)
                    with open(path) as f:
                        out.append((os.path.relpath(path, root).replace(os.sep, "/"), json.load(f)))
    for rel in cfg["golden_files"]:
        with open(os.path.join(root, rel)) as f:
            doc = json.load(f)
        cases = doc.get("cases") if isinstance(doc, dict) else None
        if not isinstance(cases, list):
            out.append((rel, None))
            continue
        out += [(f"{rel} case {i}", case) for i, case in enumerate(cases)]
    return out


def problems(label: str, trace, root: str, want: dict) -> list:
    bad = traces.trace_problems(trace)
    if bad:
        return [f"{label}: {p}" for p in bad]
    prov = trace["provenance"]
    for key, value in want.items():
        if prov[key] != value:
            bad.append(f"{label}: records {key} {prov[key]!r}, but this tree's is {value!r}")
    named = sorted(p for p, h in prov["oracle_inputs"].items() if h == prov["scenario_sha256"])
    if len(named) != 1:
        bad.append(f"{label}: its oracle_inputs name {len(named)} files with the scenario's sha256 "
                   f"{prov['scenario_sha256']}, not one, so its scenario is not a file of this tree")
        return bad
    scenario = named[0]
    now = file_sha256(os.path.join(root, scenario))
    if now != prov["scenario_sha256"]:
        bad.append(f"{label}: its scenario {scenario} has changed since it was recorded"
                   + (" (the file is gone)" if now is None else f" (sha256 {now})"))
        return bad
    with open(os.path.join(root, scenario)) as f:
        seed = json.load(f).get("seed")
    if prov["seed"] != seed:
        bad.append(f"{label}: records seed {prov['seed']!r}, but its scenario {scenario} has seed {seed!r}")
    return bad


def main(root: str) -> int:
    root = os.path.abspath(root)
    with open(os.path.join(root, "oracle", "mocks.json")) as f:
        mock_version = json.load(f)["version"]
    want = {
        "sglang_commit": pinned_commit(root),
        "requirements_lock_sha256": file_sha256(os.path.join(root, "oracle", "requirements.lock")),
        "mock_version": mock_version,
    }
    found = goldens(root)
    if not found:
        print("check_provenance: no golden trace in the golden directories or files of oracle/harness.json")
        return 1
    bad = [p for label, trace in found for p in problems(label, trace, root, want)]
    for b in bad:
        print(b)
    print(f"check_provenance: {len(found)} golden(s), {len(bad)} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1] if len(sys.argv) == 2 else os.path.dirname(HERE)))
