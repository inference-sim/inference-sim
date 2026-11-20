# Original to blis-sglang.
"""The trace format every oracle runner writes, and the rules for writing one (FID-2, FID-3).

A trace is one JSON object. Its `provenance` block says where the trace came from, so anyone can
reproduce it; every other key is what the upstream decided, and what the Go port is compared on.
A runner builds the block with provenance() and writes the trace with write():

- provenance() records the SGLang commit of the checkout, the sha256 of the committed lock
  (oracle/requirements.lock) and of the environment's `pip freeze`, the scenario's sha256 and seed,
  the manifest version (oracle/mocks.json), the model and its revision, the platform, this
  repository's commit, and the sha256 of every oracle file the run read (`oracle_inputs`). Every
  value is required: a run that cannot read one fails rather than record a placeholder (X16).
- write() refuses a path inside a golden directory or golden file of oracle/harness.json, or under
  any testdata/golden* directory, unless BLIS_SGLANG_ALLOW_GOLDEN_UPDATE=1. A commit that then
  changes what a golden decided needs a `golden-reviewed: <trace>: <reason>` trailer
  (oracle/check_goldens.py).

provenance_problems() is the format check, shared by write(), compare_goldens.py,
check_provenance.py and refresh_provenance.py. Standard library only.
"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UPDATE_FLAG = "BLIS_SGLANG_ALLOW_GOLDEN_UPDATE"

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def _hex40(v):
    return isinstance(v, str) and bool(HEX40.fullmatch(v))


def _hex64(v):
    return isinstance(v, str) and bool(HEX64.fullmatch(v))


def _text(v):
    return isinstance(v, str) and bool(v.strip())


def _count(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _repo_commit(v):
    # A run from a tree that differs from its commit in a file it read is marked, and check_goldens.py
    # refuses such a golden; the format allows the mark so the fresh trace still says so.
    return isinstance(v, str) and bool(re.fullmatch(r"[0-9a-f]{40}(\+dirty)?", v))


def _inputs(v):
    return (isinstance(v, dict) and bool(v)
            and all(isinstance(k, str) and k.startswith("oracle/") and _hex64(h) for k, h in v.items()))


# Each key a provenance block holds, and what its value must be. No other key is allowed.
PROVENANCE = {
    "tier": (_count, "a tier number"),
    "sglang_commit": (_hex40, "a full SGLang commit id"),
    "requirements_lock_sha256": (_hex64, "the sha256 of oracle/requirements.lock"),
    "lockfile_sha256": (_hex64, "the sha256 of the environment's pip freeze"),
    "scenario_sha256": (_hex64, "the sha256 of the scenario file"),
    "seed": (_count, "the scenario's seed"),
    "mock_version": (lambda v: _text(v) and v.startswith("gpu-mock/"), "oracle/mocks.json's version"),
    "model": (_text, "the scenario's model"),
    "model_revision": (_text, "the model's revision"),
    "python": (_text, "the Python version"),
    "platform": (_text, "the platform, <system>-<machine>"),
    "blis_sglang_commit": (_repo_commit, "this repository's full commit id"),
    "oracle_inputs": (_inputs, "each oracle/ file the run read, with its sha256"),
}


def provenance_problems(prov) -> list:
    """What makes prov not a provenance block of this format, each naming its key."""
    if not isinstance(prov, dict):
        return [f"provenance is a {type(prov).__name__}, not an object"]
    bad = [f"provenance.{k}: missing" for k in sorted(set(PROVENANCE) - set(prov))]
    bad += [f"provenance.{k}: not a key of the trace format" for k in sorted(set(prov) - set(PROVENANCE))]
    for key, (ok, what) in sorted(PROVENANCE.items()):
        if key in prov and not ok(prov[key]):
            bad.append(f"provenance.{key}: {prov[key]!r} is not {what}")
    return bad


def trace_problems(trace) -> list:
    """What makes trace not a trace of this format: an object with a well-formed provenance block
    and at least one other key."""
    if not isinstance(trace, dict):
        return [f"the trace is a {type(trace).__name__}, not an object"]
    if "provenance" not in trace:
        return ["the trace has no provenance block"]
    bad = provenance_problems(trace["provenance"])
    if len(trace) < 2:
        bad.append("the trace records nothing beside its provenance")
    return bad


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(root: str, *args) -> str:
    out = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"git {' '.join(args)} in {root} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def provenance(tier: int, sglang_root: str, scenario_path: str, scenario: dict, inputs: tuple) -> dict:
    """The provenance block of a run of scenario_path at tier, which read the repository files
    inputs (paths relative to the repository root). ORACLE_MODEL_REVISION and ORACLE_LOCKFILE are
    required, and so is the scenario's model and seed."""
    revision, lockfile = os.environ.get("ORACLE_MODEL_REVISION"), os.environ.get("ORACLE_LOCKFILE")
    if not revision or not lockfile:
        raise SystemExit("ORACLE_MODEL_REVISION and ORACLE_LOCKFILE must both be set")
    for key in ("model", "seed"):
        if key not in scenario:
            raise SystemExit(f"{scenario_path}: the scenario has no {key}, which its trace must record")
    with open(os.path.join(HERE, "mocks.json")) as f:
        mocks = json.load(f)
    read = {rel: sha256(os.path.join(ROOT, rel)) for rel in inputs}
    if os.path.abspath(scenario_path).startswith(ROOT + os.sep):
        read[os.path.relpath(os.path.abspath(scenario_path), ROOT).replace(os.sep, "/")] = sha256(scenario_path)
    head = git(ROOT, "rev-parse", "HEAD")
    # Dirty only if a file the run used differs from head; a golden refreshed in place is not one.
    dirty = git(ROOT, "status", "--porcelain", "--", *read)
    prov = {
        "tier": tier,
        "sglang_commit": git(sglang_root, "rev-parse", "HEAD"),
        "scenario_sha256": sha256(scenario_path),
        "mock_version": mocks["version"],
        "model": scenario["model"],
        "model_revision": revision,
        "python": sys.version.split()[0],
        "platform": platform.system().lower() + "-" + platform.machine(),
        "lockfile_sha256": sha256(lockfile),
        "requirements_lock_sha256": sha256(os.path.join(HERE, "requirements.lock")),
        "blis_sglang_commit": head + ("+dirty" if dirty else ""),
        "oracle_inputs": read,
        "seed": scenario["seed"],
    }
    bad = provenance_problems(prov)
    if bad:
        raise SystemExit("the run's provenance is incomplete: " + "; ".join(bad))
    return prov


def golden_targets(root: str = ROOT) -> tuple:
    """The golden directories and golden files oracle/harness.json names, as absolute paths."""
    with open(os.path.join(root, "oracle", "harness.json")) as f:
        cfg = json.load(f)
    dirs = tuple(os.path.join(root, d.rstrip("/")) + os.sep for d in cfg.get("golden_dirs", []))
    files = tuple(os.path.join(root, p) for p in cfg.get("golden_files", []))
    return dirs, files


def is_golden_path(path: str, root: str = ROOT) -> bool:
    """Whether writing path would write a golden: a path inside a golden directory or a golden
    file of oracle/harness.json, or under any testdata/golden* directory."""
    p = os.path.realpath(path)
    dirs, files = golden_targets(root)
    if p.startswith(tuple(os.path.realpath(d.rstrip(os.sep)) + os.sep for d in dirs)):
        return True
    if p in {os.path.realpath(f) for f in files}:
        return True
    parts = p.replace(os.sep, "/").split("/")
    return any(a == "testdata" and b.startswith("golden") for a, b in zip(parts, parts[1:]))


def refuses_write(path: str, root: str = ROOT) -> bool:
    return bool(path) and is_golden_path(path, root) and os.environ.get(UPDATE_FLAG) != "1"


REFUSAL = (f"refusing to write a golden trace: set {UPDATE_FLAG}=1, and add a "
           "golden-reviewed: <trace>: <reason> trailer for each trace the commit changes")


def dumps(trace: dict) -> str:
    return json.dumps(trace, indent=1, sort_keys=True) + "\n"


def write(out: str, trace: dict, root: str = ROOT) -> None:
    """Write trace to out, or to standard output when out is empty. Refuses a trace not of this
    format, and a golden path without the update flag."""
    bad = trace_problems(trace)
    if bad:
        raise SystemExit("not a trace: " + "; ".join(bad))
    if refuses_write(out, root):
        raise SystemExit(REFUSAL)
    if out:
        with open(out, "w") as f:
            f.write(dumps(trace))
    else:
        sys.stdout.write(dumps(trace))
