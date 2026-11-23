"""The engine-path coefficients are keyed by the code that does the work (M3.2 Table W row 6).

Every entry of cost-model-host-overheads.yaml prices work in an engine's host, scheduler or launch
path, so each states the chips' host CPU (scope.host_cpu) and the CodeIDs of the two engines'
code-path tables (code_ids, inputs/engine_code_ids.json). What is checked:

  - nothing else changed: every value, unit, method, interval and transfer is carried to its new key
    unchanged, against a snapshot of the file before it was keyed (testdata/host_overheads_before_code_keys.json);
  - the keys are the ones the catalog and the engines state, so a refit engine or a new chip cannot
    leave a stale key behind (the writer's --check is the same comparison);
  - every entry is keyed, and keyed by the path its name prices.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REG = Path(__file__).resolve().parent.parent
INPUT = json.loads((REG / "inputs" / "engine_code_ids.json").read_text())
SNAPSHOT = json.loads((Path(__file__).parent / "testdata" / "host_overheads_before_code_keys.json").read_text())
CURRENT = yaml.safe_load((REG / "coefficients" / "cost-model-host-overheads.yaml").read_text())["coefficients"]

HOST = {"host_admission_per_request", "host_admission_per_token", "host_admission_per_token_rust_frontend",
        "host_output_token", "host_completion", "host_step_per_request"}
GRAPHS = {"host_launch_eager_per_layer", "host_launch_eager_per_kernel", "host_launch_per_kernel",
          "host_replay_graph_per_step"}


# Entries the file gained after the snapshot was taken. Each is checked by the other tests here (keyed,
# scoped to its chips' host CPU) but has no snapshot value to be carried unchanged from.
ADDED_AFTER_SNAPSHOT = {
    ("host_launch_eager_per_kernel", ("h200",), ("deepseek-v4-pro",), "calibrated"):
        "row 29: DeepSeek-V4-Pro's uncompiled prefill dispatch on H200, fitted on fit-role units only",
}


def _code_id(engine: str, path: str) -> str:
    return INPUT["engines"][engine][path]


def _chip_cpu(chip: str) -> str:
    return yaml.safe_load((REG.parent / "catalog" / "hardware" / f"{chip}.yaml").read_text())["HostCPU"].lower()


def _split(entry: dict) -> tuple[str, dict]:
    ((name, body),) = entry.items()
    return name, body


def _identity(name: str, body: dict) -> tuple:
    sc = body["scope"]
    return (name, tuple(sc["hardware"]), tuple(sc.get("model", ())), body["method"])


def test_every_value_is_carried_to_its_new_key_unchanged():
    before = {_identity(e["name"], e): e for e in SNAPSHOT["entries"]}
    after = {}
    for e in CURRENT:
        name, body = _split(e)
        key = _identity(name, body)
        assert key not in after, f"{key} is stated twice"
        after[key] = body
    assert set(ADDED_AFTER_SNAPSHOT) <= set(after), "a listed later addition is gone"
    assert not set(ADDED_AFTER_SNAPSHOT) & set(before), "a listed later addition is in the snapshot"
    assert set(before) == set(after) - set(ADDED_AFTER_SNAPSHOT), "an entry was added, dropped or re-scoped"
    for key, old in before.items():
        new = dict(after[key])
        new.pop("rationale", None)
        new.pop("sources", None)
        new.pop("code_ids")
        new["scope"] = {k: v for k, v in new["scope"].items() if k != "host_cpu"}
        want = {k: v for k, v in old.items() if k != "name"}
        assert new == want, f"{key}: a field other than the keys changed"


def test_host_cpu_is_the_cpu_of_the_chips_the_entry_names():
    for e in CURRENT:
        name, body = _split(e)
        want = sorted({_chip_cpu(c) for c in body["scope"]["hardware"]})
        assert sorted(body["scope"]["host_cpu"]) == want, f"{name} {body['scope']['hardware']}"


def test_every_entry_is_keyed_by_the_code_path_its_name_prices():
    seen = set()
    for e in CURRENT:
        name, body = _split(e)
        path = "host" if name in HOST else "graphs"
        assert name in HOST | GRAPHS, f"{name} is bound to no code path; bind it in coefficient.CodePathOf"
        want = sorted(_code_id(eng, path) for eng in INPUT["engines"])
        assert sorted(body["code_ids"]) == want, name
        seen.add(name)
    assert seen <= HOST | GRAPHS


def test_the_two_engines_state_different_code_for_the_two_paths():
    ids = {(e, p): _code_id(e, p) for e in INPUT["engines"] for p in ("host", "graphs")}
    assert len(set(ids.values())) == 4, ids
    assert len(INPUT["engines"]) == 2
