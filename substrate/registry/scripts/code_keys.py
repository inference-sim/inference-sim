"""The keys an engine-path coefficient carries: the host CPU of its chips and the CodeIDs of the code it prices.

An engine-path coefficient (host admission, per-token and completion work, the scheduler's per-request
cost, eager, per-kernel and graph-replay launch) is a property of the code the engine runs to do that
work, on the CPU that runs it (M3.2 Table W row 6; design §3.2, Table DC). The registry states both:

  scope.host_cpu  the CPU architectures of the chips the entry is scoped to (the catalog's Chip.HostCPU)
  code_ids        the CodeIDs, one per engine, that the engines' code_paths.json tables state for the
                  code path the entry's name prices, as inputs/engine_code_ids.json carries them

A lookup returns the entry only for a chip whose host CPU it lists and an engine whose plan states one of
its CodeIDs; any other CodeID is refused (VAL-8, no sibling fallback). The writers of the host-overhead set
(emit_host_overheads_scope.py, fit_deepseek_v41_host.py) take both from here, so a new chip or a regenerated
code-path table is a rerun, not a hand edit. The substrate reads no file of an engine module (X7), so the CodeIDs come
from inputs/engine_code_ids.json, which tools/codepaths/export_code_ids.py writes from the tables and each
engine's own test holds to the table it embeds. code_path_of restates coefficient.CodePathOf (Go);
the Go test TestEnginePathEntriesAreKeyedByTheCodeTheyPrice holds the two together.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REGISTRY = HERE.parent
CATALOG = REGISTRY.parent / "catalog" / "hardware"
INPUT = REGISTRY / "inputs" / "engine_code_ids.json"
CPU_ORDER = ("x86", "grace")

HOST_NAMES = ("host_admission_per_request", "host_admission_per_token", "host_admission_per_token_rust_frontend",
              "host_output_token", "host_completion", "host_step_per_request")
GRAPHS_NAMES = ("host_launch_eager_per_layer", "host_launch_eager_per_kernel", "host_launch_per_kernel",
                "host_replay_graph_per_step")


def code_path_of(name: str) -> str:
    if name in HOST_NAMES:
        return "host"
    if name in GRAPHS_NAMES:
        return "graphs"
    raise SystemExit(f"{name} is bound to no engine code path; bind it in coefficient.CodePathOf and code_keys.py")


def chip_host_cpus(catalog: Path = CATALOG) -> dict[str, str]:
    out = {}
    for f in sorted(catalog.glob("*.yaml")):
        cpu = yaml.safe_load(f.read_text()).get("HostCPU")
        if not cpu:
            raise SystemExit(f"{f}: the chip states no HostCPU")
        out[f.stem] = cpu
    return out


def host_cpu_of(chips: list[str], cpus: dict[str, str]) -> list[str]:
    """The host CPUs of these chips, x86 before grace."""
    missing = [c for c in chips if c not in cpus]
    if missing:
        raise SystemExit(f"no HostCPU for {missing}: the catalog does not hold them")
    found = {cpus[c] for c in chips}
    return [c for c in CPU_ORDER if c in found]


def engine_code_ids(path: Path = INPUT) -> dict[str, dict[str, str]]:
    """{code path: {engine: CodeID}} from inputs/engine_code_ids.json."""
    doc = json.loads(path.read_text())
    if doc.get("kind") != "EngineCodeIDs" or not doc.get("engines"):
        raise SystemExit(f"{path}: not an EngineCodeIDs document that names an engine")
    out: dict[str, dict[str, str]] = {"host": {}, "graphs": {}}
    for engine, e in doc["engines"].items():
        for p in out:
            out[p][engine] = e[p]
    return out


def code_ids_of(name: str, ids: dict[str, dict[str, str]]) -> list[str]:
    """The CodeIDs an entry of this name lists: both engines', sorted, equal ids once."""
    return sorted(set(ids[code_path_of(name)].values()))


SCOPE_LINE = re.compile(r"^(      scope: \{hardware: \[)([^\]]*)(\])(, model: \[[^\]]*\])?(, host_cpu: \[[^\]]*\])?(\})$")


def keyed_scope(line: str, cpus: dict[str, str]) -> str:
    """The scope line with its host_cpu stated for the chips it names (replacing any it stated)."""
    m = SCOPE_LINE.match(line)
    if not m:
        raise SystemExit(f"not a scope line this writer can key: {line!r}")
    chips = [c.strip() for c in m.group(2).split(",")]
    return (m.group(1) + m.group(2) + m.group(3) + (m.group(4) or "")
            + ", host_cpu: [" + ", ".join(host_cpu_of(chips, cpus)) + "]" + m.group(6))


def code_ids_line(name: str, ids: dict[str, dict[str, str]]) -> str:
    return "      code_ids: [" + ", ".join(code_ids_of(name, ids)) + "]"
