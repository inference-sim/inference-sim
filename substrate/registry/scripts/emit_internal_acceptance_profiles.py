#!/usr/bin/env python3
"""Generate acceptance/internal-deployment-measurements.yaml from derived values of internal
deployment measurements.

    emit_internal_acceptance_profiles.py [--check]

The input is validator/testdata/internal/muse-dspark-vllm-gb200.json: the engine's
speculative-decoding counters (drafts verified, and drafts accepted at each draft position),
summed over each of two sweeps of one deployment. Nothing else is read but
the catalog, which must hold the model and the drafter the file names. Each sweep's per-position
rate is accepted_at_position[i] / drafts; the profile pools the two sweeps (same engine image,
same speculative configuration) by summing the counters. The acceptance
length is 1 plus the sum of the rates, the identity the schema states for per_position_rates
(AL - 1 = sum of p_i).

Nothing is typed in here but the facts of the measurement that no counter carries (THINKING,
CATEGORY, TEMPERATURE), each with its evidence. `--check` re-renders and exits non-zero if the
committed file would change.

$BLIS_CATALOG names the catalog; an empty value means the default.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "acceptance" / "internal-deployment-measurements.yaml"
SET_NAME = "internal-deployment-measurements"
INPUT = REPO / "validator" / "testdata" / "internal" / "muse-dspark-vllm-gb200.json"
CATALOG = Path(os.environ.get("BLIS_CATALOG") or REPO.parent / "catalog")
SOURCE = "internal deployment measurements"

# Facts of the measurement that no engine counter carries, each with its evidence.
#
# THINKING: the sweep replayed recorded agentic requests that carry no switch turning
# reasoning off, against a server launched with its reasoning parser and effort map, so the
# model reasoned at its default. A reasoning-only reply that reached max_tokens with empty
# content was observed in the same deployment's profiling phase. The counters are not split
# by request, so no thinking-off profile exists.
THINKING = "on"  # the file states it too; they must agree
# CATEGORY: the workload is an agentic coding replay , which is
# a coding workload but not SPEED-Bench's coding prompts, the category InferenceX's profiles
# carry. The key spells it "coding", as the profile set a run selects by.
CATEGORY = "coding"
# TEMPERATURE: the replayed requests state no sampling, so the server's default applies. The
# engine's default is 1.0 unless the checkpoint's generation_config sets another; that file was
# not inspected, so this is the engine default and not a verified value.
TEMPERATURE = 1.0


def rates(sweeps: list[dict], k: int) -> list[float]:
    drafts = sum(s["drafts"] for s in sweeps)
    for s in sweeps:
        if len(s["accepted_at_position"]) != k:
            raise SystemExit(f"sweep {s['id']}: {len(s['accepted_at_position'])} positions, K is {k}")
    out = [sum(s["accepted_at_position"][i] for s in sweeps) / drafts for i in range(k)]
    if any(a < b for a, b in zip(out, out[1:])):
        raise SystemExit("pooled rates are not non-increasing; the counters are not cumulative")
    return [round(r, 4) for r in out]


def sweep_al(s: dict) -> float:
    return 1 + sum(s["accepted_at_position"]) / s["drafts"]


def check_catalog(d: dict, catalog: Path) -> str:
    names = {p.parent.name for p in (catalog / "models").glob("*/model.yaml")}
    for key in ("model", "drafter"):
        if d[key] not in names:
            raise SystemExit(f"{key} {d[key]!r} is not in the catalog at {catalog}")
    graph = yaml.safe_load((catalog / "models" / d["model"] / "graph.yaml").read_text())
    return graph["global"]["architecture"]


def generate(input_path: Path = INPUT, catalog: Path = CATALOG) -> str:
    d = json.loads(input_path.read_text())
    k = d["num_spec_tokens"]
    arch = check_catalog(d, catalog)
    r = rates(d["sweeps"], k)
    al = round(1 + sum(r), 3)
    lo, hi = sorted(sweep_al(s) for s in d["sweeps"])
    if (d["thinking"], d["category"]) != (THINKING, CATEGORY):
        raise SystemExit("the counters file and the emitter disagree on thinking mode or category")
    caveats = [
        f"Internal deployment measurements of one deployment: {d['model']} with the {d['drafter']} "
        f"drafter on vLLM, {d['hardware']}. The profile pools two sweeps of it. "
        f"One value, at K {k}.",
        f"Covers K {k}, thinking {THINKING} and category {CATEGORY} only. It has no value at "
        f"another K, none with thinking off, and none for another category: a run at "
        f"another K borrows from it by the registry's same-drafter step, and one with "
        f"thinking off or another category is refused unless a fallback is stated.",
        f"The two sweeps give acceptance lengths {lo:.2f} and {hi:.2f}; the pooled value is "
        f"{al}. Acceptance varies with load, so a run at one load can sit outside that range: "
        "the profile is the mean over each whole sweep.",
        f"Thinking {THINKING} is the server's default reasoning, not a recorded switch: the "
        "replayed requests carried no setting that turned reasoning off, and the counters are "
        "not split by request.",
        "The workload is an agentic coding replay, which is a "
        "coding workload but not SPEED-Bench's coding prompts that InferenceX's profiles use; "
        "the two are not interchangeable.",
        f"Sampling temperature {TEMPERATURE} is the engine's default for a request that states "
        "none; the checkpoint's own generation configuration was not inspected.",
        "Per-position rates are the pooled drafts accepted at each position over the drafts "
        "verified, under adaptive verification; the engine's logged acceptance rate divides by "
        "the drafts actually proposed and differs from (AL-1)/K.",
    ]
    profile = {
        "model": d["model"], "drafter": d["drafter"],
        "draft_sampler": d["settings"]["draft_sample_method"],
        "verifier": d["settings"]["rejection_sample_method"],
        "engine": d["engine"], "architecture": arch,
        "speculative_method": d["speculative_method"], "num_spec_tokens": k,
        "thinking": THINKING, "category": CATEGORY,
        "acceptance_length": al, "per_position_rates": r, "method": "measured",
        "provenance": {
            "source": SOURCE,
            "benchmark": d["workload"],
            "hardware": d["hardware"],
            "engine_image": d["engine_image"],
            "settings": {"engine": d["engine"], "draft_sample_method": d["settings"]["draft_sample_method"],
                         "rejection_sample_method": d["settings"]["rejection_sample_method"],
                         "adaptive": d["settings"]["enable_adaptive_verification"],
                         "drafts_verified": f"{sum(s['drafts'] for s in d['sweeps']):.10g}"},
            "sampling": {"temperature": TEMPERATURE},
        },
    }
    doc = {"kind": "AcceptanceProfileSet", "name": SET_NAME, "caveats": caveats,
           "profiles": [profile]}
    header = (
        "# Speculative acceptance, from internal deployment measurements.\n"
        "#\n"
        "# GENERATED by scripts/emit_internal_acceptance_profiles.py from\n"
        "# validator/testdata/internal/muse-dspark-vllm-gb200.json.\n"
        "# Re-run it and commit the diff rather than editing values here;\n"
        "# `emit_internal_acceptance_profiles.py --check` is the drift gate.\n")
    return header + yaml.dump(doc, sort_keys=False, width=88, allow_unicode=True,
                              default_flow_style=False)


def main(argv: list[str]) -> int:
    text = generate()
    if argv[1:] == ["--check"]:
        if not TARGET.is_file() or TARGET.read_text() != text:
            print(f"{TARGET} differs from what emit_internal_acceptance_profiles.py generates",
                  file=sys.stderr)
            return 1
        return 0
    if argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    TARGET.parent.mkdir(exist_ok=True)
    TARGET.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
