"""The internal-deployment acceptance set is its counters, transcribed and nothing else.

`emit_internal_acceptance_profiles.py --check` is the drift gate. These tests re-derive the
profile's rates from the committed counters by a second path, and check that the emitter
refuses a drafter the catalog does not hold and notices a changed counter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "emit_internal_acceptance_profiles.py"
SET = yaml.safe_load((REPO / "acceptance" / "internal-deployment-measurements.yaml").read_text())
COUNTERS = REPO / "validator" / "testdata" / "internal" / "muse-dspark-vllm-gb200.json"

sys.path.insert(0, str(REPO / "scripts"))
import emit_internal_acceptance_profiles as emit  # noqa: E402


def test_the_committed_set_is_what_the_emitter_generates():
    done = subprocess.run([sys.executable, "-I", str(SCRIPT), "--check"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_rates_are_the_pooled_counters():
    d = json.loads(COUNTERS.read_text())
    drafts = sum(s["drafts"] for s in d["sweeps"])
    [p] = SET["profiles"]
    for i, rate in enumerate(p["per_position_rates"]):
        accepted = sum(s["accepted_at_position"][i] for s in d["sweeps"])
        assert rate == pytest.approx(accepted / drafts, abs=5e-5)
    assert p["acceptance_length"] == pytest.approx(1 + sum(p["per_position_rates"]), abs=1e-3)


def test_the_profile_records_how_it_was_measured():
    [p] = SET["profiles"]
    assert (p["num_spec_tokens"], p["thinking"], p["category"]) == (7, "on", "coding")
    assert (p["draft_sampler"], p["verifier"], p["engine"]) == ("probabilistic", "block", "vllm")
    assert p["provenance"]["hardware"] == "gb200-nvl72"
    assert p["provenance"]["engine_image"]
    assert p["provenance"]["source"] == "internal deployment measurements"


def test_a_changed_counter_is_drift(tmp_path):
    d = json.loads(COUNTERS.read_text())
    d["sweeps"][0]["accepted_at_position"][0] *= 0.5
    changed = tmp_path / "counters.json"
    changed.write_text(json.dumps(d))
    assert emit.generate(changed) != (REPO / "acceptance" / "internal-deployment-measurements.yaml").read_text()


def test_a_drafter_the_catalog_lacks_is_refused(tmp_path):
    d = json.loads(COUNTERS.read_text())
    d["drafter"] = "no-such-drafter"
    bad = tmp_path / "counters.json"
    bad.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="no-such-drafter"):
        emit.generate(bad)
