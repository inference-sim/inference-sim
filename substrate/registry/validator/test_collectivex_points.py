"""The digest the CollectiveX entries cite covers the points the fit reads, and nothing else.

The entries once cited the sha256 of a run body's bytes. The API then re-served every cited body
with different bytes and the same points, and the re-derivation gate failed for a change that
moved no value. These tests pin the two halves of the replacement: a change of serialisation
leaves the digest alone, and a change of any measurement the fit reads moves it.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import collectivex_points as cp  # noqa: E402


def _series(sku="h200-dgxc", mode="normal", ep=16, nodes=2, phase="decode", pts=((1, 10.0), (4, 12.5))):
    return {
        "series_id": f"{sku}-{mode}-{phase}", "backend": "deepep-v2", "precision": "bf16",
        "mode": mode, "phase": phase, "library_version": "2.1.0+local@01dc3aa", "maturity": "production",
        "system": {"sku": sku, "ep_size": ep, "nodes": nodes, "vendor": "nvidia"},
        "points": [{"tokens_per_rank": t, "components": {"roundtrip": {"latency_us": {"p50": v, "p99": v * 2}}}}
                   for t, v in pts],
    }


BODY = {"version": 1, "run": {"run_id": "1", "generated_at": "2026-10-28T00:00:00Z"},
        "series": [_series(), _series(mode="low-latency", pts=((2, 7.0),))], "kv": [], "swap_blocks": []}


def test_digest_is_a_sha256():
    assert len(cp.digest(BODY)) == 64 and int(cp.digest(BODY), 16) >= 0


def test_a_change_of_serialisation_leaves_the_digest_alone():
    d = cp.digest(BODY)
    reordered = copy.deepcopy(BODY)
    reordered["series"].reverse()
    reordered["series"][0]["points"].reverse()
    assert cp.digest(reordered) == d
    assert cp.digest_bytes(json.dumps(BODY, indent=2, sort_keys=True).encode()) == d
    assert cp.digest_bytes(json.dumps(BODY, separators=(",", ":")).encode()) == d


def test_fields_the_fit_does_not_read_leave_the_digest_alone():
    d = cp.digest(BODY)
    other = copy.deepcopy(BODY)
    other["run"]["generated_at"] = "2026-11-07T00:00:00Z"
    other["series"][0]["maturity"] = "preview"
    other["series"][0]["points"][0]["components"]["roundtrip"]["latency_us"]["p99"] = 1e9
    other["kv"] = [{"anything": 1}]
    assert cp.digest(other) == d


@pytest.mark.parametrize("edit", [
    lambda b: b["series"][0]["points"][0]["components"]["roundtrip"]["latency_us"].update(p50=10.01),
    lambda b: b["series"][0]["points"][0].update(tokens_per_rank=2),
    lambda b: b["series"][0]["system"].update(ep_size=8),
    lambda b: b["series"][0]["system"].update(nodes=4),
    lambda b: b["series"][0]["system"].update(sku="gb200"),
    lambda b: b["series"][0].update(mode="low-latency"),
    lambda b: b["series"][0].update(phase="prefill"),
    lambda b: b["series"][0].update(library_version="2.2.0"),
    lambda b: b["series"][0]["points"].pop(),
    lambda b: b["series"].pop(),
], ids=["p50", "tokens", "ep_size", "nodes", "sku", "mode", "phase", "library", "point-dropped", "series-dropped"])
def test_a_change_to_what_the_fit_reads_moves_the_digest(edit):
    other = copy.deepcopy(BODY)
    edit(other)
    assert cp.digest(other) != cp.digest(BODY)


def test_series_of_another_backend_or_precision_are_not_covered():
    d = cp.digest(BODY)
    other = copy.deepcopy(BODY)
    other["series"].append({**_series(), "backend": "nccl"})
    other["series"].append({**_series(), "precision": "fp8"})
    assert cp.digest(other) == d


def test_a_series_missing_a_field_the_fit_reads_is_an_error_not_a_skip():
    other = copy.deepcopy(BODY)
    del other["series"][0]["points"][0]["components"]["roundtrip"]
    with pytest.raises(KeyError):
        cp.digest(other)
