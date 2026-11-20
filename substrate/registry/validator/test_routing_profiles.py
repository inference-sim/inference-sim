"""The routing-profile set is the capture's FIT-split curves and nothing else.

The capture itself is third-party data read where it was downloaded, so the drift gate
(`emit_routing_profiles.py <derived.json> --check`) runs where the derivation's output is
present. These tests read the committed set and hold it to what the writer promises.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SET = yaml.safe_load((REPO / "routing" / "servingstudio-dsv41-flash.yaml").read_text())


def test_a_tier2_set_is_never_a_default():
    assert SET["role"] == "sensitivity"
    assert all(p["provenance"]["tier"] == 2 for p in SET["profiles"])


def test_every_width_has_a_decode_and_a_prefill_curve():
    keys = {(p["phase"], p["expert_parallel"]) for p in SET["profiles"]}
    assert keys == {(ph, ep) for ph in ("decode", "prefill") for ep in (2, 4, 8, 16)}


def test_the_prefill_proxy_is_not_called_measured():
    for p in SET["profiles"]:
        assert (p["method"] == "assumed") == (p["phase"] == "prefill"), p["expert_parallel"]


def test_imbalance_grows_with_width_and_falls_with_batch():
    """The busiest of more ranks is heavier; a larger step averages more tokens."""
    dec = {p["expert_parallel"]: p["points"] for p in SET["profiles"] if p["phase"] == "decode"}
    for b in range(len(dec[2])):
        row = [dec[ep][b]["token_imbalance"] for ep in (2, 4, 8, 16)]
        assert row == sorted(row), row
    for ep, pts in dec.items():
        tok = [x["token_imbalance"] for x in pts]
        assert tok[0] > tok[-1] >= 1, ep
