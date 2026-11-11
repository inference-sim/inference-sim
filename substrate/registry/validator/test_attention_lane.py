"""The trtllm-gen prefill lane as committed, and its writer's refusals.

A lane exists to price a kernel on its own rows, so what is checked here is what makes it
that: one pair per data-center Blackwell part and no other, each entry honest about not
having a kernel hash, the interval around the value it describes, and a lane that differs
from the pooled pair it replaces. The data-dependent half (the lane's rows are what the
writer says) is skipped when the AISimulate tree is absent; the rest reads only the
committed file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SET = REPO / "coefficients" / "cost-model-attention.yaml"
DATA = Path(os.environ.get(
    "AISIMULATE_DATA", "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data"))

# Chips with a trtllm-gen prefill kernel (SM100/SM103, vllm/utils/flashinfer.py:669-671 at
# vLLM 5df63a72) and a vLLM 0.25.0 context-attention sweep.
LANE_CHIPS = {"b200", "b300", "gb200-nvl72", "gb300"}
KV = ("bf16", "fp8")
NAMES = tuple(f"attention_prefill_{k}_trtllm_gen_kv_{kv}" for k in ("floor", "work_scale") for kv in KV)


def entries():
    out = {}
    for e in yaml.safe_load(SET.read_text())["coefficients"]:
        (name, body), = e.items()
        if name in NAMES:
            out.setdefault(name, []).append(body)
    return out


def pooled():
    out = {}
    for e in yaml.safe_load(SET.read_text())["coefficients"]:
        (name, body), = e.items()
        if name in ("attention_prefill_floor", "attention_prefill_work_scale"):
            for chip in body["scope"]["hardware"]:
                out[(name, chip)] = body["value"]
    return out


def test_the_lane_is_one_pair_per_cache_dtype_per_blackwell_part_and_no_other():
    got = entries()
    assert set(got) == set(NAMES)
    for name, bodies in got.items():
        chips = [c for b in bodies for c in b["scope"]["hardware"]]
        assert sorted(chips) == sorted(LANE_CHIPS), (
            f"{name} is scoped to {sorted(chips)}, not exactly {sorted(LANE_CHIPS)}: a part "
            f"without a measurement of the kernel must have no lane, so a stated true is refused")
        assert all(len(b["scope"]["hardware"]) == 1 and set(b["scope"]) == {"hardware"}
                   for b in bodies)


def test_the_lane_does_not_claim_a_kernel_hash_or_a_fit_it_cannot_show():
    """D5: no hash exists for these rows, so `transferred` and no kernel_impls; the range is recorded."""
    for bodies in entries().values():
        for b in bodies:
            assert b["method"] == "transferred"
            assert b["fitted"] is False, "the schema allows fitted: true on measured/calibrated only"
            assert "kernel_impls" not in b and "measures" not in b
            dims = {r["dim"]: (r["low"], r["high"]) for r in b["fit_range"]}
            assert set(dims) == {"tokens", "requests", "context_tokens"}, dims
            assert all(lo <= hi for lo, hi in dims.values())
            # No row has a cached prefix, so the context a row reads is its prompt length.
            assert dims["context_tokens"][1] >= dims["requests"][1]
            assert b["transferred_from"] == "hardware=" + b["scope"]["hardware"][0]
            assert "no kernel hash" in b["transfer_rule"]
            assert "vllm_flashinfer_trtllmprefill" in b["sources"][0]["cite"]
            assert "AISimulate" in b["sources"][0]["cite"]


def test_each_interval_contains_its_value_has_width_and_is_not_wider_than_half_of_it():
    for bodies in entries().values():
        for b in bodies:
            lo, hi = b["ci95"]
            assert lo <= b["value"] <= hi, (b["scope"], b["value"], b["ci95"])
            assert lo < hi, f"{b['scope']}: a zero-width interval {b['ci95']} reads as certainty (VAL-10)"
            assert hi - lo <= b["value"] / 2, (
                f"{b['scope']}: interval {b['ci95']} is wider than half the value, so the "
                f"entry would rank with assumed (VAL-10) and could not price a stated true")


def test_the_lane_differs_from_the_pooled_pair_and_by_dtype_on_every_part():
    """If the filter changed nothing the lane would be a copy; if the dtypes agreed, one pair would do."""
    p = pooled()
    got = entries()
    for chip in LANE_CHIPS:
        pair = {}
        for kv in KV:
            f = next(b for b in got[f"attention_prefill_floor_trtllm_gen_kv_{kv}"]
                     if b["scope"]["hardware"] == [chip])["value"]
            w = next(b for b in got[f"attention_prefill_work_scale_trtllm_gen_kv_{kv}"]
                     if b["scope"]["hardware"] == [chip])["value"]
            pair[kv] = (f, w)
            assert pair[kv] != (p[("attention_prefill_floor", chip)],
                                p[("attention_prefill_work_scale", chip)]), (chip, kv)
        assert pair["bf16"] != pair["fp8"], f"{chip}: the two cache dtypes fit alike, so one pair would do"


def test_the_block_sits_between_its_markers_so_the_writer_can_replace_it():
    text = SET.read_text()
    assert text.count("BEGIN trtllm-gen prefill lane") == 1
    assert text.count("END trtllm-gen prefill lane") == 1
    block = text.split("BEGIN trtllm-gen prefill lane", 1)[1].split("END trtllm-gen prefill lane", 1)[0]
    for name in NAMES:
        assert block.count(f"{name}:") == len(LANE_CHIPS)


def test_the_writer_refuses_a_sweep_that_did_not_exercise_the_kernel(tmp_path):
    """Known-bad input: a context sweep with no trtllm-gen row writes no lane.

    The sweep is built here, so the test needs no AISimulate data: it needs pyarrow, which the
    registry job installs from the fit lock file, and so it runs on every runner."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    sweep = tmp_path / "b200_sxm" / "attention" / "vllm" / "0.25.0"
    sweep.mkdir(parents=True)
    n = 2000
    pq.write_table(pa.table({
        "kernel_source": ["vllm_triton_attn"] * n, "window_size": [0] * n,
        "latency": [0.05] * n, "batch_size": [1] * n, "isl": list(range(1, n + 1)),
        "num_heads": [32] * n, "head_dim": [128] * n,
    }), sweep / "context_attention_perf.parquet")
    r = subprocess.run(
        [sys.executable, "-I", str(REPO / "scripts" / "emit_attention_trtllm_gen.py"),
         "--data", str(tmp_path), "--check"],
        capture_output=True, text=True, cwd=str(REPO), timeout=600)
    assert r.returncode != 0
    assert "did not exercise the kernel" in (r.stdout + r.stderr)


@pytest.mark.skipif(not DATA.is_dir() or next(DATA.rglob("*.parquet"), None) is None,
                    reason="AISimulate data not present")
def test_the_pooled_pair_still_reproduces_from_every_row():
    """The lane's fitter, given every row, gives the committed pooled pair: the filter is the only change."""
    r = subprocess.run(
        [sys.executable, "-I", str(REPO / "scripts" / "emit_attention_trtllm_gen.py"),
         "--data", str(DATA), "--pooled"],
        capture_output=True, text=True, cwd=str(REPO), timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr
