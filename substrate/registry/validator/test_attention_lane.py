"""The trtllm-gen prefill lane as committed, and its writer's refusals.

A lane exists to price a kernel on its own rows, so what is checked here is what makes it
that: one pair per data-center Blackwell part and no other; each entry keyed to the build its
data exercised, or moved from it with the build it was measured on, or, where the run records
no FlashInfer version, honest about having no hash (D5, VAL-8); the interval around the value
it describes; and a lane that differs from the pooled pair it replaces. The data-dependent half (the lane's rows are what the
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


def by_chip(bodies):
    out = {}
    for b in bodies:
        out.setdefault(b["scope"]["hardware"][0], []).append(b)
    return out


def test_the_lane_is_one_pair_per_cache_dtype_per_blackwell_part_and_no_other():
    got = entries()
    assert set(got) == set(NAMES)
    for name, bodies in got.items():
        assert all(len(b["scope"]["hardware"]) == 1 and set(b["scope"]) == {"hardware"} for b in bodies)
        chips = by_chip(bodies)
        assert set(chips) == LANE_CHIPS, (
            f"{name} is scoped to {sorted(chips)}, not exactly {sorted(LANE_CHIPS)}: a part "
            f"without a measurement of the kernel must have no lane, so a stated true is refused")
        for chip, bs in chips.items():
            # A keyed part has the entry measured on the run's build and the one moved to the pin's;
            # an unkeyed part has one entry. One value per part either way.
            assert len(bs) in (1, 2), (name, chip, len(bs))
            assert len({b["value"] for b in bs}) == 1, (name, chip, [b["value"] for b in bs])


def test_each_entry_is_keyed_to_the_build_its_data_exercised_or_says_why_not():
    """D5, VAL-8: a measured entry names the build the data exercised and what the number includes
    (VAL-9); a transferred one names the build it is moved to and the one it was measured on; an
    entry that names neither says that the run records no FlashInfer version."""
    for name, bodies in entries().items():
        for chip, bs in by_chip(bodies).items():
            measured = [b for b in bs if b["method"] == "measured"]
            moved = [b for b in bs if b["method"] == "transferred" and "kernel_impls" in b]
            bare = [b for b in bs if b["method"] == "transferred" and "kernel_impls" not in b]
            assert len(measured) + len(moved) + len(bare) == len(bs), (name, chip)
            for b in bs:
                dims = {r["dim"]: (r["low"], r["high"]) for r in b["fit_range"]}
                assert set(dims) == {"tokens", "requests", "context_tokens"}, dims
                assert all(lo <= hi for lo, hi in dims.values())
                # No row has a cached prefix, so the context a row reads is its prompt length.
                assert dims["context_tokens"][1] >= dims["requests"][1]
                assert "vllm_flashinfer_trtllmprefill" in b["sources"][0]["cite"]
                assert "AISimulate" in b["sources"][0]["cite"]
            if bare:
                assert not measured and not moved and len(bare) == 1, (name, chip)
                b = bare[0]
                assert b["fitted"] is False and "measures" not in b and "measured_on" not in b
                assert b["transferred_from"] == "hardware=" + chip
                assert "records no FlashInfer version" in b["transfer_rule"]
                continue
            assert len(measured) == 1, (name, chip, "a keyed part has its measured entry")
            m = measured[0]
            assert m["fitted"] is True and "transferred_from" not in m and "measured_on" not in m
            assert len(m["kernel_impls"]) == 1 and m["kernel_impls"][0]["library"] == "flashinfer"
            assert m["kernel_impls"][0]["variant"] == "trtllm_batch_context_with_kv_cache"
            assert m["measures"]["time"] == "device" and m["measures"]["execution"] == "captured"
            assert "collect_attn.py" in m["measures"]["harness"]
            assert "kernel_impls.json" in m["sources"][1]["cite"]
            for t in moved:
                assert t["fitted"] is False
                assert t["measured_on"] == m["kernel_impls"], "moved from the build it was measured on"
                assert t["kernel_impls"] != t["measured_on"], "a build equal to its source has nothing to move"
                assert t["kernel_impls"][0]["variant"] == m["kernel_impls"][0]["variant"]
                assert t["transferred_from"] == "hardware=" + chip


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
    got = entries()
    for name in NAMES:
        assert block.count(f"{name}:") == len(got[name])


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
    table = tmp_path / "kernel_impls.json"
    table.write_text('{"implementations": [], "libraries": [], "engine": {}}')
    r = subprocess.run(
        [sys.executable, "-I", str(REPO / "scripts" / "emit_attention_trtllm_gen.py"),
         "--data", str(tmp_path), "--kernel-impls", str(table), "--check"],
        capture_output=True, text=True, cwd=str(REPO), timeout=600)
    assert r.returncode != 0
    assert "did not exercise the kernel" in (r.stdout + r.stderr)


def test_the_writer_refuses_to_run_without_the_engines_table(tmp_path):
    """Known-bad input: no kernel_impls.json. The lane is keyed by the builds it lists."""
    env = {k: v for k, v in os.environ.items() if k != "BLIS_KERNEL_IMPLS"}
    r = subprocess.run(
        [sys.executable, "-I", str(REPO / "scripts" / "emit_attention_trtllm_gen.py"),
         "--data", str(tmp_path), "--check"],
        capture_output=True, text=True, cwd=str(REPO), timeout=600, env=env)
    assert r.returncode == 2
    assert "kernel_impls.json is required" in (r.stdout + r.stderr)


@pytest.mark.skipif(not DATA.is_dir() or next(DATA.rglob("*.parquet"), None) is None,
                    reason="AISimulate data not present")
def test_the_pooled_pair_still_reproduces_from_every_row():
    """The lane's fitter, given every row, gives the committed pooled pair: the filter is the only change."""
    r = subprocess.run(
        [sys.executable, "-I", str(REPO / "scripts" / "emit_attention_trtllm_gen.py"),
         "--data", str(DATA), "--pooled"],
        capture_output=True, text=True, cwd=str(REPO), timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr
