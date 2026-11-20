"""The Hopper block-FP8 grouped ramp per MoE backend (scripts/fit_hopper_fp8_grouped_ramp.py).

What is checked is what makes the family usable by the cost model: each value is bound to the build
its data exercised (VAL-8), moved to the engine pin's build only as `transferred` with that build as
`measured_on`, and the writer refuses to run without the engine's kernel table.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SET = REPO / "coefficients" / "cost-model-primitives.yaml"
CHIPS = {"h100", "h200"}
BACKENDS = ("triton", "flashinfer_cutlass")
NAMES = tuple(f"gemm_{p}_fp8_grouped_{b}" for b in BACKENDS for p in ("eps_max", "m_half", "floor"))


def entries():
    doc = yaml.safe_load(SET.read_text())
    return [(n, b) for e in doc["coefficients"] for n, b in e.items() if n in NAMES]


def test_every_value_is_keyed_to_the_build_it_was_measured_on_and_moved_only_as_transferred():
    got = entries()
    for chip in CHIPS:
        for name in NAMES:
            mine = [b for n, b in got if n == name and b["scope"] == {"hardware": [chip]}]
            measured = [b for b in mine if b["method"] == "measured"]
            moved = [b for b in mine if b["method"] == "transferred"]
            assert len(measured) == 1, f"{name} {chip}: {len(measured)} measured entries"
            m = measured[0]
            assert m["fitted"] is True and len(m["kernel_impls"]) == 1 and m["kernel_impls"][0]["code_hash"]
            assert m.get("measures", {}).get("execution") == "captured" and m.get("ci95") and m.get("fit_range")
            # A pinned build that differs is reached only through a transferred entry naming the measured one.
            assert len(moved) <= 1, f"{name} {chip}: {len(moved)} transferred entries"
            for t in moved:
                assert t["measured_on"] == m["kernel_impls"], f"{name} {chip}: moved from another build"
                assert t["kernel_impls"] != m["kernel_impls"], f"{name} {chip}: moved to the build it was measured on"
                assert t["value"] == m["value"] and t["fitted"] is False and "measures" not in t
                assert t["transferred_from"] == f"hardware={chip}"


def test_the_writer_refuses_to_run_without_the_engines_table(tmp_path):
    """Known-bad input: no kernel_impls.json. Every entry is keyed by the builds it lists."""
    env = {k: v for k, v in os.environ.items() if k != "BLIS_KERNEL_IMPLS"}
    r = subprocess.run(
        [sys.executable, "-I", str(REPO / "scripts" / "fit_hopper_fp8_grouped_ramp.py"),
         "--data", str(tmp_path), "--check"],
        capture_output=True, text=True, cwd=str(REPO), timeout=600, env=env)
    assert r.returncode == 2
    assert "kernel_impls.json is required" in (r.stdout + r.stderr)
