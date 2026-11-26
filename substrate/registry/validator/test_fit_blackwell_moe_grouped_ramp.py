"""The Blackwell NVFP4 and BF16 grouped ramp (scripts/fit_blackwell_moe_grouped_ramp.py).

Checked: the fitter recovers a known ramp and floor from rows it generated; the folds hold a geometry out whole;
the committed entries are unkeyed `assumed` ones carrying an interval that contains the value; the renderer is
idempotent and refuses an entry of the family it does not maintain.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

REG = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fit_moe", REG / "scripts" / "fit_blackwell_moe_grouped_ramp.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SET = REG / "coefficients" / "cost-model-primitives.yaml"


def test_the_fit_recovers_a_known_ramp_and_floor():
    rng = np.random.default_rng(1)
    T = np.tile(2.0 ** np.arange(0, 12), 8)
    h, i = np.repeat(rng.choice([2048., 4096., 7168.], 8), 12), np.repeat(rng.choice([512., 1024., 2048.], 8), 12)
    F = T * 8 * 6 * h * i
    peak = 9e15
    H = 256 * (1 - (248 / 256) ** T) * 3 * h * i * 0.5625 / 6e12
    truth = np.array([28e-6, 1 / 0.22, 40 / 0.22])
    S = mod.predict(truth, T, F, H, peak)
    got = mod.fit(T, F, H, S, peak)
    assert mod.to_entry(got)[0] == pytest.approx(0.22, abs=0.01)
    assert got[0] == pytest.approx(28e-6, rel=0.05)


def test_a_fold_holds_a_geometry_out_whole():
    geos = [(h, i, 8, 256) for h in (2048, 3072, 4096, 6144, 7168) for i in (512, 1024)] * 3
    fold = mod.fold_of(geos)
    for g in set(geos):
        assert len({fold[j] for j, x in enumerate(geos) if x == g}) == 1
    assert set(fold) == set(range(mod.FOLDS))


def test_committed_entries_are_unkeyed_assumed_with_an_interval():
    doc = yaml.safe_load(SET.read_text())
    got = [(n, b) for e in doc["coefficients"] for n, b in e.items() if n in mod.NAMES]
    for chip in ("b200", "b300", "gb200-nvl72", "gb300"):
        mine = [(n, b) for n, b in got if b["scope"] == {"hardware": [chip]}]
        assert sorted(n for n, _ in mine) == sorted(mod.NAMES), chip
        for n, b in mine:
            assert b["method"] == "assumed" and not b.get("kernel_impls"), (chip, n)
            lo, hi = b["ci95"]
            assert lo <= b["value"] <= hi, (chip, n)
            if "eps_max" in n:
                assert 0 < b["value"] < 1  # never above the dense peak (SUB-5)


def test_the_renderer_is_idempotent_and_refuses_a_stranger():
    fit = {"word": "nvfp4", "sku": "b200_sxm", "chip": "b200", "eps": 0.226, "mh": 1, "floor_us": 27.9, "n": 100,
           "geoms": 15, "excluded": [], "ci": {"eps": (0.2, 0.25), "mh": (1, 3), "floor": (25, 30)}, "peak_tf": 9000.0,
           "cv": {"grouped": (1.4, 1.0, 0.27), "dense": (2.4, 0.5, 0.45)}, "cv_n": 100, "other": {}}
    stub = "kind: CoefficientSet\n\ncoefficients:\n  - gemm_eps_max_nvfp4:\n      value: 0.7\n      scope: {hardware: [b200]}\n"
    out = mod.render(stub, [fit])
    for p in ("eps_max", "m_half", "floor"):
        assert f"  - gemm_{p}_nvfp4_grouped:\n" in out
    assert mod.render(out, [fit]) == out
    stranger = stub + "  - gemm_floor_bf16_grouped:\n      value: 5\n      scope: {hardware: [h100]}\n"
    with pytest.raises(SystemExit):
        mod.render(stranger, [fit])
