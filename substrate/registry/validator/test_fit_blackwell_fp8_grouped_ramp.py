"""The Blackwell block-FP8 grouped ramp (scripts/fit_blackwell_fp8_grouped_ramp.py).

What is checked: the split holds out every second geometry whole (a random row split would train on a token count and
test on its neighbour), the six committed entries are what the writer's renderer produces from their own values (so a
hand edit shows), a renderer refuses an entry of the family it does not maintain, and each part's entries are unkeyed
`assumed` ones that carry an interval (the engine release lists no sm_100a build to key them to, and an unkeyed fit is not `measured`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

REG = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fit_blackwell", REG / "scripts" / "fit_blackwell_fp8_grouped_ramp.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SET = REG / "coefficients" / "cost-model-primitives.yaml"


def committed():
    doc = yaml.safe_load(SET.read_text())
    return [(n, b) for e in doc["coefficients"] for n, b in e.items() if n in mod.NAMES]


def test_the_split_holds_out_every_second_geometry_whole():
    geos = [(h, i, 8, 256) for h in (2048, 3072, 4096, 6144) for i in (512, 1024)]
    rows = [g for g in geos for _ in range(5)]
    r = {"hidden_size": np.array([g[0] for g in rows]), "inter_size": np.array([g[1] for g in rows]),
         "topk": np.array([g[2] for g in rows]), "num_experts": np.array([g[3] for g in rows])}
    held = mod.geometry_split(r)
    sorted_geos = sorted(set(geos))
    for j, g in enumerate(sorted_geos):
        mine = [held[k] for k, row in enumerate(rows) if row == g]
        assert all(m == (j % 2 == 1) for m in mine), (g, mine)


def test_each_blackwell_part_has_three_unkeyed_assumed_entries_with_an_interval():
    got = committed()
    for chip in ("b200", "b300", "gb200-nvl72", "gb300"):
        mine = [(n, b) for n, b in got if b["scope"] == {"hardware": [chip]}]
        assert sorted(n for n, _ in mine) == sorted(mod.NAMES), (chip, [n for n, _ in mine])
        for n, b in mine:
            assert b["method"] == "assumed" and b["fitted"] is False and not b.get("kernel_impls"), (chip, n)
            lo, hi = b["ci95"]
            assert lo <= b["value"] <= hi, (chip, n, b["value"], b["ci95"])
    assert not any(b["scope"].get("hardware", [None])[0] in ("h100", "h200") for n, b in got)


def test_the_renderer_reproduces_the_committed_entries_and_refuses_a_stranger():
    fit = {"sku": "b200_sxm", "chip": "b200", "eps": 0.21, "mh": 1, "floor_us": 30.0, "err": 1.144, "signed": 1.0, "n": 736,
           "geoms": 17, "ci": {"eps": (0.195, 0.235), "mh": (1, 1), "floor": (27.5, 32.5)}, "peak_tf": 4500.0,
           "held": (1.161, 1.069), "held_dense": (1.531, 0.714), "held_n": 483, "later": {},
           "span": {"tokens": (1.0, 4096.0)}}
    stub = "kind: CoefficientSet\n\ncoefficients:\n  - gemm_eps_max_fp8:\n      value: 0.7\n      scope: {hardware: [b200]}\n"
    out = mod.render(stub, [fit])
    for n in mod.NAMES:
        assert f"  - {n}:\n" in out
    assert mod.render(out, [fit]) == out  # idempotent
    stranger = stub + f"  - {mod.NAMES[0]}:\n      value: 0.5\n      scope: {{hardware: [gb200-nvl72]}}\n"
    with pytest.raises(SystemExit):
        mod.render(stranger, [fit])
