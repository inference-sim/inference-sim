"""The per-chip launch fit (scripts/fit_host_launch_per_chip.py) on synthetic units built from a known per-launch value.

What is checked is the rule that decides whether a value is written (VAL-11): a value recovered from units that all
carry one true launch cost is written and close to it; the same fit on too few units, too few families, or units whose
launch counts do not separate the value from the device error is reported, not written; and a unit that is not on the
published fit list never reaches the fit (D7).
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "scripts"
spec = importlib.util.spec_from_file_location("fit_host_launch_per_chip", HERE / "fit_host_launch_per_chip.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def synth(true_g: float, families: int, per_family: int, noise: float = 0.0, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for f in range(families):
        for _ in range(per_family):
            n = 400 + 500 * f + rng.randint(0, 30)
            device = rng.uniform(3000, 6000)
            truth = device + n * true_g
            out.append({"chip": "bx", "family": f"fam{f}", "measured": truth * (1 + rng.gauss(0, noise)),
                        "pred0": device, "n": n})
    return out


def test_a_value_the_units_carry_is_recovered_and_written():
    r = mod.fit_chip(synth(2.0, families=4, per_family=15, noise=0.02))
    assert r["write"] and abs(r["g"] - 2.0) <= 0.25, r


def test_too_few_units_or_families_are_reported_and_not_written():
    assert not mod.fit_chip(synth(2.0, families=4, per_family=5, noise=0.02))["write"]
    assert not mod.fit_chip(synth(2.0, families=2, per_family=30, noise=0.02))["write"]


def test_units_that_cannot_separate_the_value_from_device_error_are_not_written():
    r = mod.fit_chip(synth(2.0, families=4, per_family=15, noise=0.4))
    assert not r["write"], r


def test_only_published_fit_units_are_used_and_only_up_to_concurrency_four():
    rows = [{"point_id": f"u{i}", "family": "f", "chip": "bx", "concurrency": str(c), "measured_us": "1000", "pred_us": "1100"}
            for i, c in enumerate([1, 2, 4, 8, 64])]
    zero = [dict(r, pred_us="900") for r in rows]
    got = mod.build_units(rows, zero, fit={"u0", "u1", "u3", "u4"})
    assert len(got) == 2, got  # u2 is not on the fit list, u3 and u4 are above concurrency 4
    assert abs(got[0]["n"] - 200 / 3.2) < 1e-9
