"""The offload-tier coefficients: every catalog tier, on every catalog chip, with a method that
says what the figure is.

A storage tier is a name in the catalog (devices/storage.yaml) and states no number there. What a
transfer to or from it costs is three coefficients per tier, in the set cost-model-offload-tiers, written
by scripts/emit_offload_tiers.py
(SUB-18: a fitted or assumed value is a coefficient, not a catalog fact). The kernel's TierTime
reads them, so a tier or a chip with one missing would price at nothing or refuse; the Go test
kernel/tiers_test.go checks the same completeness through the loader and the values against the
figures the catalog held. This file checks the file itself, without Go.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CATALOG = Path(os.environ.get("BLIS_CATALOG") or REPO.parent / "catalog")
SET = REPO / "coefficients" / "cost-model-offload-tiers.yaml"
TERMS = ("read_rate", "write_rate", "base_latency")
UNITS = {"read_rate": "bytes_per_us", "write_rate": "bytes_per_us", "base_latency": "us_per_transfer"}


def entries():
    out = []
    for e in yaml.safe_load(SET.read_text())["coefficients"]:
        (name, body), = e.items()
        if name.startswith("offload_tier_"):
            out.append((name, body))
    return out


def tiers():
    doc = yaml.safe_load((CATALOG / "devices" / "storage.yaml").read_text())
    return sorted(doc)


def chips():
    return sorted(Path(p).stem for p in glob.glob(str(CATALOG / "hardware" / "*.yaml")))


def test_the_writer_reproduces_the_committed_entries():
    """The entries are written by emit_offload_tiers.py and by nothing else (SUB-4)."""
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "emit_offload_tiers.py"), "--check"],
                       capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert r.returncode == 0, f"the committed offload-tier entries are not what the writer writes:\n{r.stdout}\n{r.stderr}"


def test_every_tier_has_all_three_terms_on_every_chip_exactly_once():
    found = {}
    for name, body in entries():
        for chip in body["scope"]["hardware"]:
            found.setdefault((name, chip), []).append(body)
    missing, twice = [], []
    for tier in tiers():
        for chip in chips():
            for term in TERMS:
                name = f"offload_tier_{tier}_{term}"
                n = len(found.get((name, chip), []))
                if n == 0:
                    missing.append(f"{name} on {chip}")
                elif n > 1:
                    twice.append(f"{name} on {chip}")
    assert not missing, f"tier terms absent, so the tier prices at nothing there: {missing}"
    assert not twice, f"tier terms stated twice for one chip, so no entry is chosen: {twice}"
    names = {n for n, _ in entries()}
    known = {f"offload_tier_{t}_{term}" for t in tiers() for term in TERMS}
    assert names == known, f"entries for a tier the catalog does not have: {sorted(names - known)}"


def test_each_term_carries_the_unit_the_formula_needs():
    for name, body in entries():
        term = next(t for t in TERMS if name.endswith("_" + t))
        assert body["units"] == UNITS[term], f"{name} is {body['units']}, TierTime needs {UNITS[term]}"
        assert body["value"] > 0, f"{name}: a zero term would price a transfer as free"


def test_no_tier_figure_claims_a_measurement_it_did_not_make():
    """D5, VAL-8, SUB-18. A figure is `assumed` (a judgement with its reasons) or `transferred`
    (measured on another implementation and moved by a stated rule). `measured` would claim a
    fit on the path priced, and `vendor_spec` a datasheet's figure; no tier term is either.
    A transferred entry names the scope it came from and the rule, and records nothing a fit
    on this path would (a fit range, a measurement boundary, kernel implementations); an assumed
    entry says why at length."""
    methods = set()
    for name, body in entries():
        m = body["method"]
        methods.add(m)
        assert m in ("assumed", "transferred"), f"{name} is {m}"
        assert body.get("fitted") is False, f"{name}: a tier figure is not fitted on the path priced"
        for field in ("fit_range", "measures", "kernel_impls", "ci95"):
            assert field not in body, f"{name} states {field}, which only a fit on the path priced has"
        assert body.get("sources"), f"{name} cites nothing"
        if m == "transferred":
            assert body.get("transferred_from") == "hardware=h100", name
            assert len(body.get("transfer_rule", "")) > 100, name
        else:
            assert len(body.get("rationale", "")) > 200, name
            assert "transferred_from" not in body, name
    assert methods == {"assumed", "transferred"}


def test_the_cpu_tier_is_scoped_to_the_host_it_was_fitted_on():
    """cpu_dram's rates were fitted on one H100 host. The h100 entry says so by its scope; every
    other chip carries the same figure as an entry of its own, so the transfer is on the page
    and not a generic fact (the catalog applied it to every host without saying)."""
    by_chip = {}
    for name, body in entries():
        if name.startswith("offload_tier_cpu_dram_"):
            for chip in body["scope"]["hardware"]:
                by_chip.setdefault(chip, {})[name] = body
    assert set(by_chip) == set(chips())
    for chip, got in by_chip.items():
        for name, body in got.items():
            if name.endswith("_base_latency"):
                assert body["method"] == "assumed", (chip, name)  # the floor does not transfer
            else:
                assert body["method"] == "transferred", (chip, name)
        assert (len(got) == 3), chip
        # one value on every chip: no chip has a measurement of its own
    for name in ("read_rate", "write_rate", "base_latency"):
        vals = {b[f"offload_tier_cpu_dram_{name}"]["value"] for b in by_chip.values()}
        assert len(vals) == 1, (name, vals)


def test_the_host_link_fit_refuses_a_tier_rate_that_is_not_its_median():
    """fit_host_link.py and the tier entries use the same rows. If a refit moves a median the
    tier rate stays the old number unless emit_offload_tiers.py is changed too; the fitter
    compares them (ONE FIT, NOT TWO). Its comparison, on the committed file, with the rates
    it should derive and with each one moved by a megabyte per second."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fit_host_link", REPO / "scripts" / "fit_host_link.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = SET.read_text()
    assert mod.tier_rate_problems(text, 27058.4, 46089.2) == []
    assert mod.tier_rate_problems(text, 27059.0, 46089.0)
    assert mod.tier_rate_problems(text, 27058.0, 46090.0)
    assert len(mod.tier_rate_problems(text.replace("offload_tier_cpu_dram_", "offload_tier_gone_"), 27058.0, 46089.0)) == 2
