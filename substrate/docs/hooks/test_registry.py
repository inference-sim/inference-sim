"""Tests for the documentation hook, run against the committed coefficient sets.

The hook's job is to show the data exactly. These tests check the two ways it could fail
to: by leaving an entry out of a page, and by showing an entry differently from how it is
committed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import registry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "registry"

# Loaded at import, because the parametrized tests below enumerate the sets.
registry.on_config({"config_file_path": str(ROOT / "mkdocs.yml")})


def _chomp(x):
    """Drop a block scalar's final newline, which depends on whether the excerpt ends the
    file and so carries no content."""
    if isinstance(x, str):
        return x.removesuffix("\n")
    if isinstance(x, list):
        return [_chomp(v) for v in x]
    if isinstance(x, dict):
        return {k: _chomp(v) for k, v in x.items()}
    return x


def all_sets():
    return list(registry._state["sets"].values())


def test_every_committed_set_is_loaded():
    on_disk = {yaml.safe_load(p.read_text())["name"] for p in (REGISTRY / "coefficients").glob("*.yaml")}
    assert on_disk == set(registry._state["sets"])


@pytest.mark.parametrize("s", all_sets(), ids=lambda s: s.name)
def test_every_entry_appears_in_its_matrix_and_only_under_its_own_parts(s):
    name = s.name
    rows, _ = registry.matrix(s)
    placed = {}
    for row, cells in rows.items():
        for part, entries in cells.items():
            for e in entries:
                # A fabric-only entry's columns are the fabrics it names.
                columns = e.hardware or e.fabric
                assert part in columns, f"{e.name} placed under {part}, outside its scope"
                placed.setdefault(id(e), []).append((row, part))
    for e in s.entries:
        assert id(e) in placed, f"{name}:{e.line} {e.name} is missing from the matrix"
        if "_‹part›" in registry._fold(e)[0]:
            # a name carrying its part fills exactly that part's cell
            assert len(placed[id(e)]) == 1, f"{e.name} fills {placed[id(e)]}"


def test_a_model_scoped_entry_is_its_own_row_labelled_with_its_model():
    """A model-scoped entry is admitted only for that model, so it must not share a row
    with the part-wide entry of the same name, and the label must say which model."""
    e = registry.Entry("attention_decode_rate_sparse_mla",
                       {"scope": {"hardware": ["h200"], "model": ["deepseek-v4.1-flash"]},
                        "units": "bytes_per_us", "method": "measured", "value": 1.0}, 1)
    row, columns = registry._fold(e)
    assert row == "attention_decode_rate_sparse_mla ‹model: deepseek-v4.1-flash›"
    assert columns == ["h200"]


def test_a_scalar_model_scope_is_one_label_not_its_characters():
    e = registry.Entry("x", {"scope": {"hardware": ["h200"], "model": "m-1"},
                             "units": "tokens", "method": "measured", "value": 1}, 1)
    assert registry._fold(e)[0] == "x ‹model: m-1›"


def test_a_fabric_scoped_entry_is_its_own_row_labelled_with_its_fabric():
    e = registry.Entry("moe_a2a_rate_x_16rank_h200",
                       {"scope": {"hardware": ["h200"], "fabric": ["ib-400g"]},
                        "units": "bytes_per_us", "method": "measured", "value": 1.0}, 1)
    row, columns = registry._fold(e)
    assert row == "moe_a2a_rate_x_16rank_‹part› ‹fabric: ib-400g›"
    assert columns == ["h200"]


def test_a_fabric_only_entry_fills_its_fabric_columns():
    e = registry.Entry("fabric_bandwidth_efficiency",
                       {"scope": {"fabric": ["ib-400g"]}, "units": "dimensionless",
                        "method": "literature", "value": 0.9}, 1)
    rows, _ = registry.matrix(registry.CoefficientSet("s", "s.yaml", [e]))
    assert list(rows["fabric_bandwidth_efficiency"]) == ["ib-400g"]


@pytest.mark.parametrize("scope", [{"hardware": ["h200"], "tp": [8]}, {"model": ["m"]}],
                         ids=["unshown-scope-key", "no-hardware"])
def test_the_matrix_refuses_an_entry_it_cannot_show(scope):
    e = registry.Entry("x", {"scope": scope, "units": "tokens", "method": "measured",
                             "value": 1}, 1)
    with pytest.raises(registry.PluginError):
        registry.matrix(registry.CoefficientSet("s", "s.yaml", [e]))


@pytest.mark.parametrize("s", all_sets(), ids=lambda s: s.name)
def test_every_quoted_entry_parses_back_to_the_committed_entry(s):
    for e in s.entries:
        part = e.hardware[0] if e.hardware else e.fabric[0]
        matches = [x for x in s.entries if x.name == e.name
                   and (part in x.hardware or (not x.hardware and part in x.fabric))]
        if len(matches) != 1:
            continue  # the directive refuses ambiguous quotes; nothing to compare
        html = registry.render_entry(f"{s.name} {e.name} {part}")
        quoted = re.search(r"```yaml[^\n]*\n(.*?)\n```", html, re.S).group(1)
        assert _chomp(yaml.safe_load(quoted)) == _chomp([{e.name: e.body}]), f"{s.name}:{e.line}"


@pytest.mark.parametrize("s", all_sets(), ids=lambda s: s.name)
def test_every_rendered_value_reads_back_as_the_committed_number(s):
    for e in s.entries:
        shown = registry._fmt_value(e.body["value"], e.value_text)
        assert float(shown.replace(",", "")) == float(e.body["value"]), f"{s.name}:{e.line}"
        assert shown.replace(",", "") == e.value_text, f"{s.name}:{e.line}"


def test_census_totals_match_the_files():
    total = sum(len(s.entries) for s in all_sets())
    assert registry._stat("total") == f"{total:,}"
    html = registry.render_census("")
    assert f"<strong>{total:,}</strong>" in html


@pytest.mark.parametrize("cite, expected", [
    ("NVIDIA AISimulate systems/data/h200_sxm/attention/vllm/0.25.0/generation_attention_perf.parquet",
     "AISimulate sweep, vllm lane"),
    ("NVIDIA AISimulate systems/data/a100_sxm/comm/nccl/2.27.3/nccl_perf.parquet (NCCL 2.27.3)",
     "AISimulate sweep, nccl lane"),
    ("NVIDIA AISimulate systems/b200_sxm.yaml", "AISimulate system descriptor"),
    ("Extrapolated from this part's own measured (2, 4, 8) floors in NVIDIA AISimulate systems/data/gb300/comm/nccl/",
     "derived from other registry entries"),
])
def test_source_classification(cite, expected):
    e = registry.Entry("x", {"sources": [{"kind": "model", "cite": cite, "role": "primary"}]}, 1)
    assert registry.classify_source(e) == expected


def test_every_entry_has_a_source_class():
    for s in all_sets():
        for e in s.entries:
            assert registry.classify_source(e)


def acceptance_sets():
    return list(registry._state["acceptance"].values())


def test_every_committed_acceptance_set_is_loaded():
    on_disk = {yaml.safe_load(p.read_text())["name"]
               for p in (REGISTRY / "acceptance").glob("*.yaml")}
    assert on_disk and on_disk == set(registry._state["acceptance"])


@pytest.mark.parametrize("a", acceptance_sets(), ids=lambda a: a.name)
def test_every_profile_is_rendered_with_its_committed_value_and_line(a):
    html = registry.render_acceptance(f"{a.name} values")
    lines = (ROOT / a.path).read_text().splitlines()
    for line, p in a.profiles:
        assert lines[line - 1].lstrip("- ").startswith(f"model: {p['model']}")
        cell = re.search(rf'href="[^"]*{re.escape(a.path)}#L{line}"[^>]*>([^<]+)</a>', html)
        assert cell, f"{a.path}:{line} is not rendered"
        assert float(cell.group(1)) == p["acceptance_length"]


@pytest.mark.parametrize("a", acceptance_sets(), ids=lambda a: a.name)
def test_every_caveat_and_source_is_rendered(a):
    caveats = registry.render_acceptance(f"{a.name} caveats")
    assert caveats.count("\n- ") + 1 == len(a.caveats)
    provenance = registry.render_acceptance(f"{a.name} provenance")
    for _, p in a.profiles:
        assert p["provenance"]["source"] in provenance


def test_an_unknown_acceptance_part_fails_the_build():
    name = acceptance_sets()[0].name
    with pytest.raises(registry.PluginError):
        registry.render_acceptance(f"{name} rates")
