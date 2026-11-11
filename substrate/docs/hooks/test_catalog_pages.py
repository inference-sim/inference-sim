"""Tests for the documentation page generator, docs/hooks/catalog_pages.py.

The generator's job is to restate the catalog's data without changing it, so these tests
compare what it renders against the files themselves. They need PyYAML and nothing from
MkDocs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "catalog"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog_pages as pages  # noqa: E402


@pytest.fixture(scope="module")
def cat():
    return pages.load(ROOT)


def test_loads_every_entry_on_disk(cat):
    assert [m.name for m in cat.models] == sorted(
        p.name for p in (ROOT / "models").iterdir() if p.is_dir())
    assert sorted(cat.chips) == sorted(p.stem for p in (ROOT / "hardware").glob("*.yaml"))
    assert sorted(cat.fabrics) == sorted(p.stem for p in (ROOT / "networks").glob("*.yaml"))
    assert sorted(cat.workloads) == sorted(p.stem for p in (ROOT / "workloads").glob("*.yaml"))


def test_reads_the_deriver_version_from_the_deriver(cat):
    deriver = (ROOT / "scripts/derive_graph.py").read_text()
    assert f"\nDERIVER_VERSION = {cat.deriver_version}\n" in deriver


def test_release_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_RELEASE", "v9.8.7")
    cat = pages.load(ROOT)
    assert (cat.release, cat.ref) == ("v9.8.7", "v9.8.7")
    assert "releases/tag/v9.8.7" in pages.release_text(cat)


def test_empty_release_means_main(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_RELEASE", " ")
    cat = pages.load(ROOT)
    assert (cat.release, cat.ref) == (None, "main")


def test_malformed_release_fails(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_RELEASE", "0.2.1")
    with pytest.raises(RuntimeError, match="MAJOR.MINOR.PATCH"):
        pages.load(ROOT)


@pytest.mark.parametrize("text,href", [
    ("see https://example.com/a.", "https://example.com/a"),
    ("'https://example.com/a'", "https://example.com/a"),
    ('"https://example.com/?c=1&d=2"', "https://example.com/?c=1&amp;d=2"),
    ("(https://en.wikipedia.org/wiki/Foo_(bar))", "https://en.wikipedia.org/wiki/Foo_(bar)"),
    ("(see https://example.com/x)", "https://example.com/x"),
])
def test_linkify_keeps_urls_whole(text, href):
    assert f'href="{href}"' in pages.linkify(text)


def test_summary_columns_ignore_speculator_only_kinds(cat):
    # DeepSeek-V4-Pro's MLA layer exists only in its draft module.
    v4 = next(m for m in cat.models if m.name == "deepseek-v4-pro")
    assert "mla" not in v4.attention_kinds()
    assert "sparse_mla" in v4.attention_kinds()


def test_layer_expansion_matches_every_graph(cat):
    # The expansion follows contracts/spec/model's Stack.Layers: prologue, pattern x repeat,
    # epilogue. Every expanded entry must name a kind the graph declares.
    for m in cat.models:
        s = m.graph["stack"]
        expected = (len(s.get("prologue", [])) + len(s.get("pattern", [])) * s.get("repeat", 0)
                    + len(s.get("epilogue", [])))
        assert len(m.layers) == expected > 0, m.name
        assert set(m.layers) <= {k["id"] for k in m.graph["layer_kinds"]}, m.name


def test_storage_tiers_state_no_number(cat):
    # SUB-18: a tier is a name and a description. Its rates and latency are registry
    # coefficients, so the page shows the coefficient names and no figure of its own.
    for name, tier in cat.tiers.items():
        assert set(tier) <= {"_comment"}, (name, tier)
        assert tier.get("_comment"), name
    table = pages.storage_table(cat)
    assert "offload_tier_nvme_gen4_read_rate" in table
    assert "MB/s" not in table


@pytest.mark.parametrize("value,expected", [
    (989.5, "989.5"),
    (2500.0, "2 500"),
    (80, "80"),
    (0.864, "0.864"),
    (None, "—"),
])
def test_num_states_figures_without_rounding(value, expected):
    assert pages.num(value) == expected


def test_every_model_page_renders_its_identity(cat):
    for m in cat.models:
        page = pages.model_page(cat, m)
        assert page.startswith("---\nhide:"), m.name
        assert f"# {m.name}\n" in page
        assert m.identity["source"]["revision"] in page
        assert m.graph["derived_from"]["sha256"] in page
        assert f"{len(m.layers)} layers" in page


def test_hardware_table_has_one_row_per_chip(cat):
    rows = pages.hardware_table(cat).splitlines()[2:]
    assert len(rows) == len(cat.chips)
    h100 = next(r for r in rows if r.startswith("| `h100`"))
    assert "989.5" in h100 and "3.35" in h100


def test_every_placeholder_in_the_docs_is_known(cat):
    frags = pages.fragments(cat)
    for md in (ROOT.parent / "docs" / "catalog").rglob("*.md"):
        for escaped, name in pages.PLACEHOLDER.findall(md.read_text(encoding="utf-8")):
            assert escaped or name in frags, f"{md.relative_to(ROOT.parent)}: [[catalog.{name}]]"


@pytest.mark.parametrize("text", ["[[catalog.nope]]", "[[catalog.Models.table]]",
                                  "[[catalog.models table]]"])
def test_unknown_or_misspelled_placeholder_fails(text):
    with pytest.raises(KeyError):
        pages.substitute(f"a {text} b", {"models.table": "x"}, "page.md")


def test_escaped_placeholder_renders_literally():
    assert pages.substitute("[[!catalog.NAME]]", {}, "p.md") == "[[catalog.NAME]]"


def test_svg_is_well_formed(cat):
    import xml.etree.ElementTree as ET

    for svg in (pages.hardware_figure(cat), pages.workloads_figure(cat),
                pages.layer_strip(cat.models[0])):
        ET.fromstring(svg)  # raises on malformed markup
        assert not re.search(r"\bnan\b|\binf\b", svg)


def test_workloads_split_into_shapes_and_replays(cat):
    # A replay workload has no prompt or output distribution, so it must stay out of the
    # shape figure and table, and every file must land in exactly one of the two.
    shapes, replays = pages.workload_rows(cat), pages.replay_rows(cat)
    assert sorted(shapes + replays) == sorted(cat.workloads)
    assert "agentx-mvp" in replays and "agentx-mvp" not in shapes
    assert len(pages.workloads_table(cat).splitlines()[2:]) == len(shapes)


def test_replay_tables_restate_the_corpus_pin(cat):
    rows = pages.replays_table(cat).splitlines()[2:]
    assert len(rows) == len(pages.replay_rows(cat))
    corpus = pages.replays_corpus_table(cat)
    for name in pages.replay_rows(cat):
        c = cat.workloads[name]["replay"]["corpus"]
        assert f"/tree/{c['source']['revision']})" in pages.replays_table(cat)
        # sha256 is optional in the schema; the table shows a dash without one.
        assert c.get("sha256", "—") in corpus


def test_agentx_mvp_notes_carry_its_provenance(cat):
    notes = pages.replays_notes(cat)
    assert "agentx-mvp: provenance notes" in notes
    assert "Apache-2.0" in notes and "build_weka_hf_dataset.py" in notes

