"""The shared drafter rule (scripts/drafter_links.py) on fixture catalogs and on this catalog."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import drafter_links as dl  # noqa: E402


def entry(root: Path, name: str, model_yaml: str, method: str | None = None) -> None:
    d = root / "models" / name
    d.mkdir(parents=True)
    (d / "model.yaml").write_text(f"name: {name}\n{model_yaml}")
    if method:
        (d / "graph.yaml").write_text(f"name: {name}\nspeculator:\n  num_spec_tokens: 1\n  method: {method}\n")


@pytest.fixture
def cat(tmp_path):
    entry(tmp_path, "mm3", "source:\n  provider: huggingface\n  repo: Org/MM3\ndrafters: [speculator, mm3-eagle3]\n",
          "minimax_m3_mtp")
    entry(tmp_path, "mm3-eagle3", "source:\n  repo: Org/MM3-EAGLE3\ntargets: [mm3]\n", "eagle3")
    entry(tmp_path, "other-eagle3", "source:\n  repo: Org/Other-EAGLE3\ntargets: [mm3]\n", "eagle3")
    # A DSpark head shipped in its target's repository, the target having no in-graph head.
    entry(tmp_path, "flash", "source:\n  repo: Org/Flash\ndrafters:\n- flash-dspark\n")
    entry(tmp_path, "flash-dspark", "source:\n  repo: Org/Flash\n  path: dspark/config.json\ntargets:\n- flash\n",
          "dspark")
    return dl.Catalog(tmp_path)


def test_the_models_own_speculator(cat):
    assert dl.resolve(cat, "mm3", None, "mtp") == ""
    assert dl.resolve(cat, "mm3", "native") == ""
    assert dl.resolve(cat, "mm3", "org/mm3", "mtp") == ""        # vLLM names the target itself


def test_a_separate_drafter_by_repository_case_insensitively(cat):
    assert dl.resolve(cat, "mm3", "Org/MM3-EAGLE3", "eagle3") == "mm3-eagle3"
    assert dl.resolve(cat, "mm3", "org/mm3-eagle3", "eagle3") == "mm3-eagle3"


def test_a_drafter_shipped_in_its_targets_repository_resolves(cat):
    """The engine loads a DSpark head from the target's own repository: naming that repository,
    or nothing, resolves to the drafter entry there, read from a block list."""
    assert dl.resolve(cat, "flash", "Org/Flash", "dspark") == "flash-dspark"
    assert dl.resolve(cat, "flash", "native") == "flash-dspark"
    assert dl.resolve(cat, "flash", None, "dspark") == "flash-dspark"


@pytest.mark.parametrize("model, draft, method, why", [
    ("mm3", "/data/models/MM3-EAGLE3", "eagle3", "names no repository"),
    ("mm3", "Org/Unknown", "eagle3", "not in the catalog as a drafter of mm3"),
    ("mm3", "Org/Other-EAGLE3", "eagle3", "does not list among its drafters"),
    ("mm3", "Org/MM3-EAGLE3", "dspark", "speculates by eagle3, not dspark"),
    ("mm3", None, "eagle3", "no single drafter entry in its own repository"),
    ("flash", "Org/Flash", "mtp", "no single drafter entry in its own repository"),
    ("nope", None, None, "not a catalog model"),
])
def test_a_drafter_that_does_not_resolve_is_refused_with_its_reason(cat, model, draft, method, why):
    with pytest.raises(dl.Unresolved, match=why):
        dl.resolve(cat, model, draft, method)


def test_this_catalogs_dspark_heads_resolve():
    cat = dl.Catalog(ROOT)
    assert dl.resolve(cat, "minimax-m3.1-flash", "native") == "minimax-m3.1-flash-dspark"
    assert dl.resolve(cat, "minimax-m3", "Inferact/MiniMax-M3-EAGLE3", "eagle3") == "minimax-m3-eagle3"
