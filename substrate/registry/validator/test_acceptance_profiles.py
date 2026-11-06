"""The acceptance-profile set is InferenceX's golden curves, transcribed and nothing else.

`emit_acceptance_profiles.py --check` is the drift gate, as `--check` is for the coefficient
writers. Its inputs are in the repository (validator/testdata/inferencex and the catalog),
so unlike those writers' gates it runs in every CI job rather than only in `rederive`.
The other tests re-read the vendored curves by a second path and check the refusals: a
model the catalog does not hold is left out and named, never mapped to a near name.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "emit_acceptance_profiles.py"
GOLDEN = REPO / "validator" / "testdata" / "inferencex" / "golden_al_distribution"
CATALOG = REPO.parent / "catalog"
SET = yaml.safe_load((REPO / "acceptance" / "inferencex-golden-al.yaml").read_text())

sys.path.insert(0, str(REPO / "scripts"))
import emit_acceptance_profiles as emit  # noqa: E402


def test_the_committed_set_is_what_the_emitter_generates():
    r = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True,
                       text=True, cwd=str(REPO), timeout=120,
                       env={"PATH": "", "BLIS_CATALOG": "", "INFERENCEX_GOLDEN": ""})
    assert r.returncode == 0, (
        "acceptance/inferencex-golden-al.yaml is not what emit_acceptance_profiles.py "
        f"renders from the vendored curves; re-run it.\n{r.stderr}")


def test_the_vendored_curves_are_the_commit_the_emitter_cites():
    vendored = (GOLDEN.parent / "VENDORED.md").read_text()
    assert emit.INFERENCEX_SHA in vendored
    for p in SET["profiles"]:
        assert emit.INFERENCEX_SHA in p["provenance"]["source"]


def test_every_profile_is_the_value_its_source_file_states():
    """A second path: read each cited file directly and compare cell by cell."""
    for p in SET["profiles"]:
        stem = p["provenance"]["source"].rsplit("/", 1)[-1].removesuffix(".yaml")
        [(_, modes)] = yaml.safe_load((GOLDEN / f"{stem}.yaml").read_text()).items()
        stated = modes[f"thinking_{p['thinking']}"][p["num_spec_tokens"]]
        assert p["acceptance_length"] == stated, (stem, p["thinking"], p["num_spec_tokens"])


def test_every_cell_of_a_carried_curve_is_carried():
    carried = {p["provenance"]["source"].rsplit("/", 1)[-1] for p in SET["profiles"]}
    got = {(p["provenance"]["source"].rsplit("/", 1)[-1], p["thinking"], p["num_spec_tokens"])
           for p in SET["profiles"]}
    for name in carried:
        [(_, modes)] = yaml.safe_load((GOLDEN / name).read_text()).items()
        for mode, cells in modes.items():
            for k in cells:
                assert (name, mode.removeprefix("thinking_"), k) in got


def test_every_model_is_a_catalog_model_and_every_curve_left_out_is_named():
    catalog = {p.parent.name for p in (CATALOG / "models").glob("*/model.yaml")}
    assert {p["model"] for p in SET["profiles"]} <= catalog
    carried = {p["provenance"]["source"].rsplit("/", 1)[-1] for p in SET["profiles"]}
    for f in sorted(GOLDEN.glob("*.yaml")):
        if f.name not in carried:
            assert any(c.startswith(f"Not carried: {f.name} ") for c in SET["caveats"]), f.name


def test_a_copied_profile_carries_the_value_it_names():
    by_key = {emit.key_string(p["model"], p["speculative_method"], p["num_spec_tokens"],
                              p["thinking"], p["category"]): p for p in SET["profiles"]}
    copied = [p for p in SET["profiles"] if p["method"] == "copied"]
    assert copied
    for p in copied:
        assert by_key[p["copied_from"]]["method"] == "measured"
        assert by_key[p["copied_from"]]["acceptance_length"] == p["acceptance_length"]


def test_a_model_key_resolves_by_repository_basename_or_is_refused():
    models = emit.catalog_models(CATALOG)
    assert emit.catalog_name("glm-5.2-fp8", models) == ("glm-5.2-fp8", None)
    assert emit.catalog_name("deepseek-v4.1-flash", models) == ("deepseek-v4.1-flash", None)
    with pytest.raises(emit.Refused):
        # The catalog holds Qwen/Qwen3.5-397B-A17B, not the NVFP4 checkpoint InferenceX ran.
        emit.catalog_name("qwen3.5-397b-a17b-nvfp4", models)


def test_an_alias_whose_catalog_evidence_moved_is_an_error():
    models = dict(emit.catalog_models(CATALOG), **{"glm-5.3": "zai-org/GLM-5.3-Other"})
    with pytest.raises(SystemExit, match="alias glm-5.3-fp8"):
        emit.catalog_name("glm-5.3-fp8", models)


def test_an_unclassified_curve_fails_rather_than_being_skipped(tmp_path):
    golden = tmp_path / "golden"
    shutil.copytree(GOLDEN, golden)
    shutil.copy(golden / "glm5.2_mtp.yaml", golden / "glm5.4_mtp.yaml")
    with pytest.raises(SystemExit, match="unclassified golden curves"):
        emit.collect(golden, CATALOG)


def test_an_untranscribed_header_setting_fails_rather_than_being_dropped(tmp_path):
    golden = tmp_path / "golden"
    shutil.copytree(GOLDEN, golden)
    f = golden / "glm5.2_mtp.yaml"
    f.write_text(f.read_text().replace("| output_len: 4096", "| output_len: 4096 | min_p: 0.1", 1))
    with pytest.raises(SystemExit, match="min_p"):
        emit.collect(golden, CATALOG)


def test_k_values_the_image_rejected_have_no_profile():
    caveat = next(c for c in SET["caveats"] if "rejected by the image" in c)
    model, lo, hi = re.match(r"(\S+) with \S+ at K (\d+)-(\d+) ", caveat).groups()
    ks = {p["num_spec_tokens"] for p in SET["profiles"] if p["model"] == model}
    assert ks and not ks & set(range(int(lo), int(hi) + 1))


def _mutated(tmp_path, stem, old, new):
    golden = tmp_path / "golden"
    shutil.copytree(GOLDEN, golden)
    f = golden / f"{stem}.yaml"
    assert old in f.read_text()
    f.write_text(f.read_text().replace(old, new, 1))
    return golden


@pytest.mark.parametrize("piece", ["min-p: 0.1", "min_p=0.1", "top_p2: 0.9"])
def test_a_header_value_of_any_spelling_fails_unless_transcribed(tmp_path, piece):
    """Hyphens, digits and `=` in a key must not let a stated value slip past the parser."""
    golden = _mutated(tmp_path, "glm5.2_mtp", "| output_len: 4096", f"| output_len: 4096 | {piece}")
    with pytest.raises(SystemExit, match="not transcribed|does not transcribe"):
        emit.collect(golden, CATALOG)


def test_an_uncarried_curve_header_is_still_read_strictly(tmp_path):
    golden = _mutated(tmp_path, "qwen3.8next_mtp", "| output_len: 4096",
                      "| output_len: 4096 | min-p: 0.1")
    with pytest.raises(SystemExit, match="min-p"):
        emit.collect(golden, CATALOG)


def test_speculative_config_pieces_are_transcribed():
    h = emit.read_header(GOLDEN / "dsv4-pro-0813-dspark.yaml")
    assert h.get("method") == "dspark"
    assert h.get("draft_sample_method") == "probabilistic"


def test_a_stated_method_must_be_the_curves_method(tmp_path):
    golden = _mutated(tmp_path, "dsv41flash_dspark", "# method: dspark |", "# method: mtp |")
    with pytest.raises(SystemExit, match="states method 'mtp'"):
        emit.collect(golden, CATALOG)


def test_a_header_with_no_method_field_is_checked_by_its_measured_on_line(tmp_path):
    assert emit.read_header(GOLDEN / "glm5.2_mtp.yaml").get("method") is None
    golden = _mutated(tmp_path, "glm5.2_mtp", "(B300, vLLM MTP)", "(B300, vLLM EAGLE3)")
    with pytest.raises(SystemExit, match="measured with 'vLLM EAGLE3'"):
        emit.collect(golden, CATALOG)


def test_an_unreviewed_collector_fails_rather_than_its_precision_going_unstated(tmp_path):
    golden = _mutated(tmp_path, "glm5.2_mtp", "glm52_fp4_b300_vllm.sh", "glm53_fp4_b300_vllm.sh")
    with pytest.raises(SystemExit, match="not in COLLECTORS"):
        emit.collect(golden, CATALOG)
