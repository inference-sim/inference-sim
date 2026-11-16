"""The acceptance-profile set is InferenceX's golden curves, transcribed and nothing else.

`emit_acceptance_profiles.py --check` is the drift gate, as `--check` is for the coefficient
writers. Its inputs are in the repository (validator/testdata/inferencex and the catalog),
so unlike those writers' gates it runs in every CI job rather than only in `rederive`.
The other tests re-read the vendored curves by a second path, and check what the emitter
does with a curve it has never seen: one whose model and drafter resolve is carried with
its drafter, and one that does not resolve is left out and named, never mapped to a near
name.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "emit_acceptance_profiles.py"
CATALOG = REPO.parent / "catalog"
SET = yaml.safe_load((REPO / "acceptance" / "inferencex-golden-al.yaml").read_text())

sys.path.insert(0, str(REPO / "scripts"))
import emit_acceptance_profiles as emit  # noqa: E402

# The pinned InferenceX files, fetched by scripts/fetch_inferencex_curves.py (or
# $INFERENCEX_GOLDEN). Absent, every test here fails, naming the fetch: none skips (X16).
GOLDEN = emit.GOLDEN
emit.verify_inputs(GOLDEN)

GLM52 = "glm-5.2-fp8__mtp__native__default-default"
EAGLE = "minimax-m3__eagle3__minimax-m3-eagle3__default-default"
GQA = "minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-default"
GQA_BLOCK = "minimax-m3__eagle3__minimax-m3-eagle3-gqa__default-block"
MUSE = "muse-spark-open-816b-a42b-nvfp4__dflash__muse-spark-open-dflash2__default-default"


def stem_of(p: dict) -> str:
    return p["provenance"]["source"].rsplit("/", 1)[-1].removesuffix(".yaml")


def generated(golden: Path) -> dict:
    return yaml.safe_load(emit.generate(golden, CATALOG))


def copy_golden(tmp_path: Path) -> Path:
    golden = tmp_path / "golden"
    shutil.copytree(GOLDEN, golden)
    return golden


def add_identity(golden: Path, stem: str, identity: dict) -> None:
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    ids[stem] = identity
    (golden / emit.IDENTITIES).write_text(yaml.safe_dump(ids, sort_keys=False))


def remove_curve(golden: Path, stem: str) -> None:
    (golden / f"{stem}.yaml").unlink()
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    del ids[stem]
    (golden / emit.IDENTITIES).write_text(yaml.safe_dump(ids, sort_keys=False))


def test_the_committed_set_is_what_the_emitter_generates():
    r = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True,
                       text=True, cwd=str(REPO), timeout=120,
                       env={"PATH": "", "BLIS_CATALOG": "", "INFERENCEX_GOLDEN": str(GOLDEN)})
    assert r.returncode == 0, (
        "acceptance/inferencex-golden-al.yaml is not what emit_acceptance_profiles.py "
        f"renders from the vendored curves; re-run it.\n{r.stderr}")


def test_every_profile_cites_the_pinned_file_by_commit_and_hash():
    lock = json.loads(emit.LOCK.read_text())
    for p in SET["profiles"]:
        assert lock["commit"] in p["provenance"]["source"]
        body = (GOLDEN / f"{stem_of(p)}.yaml").read_bytes()
        assert p["provenance"]["sha256"] == hashlib.sha256(body).hexdigest() \
            == lock["files"][f"{stem_of(p)}.yaml"], stem_of(p)


def test_no_inferencex_file_is_committed():
    """X12: the repository holds the pin (identifiers and hashes), never the files."""
    tracked = subprocess.run(["git", "ls-files", "validator/testdata/inferencex"], cwd=str(REPO),
                             capture_output=True, text=True, check=True).stdout.split()
    assert tracked and all(not t.endswith(tuple(json.loads(emit.LOCK.read_text())["files"]))
                           for t in tracked), tracked


@pytest.mark.parametrize("change", ["edit", "extra", "missing"])
def test_the_emitter_refuses_files_that_are_not_the_pinned_ones(tmp_path, change):
    golden = copy_golden(tmp_path)
    if change == "edit":
        f = golden / f"{GLM52}.yaml"
        f.write_text(f.read_text().replace("1.84", "1.85", 1))
    elif change == "extra":
        shutil.copy(golden / f"{GLM52}.yaml", golden / "glm-5.4__mtp__native__default-default.yaml")
    else:
        (golden / f"{GLM52}.yaml").unlink()
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "--check"], capture_output=True, text=True,
                       cwd=str(REPO), timeout=120, env={"PATH": "", "INFERENCEX_GOLDEN": str(golden)})
    assert r.returncode != 0 and "curves.json" in r.stderr, r.stderr


def test_every_profile_is_the_value_its_source_file_states():
    """A second path: read each cited file directly and compare cell by cell."""
    for p in SET["profiles"]:
        doc = yaml.safe_load((GOLDEN / f"{stem_of(p)}.yaml").read_text())
        [modes] = [v for k, v in doc.items() if k not in ("source_run", "collection")]
        stated = modes[f"thinking_{p['thinking']}"][p["num_spec_tokens"]]
        assert p["acceptance_length"] == stated, (stem_of(p), p["thinking"], p["num_spec_tokens"])


def test_every_cell_of_a_carried_curve_is_carried():
    got = {(stem_of(p), p["thinking"], p["num_spec_tokens"]) for p in SET["profiles"]}
    for stem in {s for s, _, _ in got}:
        doc = yaml.safe_load((GOLDEN / f"{stem}.yaml").read_text())
        [modes] = [v for k, v in doc.items() if k not in ("source_run", "collection")]
        for mode, cells in modes.items():
            for k in cells:
                assert (stem, mode.removeprefix("thinking_"), k) in got


def test_every_curve_left_out_is_named_with_its_reason():
    catalog = {p.parent.name for p in (CATALOG / "models").glob("*/model.yaml")}
    assert {p["model"] for p in SET["profiles"]} <= catalog
    assert {p["drafter"] for p in SET["profiles"] if "drafter" in p} <= catalog
    carried = {stem_of(p) for p in SET["profiles"]}
    for f in sorted(GOLDEN.glob("*__*.yaml")):
        if f.stem not in carried:
            assert any(c.startswith(f"Not carried: {f.name} (") and ", because " in c
                       for c in SET["caveats"]), f.name


def test_a_new_curve_whose_model_resolves_is_carried_with_no_script_edit(tmp_path):
    """InferenceX adds a curve file and its identity; the emitter carries it as it is."""
    golden = copy_golden(tmp_path)
    new = "glm-5.2__mtp__native__default-default"
    text = (golden / f"{GLM52}.yaml").read_text().replace("glm-5.2-fp8", "glm-5.2")
    (golden / f"{new}.yaml").write_text(text)
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    add_identity(golden, new, ids[GLM52])
    out = generated(golden)
    ours = [p for p in out["profiles"] if stem_of(p) == new]
    assert ours and {p["model"] for p in ours} == {"glm-5.2"}
    assert all(p["method"] == "measured" and "drafter" not in p for p in ours)


def test_a_curve_measured_with_a_separate_drafter_names_it():
    eagle = [p for p in SET["profiles"] if stem_of(p) == EAGLE]
    assert eagle and {p["drafter"] for p in eagle} == {"minimax-m3-eagle3"}
    assert {p["provenance"]["settings"]["draft_model"] for p in eagle} == {
        "Inferact/MiniMax-M3-EAGLE3"}


def test_two_drafters_of_one_target_are_two_profiles(tmp_path):
    """The base and GQA EAGLE3 curves share every key field but the drafter. With the drafter
    in the key, both are carried; neither stands for the other."""
    golden = copy_golden(tmp_path)
    remove_curve(golden, GQA_BLOCK)
    out = generated(golden)
    by = {}
    for p in out["profiles"]:
        if p["model"] == "minimax-m3" and p["thinking"] == "on" and p["num_spec_tokens"] == 3:
            by[p["drafter"]] = p["acceptance_length"]
    stated = {d: yaml.safe_load((golden / f"{s}.yaml").read_text())["minimax-m3"]["thinking_on"][3]
              for d, s in (("minimax-m3-eagle3", EAGLE), ("minimax-m3-eagle3-gqa", GQA))}
    assert stated["minimax-m3-eagle3"] != stated["minimax-m3-eagle3-gqa"]
    assert by == stated


def test_curves_that_differ_in_sampler_or_verifier_are_carried_with_them():
    """The GQA drafter's two curves differ only in the verifier, and Kimi-K3's three DSpark
    curves in the sampler and verifier (inference-sim/blis-substrate#174): each is carried under
    the key that states what it was measured with, the engine's default left out."""
    got = {}
    for p in SET["profiles"]:
        got.setdefault(stem_of(p), set()).add((p.get("draft_sampler", ""), p.get("verifier", "")))
    assert got[GQA] == {("", "")} and got[GQA_BLOCK] == {("", "block")}
    kimi = "kimi-k3__dspark__kimi-k3-dspark__"
    assert got[kimi + "default-default"] == {("", "")}
    assert got[kimi + "probabilistic-block"] == {("probabilistic", "block")}
    assert got[kimi + "probabilistic-default"] == {("probabilistic", "")}


def test_a_curve_with_its_methods_only_curve_states_neither():
    """DeepSeek-V4.1-Flash was measured with probabilistic drafting and block verification, but it
    is its target's only DSpark curve, so its key states neither and every run's lookup finds it."""
    p = next(p for p in SET["profiles"] if stem_of(p).startswith("deepseek-v4.1-flash__"))
    assert "draft_sampler" not in p and "verifier" not in p
    assert p["provenance"]["settings"]["verify"] == "block"


def test_curves_that_claim_one_key_are_both_refused(tmp_path):
    """Two curves of one target, drafter, method, sampler and verifier: nothing tells them apart."""
    golden = copy_golden(tmp_path)
    new = "glm-5.2-fp8__mtp__native__default-relaxed"
    (golden / f"{new}.yaml").write_text((golden / f"{GLM52}.yaml").read_text())
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    add_identity(golden, new, ids[GLM52])
    out = generated(golden)
    carried = {stem_of(p) for p in out["profiles"]}
    assert GLM52 not in carried and new not in carried
    assert any(c.startswith(f"Not carried: {new}.yaml") and f"{GLM52}.yaml" in c for c in out["caveats"])


def test_a_verifier_the_key_has_no_word_for_fails(tmp_path):
    golden = copy_golden(tmp_path)
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    ids[GQA_BLOCK]["collection"]["verify"] = {"value": "typical", "source": "explicit"}
    (golden / emit.IDENTITIES).write_text(yaml.safe_dump(ids, sort_keys=False))
    with pytest.raises(SystemExit, match="typical"):
        emit.collect(golden, CATALOG)


def test_a_drafter_the_catalog_lacks_is_refused_and_named(tmp_path):
    """A curve measured with a drafter no catalog entry has (a fixture's) is not carried, and
    the caveat names the drafter; onboarding a real drafter cannot break this."""
    golden = copy_golden(tmp_path)
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    ids[EAGLE]["collection"]["draft"] = "Org/Minimax-M3-Eagle3"
    (golden / emit.IDENTITIES).write_text(yaml.safe_dump(ids, sort_keys=False))
    out = generated(golden)
    assert EAGLE not in {stem_of(p) for p in out["profiles"]}
    caveat = next(c for c in out["caveats"] if c.startswith(f"Not carried: {EAGLE}.yaml"))
    assert "Org/Minimax-M3-Eagle3 is not in the catalog" in caveat


def test_a_drafter_measured_at_another_revision_is_refused(tmp_path):
    golden = copy_golden(tmp_path)
    stem = "kimi-k3.1-pro__dspark__kimi-k3.1-pro-dspark-preview__probabilistic-block"
    f = golden / f"{stem}.yaml"
    assert "@ 6a2ef20a" in f.read_text()
    f.write_text(f.read_text().replace("@ 6a2ef20a", "@ 0123abcd"))
    out = generated(golden)
    assert stem not in {stem_of(p) for p in out["profiles"]}
    assert any(c.startswith(f"Not carried: {stem}.yaml") and "0123abcd" in c
               for c in out["caveats"])


def test_an_explicit_skip_needs_a_reason(monkeypatch):
    monkeypatch.setattr(emit, "SKIP", {GLM52: " "})
    with pytest.raises(SystemExit, match="states a reason"):
        emit.collect(GOLDEN, CATALOG)


def test_an_explicit_skip_is_left_out_with_its_reason(monkeypatch):
    monkeypatch.setattr(emit, "SKIP", {GLM52: "a test reason"})
    carried, skipped, _, _ = emit.collect(GOLDEN, CATALOG)
    assert GLM52 not in {c.curve.stem for c in carried}
    assert (GLM52, "glm-5.2-fp8", "a test reason") in skipped


def test_a_provisional_curve_is_refused(tmp_path):
    golden = copy_golden(tmp_path)
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    ids[GLM52]["collection"]["provisional"] = True
    (golden / emit.IDENTITIES).write_text(yaml.safe_dump(ids, sort_keys=False))
    out = generated(golden)
    assert GLM52 not in {stem_of(p) for p in out["profiles"]}


def test_a_curve_file_with_no_identity_fails(tmp_path):
    golden = copy_golden(tmp_path)
    shutil.copy(golden / f"{GLM52}.yaml", golden / "glm-5.4__mtp__native__default-default.yaml")
    with pytest.raises(SystemExit, match="identities.yaml states no identity"):
        emit.collect(golden, CATALOG)


def test_an_untranscribed_identity_fact_fails_rather_than_being_dropped(tmp_path):
    golden = copy_golden(tmp_path)
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    ids[GLM52]["collection"]["tree_width"] = {"value": 4, "source": "explicit"}
    (golden / emit.IDENTITIES).write_text(yaml.safe_dump(ids, sort_keys=False))
    with pytest.raises(SystemExit, match="tree_width"):
        emit.collect(golden, CATALOG)


def test_an_untranscribed_sampling_parameter_fails_rather_than_being_dropped(tmp_path):
    golden = copy_golden(tmp_path)
    f = golden / f"{GLM52}.yaml"
    assert "top_p: 0.95, repetition_penalty" in f.read_text()
    f.write_text(f.read_text().replace("top_p: 0.95, repetition_penalty",
                                       "top_p: 0.95, min_p: 0.1, repetition_penalty"))
    with pytest.raises(SystemExit, match="min_p"):
        emit.collect(golden, CATALOG)


def test_identity_facts_and_per_mode_sampling_are_transcribed():
    p = next(p for p in SET["profiles"] if stem_of(p).startswith("deepseek-v4.1-flash__")
             and p["thinking"] == "off")
    s = p["provenance"]["settings"]
    assert (s["draft_sample"], s["verify"], s["adaptive"]) == ("probabilistic", "block", "false")
    assert s["chat_template_kwargs"] == '{"thinking": false}'
    assert p["provenance"]["sampling"]["temperature"] == 1.0


def test_a_transferred_profile_cites_its_measured_source(tmp_path):
    """With a catalog that lacks glm-5.3-nvfp4 (a fixture, so the real entry cannot change it),
    GLM-5.3's curve is transferred, and each transferred profile names its rule and source."""
    catalog = tmp_path / "catalog"
    shutil.copytree(CATALOG, catalog)
    shutil.rmtree(catalog / "models" / "glm-5.3-nvfp4", ignore_errors=True)
    out = yaml.safe_load(emit.generate(GOLDEN, catalog))
    moved = [p for p in out["profiles"] if p["method"] == "transferred"]
    assert moved
    for p in moved:
        assert p["transfer_rule"] and p["rationale"]
        assert p["transferred_from"].split("/", 1)[1] == p_key_tail(p)


def p_key_tail(p: dict) -> str:
    return f"{p['speculative_method']}/k{p['num_spec_tokens']}/thinking-{p['thinking']}/{p['category']}"


def test_k_values_the_image_rejected_have_no_profile():
    caveat = next(c for c in SET["caveats"] if "rejected by the image" in c)
    model, lo, hi = re.match(r"(\S+) with \S+ at K (\d+)-(\d+) ", caveat).groups()
    ks = {p["num_spec_tokens"] for p in SET["profiles"] if p["model"] == model}
    assert ks and not ks & set(range(int(lo), int(hi) + 1))


def test_a_model_key_resolves_by_repository_basename_or_is_refused():
    targets = emit.load_catalog(CATALOG).targets()
    assert emit.catalog_name("glm-5.2-fp8", targets) == "glm-5.2-fp8"
    assert emit.catalog_name("kimi-k3.1-pro", targets) == "kimi-k3.1"
    with pytest.raises(emit.Refused):
        # A key no catalog repository's basename is (a near name is not a match).
        emit.catalog_name("glm-5.2-fp8-not-in-any-catalog", targets)


def test_a_native_drafter_in_the_targets_own_repository_is_that_entry():
    """MiniMax-M3.1-Flash ships its DSpark drafter under dspark/ in its own repository, which
    InferenceX calls native and the catalog holds as minimax-m3.1-flash-dspark."""
    cat = emit.load_catalog(CATALOG)
    assert emit.catalog_drafter("minimax-m3.1-flash", "native", cat) == "minimax-m3.1-flash-dspark"
    assert emit.catalog_drafter("deepseek-v4.1-flash", "native", cat) == ""


def test_hardware_a_file_states_must_agree_with_its_run(tmp_path):
    golden = copy_golden(tmp_path)
    f = golden / f"{GLM52}.yaml"
    assert "(B300, vLLM MTP)" in f.read_text()
    f.write_text(f.read_text().replace("(B300, vLLM MTP)", "(H200, vLLM MTP)"))
    with pytest.raises(SystemExit, match="ran on b300"):
        emit.collect(golden, CATALOG)


GLM53 = "glm-5.3-nvfp4__mtp__native__default-default"


def test_a_measured_curve_for_a_transfers_model_makes_the_transfer_stale(tmp_path):
    """If InferenceX measures GLM-5.3 itself, the transfer from its NVFP4 checkpoint must not
    quietly collide with it and leave glm-5.3 with no profile: the run fails and names both."""
    golden = copy_golden(tmp_path)
    new = "glm-5.3__mtp__native__default-default"
    (golden / f"{new}.yaml").write_text((golden / f"{GLM53}.yaml").read_text()
                                        .replace("glm-5.3-nvfp4", "glm-5.3"))
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    add_identity(golden, new, ids[GLM53])
    with pytest.raises(SystemExit, match=f"TRANSFERS {GLM53} is stale: {new} measures glm-5.3"):
        emit.collect(golden, CATALOG)


def test_a_transfer_whose_curve_is_gone_fails(tmp_path):
    golden = copy_golden(tmp_path)
    remove_curve(golden, GLM53)
    with pytest.raises(SystemExit, match=f"TRANSFERS {GLM53} names no curve file"):
        emit.collect(golden, CATALOG)


def _header(tmp_path, stem, old, new):
    golden = copy_golden(tmp_path)
    f = golden / f"{stem}.yaml"
    assert old in f.read_text()
    f.write_text(f.read_text().replace(old, new, 1))
    return golden


def test_an_untranscribed_header_sampling_parameter_fails(tmp_path):
    golden = _header(tmp_path, GLM53, "# thinking_on: chat template default",
                     "# thinking_on: temp 1.0 min_p 0.05 | chat template default")
    with pytest.raises(SystemExit, match="min_p"):
        emit.collect(golden, CATALOG)


def test_a_header_repetition_penalty_is_transcribed(tmp_path):
    golden = _header(tmp_path, GLM53, "# thinking_on: chat template default",
                     "# thinking_on: temp 1.0 repetition_penalty 1.05 | chat template default")
    out = generated(golden)
    p = next(p for p in out["profiles"] if stem_of(p) == GLM53)
    assert p["provenance"]["settings"]["repetition_penalty"] == "1.05"


def test_a_header_value_stated_twice_differently_fails(tmp_path):
    golden = _header(tmp_path, GLM53, "| top_p: 0.95 |", "| top_p: 0.95 | top_p: 0.9 |")
    with pytest.raises(SystemExit, match="stated twice, differently"):
        emit.collect(golden, CATALOG)


def test_two_different_measured_on_lines_fail(tmp_path):
    line = "# Measured on glm-5.3-nvfp4 (NVIDIA B300 SXM6 AC, SGLang MTP), per num_speculative_tokens."
    golden = _header(tmp_path, GLM53, line, line + "\n" + line.replace("B300 SXM6 AC", "B200"))
    with pytest.raises(SystemExit, match="two different measured-on lines"):
        emit.collect(golden, CATALOG)


def test_a_curve_no_catalog_model_resolves_is_still_checked(tmp_path):
    """Qwen3.8-Flash-Next is not in the catalog, so it is not carried; a format change in it
    still fails the run, rather than on the day the model is added."""
    golden = copy_golden(tmp_path)
    stem = "qwen3.8-flash-next-fp8__mtp__native__default-default"
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    ids[stem]["collection"]["tree_width"] = {"value": 4, "source": "explicit"}
    (golden / emit.IDENTITIES).write_text(yaml.safe_dump(ids, sort_keys=False))
    with pytest.raises(SystemExit, match="tree_width"):
        emit.collect(golden, CATALOG)


def test_an_identity_with_no_draft_fails(tmp_path):
    golden = copy_golden(tmp_path)
    ids = yaml.safe_load((golden / emit.IDENTITIES).read_text())
    del ids[EAGLE]["collection"]["draft"]
    (golden / emit.IDENTITIES).write_text(yaml.safe_dump(ids, sort_keys=False))
    with pytest.raises(SystemExit, match="no collection.draft"):
        emit.collect(golden, CATALOG)


def test_a_stated_drafter_revision_is_recorded_and_an_unstated_one_is_named():
    k31 = [p for p in SET["profiles"] if p.get("drafter") == "kimi-k3.1-dspark"]
    assert k31 and {p["provenance"]["settings"]["draft_revision"] for p in k31} == {"6a2ef20a"}
    caveat = next(c for c in SET["caveats"] if c.startswith("Drafter revision unstated"))
    assert "minimax-m3+minimax-m3-eagle3" in caveat and "kimi-k3.1" not in caveat


def test_the_identities_file_is_cited_with_its_hash():
    lock = json.loads(emit.LOCK.read_text())
    assert any(lock["files"]["identities.yaml"] in c and "identities.yaml" in c for c in SET["caveats"])


def test_a_curve_is_carried_on_its_own_model_and_transferred_until_measured(tmp_path):
    """While the catalog lacks glm-5.3-nvfp4, its curve is transferred to glm-5.3. Once the
    catalog has it, the curve is carried as measured on it and still transferred to glm-5.3,
    with the same values, since no curve measures glm-5.3 itself. Both catalogs are fixtures,
    so the test holds with or without the real entry."""
    without = tmp_path / "without"
    shutil.copytree(CATALOG, without)
    shutil.rmtree(without / "models" / "glm-5.3-nvfp4", ignore_errors=True)
    out = yaml.safe_load(emit.generate(GOLDEN, without))
    mine = [p for p in out["profiles"] if stem_of(p) == GLM53]
    assert mine and {(p["model"], p["method"]) for p in mine} == {("glm-5.3", "transferred")}
    with_ = tmp_path / "with"
    shutil.copytree(without, with_)
    d = with_ / "models" / "glm-5.3-nvfp4"
    shutil.copytree(with_ / "models" / "glm-5.3", d)
    doc = yaml.safe_load((d / "model.yaml").read_text())
    doc["name"] = "glm-5.3-nvfp4"
    doc["source"]["repo"] = "nvidia/GLM-5.3-NVFP4"
    (d / "model.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    g = yaml.safe_load((d / "graph.yaml").read_text())
    g["name"] = "glm-5.3-nvfp4"
    (d / "graph.yaml").write_text(yaml.safe_dump(g, sort_keys=False))
    out2 = yaml.safe_load(emit.generate(GOLDEN, with_))
    mine2 = [p for p in out2["profiles"] if stem_of(p) == GLM53]
    assert {(p["model"], p["method"]) for p in mine2} == {("glm-5.3-nvfp4", "measured"),
                                                          ("glm-5.3", "transferred")}
    value = lambda ps, m: {(p["thinking"], p["num_spec_tokens"]): p["acceptance_length"]
                           for p in ps if p["model"] == m}
    assert value(mine2, "glm-5.3") == value(mine, "glm-5.3") == value(mine2, "glm-5.3-nvfp4")


def step_rates(al: float, k: int) -> list[float]:
    """acceptance.MinimumVarianceRates, the engine's synthetic schedule."""
    full = int(al - 1)
    return [1.0 if i < full else (al - 1 - full if i == full else 0.0) for i in range(k)]


def test_the_set_states_a_shape_for_every_method_and_one_for_all():
    methods = {p["speculative_method"] for p in SET["profiles"]}
    stated = {s.get("speculative_method", "") for s in SET["shapes"]}
    assert methods | {""} <= stated
    for s in SET["shapes"]:
        assert s["fitted_from"] and all(emit.INFERENCEX_SHA in f for f in s["fitted_from"])


def test_a_natural_shape_predicts_a_held_out_curve_better_than_the_step():
    """Leave one source file out (both its thinking modes): fit its method's exponent on the
    other files, spread each held-out curve's largest-K acceptance length with it, and compare
    with the lengths it measured at every smaller K. The decay must beat the engine's step
    schedule on every curve, and by a wide margin overall."""
    tests = emit.shape_tests()
    flat = [(m, src, c) for m in tests for src, c in tests[m]]
    natural, step = [], []
    for m, src, cells in flat:
        rest = [c for mm, s2, c in flat if s2 != src and mm == m] or \
               [c for _, s2, c in flat if s2 != src]
        b = emit.fit_exponent(rest)
        k0 = max(cells)
        s = step_rates(cells[k0], k0)
        natural.append(emit.worst_error(cells, b))
        step.append(max(abs(1 + sum(s[:k]) - al) for k, al in cells.items()))
        assert natural[-1] < step[-1], (src, natural[-1], step[-1])
    natural.sort(), step.sort()
    assert natural[len(natural) // 2] * 4 < step[len(step) // 2]


def test_a_rate_only_pin_at_k7_recovers_the_measured_per_position_profile():
    """Muse-Spark-Open with DFlash2, measured at K 1-7, held out of the fit: with the dflash
    exponent fitted to the method's other curves, the rates its acceptance length at K 7 alone
    derives are within 0.05, at every position, of the per-position rates its curve measures
    (AL(k) - AL(k-1)), and sum to the pinned length. The step misses by far more."""
    tests = emit.shape_tests()["dflash"]
    [cells] = [c for src, c in tests if MUSE in src]
    others = [c for src, c in tests if MUSE not in src]
    assert others, "the dflash family has no other curve to fit on"
    r = emit.natural_rates(cells[7], 7, emit.fit_exponent(others))
    measured = [cells[k] - cells.get(k - 1, 1.0) for k in range(1, 8)]
    assert max(abs(a - b) for a, b in zip(r, measured)) <= 0.05
    assert abs(1 + sum(r) - cells[7]) < 1e-9
    assert max(abs(a - b) for a, b in zip(step_rates(cells[7], 7), measured)) > 0.05


def test_every_shape_is_an_estimate():
    assert SET["shapes"] and all(s["method"] in ("calibrated", "assumed") for s in SET["shapes"])
