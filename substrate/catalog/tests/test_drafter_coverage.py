"""Tests for the drafter coverage report (blis-catalog drafters, design §12.1)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "drafter_coverage.py"
sys.path.insert(0, str(SCRIPT.parent))
import drafter_coverage as dc  # noqa: E402


def catalog(tmp_path, models: dict, methods: dict | None = None) -> Path:
    """A catalog of cards. `methods` maps a model to the method its graph states for its
    speculator; a drafter without an entry gets `eagle3`, a target without one gets `mtp`
    (the committed graphs' spelling is deepseek_mtp, minimax_m3_mtp ...: vLLM's `mtp`)."""
    methods = methods or {}
    for name, card in models.items():
        d = tmp_path / "models" / name
        d.mkdir(parents=True)
        (d / "model.yaml").write_text(yaml.safe_dump({"name": name, **card}))
        method = methods.get(name, "eagle3" if card.get("targets") else "mtp")
        if method:
            (d / "graph.yaml").write_text(yaml.safe_dump({"name": name, "speculator": {"method": method}}))
    return tmp_path


TARGET = {"source": {"repo": "Org/Target"}, "drafters": ["speculator", "t-eagle3"]}
DRAFTER = {"source": {"repo": "Lab/T-EAGLE3"}, "targets": ["target"]}


def dep(draft="Lab/T-EAGLE3", method="eagle3", target="Org/Target"):
    return dc.Deployment(target, method, draft, "recipe.yaml")


def gaps(tmp_path, models, deployments, methods=None):
    return dc.check(dc.Catalog(catalog(tmp_path, models, methods)), deployments)


def test_a_linked_drafter_is_no_gap(tmp_path):
    g = gaps(tmp_path, {"target": TARGET, "t-eagle3": DRAFTER}, [dep(), dep(draft=None, method="mtp")])
    assert not any(g.values())


def test_a_drafter_the_catalog_lacks_is_reported(tmp_path):
    g = gaps(tmp_path, {"target": TARGET}, [dep()])
    assert g["missing_drafter"] == ["Lab/T-EAGLE3 (eagle3) drafts for Org/Target"]


def test_a_drafter_that_does_not_name_the_target_is_reported(tmp_path):
    g = gaps(tmp_path, {"target": TARGET, "t-eagle3": {**DRAFTER, "targets": ["other"]}}, [dep()])
    assert len(g["unlinked_drafter"]) == 1


def test_a_target_that_does_not_list_the_drafter_is_reported(tmp_path):
    g = gaps(tmp_path, {"target": {**TARGET, "drafters": ["speculator"]}, "t-eagle3": DRAFTER}, [dep()])
    assert len(g["unlinked_drafter"]) == 1


def test_a_speculator_the_target_does_not_list_is_reported(tmp_path):
    g = gaps(tmp_path, {"target": {**TARGET, "drafters": []}, "t-eagle3": DRAFTER}, [dep(draft=None, method="mtp")])
    assert len(g["missing_speculator"]) == 1


def test_a_missing_model_is_reported_apart_from_a_missing_drafter(tmp_path):
    g = gaps(tmp_path, {}, [dep()])
    assert g["missing_model"] == ["Org/Target"] and not g["missing_drafter"]


def test_repositories_compare_without_regard_to_case(tmp_path):
    g = gaps(tmp_path, {"target": TARGET, "t-eagle3": DRAFTER}, [dep(draft="lab/t-eagle3", target="org/target")])
    assert not any(g.values())


def test_vllm_recipe_flag_form_and_inferencex_key_form_are_read(tmp_path):
    v = tmp_path / "recipes" / "Org"
    v.mkdir(parents=True)
    (v / "Target.yaml").write_text(yaml.safe_dump({"args": ["--speculative-config",
        json.dumps({"method": "eagle3", "model": "Lab/T-EAGLE3", "num_speculative_tokens": 3})]}))
    x = tmp_path / "ix"
    x.mkdir()
    (x / "r.yaml").write_text("served-model-name: Org/Target\nspeculative-config: '{\"method\":\"dspark\"}'\n")
    assert dc.read_vllm_recipes(tmp_path / "recipes") == [
        dc.Deployment("Org/Target", "eagle3", "Lab/T-EAGLE3", str(v / "Target.yaml"))]
    assert dc.read_inferencex(x) == [dc.Deployment("Org/Target", "dspark", None, str(x / "r.yaml"))]


def test_the_committed_catalog_holds_the_minimax_m3_eagle3_heads(tmp_path):
    """Both heads vLLM's MiniMax-M3 recipe serves are in the catalog and linked to both
    quantisations; the Kimi-K3 DSpark head is found, and RedHatAI's separate Kimi-K3 speculator, which vLLM's recipe also names, is reported while the catalog lacks it."""
    cat = dc.Catalog(ROOT)
    ds = [dc.Deployment("MiniMaxAI/MiniMax-M3", "eagle3", f"Inferact/MiniMax-M3-EAGLE3{s}", "r")
          for s in ("", "-GQA")]
    ds += [dc.Deployment("MiniMaxAI/MiniMax-M3-MXFP8", "eagle3", "Inferact/MiniMax-M3-EAGLE3-GQA", "r")]
    assert not any(dc.check(cat, ds).values())
    assert not any(dc.check(cat, [dc.Deployment("moonshotai/Kimi-K3", "dspark", "Inferact/Kimi-K3-DSpark", "r")]).values())
    bad = dc.check(cat, [dc.Deployment("moonshotai/Kimi-K3", "dspark", "RedHatAI/Kimi-K3-speculator.dspark", "r")])
    assert bad["missing_drafter"]


def test_the_command_exits_nonzero_on_a_gap_unless_report_only(tmp_path):
    cat = catalog(tmp_path / "c", {"target": TARGET})
    r = tmp_path / "recipes" / "Org"
    r.mkdir(parents=True)
    (r / "Target.yaml").write_text(yaml.safe_dump({"args": ["--speculative-config",
        json.dumps({"method": "eagle3", "model": "Lab/T-EAGLE3"})]}))
    run = lambda *extra: subprocess.run([sys.executable, str(SCRIPT), "--catalog", str(cat),
                                         "--vllm-recipes", str(tmp_path / "recipes"), *extra],
                                        capture_output=True, text=True)
    assert run().returncode == 1
    assert run("--report-only").returncode == 0


def test_a_draft_shipped_in_the_targets_repository_is_found_and_never_taken_for_the_target(tmp_path):
    """MiniMax-M3.1-Flash's DSpark lives in dspark/ of the target's repository: the target is the
    entry without targets, a deployment with no draft checkpoint is satisfied by the shipped
    drafter, and one naming the shared repository as its draft finds the drafter entry."""
    shipped = {"source": {"repo": "Org/Flash", "path": "dspark/config.json"}, "targets": ["flash"]}
    models = {"flash": {"source": {"repo": "Org/Flash"}, "drafters": ["flash-dspark"]},
              "flash-dspark": shipped}
    ms = {"flash": None, "flash-dspark": "dspark"}
    assert not any(gaps(tmp_path, models, [dep(draft=None, method="dspark", target="Org/Flash")], ms).values())
    assert not any(gaps(tmp_path / "x", models, [dep(draft="Org/Flash", method="dspark", target="Org/Flash")], ms).values())
    # A target that lists no speculator and ships no draft is still reported.
    bare = {"flash": {"source": {"repo": "Org/Flash"}, "drafters": []}}
    assert len(gaps(tmp_path / "y", bare, [dep(draft=None, method="dspark", target="Org/Flash")])["missing_speculator"]) == 1


def test_an_in_graph_speculator_of_another_method_does_not_satisfy_a_deployment(tmp_path):
    """A target listing `speculator` whose graph states MTP does not run EAGLE3 or DSpark."""
    for method in ("eagle3", "dspark", "dflash"):
        g = gaps(tmp_path / method, {"target": TARGET}, [dep(draft=None, method=method)])
        assert len(g["missing_speculator"]) == 1, method
        assert "mtp" in g["missing_speculator"][0]
    # The same speculator does satisfy the deployment that states `mtp`, and one that states
    # the model-specific name vLLM rewrites to it.
    for method in ("mtp", "deepseek_mtp"):
        assert not any(gaps(tmp_path / method, {"target": TARGET}, [dep(draft=None, method=method)],
                            {"target": "deepseek_mtp"}).values())


def test_a_target_with_no_speculator_in_its_graph_runs_no_method_in_graph(tmp_path):
    g = gaps(tmp_path, {"target": TARGET}, [dep(draft=None, method="mtp")], {"target": None})
    assert len(g["missing_speculator"]) == 1


def test_a_shipped_drafter_of_another_method_does_not_satisfy_a_deployment(tmp_path):
    """The target ships a DSpark drafter in its repository; an eagle3 or mtp deployment with no
    draft checkpoint is not served by it, a dspark one is."""
    models = {"flash": {"source": {"repo": "Org/Flash"}, "drafters": ["flash-dspark"]},
              "flash-dspark": {"source": {"repo": "Org/Flash", "path": "dspark/config.json"}, "targets": ["flash"]}}
    ms = {"flash": None, "flash-dspark": "dspark"}
    for method in ("eagle3", "mtp"):
        g = gaps(tmp_path / method, models, [dep(draft=None, method=method, target="Org/Flash")], ms)
        assert len(g["missing_speculator"]) == 1, method
    assert not any(gaps(tmp_path / "ok", models, [dep(draft=None, method="dspark", target="Org/Flash")], ms).values())


def test_in_graph_mtp_beside_a_same_repository_dspark_drafter_does_not_certify_eagle3(tmp_path):
    """The review's case: in-graph MTP plus a same-repository DSpark drafter, deployment eagle3."""
    models = {"flash": {"source": {"repo": "Org/Flash"}, "drafters": ["speculator", "flash-dspark"]},
              "flash-dspark": {"source": {"repo": "Org/Flash", "path": "dspark/config.json"}, "targets": ["flash"]}}
    ms = {"flash": "minimax_m3_mtp", "flash-dspark": "dspark"}
    g = gaps(tmp_path, models, [dep(draft=None, method="eagle3", target="Org/Flash")], ms)
    assert len(g["missing_speculator"]) == 1
    for method in ("mtp", "dspark"):
        assert not any(gaps(tmp_path / method, models, [dep(draft=None, method=method, target="Org/Flash")], ms).values())


def test_a_named_draft_checkpoint_must_run_the_deployments_method(tmp_path):
    """A linked drafter whose graph is DSpark is no EAGLE3 head."""
    models = {"target": TARGET, "t-eagle3": DRAFTER}
    g = gaps(tmp_path, models, [dep(method="dspark")])
    assert len(g["wrong_method"]) == 1 and "eagle3" in g["wrong_method"][0]
    assert not any(gaps(tmp_path / "ok", models, [dep(method="eagle3")]).values())
    # A drafter whose graph has no speculator runs nothing.
    g = gaps(tmp_path / "none", models, [dep()], {"t-eagle3": None})
    assert len(g["wrong_method"]) == 1


def test_a_wrong_method_fails_the_command_unless_report_only(tmp_path):
    cat = catalog(tmp_path / "c", {"target": TARGET})
    r = tmp_path / "recipes" / "Org"
    r.mkdir(parents=True)
    (r / "Target.yaml").write_text(yaml.safe_dump({"args": ["--speculative-config", json.dumps({"method": "eagle3"})]}))
    run = lambda *extra: subprocess.run([sys.executable, str(SCRIPT), "--catalog", str(cat),
                                         "--vllm-recipes", str(tmp_path / "recipes"), *extra],
                                        capture_output=True, text=True)
    assert run().returncode == 1 and "missing_speculator" in run().stdout
    assert run("--report-only").returncode == 0


def test_the_committed_catalog_does_not_certify_the_wrong_method():
    cat = dc.Catalog(ROOT)
    # DeepSeek-V3 holds a DeepSeek MTP speculator in its graph: mtp passes, eagle3 and dspark do not.
    ok = dc.check(cat, [dc.Deployment("deepseek-ai/DeepSeek-V3", "mtp", None, "r")])
    assert not any(ok.values())
    for method in ("eagle3", "dspark"):
        assert dc.check(cat, [dc.Deployment("deepseek-ai/DeepSeek-V3", method, None, "r")])["missing_speculator"]
    # MiniMax-M3.1-Flash ships a DSpark drafter and no MTP.
    assert not any(dc.check(cat, [dc.Deployment("MiniMaxAI/MiniMax-M3.1-Flash", "dspark", None, "r")]).values())
    assert dc.check(cat, [dc.Deployment("MiniMaxAI/MiniMax-M3.1-Flash", "mtp", None, "r")])["missing_speculator"]
    # An EAGLE3 deployment naming a DSpark checkpoint of the right target has the wrong method.
    assert dc.check(cat, [dc.Deployment("moonshotai/Kimi-K3", "eagle3", "Inferact/Kimi-K3-DSpark", "r")])["wrong_method"]
