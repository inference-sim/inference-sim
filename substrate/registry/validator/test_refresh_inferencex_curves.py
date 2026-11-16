"""The scheduled refresh moves every statement of the InferenceX pin together, records the new
files' hashes, and regenerates the set; a second run with the same files changes nothing, and a
run whose commit moved but whose files did not changes nothing either. A failed refresh leaves
the tree as it was.

It runs on a copy of the files it edits, so the repository is untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO.parent
NEW_SHA = "0123456789abcdef0123456789abcdef01234567"
GLM52 = "glm-5.2-fp8__mtp__native__default-default"

sys.path.insert(0, str(REPO / "scripts"))
import emit_acceptance_profiles as emit  # noqa: E402


def tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in ("registry/scripts/emit_acceptance_profiles.py",
                "registry/scripts/refresh_inferencex_curves.py",
                "registry/acceptance/inferencex-golden-al.yaml",
                "registry/validator/testdata/inferencex/curves.json",
                ".github/workflows/rederive.yml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, root / rel)
    return root


def run(root: Path, src: Path, sha: str = NEW_SHA) -> subprocess.CompletedProcess:
    env = dict(os.environ, BLIS_CATALOG=str(ROOT / "catalog"), INFERENCEX_GOLDEN="")
    return subprocess.run([sys.executable, "-I",
                           str(root / "registry/scripts/refresh_inferencex_curves.py"),
                           str(src), sha, "2026-12-01"],
                          capture_output=True, text=True, env=env, timeout=120)


def upstream(tmp_path: Path, new_curve: bool = True) -> Path:
    src = tmp_path / "upstream"
    shutil.copytree(emit.GOLDEN, src)
    if new_curve:
        new = "glm-5.2__mtp__native__default-default"
        (src / f"{new}.yaml").write_text(
            (src / f"{GLM52}.yaml").read_text().replace("glm-5.2-fp8", "glm-5.2"))
        ids = yaml.safe_load((src / "identities.yaml").read_text())
        ids[new] = ids[GLM52]
        (src / "identities.yaml").write_text(yaml.safe_dump(ids, sort_keys=False))
    return src


def snapshot(root: Path) -> dict:
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_a_refresh_moves_every_pin_and_carries_the_new_curve(tmp_path):
    root = tree(tmp_path)
    r = run(root, upstream(tmp_path))
    assert r.returncode == 0, r.stderr
    lock = json.loads((root / "registry/validator/testdata/inferencex/curves.json").read_text())
    assert (lock["commit"], lock["date"]) == (NEW_SHA, "2026-12-01")
    assert "glm-5.2__mtp__native__default-default.yaml" in lock["files"]
    assert f"INFERENCEX_SHA: {NEW_SHA}" in (root / ".github/workflows/rederive.yml").read_text()
    out = yaml.safe_load((root / "registry/acceptance/inferencex-golden-al.yaml").read_text())
    sources = {p["provenance"]["source"] for p in out["profiles"]}
    assert all(NEW_SHA in s for s in sources)
    assert any(s.endswith("/glm-5.2__mtp__native__default-default.yaml") for s in sources)


def test_a_second_refresh_with_the_same_files_changes_nothing(tmp_path):
    root = tree(tmp_path)
    src = upstream(tmp_path)
    assert run(root, src).returncode == 0
    before = snapshot(root)
    r = run(root, src)
    assert r.returncode == 0 and r.stdout.strip() == "unchanged"
    assert before == snapshot(root)


def test_a_commit_that_changed_no_curve_moves_no_pin(tmp_path):
    root = tree(tmp_path)
    before = snapshot(root)
    r = run(root, upstream(tmp_path, new_curve=False), sha="f" * 40)
    assert r.returncode == 0 and r.stdout.strip() == "unchanged", r.stderr
    assert before == snapshot(root)


def test_a_curve_upstream_removed_leaves_the_pin(tmp_path):
    root = tree(tmp_path)
    src = upstream(tmp_path)
    (src / f"{GLM52}.yaml").unlink()
    ids = yaml.safe_load((src / "identities.yaml").read_text())
    del ids[GLM52]
    (src / "identities.yaml").write_text(yaml.safe_dump(ids, sort_keys=False))
    assert run(root, src).returncode == 0
    lock = json.loads((root / "registry/validator/testdata/inferencex/curves.json").read_text())
    assert f"{GLM52}.yaml" not in lock["files"]


def test_a_failed_refresh_leaves_the_tree_as_it_was(tmp_path):
    """A refresh whose emitter fails (here: a transfer made stale by a new GLM-5.3 curve) restores
    every file it wrote."""
    root = tree(tmp_path)
    src = upstream(tmp_path, new_curve=False)
    glm53 = "glm-5.3-nvfp4__mtp__native__default-default"
    new = "glm-5.3__mtp__native__default-default"
    (src / f"{new}.yaml").write_text((src / f"{glm53}.yaml").read_text().replace("glm-5.3-nvfp4", "glm-5.3"))
    ids = yaml.safe_load((src / "identities.yaml").read_text())
    ids[new] = ids[glm53]
    (src / "identities.yaml").write_text(yaml.safe_dump(ids, sort_keys=False))
    before = snapshot(root)
    r = run(root, src)
    assert r.returncode != 0 and "is stale" in r.stderr
    assert before == snapshot(root)


def test_a_missing_pin_site_fails_before_anything_is_written(tmp_path):
    root = tree(tmp_path)
    wf = root / ".github/workflows/rederive.yml"
    wf.write_text(wf.read_text().replace("      INFERENCEX_SHA:", "      INFERENCEX_COMMIT:"))
    before = snapshot(root)
    r = run(root, upstream(tmp_path))
    assert r.returncode != 0 and "pin line" in r.stderr
    assert before == snapshot(root)
