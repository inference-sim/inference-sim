"""The scheduled refresh moves every statement of the InferenceX pin together, vendors the
new curves, and regenerates the set; a second run with the same input changes nothing.

It runs on a copy of the files it edits, so the repository is untouched.
"""

from __future__ import annotations

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


def tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in ("registry/scripts/emit_acceptance_profiles.py",
                "registry/scripts/refresh_inferencex_curves.py",
                "registry/acceptance/inferencex-golden-al.yaml",
                ".github/workflows/rederive.yml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, root / rel)
    shutil.copytree(REPO / "validator/testdata/inferencex",
                    root / "registry/validator/testdata/inferencex")
    return root


def run(root: Path, src: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, BLIS_CATALOG=str(ROOT / "catalog"), INFERENCEX_GOLDEN="")
    return subprocess.run([sys.executable, "-I",
                           str(root / "registry/scripts/refresh_inferencex_curves.py"),
                           str(src), NEW_SHA, "2026-12-01"],
                          capture_output=True, text=True, env=env, timeout=120)


def upstream_with_a_new_curve(tmp_path: Path) -> Path:
    src = tmp_path / "upstream"
    shutil.copytree(REPO / "validator/testdata/inferencex/golden_al_distribution", src)
    new = "glm-5.2__mtp__native__default-default"
    (src / f"{new}.yaml").write_text(
        (src / f"{GLM52}.yaml").read_text().replace("glm-5.2-fp8", "glm-5.2"))
    ids = yaml.safe_load((src / "identities.yaml").read_text())
    ids[new] = ids[GLM52]
    (src / "identities.yaml").write_text(yaml.safe_dump(ids, sort_keys=False))
    return src


def test_a_refresh_moves_every_pin_and_carries_the_new_curve(tmp_path):
    root = tree(tmp_path)
    r = run(root, upstream_with_a_new_curve(tmp_path))
    assert r.returncode == 0, r.stderr
    assert f'INFERENCEX_SHA = "{NEW_SHA}"' in (
        root / "registry/scripts/emit_acceptance_profiles.py").read_text()
    assert f"INFERENCEX_SHA: {NEW_SHA}" in (root / ".github/workflows/rederive.yml").read_text()
    assert f"`{NEW_SHA}` (2026-12-01)" in (
        root / "registry/validator/testdata/inferencex/VENDORED.md").read_text()
    out = yaml.safe_load((root / "registry/acceptance/inferencex-golden-al.yaml").read_text())
    sources = {p["provenance"]["source"] for p in out["profiles"]}
    assert all(NEW_SHA in s for s in sources)
    assert any(s.endswith("/glm-5.2__mtp__native__default-default.yaml") for s in sources)


def test_a_second_refresh_with_the_same_input_changes_nothing(tmp_path):
    root = tree(tmp_path)
    src = upstream_with_a_new_curve(tmp_path)
    assert run(root, src).returncode == 0
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    r = run(root, src)
    assert r.returncode == 0 and r.stdout.strip() == "unchanged"
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_a_curve_upstream_removed_is_removed_from_the_vendored_copy(tmp_path):
    root = tree(tmp_path)
    src = upstream_with_a_new_curve(tmp_path)
    (src / f"{GLM52}.yaml").unlink()
    ids = yaml.safe_load((src / "identities.yaml").read_text())
    del ids[GLM52]
    (src / "identities.yaml").write_text(yaml.safe_dump(ids, sort_keys=False))
    assert run(root, src).returncode == 0
    golden = root / "registry/validator/testdata/inferencex/golden_al_distribution"
    assert not (golden / f"{GLM52}.yaml").exists()
