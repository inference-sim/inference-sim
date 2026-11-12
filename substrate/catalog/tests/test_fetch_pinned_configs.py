"""Tests for the pull-request fetch of hash-pinned configs from public repositories."""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import urllib.error
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "fetch_pinned_configs.py"
DERIVER = ROOT / "scripts" / "derive_graph.py"
sys.path.insert(0, str(SCRIPT.parent))
import fetch_pinned_configs as fp  # noqa: E402


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def tree(tmp_path, entries: dict) -> Path:
    for name, (repo, body) in entries.items():
        d = tmp_path / "models" / name
        d.mkdir(parents=True)
        card = {"name": name, "source": {"repo": repo, "revision": "r" * 40}}
        if body is not None:
            card["config"] = {"sha256": hashlib.sha256(body).hexdigest()}
        (d / "model.yaml").write_text(yaml.safe_dump(card))
    return tmp_path


def opener(files: dict):
    def open_(req, timeout=0):
        for repo, reply in files.items():
            if f"/{repo}/" in req.full_url:
                if isinstance(reply, int):
                    raise urllib.error.HTTPError(req.full_url, reply, "x", {}, None)
                return Reply(reply)
        raise urllib.error.HTTPError(req.full_url, 404, "x", {}, None)
    return open_


def test_a_public_file_is_fetched_and_its_hash_checked(tmp_path):
    body = b'{"a": 1}'
    cat = tree(tmp_path / "c", {"m": ("org/m", body), "committed": ("org/c", None)})
    out = tmp_path / "out"
    fetched, unfetched, bad = fp.run(cat, out, opener({"org/m": body}))
    assert fetched == ["m"] and not unfetched and not bad
    assert (out / "m" / "config.json").read_bytes() == body


def test_a_file_whose_hash_differs_from_the_pin_is_an_error(tmp_path):
    cat = tree(tmp_path / "c", {"m": ("org/m", b"pinned")})
    fetched, unfetched, bad = fp.run(cat, tmp_path / "out", opener({"org/m": b"different"}))
    assert not fetched and len(bad) == 1 and "not the pinned" in bad[0]


def test_a_gated_repository_is_left_for_the_token_job(tmp_path):
    cat = tree(tmp_path / "c", {"m": ("org/m", b"x")})
    fetched, unfetched, bad = fp.run(cat, tmp_path / "out", opener({"org/m": 401}))
    assert not fetched and not bad and unfetched == ["m (HTTP 401)"]


def test_a_network_failure_is_listed_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(fp.time, "sleep", lambda s: None)
    def boom(req, timeout=0):
        raise urllib.error.URLError("down")
    cat = tree(tmp_path / "c", {"m": ("org/m", b"x")})
    fetched, unfetched, bad = fp.run(cat, tmp_path / "out", boom)
    assert unfetched == ["m (network)"] and not bad


def test_derive_graph_rederives_what_was_fetched_and_counts_the_rest_apart():
    """The committed catalog with nothing fetched: --allow-unfetched leaves every pinned entry
    unverified and still exits 0; without the flag the same run fails each entry."""
    import tempfile
    with tempfile.TemporaryDirectory() as empty:
        r = subprocess.run([sys.executable, str(DERIVER), str(ROOT), "--check", "--fetched", empty,
                            "--allow-unfetched"], capture_output=True, text=True)
        assert r.returncode == 0 and "not re-derivable offline" in r.stdout, (r.stdout, r.stderr)
        r = subprocess.run([sys.executable, str(DERIVER), str(ROOT), "--check", "--fetched", empty],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "no fetched config.json" in r.stderr
