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


def _header_file(header: bytes) -> bytes:
    import struct
    return struct.pack("<Q", len(header)) + header + b"\0" * 64   # the tensor data, never read


def ranged(files: dict):
    """An opener that serves byte ranges and records each request's Range."""
    seen = []

    def open_(req, timeout=0):
        rng = req.get_header("Range")
        seen.append(rng)
        for path, body in files.items():
            if req.full_url.endswith("/" + path):
                if rng is None:
                    return Reply(body)
                lo, hi = (int(x) for x in rng.removeprefix("bytes=").split("-"))
                return Reply(body[lo:hi + 1])
        raise urllib.error.HTTPError(req.full_url, 404, "x", {}, None)
    return open_, seen


def _index(*shards) -> bytes:
    import json
    return json.dumps({"weight_map": {f"t{i}": s for i, s in enumerate(shards)}}).encode()


def _headers_tree(tmp_path, pins, index: bytes):
    d = tmp_path / "models" / "m"
    d.mkdir(parents=True)
    card = {"name": "m", "source": {"repo": "org/m", "revision": "r" * 40}, "headers": pins,
            "shard_index": {"path": "model.safetensors.index.json",
                            "sha256": hashlib.sha256(index).hexdigest()}}
    (d / "model.yaml").write_text(yaml.safe_dump(card))
    return tmp_path


HEADER = b'{"a.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}'
PIN = {"path": "model-1.safetensors", "sha256": hashlib.sha256(HEADER).hexdigest()}


def _served(index: bytes, header: bytes = HEADER) -> dict:
    return {"model-1.safetensors": _header_file(header), "model.safetensors.index.json": index}


def test_pinned_headers_are_fetched_by_range_and_hash_checked(tmp_path):
    index = _index("model-1.safetensors")
    cat = _headers_tree(tmp_path / "c", [PIN], index)
    open_, seen = ranged(_served(index))
    fetched, unfetched, bad = fp.run(cat, tmp_path / "out", open_)
    assert fetched == ["m (headers)"] and not unfetched and not bad
    assert (tmp_path / "out" / "m" / "headers" / "model-1.safetensors.json").read_bytes() == HEADER
    assert (tmp_path / "out" / "m" / "headers" / "index.json").read_bytes() == index
    # The index whole, then the length and exactly the header: the tensor data is never requested.
    assert seen == [None, "bytes=0-7", f"bytes=8-{8 + len(HEADER) - 1}"]


def test_a_header_whose_hash_differs_from_its_pin_is_an_error(tmp_path):
    index = _index("model-1.safetensors")
    cat = _headers_tree(tmp_path / "c", [{"path": "model-1.safetensors", "sha256": "0" * 64}], index)
    open_, _ = ranged(_served(index))
    fetched, unfetched, bad = fp.run(cat, tmp_path / "out", open_)
    assert not fetched and len(bad) == 1 and "not the pinned" in bad[0]
    assert not (tmp_path / "out" / "m").exists()


def test_pins_that_are_not_the_indexs_shards_are_an_error(tmp_path):
    index = _index("model-1.safetensors", "model-2.safetensors")
    cat = _headers_tree(tmp_path / "c", [PIN], index)
    open_, _ = ranged(_served(index))
    fetched, unfetched, bad = fp.run(cat, tmp_path / "out", open_)
    assert not fetched and len(bad) == 1 and "missing ['model-2.safetensors']" in bad[0]


def test_a_refused_or_failed_header_fetch_writes_nothing(tmp_path, monkeypatch):
    """A 401 or an outage on one header is reported as unfetched, and leaves no headers
    directory, so the deriver counts the entry as not re-derivable rather than failing it."""
    monkeypatch.setattr(fp.time, "sleep", lambda s: None)
    index = _index("model-1.safetensors")
    cat = _headers_tree(tmp_path / "c", [PIN], index)
    open_, _ = ranged({"model.safetensors.index.json": index})   # the shard answers 404

    def refusing(req, timeout=0):
        if req.full_url.endswith("model-1.safetensors"):
            raise urllib.error.HTTPError(req.full_url, 401, "x", {}, None)
        return open_(req, timeout)
    fetched, unfetched, bad = fp.run(cat, tmp_path / "out", refusing)
    assert unfetched == ["m headers (HTTP 401)"] and not bad and not fetched
    assert not (tmp_path / "out" / "m").exists()
    r = subprocess.run([sys.executable, str(DERIVER), str(cat), "--check", "--fetched", str(tmp_path / "out"),
                        "--allow-unfetched"], capture_output=True, text=True)
    assert "1 not re-derivable offline" in r.stdout, (r.stdout, r.stderr)


def test_the_token_is_sent_for_config_files_and_not_for_headers(tmp_path):
    """The nightly job's call: --token-env sends the token for a pinned config.json, which may be
    gated, and fetches the public headers without it."""
    index = _index("model-1.safetensors")
    cat = _headers_tree(tmp_path / "c", [PIN], index)
    card = yaml.safe_load((cat / "models" / "m" / "model.yaml").read_text())
    card["config"] = {"sha256": hashlib.sha256(b"{}").hexdigest()}
    (cat / "models" / "m" / "model.yaml").write_text(yaml.safe_dump(card))
    auth = {}

    def open_(req, timeout=0):
        url = req.full_url
        auth[url.rsplit("/", 1)[-1]] = req.get_header("Authorization")
        if url.endswith("config.json"):
            return Reply(b"{}")
        return ranged(_served(index))[0](req, timeout)
    fetched, unfetched, bad = fp.run(cat, tmp_path / "out", open_, token="secret")
    assert not bad and not unfetched and sorted(fetched) == ["m", "m (headers)"]
    assert auth["config.json"] == "Bearer secret"
    assert auth["model-1.safetensors"] is None and auth["model.safetensors.index.json"] is None
