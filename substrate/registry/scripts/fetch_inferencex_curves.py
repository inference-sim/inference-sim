#!/usr/bin/env python3
"""Fetch InferenceX's golden acceptance curves at the pinned commit, and check each file's hash.

    fetch_inferencex_curves.py [DEST]

The repository commits no InferenceX file (X12). It commits validator/testdata/inferencex/
curves.json: the InferenceX commit, the directory, and the SHA-256 of every file the emitter
reads (the curve files and identities.yaml). This script downloads each of them from
raw.githubusercontent.com at that commit into DEST (default: the emitter's cache directory,
validator/testdata/inferencex/cache/<commit>/), and fails, naming the file, when a download's
hash is not the pinned one. A file already present with the pinned hash is not downloaded again.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent
LOCK = REGISTRY / "validator" / "testdata" / "inferencex" / "curves.json"
CACHE = REGISTRY / "validator" / "testdata" / "inferencex" / "cache"
RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}/{name}"


def load_lock(path: Path = LOCK) -> dict:
    lock = json.loads(path.read_text())
    for key in ("repository", "commit", "path", "files"):
        if key not in lock:
            raise SystemExit(f"{path}: no {key!r}")
    return lock


def default_dest(lock: dict) -> Path:
    return CACHE / lock["commit"] / "golden_al_distribution"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(dest: Path, lock: dict, opener=urllib.request.urlopen) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    repo = lock["repository"].removeprefix("https://github.com/")
    fetched = 0
    for name, want in sorted(lock["files"].items()):
        target = dest / name
        if target.is_file() and sha256(target) == want:
            continue
        url = RAW.format(repo=repo, commit=lock["commit"], path=lock["path"], name=name)
        with opener(url, timeout=60) as r:
            body = r.read()
        got = hashlib.sha256(body).hexdigest()
        if got != want:
            raise SystemExit(f"{url}: SHA-256 {got}, but curves.json pins {want}")
        target.write_bytes(body)
        fetched += 1
    return fetched


def main(argv: list[str]) -> int:
    lock = load_lock()
    dest = Path(argv[1]) if len(argv) > 1 else default_dest(lock)
    n = fetch(dest, lock)
    print(f"{dest}: {len(lock['files'])} file(s) at {lock['commit'][:12]}, {n} downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
