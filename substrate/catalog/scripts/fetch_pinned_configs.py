#!/usr/bin/env python3
"""Fetch the config.json files that catalog entries pin by hash, from public repositories.

    python3 scripts/fetch_pinned_configs.py OUTDIR [--catalog ROOT]

A model whose licence keeps config.json out of the tree commits only its SHA-256 (X12). The
nightly job fetches those files with a token and re-derives the graphs. A pull request can do the
same for every repository that is public, with no token: this script fetches each pinned file at
its revision (`source.path`, else config.json), checks its hash against model.yaml, and writes it
to OUTDIR/<entry>/config.json. Then `derive_graph.py --check --fetched OUTDIR --allow-unfetched`
re-derives those entries, so a hand edit of one of their graphs fails the pull request.

An entry whose repository answers 401, 403 or 404 (gated or private) is listed and left unfetched
for the token job. A file whose hash differs from the pin is an error: the pin or the revision is
wrong. A network failure after retries is listed as unfetched and noted on stderr, because a
registry outage must not fail an unrelated pull request; the nightly job still re-derives all of
them.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

HOST = "https://huggingface.co"


def url_of(source: dict) -> str:
    return f"{HOST}/{source['repo']}/resolve/{source['revision']}/{source.get('path', 'config.json')}"


def fetch(url: str, opener=urllib.request.urlopen, retries: int = 3) -> bytes | int | None:
    """The file's bytes; the HTTP status for a refusal (401, 403, 404); None after the network
    failed `retries` times."""
    for attempt in range(retries):
        try:
            with opener(urllib.request.Request(url), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return e.code
            err = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err = e
        print(f"fetch {url}: {err}; attempt {attempt + 1} of {retries}", file=sys.stderr)
        time.sleep(2 * (attempt + 1))
    return None


def run(catalog: Path, out: Path, opener=urllib.request.urlopen) -> tuple[list[str], list[str], list[str]]:
    """Returns (fetched, unfetched, hash errors) entry names."""
    fetched, unfetched, bad = [], [], []
    for card in sorted((catalog / "models").glob("*/model.yaml")):
        m = yaml.safe_load(card.read_text()) or {}
        pin = (m.get("config") or {}).get("sha256")
        if not pin:
            continue
        name = card.parent.name
        got = fetch(url_of(m["source"]), opener)
        if not isinstance(got, bytes):
            unfetched.append(f"{name} ({'network' if got is None else 'HTTP ' + str(got)})")
            continue
        if hashlib.sha256(got).hexdigest() != pin:
            bad.append(f"{name}: the file at {m['source']['revision']} hashes to "
                       f"{hashlib.sha256(got).hexdigest()}, not the pinned {pin}")
            continue
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_bytes(got)
        fetched.append(name)
    return fetched, unfetched, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--catalog", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    fetched, unfetched, bad = run(Path(args.catalog), Path(args.out))
    print(f"{len(fetched)} pinned config(s) fetched and hash-checked")
    if unfetched:
        print("left for the token job: " + ", ".join(unfetched))
    for line in bad:
        print(line, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
