#!/usr/bin/env python3
"""Fetch the config.json files that catalog entries pin by hash, from public repositories.

    python3 scripts/fetch_pinned_configs.py OUTDIR [--catalog ROOT]

A model whose licence keeps config.json out of the tree commits only its SHA-256 (X12). The
nightly job fetches those files with a token and re-derives the graphs. A pull request can do the
same for every repository that is public, with no token: this script fetches each pinned file at
its revision (`source.path`, else config.json), checks its hash against model.yaml, and writes it
to OUTDIR/<entry>/config.json. Then `derive_graph.py --check --fetched OUTDIR --allow-unfetched`
re-derives those entries, so a hand edit of one of their graphs fails the pull request.

An entry may also pin the safetensors headers of its shards (model.yaml `headers`), with the
shard index that names them (`shard_index`), which the deriver reads stored widths from when
config.json states no quantization (SUB-15). These are fetched without a token, by both the
pull-request job and the nightly job. The index is fetched whole and the pinned shards must be
exactly the files its weight_map names; each header is fetched by HTTP range request, the 8-byte
length and then the header JSON, never the weights. Every hash is checked against its pin, and
only when all of them verify are they written, to OUTDIR/<entry>/headers/index.json and
OUTDIR/<entry>/headers/<shard>.json.

The nightly job runs this script with --token-env HF_TOKEN, which sends the token for config
files, and --require-all, which fails on anything left unfetched.

An entry whose repository answers 401, 403 or 404 (gated or private) is listed and left unfetched
for the token job. A file whose hash differs from the pin is an error: the pin or the revision is
wrong. A network failure after retries is listed as unfetched and noted on stderr, because a
registry outage must not fail an unrelated pull request; the nightly job still re-derives all of
them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

HOST = "https://huggingface.co"


def url_of(source: dict) -> str:
    return f"{HOST}/{source['repo']}/resolve/{source['revision']}/{source.get('path', 'config.json')}"


def fetch(url: str, opener=urllib.request.urlopen, retries: int = 3,
          span: tuple[int, int] | None = None, token: str | None = None) -> bytes | int | None:
    """The file's bytes, or bytes span[0] to span[1] inclusive by range request; the HTTP status
    for a refusal (401, 403, 404); None after the network failed `retries` times. `token`, where
    given, is sent as a bearer token (the nightly job's, for a gated repository)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            if span:
                req.add_header("Range", f"bytes={span[0]}-{span[1]}")
            if token:
                req.add_header("Authorization", "Bearer " + token)
            with opener(req, timeout=60) as r:
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


def header_of(source: dict, shard: str, opener) -> bytes | int | None:
    """A shard's safetensors header: its first 8 bytes give the header's length (little-endian),
    and the header JSON follows. Two range requests; the tensor data is never read."""
    url = f"{HOST}/{source['repo']}/resolve/{source['revision']}/{shard}"
    head = fetch(url, opener, span=(0, 7))
    if not isinstance(head, bytes):
        return head
    if len(head) != 8:
        return None
    n = struct.unpack("<Q", head)[0]
    return fetch(url, opener, span=(8, 8 + n - 1))


def fetch_headers(name: str, m: dict, out: Path, opener) -> str | None:
    """Fetch and check an entry's shard index and every pinned header, without a token (a
    header-pinned repository is public). Returns why it failed, None when all were written. A
    refusal or an outage is reported as "unfetched: ...", a hash that differs, or pins that are
    not the index's shards, as the error itself. Nothing is written unless every file verifies,
    so a partial fetch is never read as a complete one."""
    def check(path, got, pin):
        if not isinstance(got, bytes):
            return f"unfetched: {'network' if got is None else 'HTTP ' + str(got)}"
        digest = hashlib.sha256(got).hexdigest()
        if digest != pin:
            return (f"{name}: {path} at {m['source']['revision']} hashes to {digest}, not the "
                    f"pinned {pin}")
        return None

    index_pin = m.get("shard_index") or {}
    index_url = f"{HOST}/{m['source']['repo']}/resolve/{m['source']['revision']}/{index_pin.get('path')}"
    index = fetch(index_url, opener)
    why = check(index_pin.get("path"), index, index_pin.get("sha256"))
    if why:
        return why
    shards = sorted(set((json.loads(index).get("weight_map") or {}).values()))
    pinned = sorted(p["path"] for p in m["headers"])
    if shards != pinned:
        return (f"{name}: the headers pinned are not the shards {index_pin['path']} names "
                f"(missing {sorted(set(shards) - set(pinned))}, extra {sorted(set(pinned) - set(shards))})")

    def one(pin):
        return pin, header_of(m["source"], pin["path"], opener)

    got_all = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for pin, got in pool.map(one, m["headers"]):
            why = check(f"the header of {pin['path']}", got, pin["sha256"])
            if why:
                return why
            got_all[pin["path"]] = got
    d = out / name / "headers"
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.json").write_bytes(index)
    for path, got in got_all.items():
        (d / (path.replace("/", "__") + ".json")).write_bytes(got)
    return None


def run(catalog: Path, out: Path, opener=urllib.request.urlopen,
        token: str | None = None) -> tuple[list[str], list[str], list[str]]:
    """Returns (fetched, unfetched, hash errors) entry names. `token` is sent for config files."""
    fetched, unfetched, bad = [], [], []
    for card in sorted((catalog / "models").glob("*/model.yaml")):
        m = yaml.safe_load(card.read_text()) or {}
        name = card.parent.name
        if m.get("headers"):
            why = fetch_headers(name, m, out, opener)
            if why is None:
                fetched.append(f"{name} (headers)")
            elif why.startswith("unfetched: "):
                unfetched.append(f"{name} headers ({why[len('unfetched: '):]})")
            else:
                bad.append(why)
        pin = (m.get("config") or {}).get("sha256")
        if not pin:
            continue
        got = fetch(url_of(m["source"]), opener, token=token)
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
    ap.add_argument("--token-env", default=None,
                    help="the environment variable holding the access token sent for config files "
                         "(the nightly job's); headers are fetched without one")
    ap.add_argument("--require-all", action="store_true",
                    help="fail on any file left unfetched (the nightly job, which must re-derive all)")
    args = ap.parse_args()
    token = None
    if args.token_env:
        token = os.environ.get(args.token_env)
        if not token:
            print(f"{args.token_env} is empty; the token job cannot run (X16)", file=sys.stderr)
            return 1
    fetched, unfetched, bad = run(Path(args.catalog), Path(args.out), token=token)
    print(f"{len(fetched)} pinned file set(s) fetched and hash-checked")
    if unfetched:
        print("left unfetched: " + ", ".join(unfetched))
    for line in bad:
        print(line, file=sys.stderr)
    return 1 if bad or (args.require_all and unfetched) else 0


if __name__ == "__main__":
    sys.exit(main())
