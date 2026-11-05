#!/usr/bin/env python3
"""Fetch the InferenceX CollectiveX run bodies that committed entries cite and the API still
serves, and refuse unless each is byte-for-byte the body pinned below.

    python -I scripts/fetch_cited_inferencex.py <out dir>     # then COLLECTIVEX_DATA=<out dir>

The bodies are third-party data and are never committed (X12, D6: the ledger's InferenceX
licence has no sign-off); CI fetches them and checks their sha256 instead. Each is saved as
cx_run_<id>.body, the name the writers read.

WHICH BODIES. fit_deepep_collectivex.py's GB200 / H200-EP16 entries cite run 1793303708 by
its full sha256 and name the runs that publish the same points byte-identically; the writer
re-derives that list, so all of them must be present. The list and hashes are those served
on 2026-11-05; 1793303708's matches the hash the entries cite.

WHICH ARE NOT HERE. fit_deepep_ll_graph.py and fit_host_link.py cite runs the API no longer
serves (33356406487, 36334392363, 36367074689, 36402856875, and the swap_blocks bodies sha256
26cad6ccbdaa): each answers 404. Those writers then refuse (exit 3), their tests skip, and the
skips are on .github/quarantine.yaml (X16, X25) under inference-sim/blis-registry#57 until the
bodies are restored or the entries are re-cited through a reviewed refit. Nor is the OperatorX
response fit_marlin_nvfp4_grouped_ramp.py cites (sha256 ea8fb4078168): the URL now serves a
different response, so $OPERATORX_DATA is left unset and that test is quarantined too.

The API keeps a bounded window of recent runs, so a body below can stop being served. Then
this fails, naming it, and the gate fails rather than skips: restore the body, or quarantine
the test with an owning issue.
"""

from __future__ import annotations

import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://inferencex.semianalysis.com/api/v1/collectivex/runs/{run}?version=1"

# run id -> sha256 of its body, as served on 2026-11-05.
BODIES = {
    "1793201996": "3a250c0d2152aae95e3de7db1eb469f69258f51a6c454133f59fb0799999081b",
    "1793209284": "43f204c13aaafb5a5c0f59088db1d085799b018fda3b14544d63b4b4aae320e6",
    "1793224262": "c6153f36e9629902c66d91923a3ae9b316c715e9f94b81c9b07688ce9d8d4ed5",
    "1793237711": "435c7b6824b60f7cfe8e2ba4fa399670167448325a0d022752c8c59047db831b",
    "1793244384": "57317e1fdf671f7afb43a9fb30d920f004b32f487a4c458371acd54b7b9867c5",
    "1793264366": "fab9b59c2e25bc4485ef9dd696b3874952ed1962e43e964ed65b24d2f6174756",
    "1793264384": "7f0cbe74f38ab4d35dee3210427b4d7cbb44e98973d32e992bfdac7055c9ccd1",
    "1793264397": "66548896e6e94ecc1cb7b80b3dabea6d191c3cbfd6e5c26738a234677ff5143e",
    "1793270219": "65909914ee2d63ed6b742b5c013020716d9582454d8d9682da0560b4f54befa5",
    "1793270233": "33d1a3b2d0bdd0d04358c49498e4c2b4e1a317926d76b85a3e56fd54de743d39",
    "1793295596": "1238ecd8904124e5e00af7fed58bcd68aa370e8945d3beb8a3988aedefad7c0f",
    "1793295612": "2d1d543bcfe53dfe14bc6cee9d4298f60138d30ed329f26fd02f82ebda4f3287",
    "1793300196": "b688efab93cad5f9e3f968e80428cb726cd9ef5cb54de4cc412d710a3b841cdd",
    "1793303708": "78d991b356a157fbe50b59f62724331fd69a68b7e5c38610a2dbb305f36454af",
}
ATTEMPTS = 4


def fetch(run: str) -> bytes:
    req = urllib.request.Request(URL.format(run=run), headers={"User-Agent": "blis-rederive"})
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == ATTEMPTS:
                raise SystemExit(f"run {run}: HTTP {e.code} from {req.full_url}; the cited body "
                                 "is not served") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == ATTEMPTS:
                raise SystemExit(f"run {run}: {e} from {req.full_url}") from None
        time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    bad = []
    for run, want in sorted(BODIES.items()):
        body = fetch(run)
        got = hashlib.sha256(body).hexdigest()
        if got != want:
            bad.append(f"run {run}: body sha256 {got}, not the pinned {want}")
            continue
        (out / f"cx_run_{run}.body").write_bytes(body)
        print(f"run {run}: {len(body)} bytes, sha256 {got[:12]} ok")
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1
    # X22: the job states the number of items it must produce, and fails below it.
    n = len(list(out.glob("cx_run_*.body")))
    if n != len(BODIES):
        print(f"{out} holds {n} bodies, not {len(BODIES)}", file=sys.stderr)
        return 1
    print(f"{n} CollectiveX bodies fetched and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
