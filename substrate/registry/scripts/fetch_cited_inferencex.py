#!/usr/bin/env python3
"""Fetch the InferenceX CollectiveX run bodies that committed entries cite and the API still
serves, and refuse unless the points of each are the points pinned below.

    python -I scripts/fetch_cited_inferencex.py <out dir>     # then COLLECTIVEX_DATA=<out dir>

The bodies are third-party data and are never committed (X12, D6: the ledger's InferenceX
licence has no sign-off); CI fetches them and checks a digest of their points instead
(collectivex_points.py). The digest covers what the fit reads and not the bytes: on 2026-11-07
the API began serving all fourteen bodies with different bytes and the same points, and a pin
on bytes failed the gate for a change that moved no value. Each is saved as
cx_run_<id>.body, the name the writers read.

WHICH BODIES. fit_deepep_collectivex.py's GB200 / H200-EP16 entries cite run 1793303708 by
the sha256 of its points and name the runs that publish the same points; the writer
re-derives that list, so all of them must be present. The list is the one served on
2026-11-05. The digests were taken on 2026-11-09, and the fit gives the same values from the
bodies of both dates.

WHICH ARE NOT HERE. fit_deepep_ll_graph.py and fit_host_link.py cite runs the API no longer
serves (33356406487, 36334392363, 36367074689, 36402856875, and the swap_blocks bodies sha256
26cad6ccbdaa): each answers 404. Those writers then refuse (exit 3), their tests skip, and the
skips are on .github/quarantine.yaml (X16, X25) under inference-sim/blis-registry#57 until the
bodies are restored or the entries are re-cited through a reviewed refit. The OperatorX
response fit_marlin_nvfp4_grouped_ramp.py cites is served and is fetched here too, as
moe_MiniMax-M3.json, checked against a digest of the rows that fit reads (operatorx_points.py):
its bytes changed by 2026-11-05 and its rows did not.

The API keeps a bounded window of recent runs, so a body below can stop being served. Then
this fails, naming it, and the gate fails rather than skips: restore the body, or quarantine
the test with an owning issue.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "collectivex_points", Path(__file__).resolve().parent / "collectivex_points.py")
cpts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpts)
_ospec = importlib.util.spec_from_file_location(
    "operatorx_points", Path(__file__).resolve().parent / "operatorx_points.py")
opx = importlib.util.module_from_spec(_ospec)
_ospec.loader.exec_module(opx)

URL = "https://inferencex.semianalysis.com/api/v1/collectivex/runs/{run}?version=1"
OPERATORX_URL = ("https://inferencex.semianalysis.com/api/v1/operatorx/compare"
                 "?op=moe&workload=moe:MiniMax-M3")
OPERATORX_ROWS = "82c64246b4135bbf66348411c6732e46d5e12be02979ec6b580100a375450eaf"

# run id -> sha256 of the points of its body (collectivex_points.digest), as served on 2026-11-09.
BODIES = {
    "1793201996": "524ad4a5c24a6fc496ff180d8709070f8b668a4c60db3725c621be79277ce757",
    "1793209284": "3493f5b1c214a3c5a92f820e8086213f7cc23538ef73ad375e33cbea22449c17",
    "1793224262": "5bd19a8a0f0e37d1b9fea8a80b46238d9003f3278a196ece24686fd62549460a",
    "1793237711": "648cccba78aec6337a9e60dfd7bf523e1ba89b8e53e56911e1195758995091eb",
    "1793244384": "f6ff7a5ba7ec095988fb033d7af1c34a23ce7973b58b2c7b13d20773b3f2a5e3",
    "1793264366": "f6ff7a5ba7ec095988fb033d7af1c34a23ce7973b58b2c7b13d20773b3f2a5e3",
    "1793264384": "f6ff7a5ba7ec095988fb033d7af1c34a23ce7973b58b2c7b13d20773b3f2a5e3",
    "1793264397": "f6ff7a5ba7ec095988fb033d7af1c34a23ce7973b58b2c7b13d20773b3f2a5e3",
    "1793270219": "53f25c667cca54d5a93c9c7c4982e60e84d27e034d6e0c7819fc63584421514a",
    "1793270233": "762780b8ae07e6bf7374ef762d20b5c0047a70ed2c06532495685179baf33232",
    "1793295596": "38b396d583c7036b2b25e93c4cbbfb1f4c753a8b5e2fa7ddb1716292c22be0a6",
    "1793295612": "eebb7982403a8056cc53cdaef4a82817a2382ee457d56f99fd49d16fc530002a",
    "1793300196": "eebb7982403a8056cc53cdaef4a82817a2382ee457d56f99fd49d16fc530002a",
    "1793303708": "eebb7982403a8056cc53cdaef4a82817a2382ee457d56f99fd49d16fc530002a",
}
ATTEMPTS = 4


def fetch(run: str, url: str = URL) -> bytes:
    req = urllib.request.Request(url.format(run=run), headers={"User-Agent": "blis-rederive"})
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
        got = cpts.digest_bytes(body)
        if got != want:
            bad.append(f"run {run}: points sha256 {got}, not the pinned {want}")
            continue
        (out / f"cx_run_{run}.body").write_bytes(body)
        print(f"run {run}: {len(body)} bytes, points sha256 {got[:12]} ok")
    body = fetch("operatorx", OPERATORX_URL)
    got = opx.digest_bytes(body)
    if got != OPERATORX_ROWS:
        bad.append(f"OperatorX MoE MiniMax-M3: rows sha256 {got}, not the pinned {OPERATORX_ROWS}")
    else:
        (out / "moe_MiniMax-M3.json").write_bytes(body)
        print(f"OperatorX MoE MiniMax-M3: {len(body)} bytes, rows sha256 {got[:12]} ok")
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
