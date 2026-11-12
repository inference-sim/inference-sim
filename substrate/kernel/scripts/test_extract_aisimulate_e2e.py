#!/usr/bin/env python3
"""Tests for extract_aisimulate_e2e.py: the input is the pinned artifact run (D3) and a corpus
of nothing is a failure (X22).

Each case runs the extractor as a program, with a pin file written for a small summary made
here, so the snapshot rules can be tried one at a time while the digest check passes. The last
cases use the committed pin: an arbitrary summary.json is refused. With AISIM_E2E_SUMMARY set to
a downloaded summary.json of the pinned run, the real artifact is extracted and its counts are
required to equal the pin's.

Run: python3 -I scripts/test_extract_aisimulate_e2e.py
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "extract_aisimulate_e2e.py"
COMMITTED_PIN = HERE / "aisimulate_e2e.pin.json"
FAILURES = []

COMMIT = "a" * 40


def ck(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def point(c):
    return {"concurrency": c, "status": "success",
            "measured": {"tpot_relative": 1.0 + c / 100, "ttft_relative": 1.0 + c / 50}}


def good_summary(model="gptoss120b", points=(1, 4, 16)):
    return {
        "snapshot": {
            "evaluated_revision": {"branch": "main", "commit_sha": COMMIT},
            "release_tag": "db-dump/2026-09-28",
            "campaign": {"run_id": "36849388864", "run_attempt": "1", "status": "complete"},
        },
        "scope": {}, "totals": {},
        "models": [{"model": model, "workloads": [{
            "identity": "1024:1024", "label": "1k1k",
            "gpus": [{"gpu": "h100", "topologies": [{
                "id": "t1", "serving": "aggregated", "spec_method": None, "precision": "fp8",
                "framework": "vllm",
                "parallelism": {"tp_size": 4, "pp_size": 1, "attention_dp_size": 1,
                                "moe_ep_size": 1, "moe_tp_size": 4},
                "points": [point(c) for c in points]}]}]}]}],
    }


def pin_for(raw: bytes, sweeps=1, points=3):
    return {"source": "aisimulate", "artifact_id": "e2e-accuracy-web",
            "repository": "ai-dynamo/aisimulate", "run_id": "36849388864", "run_attempt": 1,
            "commit": COMMIT, "release_tag": "db-dump/2026-09-28",
            "artifact": "e2e-accuracy-web-0d6e4079e36703eb", "zip_sha256": "b" * 64,
            "summary_sha256": hashlib.sha256(raw).hexdigest(),
            "expected": {"sweeps": sweeps, "points": points}}


def run(summary, mutate_pin=None, use_committed_pin=False, raw=None):
    """Run the extractor on a summary; return (exit code, stderr, whether it wrote a corpus)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = raw if raw is not None else json.dumps(summary).encode()
        (tmp / "summary.json").write_bytes(raw)
        pin = pin_for(raw)
        if mutate_pin:
            mutate_pin(pin)
        (tmp / "pin.json").write_text(json.dumps(pin))
        out = tmp / "corpus.json"
        cmd = [sys.executable, "-I", str(SCRIPT), str(tmp / "summary.json"), "-o", str(out)]
        if not use_committed_pin:
            cmd += ["--pin", str(tmp / "pin.json")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode, r.stderr, out.exists()


def refused(label, summary, needle, **kw):
    code, err, wrote = run(summary, **kw)
    ck(label, code == 1 and needle in err and not wrote, f"exit {code}: {err.strip()[:200]}")


def edit(fn):
    s = copy.deepcopy(good_summary())
    fn(s)
    return s


def main() -> int:
    code, err, wrote = run(good_summary())
    ck("the pinned artifact is extracted", code == 0 and wrote, f"exit {code}: {err.strip()}")

    # A changed byte: the digest no longer matches the pin.
    refused("a summary with another SHA-256 is refused",
            good_summary(), "SHA-256",
            mutate_pin=lambda p: p.update(summary_sha256="0" * 64))
    refused("a changed summary is refused by the committed pin, whatever it says",
            good_summary(), "SHA-256", use_committed_pin=True)

    # The snapshot's own statements, with the digest made to match so only they decide.
    refused("a snapshot with no run id is refused",
            edit(lambda s: s["snapshot"]["campaign"].pop("run_id")), "states no campaign run_id")
    refused("a snapshot with no campaign block is refused",
            edit(lambda s: s["snapshot"].pop("campaign")), "states no campaign run_id")
    refused("another run id is refused",
            edit(lambda s: s["snapshot"]["campaign"].update(run_id="1")), "run_id")
    refused("another attempt is refused",
            edit(lambda s: s["snapshot"]["campaign"].update(run_attempt="2")), "run_attempt")
    refused("an evaluated_revision of 'latest' is refused",
            edit(lambda s: s["snapshot"].update(evaluated_revision="latest")), "not a commit record")
    refused("a commit_sha of 'latest' is refused",
            edit(lambda s: s["snapshot"]["evaluated_revision"].update(commit_sha="latest")),
            "pinned by commit")
    refused("another commit is refused",
            edit(lambda s: s["snapshot"]["evaluated_revision"].update(commit_sha="c" * 40)),
            "commit_sha")
    refused("no snapshot at all is refused", edit(lambda s: s.pop("snapshot")), "snapshot")
    refused("another release tag is refused",
            edit(lambda s: s["snapshot"].update(release_tag="db-dump/2026-10-05")), "release_tag")
    refused("an incomplete campaign is refused",
            edit(lambda s: s["snapshot"]["campaign"].update(status="running")), "status")

    # X22: nothing read is a failure, and the pin's counts are a minimum that is also a maximum.
    refused("a pinned artifact whose slugs match nothing reads 0 points and fails",
            good_summary(model="unknown-slug"), "slugs no longer match")
    refused("a sweep below three points yields nothing and fails",
            good_summary(points=(1, 4)), "slugs no longer match")
    refused("a different number of points than the pin records is refused",
            good_summary(), "pin records", mutate_pin=lambda p: p["expected"].update(points=4))
    refused("a different number of sweeps than the pin records is refused",
            good_summary(), "pin records", mutate_pin=lambda p: p["expected"].update(sweeps=2))

    # A pin that is not itself exact.
    for label, (fn, why) in {
        "a pin with release_tag 'latest'": (lambda p: p.update(release_tag="latest"), "release_tag is empty or unpinned"),
        "a pin with a short commit": (lambda p: p.update(commit="cab77b9"), "commit is not a full 40-digit commit"),
        "a pin with a non-numeric run id": (lambda p: p.update(run_id="latest"), "run_id is not a workflow run id"),
        "a pin with no digest": (lambda p: p.update(summary_sha256=""), "summary_sha256 is not a SHA-256"),
        "a pin expecting nothing": (lambda p: p.update(expected={"sweeps": 0, "points": 0}), "expected needs positive"),
    }.items():
        code, err, wrote = run(good_summary(), mutate_pin=fn)
        ck(f"{label} is refused", code != 0 and why in err and not wrote,
           f"exit {code}: {err.strip()[:160]}")

    # The reviewer's repro: a document whose snapshot says latest, with one valid sweep.
    code, err, wrote = run(edit(lambda s: s["snapshot"].update(evaluated_revision="latest")),
                           use_committed_pin=True)
    ck("the committed pin refuses a 'latest' snapshot with one valid sweep",
       code == 1 and not wrote, f"exit {code}")

    # The committed pin is itself exact.
    pin = json.loads(COMMITTED_PIN.read_text())
    ck("the committed pin names a run, a commit and two digests",
       pin["run_id"].isdigit() and len(pin["commit"]) == 40
       and len(pin["summary_sha256"]) == 64 and len(pin["zip_sha256"]) == 64)

    real = os.environ.get("AISIM_E2E_SUMMARY")
    if real:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "corpus.json"
            r = subprocess.run([sys.executable, "-I", str(SCRIPT), real, "-o", str(out)],
                               capture_output=True, text=True)
            ck("the real pinned artifact extracts to the pinned counts",
               r.returncode == 0 and out.exists(), f"exit {r.returncode}: {r.stderr.strip()}")
            if out.exists():
                c = json.loads(out.read_text())
                ck("and the counts are the pin's",
                   (len(c["sweeps"]), sum(len(x["points"]) for x in c["sweeps"]))
                   == (pin["expected"]["sweeps"], pin["expected"]["points"]))
    else:
        print("  SKIP the real artifact (set AISIM_E2E_SUMMARY to a downloaded summary.json)")

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("PASS: 0 failure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
