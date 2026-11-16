#!/usr/bin/env python3
"""Move the InferenceX golden-curve pin to another InferenceX commit.

    refresh_inferencex_curves.py <golden_al_distribution dir> <commit> <YYYY-MM-DD>

The repository commits no InferenceX file (X12), only validator/testdata/inferencex/curves.json:
the commit, its date and the SHA-256 of every *.yaml file the emitter reads. Given a checkout's
golden_al_distribution/ at <commit>, this script rewrites curves.json from it and moves
INFERENCEX_SHA in .github/workflows/rederive.yml, then re-runs the emitter against the checkout,
which regenerates acceptance/inferencex-golden-al.yaml.

It prints "unchanged" and changes nothing when the checkout's files are the pinned ones, whatever
the commit: a commit that touched no curve moves no pin. Every pin site is checked before any
file is written, and the files are restored if the emitter fails, so a failed refresh leaves the
tree as it was.

The scheduled workflow acceptance-refresh.yml runs it and opens a pull request on change; the
emitter's --check stays the drift gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent
LOCK = REGISTRY / "validator" / "testdata" / "inferencex" / "curves.json"
EMITTER = REGISTRY / "scripts" / "emit_acceptance_profiles.py"
TARGET = REGISTRY / "acceptance" / "inferencex-golden-al.yaml"
REDERIVE = REGISTRY.parent / ".github" / "workflows" / "rederive.yml"
REDERIVE_PIN = re.compile(r"^      INFERENCEX_SHA: ([0-9a-f]{40})$", re.M)
SHA = re.compile(r"[0-9a-f]{40}")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def hashes(src: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(src.glob("*.yaml"))}


def refresh(src: Path, sha: str, date: str) -> bool:
    if not SHA.fullmatch(sha) or not DATE.fullmatch(date):
        raise SystemExit("usage: refresh_inferencex_curves.py <dir> <40-hex commit> <YYYY-MM-DD>")
    if not (src / "identities.yaml").is_file():
        raise SystemExit(f"{src}: no identities.yaml; is it a golden_al_distribution checkout?")
    lock = json.loads(LOCK.read_text())
    rederive = REDERIVE.read_text()
    if not REDERIVE_PIN.search(rederive):
        raise SystemExit(f"{REDERIVE}: the INFERENCEX_SHA pin line is not where this script expects it")
    files = hashes(src)
    if files == lock["files"]:
        print("unchanged")
        return False
    saved = {LOCK: LOCK.read_bytes(), REDERIVE: rederive.encode(), TARGET: TARGET.read_bytes()}
    try:
        LOCK.write_text(json.dumps(dict(lock, commit=sha, date=date, files=files),
                                   indent=2, sort_keys=True) + "\n")
        REDERIVE.write_text(REDERIVE_PIN.sub(f"      INFERENCEX_SHA: {sha}", rederive, count=1))
        subprocess.run([sys.executable, "-I", str(EMITTER)], check=True, cwd=str(REGISTRY),
                       env=dict(os.environ, INFERENCEX_GOLDEN=str(src)))
    except BaseException:
        for path, body in saved.items():
            path.write_bytes(body)
        raise
    print(f"moved to {sha}")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    refresh(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
