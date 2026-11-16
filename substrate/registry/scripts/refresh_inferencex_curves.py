#!/usr/bin/env python3
"""Move the vendored InferenceX golden curves to another InferenceX commit.

    refresh_inferencex_curves.py <golden_al_distribution dir> <commit> <YYYY-MM-DD>

Copies every *.yaml of the given checkout's golden_al_distribution/ (the curve files and
identities.yaml) over validator/testdata/inferencex/golden_al_distribution/, removing any the
checkout no longer has, and moves the pin to <commit> in the three places that state it:
INFERENCEX_SHA in emit_acceptance_profiles.py, the commit line of VENDORED.md, and
INFERENCEX_SHA in .github/workflows/rederive.yml. It then re-runs the emitter, which
regenerates acceptance/inferencex-golden-al.yaml. It prints "unchanged" and changes nothing
when the checkout's curves and the pin are already the vendored ones.

The scheduled workflow acceptance-refresh.yml runs it and opens a pull request on change; the
emitter's --check stays the drift gate.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent
VENDORED = REGISTRY / "validator" / "testdata" / "inferencex"
EMITTER = REGISTRY / "scripts" / "emit_acceptance_profiles.py"
REDERIVE = REGISTRY.parent / ".github" / "workflows" / "rederive.yml"

SHA = re.compile(r"[0-9a-f]{40}")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
PINS = (
    (EMITTER, re.compile(r'^INFERENCEX_SHA = "([0-9a-f]{40})"$', re.M), 'INFERENCEX_SHA = "{sha}"'),
    (REDERIVE, re.compile(r"^      INFERENCEX_SHA: ([0-9a-f]{40})$", re.M),
     "      INFERENCEX_SHA: {sha}"),
)
VENDORED_LINE = re.compile(r"at commit `([0-9a-f]{40})` \((\d{4}-\d{2}-\d{2})\)")


def same_tree(src: Path, dst: Path) -> bool:
    a = sorted(p.name for p in src.glob("*.yaml"))
    b = sorted(p.name for p in dst.glob("*.yaml"))
    return a == b and all((src / n).read_bytes() == (dst / n).read_bytes() for n in a)


def refresh(src: Path, sha: str, date: str) -> bool:
    if not SHA.fullmatch(sha) or not DATE.fullmatch(date):
        raise SystemExit("usage: refresh_inferencex_curves.py <dir> <40-hex commit> <YYYY-MM-DD>")
    if not (src / "identities.yaml").is_file():
        raise SystemExit(f"{src}: no identities.yaml; is it a golden_al_distribution checkout?")
    golden = VENDORED / "golden_al_distribution"
    pinned = {path: rx.search(path.read_text()) for path, rx, _ in PINS}
    for path, m in pinned.items():
        if not m:
            raise SystemExit(f"{path}: the INFERENCEX_SHA pin line is not where this script "
                             "expects it")
    if same_tree(src, golden) and all(m.group(1) == sha for m in pinned.values()):
        print("unchanged")
        return False
    for p in golden.glob("*.yaml"):
        p.unlink()
    for p in sorted(src.glob("*.yaml")):
        shutil.copyfile(p, golden / p.name)
    for path, rx, line in PINS:
        text = path.read_text()
        path.write_text(rx.sub(line.format(sha=sha), text, count=1))
    md = VENDORED / "VENDORED.md"
    text = md.read_text()
    if not VENDORED_LINE.search(text):
        raise SystemExit(f"{md}: no 'at commit `<sha>` (<date>)' line")
    md.write_text(VENDORED_LINE.sub(f"at commit `{sha}` ({date})", text, count=1))
    subprocess.run([sys.executable, "-I", str(EMITTER)], check=True, cwd=str(REGISTRY))
    print(f"moved to {sha}")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    refresh(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
