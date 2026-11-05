#!/usr/bin/env python3
"""Score two registries against the InferenceX corpus and diff the result.

WHY THIS SCRIPT EXISTS. A coefficient change is judged by what it does end to end, and
that judgement is only as good as the controls. Three things other than the registry move
the numbers — the metricscore binary, the blis-catalog revision and the blis-latency-kernel
revision — so a comparison that varies any of them silently measures the wrong thing. This
once produced an apparent 0.03pp "effect" that was really two new catalog commits. The
script pins all three by construction: one binary, one invocation, one checkout, and it
prints the revisions it used so a reader can tell what was actually compared.

It does not fit, write or select anything. It reports, and it reports both configuration
tiers separately — `measured` (the run's own command line) and `resolved` (vLLM's defaults)
— because a mean over the two mixes two kinds of evidence. In this corpus that split is
also the MoE/dense split, so a difference between them is not attributable to tier alone.

Usage:
    python scripts/compare_registries.py --before /path/to/reg-a --after .
    python scripts/compare_registries.py --before HEAD --after .        # HEAD = git worktree
    python scripts/compare_registries.py --after . --tier measured
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

METRICS = ("TPOT shape", "TPOT mape", "TTFT shape", "TTFT mape")
DEFAULT_SCORER = Path("/private/tmp/blis-kernel-wt")
REPO = Path(__file__).resolve().parent.parent


# The recorded figures live beside the documents that quote them, so a reader who
# checks out this repository has both the claim and the means to test it.
BASELINE = REPO / "docs" / "evaluation-baseline.json"


def revisions(scorer: Path, catalog: Path) -> list[str]:
    def rev(p: Path) -> str:
        try:
            out = subprocess.run(["git", "-C", str(p), "log", "-1", "--format=%h %s"],
                                 capture_output=True, text=True, check=False)
            return out.stdout.strip()[:72] or "(not a git checkout)"
        except OSError:
            return "(unavailable)"
    return [f"scorer   {scorer}: {rev(scorer)}",
            f"catalog  {catalog}: {rev(catalog)}"]


def score(scorer: Path, registry: Path, catalog: Path, tier: str,
          framework: str, ratio: str, out_path: Path) -> None:
    """Run metricscore once. Deliberately `go run` from one checkout for both arms."""
    cmd = ["go", "run", "./cmd/metricscore",
           "-registry", str(registry), "-catalog", str(catalog),
           "-config-tier", tier, "-framework", framework,
           "-length-range-ratio", ratio]
    with out_path.open("w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=scorer, stdout=fh, stderr=subprocess.STDOUT, check=False)
    if r.returncode != 0:
        tail = "\n".join(out_path.read_text(encoding="utf-8").splitlines()[-15:])
        raise SystemExit(f"metricscore failed for {registry}:\n{tail}")


def headline(path: Path) -> dict[str, tuple[float, str]]:
    t = path.read_text(encoding="utf-8")
    out: dict[str, tuple[float, str]] = {}
    for m in METRICS:
        hit = re.search(
            r"=== " + m + r".*?\nblis-latency-kernel\s+(\d+)\s+\S+\s+\S+\s+([0-9.]+)%",
            t, re.S)
        if hit:
            out[m] = (float(hit.group(2)), hit.group(1))
    return out


def by_model(path: Path) -> dict[tuple[str, str], float]:
    t = path.read_text(encoding="utf-8")
    parts = t.split("=== By model")
    if len(parts) < 2:
        return {}
    out: dict[tuple[str, str], float] = {}
    for line in parts[1].splitlines():
        f = line.split()
        if len(f) >= 3 and f[1] in ("TPOT", "TTFT"):
            try:
                out[(f[0], f[1])] = float(f[2].rstrip("%"))
            except ValueError:
                pass
    return out


def materialise(spec: str, tmp: Path) -> Path:
    """A path, or `HEAD`/a git revision exported into a temporary directory."""
    p = Path(spec)
    if p.is_dir():
        return p.resolve()
    dest = tmp / f"reg-{spec.replace('/', '_')}"
    dest.mkdir(parents=True)
    tar = subprocess.run(["git", "-C", str(REPO), "archive", spec],
                         capture_output=True, check=False)
    if tar.returncode != 0:
        raise SystemExit(f"{spec!r} is neither a directory nor a git revision of {REPO}")
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar.stdout, check=True)
    return dest


def single(scorer: Path, registry: Path, catalog: Path, tiers: list[str],
           args, keep: Path) -> int:
    """Score ONE registry and either record its headline figures or check them.

    This is the end-to-end reproducibility path. It reuses the same controls the
    two-registry comparison uses -- one scorer checkout, one catalog, an explicit
    `-registry`, and the revisions printed above -- because those are exactly the
    things that silently move a figure.

    A recorded file is a claim about a specific triple of revisions, not about the
    registry alone. `--check` therefore compares the revisions too and says so when
    they differ, rather than reporting a drift it cannot attribute.
    """
    live: dict = {"framework": args.framework,
                  "length_range_ratio": args.length_range_ratio,
                  "revisions": revisions(scorer, catalog),
                  "tiers": {}}
    for tier in tiers:
        path = keep / f"single-{tier}.txt"
        score(scorer, registry, catalog, tier, args.framework,
              args.length_range_ratio, path)
        H = headline(path)
        if not H:
            print(f"=== {tier}: no headline table parsed", file=sys.stderr)
            return 1
        live["tiers"][tier] = {m: [round(H[m][0], 2), H[m][1]] for m in METRICS if m in H}

    dest = Path(args.record or args.check)
    if args.record:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(live, indent=2) + "\n", encoding="utf-8")
        print(f"\nrecorded {dest}")
        for tier, ms in live["tiers"].items():
            print(f"  {tier}: " + "  ".join(f"{m} {v[0]:.2f}%" for m, v in ms.items()))
        return 0

    if not dest.exists():
        print(f"no recorded baseline at {dest}; run --record first", file=sys.stderr)
        return 1
    rec = json.loads(dest.read_text())
    rc = 0
    for key in ("framework", "length_range_ratio"):
        if rec.get(key) != live[key]:
            print(f"FAIL {key}: recorded={rec.get(key)!r} now={live[key]!r}")
            rc = 1
    if rec.get("revisions") != live["revisions"]:
        print("NOTE the scorer or catalog revision moved since the baseline was recorded:")
        for line in rec.get("revisions", []):
            print(f"  recorded  {line}")
        for line in live["revisions"]:
            print(f"  now       {line}")
    for tier, ms in live["tiers"].items():
        recorded = rec.get("tiers", {}).get(tier, {})
        for m, (val, n) in ms.items():
            if m not in recorded:
                print(f"FAIL {tier} {m}: not in the recorded baseline")
                rc = 1
                continue
            rval, rn = recorded[m]
            if abs(val - rval) > 0.005 or n != rn:
                print(f"FAIL {tier} {m}: computed={val:.2f}% n={n} "
                      f"recorded={rval:.2f}% n={rn}")
                rc = 1
            else:
                print(f"OK   {tier} {m}: {val:.2f}% (n={n})")
    return rc


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default="HEAD",
                    help="registry path or git revision (default HEAD)")
    ap.add_argument("--after", default=str(REPO), help="registry path (default this repo)")
    ap.add_argument("--scorer", default=str(DEFAULT_SCORER),
                    help="checkout holding cmd/metricscore")
    ap.add_argument("--catalog", default=os.environ.get(
        "BLIS_CATALOG", "/Users/sri/Documents/Projects/blis-catalog"))
    ap.add_argument("--tier", default="both", choices=["measured", "resolved", "both"])
    ap.add_argument("--framework", default="vllm")
    ap.add_argument("--length-range-ratio", default="1.0")
    ap.add_argument("--keep", metavar="DIR", help="write the raw score reports here")
    ap.add_argument("--record", metavar="FILE", nargs="?", const=str(BASELINE),
                    help="score --after alone and write its headline figures to FILE "
                         f"(default {BASELINE.name}), instead of comparing two registries")
    ap.add_argument("--check", metavar="FILE", nargs="?", const=str(BASELINE),
                    help="score --after alone and FAIL if any headline figure differs "
                         f"from FILE (default {BASELINE.name})")
    args = ap.parse_args(argv[1:])

    scorer, catalog = Path(args.scorer), Path(args.catalog)
    if not (scorer / "cmd" / "metricscore").is_dir():
        raise SystemExit(f"--scorer {scorer} has no cmd/metricscore")
    for line in revisions(scorer, catalog):
        print(f"# {line}")
    print(f"# framework={args.framework} length-range-ratio={args.length_range_ratio}")

    tiers = ["measured", "resolved"] if args.tier == "both" else [args.tier]
    tmp = Path(tempfile.mkdtemp(prefix="cmpreg-"))
    keep = Path(args.keep) if args.keep else tmp
    keep.mkdir(parents=True, exist_ok=True)
    try:
        after = Path(args.after).resolve()
        if args.record or args.check:
            return single(scorer, after, catalog, tiers, args, keep)
        before = materialise(args.before, tmp)
        print(f"# before  {before}\n# after   {after}")
        rc = 0
        for tier in tiers:
            paths = {}
            for label, reg in (("before", before), ("after", after)):
                paths[label] = keep / f"{label}-{tier}.txt"
                score(scorer, reg, catalog, tier, args.framework,
                      args.length_range_ratio, paths[label])
            B, A = headline(paths["before"]), headline(paths["after"])
            if not B or not A:
                print(f"\n=== {tier}: no headline table parsed", file=sys.stderr)
                rc = 1
                continue
            print(f"\n=== {tier} tier")
            print(f"{'metric':12} {'before':>9} {'after':>9} {'delta':>8}   n")
            for m in METRICS:
                if m in B and m in A:
                    print(f"{m:12} {B[m][0]:8.2f}% {A[m][0]:8.2f}% "
                          f"{A[m][0] - B[m][0]:+7.2f}   {B[m][1]}")
            MB, MA = by_model(paths["before"]), by_model(paths["after"])
            shared = sorted(set(MB) & set(MA))
            if shared:
                print(f"\n  {'model':24} {'':4} {'before':>8} {'after':>8} {'delta':>8}")
                for k in shared:
                    d = MA[k] - MB[k]
                    print(f"  {k[0]:24} {k[1]:4} {MB[k]:7.2f}% {MA[k]:7.2f}% {d:+7.2f}"
                          + ("" if abs(d) > 0.005 else "   (unchanged)"))
        return rc
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
