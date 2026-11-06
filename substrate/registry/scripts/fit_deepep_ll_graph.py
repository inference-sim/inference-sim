#!/usr/bin/env python3
"""Fit DeepEP low-latency's dispatch + combine price under CUDA-graph replay, per part.

WHAT THIS CLOSES. moe_a2a_{floor,rate}_deepep_low_latency_8rank_<part> (fit_deepep_a2a.py)
is fitted on vLLM v0.24.0's prepare()/finalize() timed EAGERLY, host issue included; its
rationale says that under CUDA graphs it is an upper bound. vLLM v0.31.0 captures the
low-latency kernels in a decode step's graph. InferenceX CollectiveX times the same legacy
kernels replayed as graph-compatible chained pairs, and the eager entries over-price it by
+165% (h100) and +151% (h200) at the signed median. This writes a separate graph-mode pair;
blis-latency-kernel takes it for a captured step and the eager pair otherwise.

THE DATA. InferenceX CollectiveX (inferencex.semianalysis.com /api/v1/collectivex/runs/<id>,
contract v1), backend deepep-v2 in mode low-latency, which runs DeepEP's legacy Buffer
low-latency kernels (InferenceX collectivex/bench/ep_deepep_v2.py), bf16, EP8 on one node,
shape hidden 7168, top-k 6, 384 experts, uniform routing; the roundtrip (dispatch + combine)
p50 per tokens-per-rank point, median over the runs that measured it.

WHY DeepEP 01dc3aa MEASURES v0.31.0's d4f41e4e93. CollectiveX pins DeepEP 01dc3aaac820
(ep_deepep_v2.py:30); vLLM v0.31.0 pins d4f41e4e93 (docker/Dockerfile:516). The GitHub
compare d4f41e4e93...01dc3aa is 6 commits ahead, 0 behind, and touches no file under
csrc/kernels/legacy, where the low-latency kernels live; every change is to the elastic
(v2) path.

THE FORM: latency_us = floor + tokens_per_rank x hidden x 2 x top_k / rate, the basis the
eager entries and the kernel use. Fitted by mean |log ratio| on tokens {1, 4, 16, 64, 256};
held out {2, 8, 32, 128}, declared before the fit; band: held-out median |rel err| <= 15%.

THE BODIES MUST BE THE CITED ONES. CollectiveX's API serves only its latest 15 runs, and the
run bodies are fetched from https://inferencex.semianalysis.com/api/v1/collectivex/runs/<id>?version=1
(the bare `version=1` query is required; without it the API answers "Unknown version"). A
re-fit on whatever runs are served today is a different measurement, so it must not overwrite
an entry that cites other runs. Before fitting, this checks that every run the committed entries
cite is present and that those bodies hash to the cited sha256; if not, it refuses (exit 3) and
names what is missing, rather than re-citing the entries to new runs.

Usage:
    python scripts/fit_deepep_ll_graph.py --data <dir of CollectiveX run bodies> [--check]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SET_PATH = REPO / "coefficients" / "cost-model-collectives-inferencex.yaml"
PARTS = {"h100": "h100-dgxc", "h200": "h200-dgxc"}
HIDDEN, TOP_K, RANKS = 7168, 6, 8
FIT, HELD = {1, 4, 16, 64, 256}, {2, 8, 32, 128}
BAND = 0.15
LIBRARY_DIFF = ("DeepEP d4f41e4e93...01dc3aa: 6 commits ahead, 0 behind, no file under "
                "csrc/kernels/legacy")

_spec = importlib.util.spec_from_file_location(
    "coefficient_blocks", Path(__file__).parent / "coefficient_blocks.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def points(data: Path, sku: str) -> tuple[dict[int, float], list[str]]:
    by: dict[int, list[float]] = {}
    runs = set()
    for f in sorted(glob.glob(str(data / "cx_run_*.body"))):
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        for s in d.get("series") or []:
            sy = s.get("system") or {}
            if (s.get("backend"), s.get("mode"), s.get("precision"), sy.get("sku"),
                    sy.get("ep_size"), sy.get("nodes")) != (
                    "deepep-v2", "low-latency", "bf16", sku, RANKS, 1):
                continue
            runs.add(d["run"]["run_id"])
            for p in s["points"]:
                by.setdefault(p["tokens_per_rank"], []).append(
                    p["components"]["roundtrip"]["latency_us"]["p50"])
    return {t: statistics.median(v) for t, v in by.items()}, sorted(runs)


def fit(data: dict[int, float]) -> tuple[float, float, float]:
    rows = [(t * HIDDEN * 2 * TOP_K, data[t]) for t in sorted(data) if t in FIT]
    best = None
    for i in range(50, 600):
        fl = i * 0.1
        for j in range(601):
            rate = math.exp(math.log(1e3) + j * (math.log(1e6) - math.log(1e3)) / 600)
            e = sum(abs(math.log((fl + b / rate) / y)) for b, y in rows) / len(rows)
            if best is None or e < best[0]:
                best = (e, fl, rate)
    return best[1], best[2], best[0]


def held_out(data, fl, rate):
    return {t: (fl + t * HIDDEN * 2 * TOP_K / rate) / data[t] - 1
            for t in sorted(data) if t in HELD}


def entry(name, value, units, part, runs, sha, note):
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{part}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "InferenceX CollectiveX runs {", ".join(runs)} '
        f'(api/v1/collectivex, contract v1; bodies sha256 {sha}), deepep-v2 low-latency '
        f'(DeepEP legacy Buffer, 01dc3aa), bf16, EP8 one node, roundtrip p50", role: primary}}',
        "      rationale: >",
        "        DeepEP low-latency dispatch plus combine as one MoE layer's price under CUDA-graph "
        "replay, which vLLM v0.31.0 uses for a captured decode step; the eager pair "
        "(moe_a2a_*_deepep_low_latency_8rank) prices a step with no graph. floor + tokens x "
        "hidden x 2 x top_k / rate, per-rank tokens, fitted on tokens 1/4/16/64/256. "
        f"{note} {LIBRARY_DIFF}. Reproduce with scripts/fit_deepep_ll_graph.py.",
    ]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("COLLECTIVEX_DATA"),
                    help="directory of CollectiveX run bodies (default $COLLECTIVEX_DATA)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv[1:])
    if not args.data or not Path(args.data).is_dir():
        print("no CollectiveX data: pass --data or set COLLECTIVEX_DATA", file=sys.stderr)
        return 2
    before = SET_PATH.read_text(encoding="utf-8")
    missing = cited_bodies_absent(before, Path(args.data))
    if missing:
        print(f"refusing: the cited CollectiveX bodies are not present ({missing}). The API "
              "serves only its latest 15 runs; restore the cited bodies to re-derive, or fit "
              "new runs into new entries rather than re-citing these.", file=sys.stderr)
        return 3
    blocks = cb.split(before)
    for part, sku in PARTS.items():
        data, runs = points(Path(args.data), sku)
        fl, rate, err = fit(data)
        held = held_out(data, fl, rate)
        med = statistics.median(abs(x) for x in held.values())
        sha = _sha(args.data, runs)
        line = ", ".join(f"T{t} {x:+.1%}" for t, x in held.items())
        print(f"{part}: floor {fl:.1f} us rate {rate / 1e3:.1f} GB/s fit {err:.3f}; "
              f"held out {line}; median {med:.1%} ({'pass' if med <= BAND else 'FAIL'})")
        if med > BAND:
            continue
        note = f"Held out (tokens 2/8/32/128): median |rel err| {med:.1%}; {line}."
        scope = f"{{hardware: [{part}]}}"
        for name, value, units in (
                (f"moe_a2a_floor_deepep_low_latency_graph_{RANKS}rank_{part}", round(fl, 1),
                 "us_per_transfer"),
                (f"moe_a2a_rate_deepep_low_latency_graph_{RANKS}rank_{part}",
                 float(f"{rate:.4g}"), "bytes_per_us")):
            cb.upsert(blocks, entry(name, value, units, part, runs, sha, note),
                      after=lambda b: b[0] == "coefficients:" or cb.name_of(b) is not None)
    after = cb.join(blocks)
    if args.check:
        if after != before:
            print(f"{SET_PATH.name} would change", file=sys.stderr)
            return 1
        print(f"{SET_PATH.name} already matches the fit")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    yaml.safe_load(after)
    print(f"{SET_PATH.name} written")
    return 0


CITE = re.compile(r"InferenceX CollectiveX runs ([0-9, ]+) \(api/v1/collectivex, contract v1; "
                  r"bodies sha256 ([0-9a-f]{12})\)")


def cited_bodies_absent(text: str, data: Path) -> str:
    """What of the committed entries' cited bodies is absent or differs, or '' if none."""
    problems = []
    for e in yaml.safe_load(text)["coefficients"]:
        (name, body), = e.items()
        if not name.startswith("moe_a2a_") or "_low_latency_graph_8rank_" not in name:
            continue
        for s in body.get("sources") or []:
            m = CITE.search(s["cite"])
            if not m:
                continue
            runs = [r.strip() for r in m.group(1).split(",")]
            absent = [r for r in runs if not (data / f"cx_run_{r}.body").is_file()]
            if absent:
                problems.append(f"{name}: runs {', '.join(absent)} absent")
            elif _sha(str(data), runs) != m.group(2):
                problems.append(f"{name}: cited bodies hash to {_sha(str(data), runs)}, "
                                f"not the cited {m.group(2)}")
    return "; ".join(sorted(set(problems)))


def _sha(data: str, runs: list[str]) -> str:
    h = hashlib.sha256()
    for r in runs:
        h.update((Path(data) / f"cx_run_{r}.body").read_bytes())
    return h.hexdigest()[:12]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
