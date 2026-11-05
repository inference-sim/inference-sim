#!/usr/bin/env python3
"""Write the direct corpus's measured latencies in the loader's own absolutes format.

`harness.LoadAbsolutes` reads a per-sweep record of concurrency -> microseconds for TPOT
and
TTFT, which is what makes a mean ABSOLUTE percentage error computable against
predictions in
real units. The committed file of that shape comes from
extract_inferencex_absolutes.py, which validates each sweep against NVIDIA's published
relatives -- the check that establishes those are the same rows AISimulate was scored
on.

This corpus has no such check available, and says so rather than implying one. Its
sweeps
are deployments NVIDIA does not publish, so there are no relatives to reproduce. What
replaces that check is narrower and stated per sweep:

  - every row sharing a stated configuration is kept and averaged, so no run-selection
    policy chooses between them and none can choose wrong;
  - `relative_agreement` is written as the measured within-point spread rather than as
  an
    agreement with NVIDIA, because agreeing with NVIDIA is not a thing these rows can
    do.
    A reader comparing the two files must not read the field as the same quantity.

Usage:
    python scripts/gen_direct_absolutes.py \
        -c testdata/measurements/inferencex_direct.json \
        -o testdata/measurements/inferencex_direct_absolutes.json
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--corpus", required=True)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    anchors = []
    for sw in corpus["sweeps"]:
        spreads = [p["tpot_spread"] for p in sw["points"]
                   if p["repeats"] > 1 and p["tpot_spread"]]
        anchors.append({
            "scenario": sw["scenario"],
            "label": sw["label"],
            "framework": sw["framework"],
            "gpu": sw["gpu"],
            "absolute_tpot_us": {str(p["concurrency"]): p["measured_tpot_us"]
                                 for p in sw["points"]},
            "absolute_ttft_us": {str(p["concurrency"]): p["measured_ttft_us"]
                                 for p in sw["points"]},
            "source_run_date": sw["dates"][-1] if sw.get("dates") else "",
            "source_config_id": sw["config_ids"][0] if sw.get("config_ids") else 0,
            # NOT an agreement with NVIDIA: this corpus has no published relatives to
            # reproduce. It is the worst within-point spread across this sweep's
            # repeated
            # runs under one stated configuration, as a fraction, or 0.0 where every
            # point
            # ran once. A reader must not compare it with the other file's field.
            "relative_agreement": round(max(spreads) - 1.0, 4) if spreads else 0.0,
        })

    anchors.sort(key=lambda a: (a["scenario"], a["label"]))
    doc = {
        "source": corpus["source"],
        "source_url": "",
        "release_tag": "",
        "metrics": "mean_tpot and mean_ttft, in microseconds, from the dump itself",
        "validation": (
            "No agreement with NVIDIA's published relatives is claimed or possible: "
            "these are deployments NVIDIA does not publish. What stands in its place "
            "is that no run-selection policy is applied -- every row sharing a stated "
            "configuration is kept and averaged -- and that relative_agreement here "
            "reports the within-point spread of those repeats, not an agreement with "
            "another source."),
        "anchors": anchors,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    npts = sum(len(a["absolute_tpot_us"]) for a in anchors)
    print(f"{len(anchors)} sweeps, {npts} points -> {args.out}")
    worst = max((a["relative_agreement"] for a in anchors), default=0.0)
    print(f"  worst within-point repeat spread: {worst * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
