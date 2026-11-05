#!/usr/bin/env python3
"""Extract ABSOLUTE measured latencies from SemiAnalysis InferenceX's published database dump.

# Why this exists

NVIDIA's accuracy artifact publishes every latency normalised to its sweep's lowest concurrency,
so a mean ABSOLUTE percentage error cannot be computed from it for a model whose predictions are
in real units. The quantity it withholds is one number per sweep: the measured latency at the
anchor concurrency.

That number is published, by the measurement's own source. The artifact's snapshot block names it:
measurement_source "SemiAnalysis InferenceX", measurement_source_url pointing at the
InferenceX-app release db-dump/2026-09-28, pinned by measurement_sha256. This script reads that
dump and writes the anchors, so a mape against the same ground truth both simulators were scored
on becomes computable.

# The join, and the check that validates it

    benchmark_results  keyed (config_id, benchmark_type, isl, osl, conc), metrics jsonb
    configs            hardware, framework, model, precision, decode_tp, decode_ep, disagg, ...

A sweep is matched on model slug, hardware, framework, decode_tp, non-disagg and non-multinode,
restricted to benchmark_type 'single_turn', at the sweep's isl:osl. Among the dated runs for that
config the one selected is whichever reproduces the artifact's own measured relatives.

That last step is the point. NVIDIA applies selection_policy latest-complete-config-run-v1 and
filters 33,140 incomplete, 24,649 stale and 10,524 multinode rows, none of which this script can
see. Rather than guess the policy, it picks the run whose relatives match what the artifact
publishes, and REFUSES a sweep where no run matches within the tolerance. Reproducing the
baseline's own published numbers from the baseline's own source is the only check that the rows
are the same rows.

Two facts about the dump that are easy to get wrong, recorded because both cost time:

  - The artifact uses mean_tpot and mean_ttft, not the medians. The medians are present and
    differ: on gpt-oss-120b h200 tp4 1k1k the mean TTFT relatives reproduce the artifact exactly
    while the medians are off by 50% at the top of the sweep.
  - Metrics are in SECONDS. median_e2el reads 3.79 for a 1024-token generation at 4 ms/token.
  - benchmark_type 'agentic_traces' rows (2,431 of 85,309) carry NULL isl and osl. Only
    'single_turn' is the benchmark the artifact scores.

Requires a restored dump. See docs for the reassembly and pg_restore commands; only
benchmark_results, configs and workflow_runs are needed.

Usage:
    python scripts/extract_inferencex_absolutes.py \
        -c testdata/measurements/aisimulate_e2e.json \
        -o testdata/measurements/inferencex_absolutes.json
"""

import argparse
import collections
import json
import subprocess
import sys

PSQL = "psql"
DB = "inferencex"

# Artifact model slug -> blis-catalog model name, as scripts/extract_aisimulate_e2e.py defines it.
# Imported rather than restated so one re-map cannot diverge from the other.
sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
from extract_aisimulate_e2e import MODELS  # noqa: E402

TO_SLUG = {catalog: slug for slug, catalog in MODELS.items()}

# A run reproduces the artifact when every point's relative agrees to this fraction. 0.5% is far
# tighter than the spread BETWEEN dated runs of the same config (several percent), so it
# identifies one run rather than admitting a family of them.
TOLERANCE = 0.005


def query(sql: str) -> list[list[str]]:
    out = subprocess.run([PSQL, "-d", DB, "-t", "-A", "-F", "|", "-c", sql],
                         capture_output=True, text=True, check=True)
    return [line.split("|") for line in out.stdout.strip().split("\n") if line]


def candidates(sweep: dict) -> dict[str, dict[int, tuple[float, float]]]:
    """Every (date, config) run for this deployment, as {key: {conc: (tpot_s, ttft_s)}}."""
    isl, osl = sweep["workload"].split(":")
    sql = f"""
    select b.date || '/' || c.id, b.conc,
           (b.metrics->>'mean_tpot')::numeric, (b.metrics->>'mean_ttft')::numeric
    from benchmark_results b join configs c on c.id = b.config_id
    where c.model = '{TO_SLUG[sweep["model"]]}'
      and c.hardware = '{sweep["gpu"]}'
      and c.framework = '{sweep["framework"]}'
      and c.decode_tp = {sweep["parallelism"]["tp_size"]}
      and c.disagg = false and c.is_multinode = false
      and b.benchmark_type = 'single_turn'
      and b.isl = {isl} and b.osl = {osl}
      and b.error is null
      and b.metrics ? 'mean_tpot' and b.metrics ? 'mean_ttft'
    """
    runs: dict[str, dict[int, tuple[float, float]]] = collections.defaultdict(dict)
    for key, conc, tpot, ttft in query(sql):
        runs[key][int(conc)] = (float(tpot), float(ttft))
    return runs


def match(sweep: dict, runs: dict) -> tuple[str, dict, float] | None:
    """The run whose relatives reproduce the artifact's, or None."""
    want = {p["concurrency"]: (p["measured_tpot_relative"], p["measured_ttft_relative"])
            for p in sweep["points"]}
    best = None
    for key, points in runs.items():
        if not all(c in points for c in want):
            continue
        concs = sorted(points)
        anchor_tpot, anchor_ttft = points[concs[0]]
        if anchor_tpot <= 0 or anchor_ttft <= 0:
            continue
        worst = 0.0
        for c, (wt, wf) in want.items():
            tp, tt = points[c]
            worst = max(worst, abs(tp / anchor_tpot - wt) / wt, abs(tt / anchor_ttft - wf) / wf)
        if best is None or worst < best[2]:
            best = (key, points, worst)
    if best is None or best[2] > TOLERANCE:
        return None
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--corpus", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--framework", default="", help="restrict to one framework")
    args = ap.parse_args()

    corpus = json.load(open(args.corpus))
    out = []
    unmatched = []
    for sweep in corpus["sweeps"]:
        if args.framework and sweep["framework"] != args.framework:
            continue
        if sweep["model"] not in TO_SLUG:
            unmatched.append((sweep["scenario"], sweep["label"], "model not in the slug map"))
            continue
        found = match(sweep, candidates(sweep))
        if found is None:
            unmatched.append((sweep["scenario"], sweep["label"], "no run reproduces the artifact"))
            continue
        key, points, worst = found
        date, config_id = key.split("/")
        concs = sorted(points)
        out.append({
            "scenario": sweep["scenario"],
            "label": sweep["label"],
            "framework": sweep["framework"],
            "gpu": sweep["gpu"],
            # Seconds in the dump; microseconds here, the unit the simulator reports.
            "anchor_concurrency": concs[0],
            "anchor_tpot_us": points[concs[0]][0] * 1e6,
            "anchor_ttft_us": points[concs[0]][1] * 1e6,
            "absolute_tpot_us": {str(c): points[c][0] * 1e6 for c in concs},
            "absolute_ttft_us": {str(c): points[c][1] * 1e6 for c in concs},
            "source_run_date": date,
            "source_config_id": int(config_id),
            "relative_agreement": worst,
        })

    doc = {
        "source": "SemiAnalysis InferenceX database dump",
        "source_url": corpus["snapshot"].get("measurement_source_url"),
        "release_tag": corpus["snapshot"].get("release_tag"),
        "metrics": "mean_tpot and mean_ttft, the fields NVIDIA's artifact normalises; "
                   "SECONDS in the dump, microseconds here",
        "benchmark_type": "single_turn",
        "validation": f"each sweep's run reproduces the artifact's own measured relatives to "
                      f"within {TOLERANCE * 100:.1f}%; a sweep with no such run is omitted",
        "sweeps_matched": len(out),
        "sweeps_omitted": len(unmatched),
        "omitted": [{"scenario": s, "label": l, "reason": r} for s, l, r in unmatched],
        "anchors": out,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{len(out)} sweeps matched, {len(unmatched)} omitted -> {args.out}")
    if out:
        print(f"worst relative disagreement among matched sweeps: "
              f"{max(a['relative_agreement'] for a in out) * 100:.3f}%")
    for s, l, r in unmatched[:12]:
        print(f"  omitted {s} {l}: {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
