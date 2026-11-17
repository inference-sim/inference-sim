#!/usr/bin/env python3
"""Extract the scoreable per-point corpus from NVIDIA AISimulate's e2e accuracy artifact.

The artifact is published by the `E2E Accuracy Matrix` workflow in ai-dynamo/aisim (the
repository was ai-dynamo/aisimulate until it was renamed) as `e2e-accuracy-web`, and carries a
nested summary:

    models -> workloads -> gpus -> topologies -> points

Each point holds `measured`, `aic` and `aisim` blocks of TTFT/TPOT relatives. A topology
is one deployment at one workload; its points are a concurrency sweep. The simulator's block
was named `aisimulate` before the rename and is `aisim` now; the corpus this writes keeps the
`aisimulate_*` field names its readers use, because the estimator is the same one. A summary
whose totals have no `aisim` block is refused rather than read as an estimator with no figures.

# Provenance, stated because it is mixed

The predictions and the snapshot are NVIDIA's. The `measured_*` ground truth is NOT: the
artifact's own `measurement_source` is "SemiAnalysis InferenceX", with a release URL. This
project's REGISTRY is single-sourced from AISimulate and contains no InferenceX data -- that
constraint is about coefficients, the numbers the model is built from. This file is an
evaluation set, a different role: it is what the model is scored against, never fitted to. The
distinction is recorded here so a later reader does not mistake one for the other.

# Why relatives, and what that costs

Absolute latencies are not disclosed, so every value is normalised to its sweep's lowest
concurrency. That is why no simulator is needed to score it -- a ratio of step times at one
fixed deployment is a pure function of the kernel -- and also why it measures shape only.

# Selection

A topology is emitted when it has at least three successful points. Two points carry a single
ratio, which pins a level rather than a curvature -- scoring a shape against it measures almost
nothing and dilutes the reported figure with sweeps that cannot disagree. Aggregated serving
with no speculation and one pipeline stage is the whole of the reachable set here: a step-time
ratio at one deployment cannot price a prefill pool and a decode pool, or a drafter. The
artifact carries disaggregated and speculative topologies; each is counted under its reason in
`selection.skipped`, never scored under a deployment it is not.

A model is selected by its checkpoint. The artifact names each by its Hugging Face path, and a
topology is kept only when exactly one catalog model's `source.repo` is that path, read from
catalog/models at run time rather than copied here. A checkpoint the catalog does not carry,
such as an NVFP4 export whose base model it does, is skipped: pricing it on the base graph would
score another checkpoint's weights.

# The input is pinned (D3)

The extractor reads one artifact run, named in catalog/ledger/sources.yaml (source aisimulate,
artifact e2e-accuracy-web) and copied to kernel/scripts/aisimulate_e2e.pin.json, which a Go test
requires to equal that entry. It refuses a summary.json whose SHA-256 is not the pinned one,
whose snapshot does not state the pinned run id, attempt, evaluated commit and release tag (a
"latest", a branch, or a missing field is refused), or whose extraction does not give the pinned
number of sweeps and points. To move to a newer artifact, change the ledger entry and the pin
file together in one pull request.

Usage:

    python scripts/extract_aisimulate_e2e.py summary.json -o testdata/measurements/aisimulate_e2e.json
"""

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

PIN_PATH = Path(__file__).resolve().with_name("aisimulate_e2e.pin.json")
UNPINNED = {"latest", "main", "unknown", ""}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

CATALOG_MODELS = Path(__file__).resolve().parents[2] / "catalog" / "models"
REPO_LINE = re.compile(r"^\s+repo:\s*(\S+)\s*$")


def catalog_models(root: Path = CATALOG_MODELS) -> dict:
    """Hugging Face path -> the catalog models whose model.yaml names it as source.repo.

    Read line by line rather than with a YAML parser, so the extractor needs only the standard
    library: `repo:` appears once in a model.yaml, under `source:`. A path can name more than
    one catalog model (a speculator shipped inside its target's checkpoint has its own entry);
    extract refuses such a path when the artifact selects it.
    """
    out: dict = {}
    for f in sorted(root.glob("*/model.yaml")):
        lines = f.read_text().splitlines()
        try:
            src = lines.index("source:")
        except ValueError:
            continue
        for line in lines[src + 1:]:
            if line and not line.startswith(" "):
                break
            m = REPO_LINE.match(line)
            if m:
                out.setdefault(m.group(1), []).append(f.parent.name)
                break
    return out


# Artifact GPU slug -> blis-catalog hardware name.
CHIPS = {"h100": "h100", "h200": "h200", "b200": "b200", "b300": "b300",
         "gb200": "gb200-nvl72", "gb300": "gb300"}

MIN_POINTS = 3


def scenario_name(model: str, gpu: str, precision: str, framework: str, par: dict) -> str:
    """One file per distinct deployment, named from the fields that define it.

    A DENSE model has no experts, and the snapshot records that as moe_ep_size null rather
    than 1: every one of llama70b's 81 topologies carries None where an MoE model carries an
    integer. None is not 1 -- a dense deployment has no expert group at all, where ep=1 means
    one expert group that is not sharded -- so it is read as absent rather than coerced.
    """
    ep = par["moe_ep_size"]
    parts = [model, gpu, precision, framework, f"tp{par['tp_size']}"]
    if ep is not None and ep > 1:
        parts.append(f"ep{ep}")
    if par["attention_dp_size"] > 1:
        parts.append(f"dp{par['attention_dp_size']}")
    return "-".join(parts) + ".yaml"


def count_points(node: dict, serving_not: str | None = None) -> int:
    """How many points lie under one models, workloads or gpus record, leaving out the
    topologies of one serving mode (the disaggregated ones are carried apart, not skipped)."""
    if "points" in node:
        return 0 if node.get("serving") == serving_not and serving_not else len(node["points"])
    for child in ("workloads", "gpus", "topologies"):
        if child in node:
            return sum(count_points(c, serving_not) for c in node[child])
    return 0


def disaggregated_record(model: dict, name: str | None, workload: dict, gpu: dict, topology: dict) -> dict:
    """One prefill/decode topology, whole: every point, scored or failed, with the measured and
    the estimator's absolute figures (this artifact states them in ms) beside the relatives.

    A topology the artifact lists is kept even when BLIS cannot yet price it (`model` null for a
    checkpoint the catalog lacks): the comparison counts the estimator's points over all of them,
    and a point BLIS does not score is uncovered, not absent. The row of the InferenceX dump a
    topology measures is found by the matcher (accuracy-dashboard matched.py), not here."""
    points = []
    for p in topology["points"]:
        rec = point_record(p) if p.get("status") == "success" else {
            "concurrency": p["concurrency"], "status": p["status"]}
        for who, key in (("measured", "measured"), ("aisimulate", ESTIMATOR_BLOCK)):
            block = p.get(key)
            if isinstance(block, dict):
                for m in METRICS:
                    if block.get(f"{m}_ms") is not None:
                        rec[f"{who}_{m}_ms"] = block[f"{m}_ms"]
        if p.get("aisim_error"):
            rec["aisimulate_error"] = str(p["aisim_error"]).splitlines()[0][:200]
        points.append(rec)
    points.sort(key=lambda r: r["concurrency"])
    return {
        "topology_id": topology["id"],
        "checkpoint": model["model"],
        "model": name,
        "gpu": CHIPS.get(gpu["gpu"], gpu["gpu"]),
        "workload": workload["identity"],
        "label": workload["label"],
        "framework": topology["framework"],
        "precision": topology["precision"],
        "spec_method": topology.get("spec_method") or "none",
        "total_gpus": topology["total_gpus"],
        "is_multinode": topology["is_multinode"],
        "parallelism": topology["parallelism"],
        "serving": "disaggregated",
        "points": points,
    }


def extract(summary: dict, models: dict) -> dict:
    """The reachable sweeps. `selection.skipped` counts POINTS per reason, every point of the
    artifact landing in a sweep or under exactly one reason."""
    sweeps = []
    disaggregated = []
    skipped = collections.Counter()
    for model in summary["models"]:
        names = models.get(model["model"], [])
        if len(names) > 1:
            raise SystemExit(f"error: the artifact's checkpoint {model['model']} is the source of "
                             f"{len(names)} catalog models ({', '.join(names)}); which one it "
                             "measured is not stated")
        name = names[0] if names else None
        for workload in model["workloads"]:
            for gpu in workload["gpus"]:
                disaggregated += [disaggregated_record(model, name, workload, gpu, t)
                                  for t in gpu["topologies"] if t["serving"] == "disaggregated"]
        if name is None:
            skipped["checkpoint not in catalog"] += count_points(model, "disaggregated")
            continue
        for workload in model["workloads"]:
            for gpu in workload["gpus"]:
                chip = CHIPS.get(gpu["gpu"])
                if chip is None:
                    skipped["gpu not in catalog"] += count_points(gpu)
                    continue
                for topology in gpu["topologies"]:
                    serving = topology["serving"]
                    spec = topology.get("spec_method") or "none"
                    if serving == "disaggregated":
                        continue  # carried whole in `disaggregated`, counted by the pin
                    if serving != "aggregated" or spec != "none":
                        # Not reachable by a step-time ratio at one deployment.
                        skipped[f"serving={serving} spec={spec}"] += len(topology["points"])
                        continue
                    par = topology["parallelism"]
                    if par["pp_size"] != 1:
                        skipped["pp_size>1"] += len(topology["points"])
                        continue
                    points = [point_record(p) for p in topology["points"]
                              if p.get("status") == "success"]
                    if len(points) < MIN_POINTS:
                        skipped["fewer than three successful points"] += len(topology["points"])
                        continue
                    if len(points) < len(topology["points"]):
                        skipped["failed point in a kept sweep"] += (
                            len(topology["points"]) - len(points))
                    points.sort(key=lambda p: p["concurrency"])
                    sweeps.append(
                        {
                            "scenario": scenario_name(
                                name, chip, topology["precision"],
                                topology["framework"], par,
                            ),
                            "model": name,
                            "gpu": chip,
                            "workload": workload["identity"],
                            "label": workload["label"],
                            "framework": topology["framework"],
                            "precision": topology["precision"],
                            "serving": serving,
                            "spec_method": spec,
                            "parallelism": par,
                            # Stated by the topology; neither is a node size.
                            "total_gpus": topology.get("total_gpus"),
                            "is_multinode": topology.get("is_multinode"),
                            "topology_id": topology["id"],
                            "points": points,
                        }
                    )
    sweeps.sort(key=lambda s: (s["model"], s["gpu"], s["scenario"], s["label"]))
    disaggregated.sort(key=lambda d: (d["checkpoint"], d["gpu"], d["label"], d["topology_id"]))
    return {
        "source": "NVIDIA AISimulate e2e accuracy artifact (e2e-accuracy-web)",
        "measurement_source": summary["snapshot"].get("measurement_source"),
        "measurement_source_url": summary["snapshot"].get("measurement_source_url"),
        "snapshot": summary["snapshot"],
        "scope": summary["scope"],
        # Carries all four published figures per estimator. The *_shape_error_pct ones are
        # comparable to a normalised-curve score; the *_mape_pct ones are absolute and cannot be
        # recomputed from this corpus, which ships no absolute latencies.
        "aisimulate_totals": summary["totals"][ESTIMATOR_BLOCK],
        "aic_totals": summary["totals"].get("aic", {}),
        "selection": {
            "models": sorted({s["model"] for s in sweeps}),
            "min_points_per_sweep": MIN_POINTS,
            "serving": "aggregated",
            "spec_method": "none",
            "skipped": dict(skipped),
        },
        "sweeps": sweeps,
        "disaggregated": disaggregated,
    }


# The artifact's own field names, kept verbatim so a reader can trace a value back.
#
# TTFT is carried alongside TPOT, and AIC alongside AISIMULATE, because the snapshot publishes
# four accuracy figures and an earlier revision of this extractor kept one. TTFT is where the
# baseline is weakest -- 22.82% shape against 10.05% for TPOT over the whole snapshot -- so
# discarding it discarded the interesting half of the comparison.
#
# AIC and AISIMULATE are the same repository (snapshot.aic_source names ai-dynamo/aisimulate at a
# pinned commit): AIC is the analytic configurator path, AISimulate the full simulator. Both are
# kept so a score can be read against either.
#
# Only RELATIVES exist. Every point in the artifact carries exactly tpot_relative and
# ttft_relative per estimator, and no absolute latency field appears anywhere in it -- verified by
# enumerating every key of every point. So a mean ABSOLUTE percentage error cannot be computed
# against this corpus at all; the mape figures the snapshot publishes come from rows it does not
# ship. Shape error is the only per-point metric this file can support, which is a property of
# the data rather than a choice.
#
# The artifact's block for the simulator is `aisim`; the corpus field it becomes keeps the
# `aisimulate_` prefix the corpus readers use (OUTPUT_PREFIX).
ESTIMATOR_BLOCK = "aisim"
ESTIMATORS = (ESTIMATOR_BLOCK, "aic")
OUTPUT_PREFIX = {ESTIMATOR_BLOCK: "aisimulate", "aic": "aic"}
METRICS = ("tpot", "ttft")


def point_record(p: dict) -> dict:
    """One concurrency point: the measured relatives and every estimator's prediction."""
    rec = {"concurrency": p["concurrency"], "status": p["status"]}
    for metric in METRICS:
        rec[f"measured_{metric}_relative"] = p["measured"][f"{metric}_relative"]
    for est in ESTIMATORS:
        block = p.get(est)
        if not isinstance(block, dict):
            continue
        for metric in METRICS:
            rel = block.get(f"{metric}_relative")
            if rel is None:
                continue
            out = OUTPUT_PREFIX[est]
            rec[f"{out}_{metric}_relative"] = rel
            err = block.get(f"{metric}_error_pct")
            if err is not None:
                rec[f"{out}_{metric}_error_pct"] = err
    return rec


def load_pin(path) -> dict:
    """Read the pin and refuse one that is itself not an exact revision (D3)."""
    with open(path) as fh:
        pin = json.load(fh)
    bad = []
    if not re.fullmatch(r"[0-9]+", str(pin.get("run_id", ""))):
        bad.append("run_id is not a workflow run id")
    if not isinstance(pin.get("run_attempt"), int) or pin["run_attempt"] < 1:
        bad.append("run_attempt is not a positive integer")
    if not HEX40.match(str(pin.get("commit", ""))):
        bad.append("commit is not a full 40-digit commit")
    for key in ("zip_sha256", "summary_sha256"):
        if not HEX64.match(str(pin.get(key, ""))):
            bad.append(f"{key} is not a SHA-256")
    for key in ("artifact", "release_tag"):
        if str(pin.get(key, "")).strip().lower() in UNPINNED:
            bad.append(f"{key} is empty or unpinned")
    exp = pin.get("expected")
    if not (isinstance(exp, dict) and isinstance(exp.get("sweeps"), int)
            and isinstance(exp.get("points"), int) and exp["sweeps"] > 0 and exp["points"] > 0):
        bad.append("expected needs positive integer sweeps and points")
    if bad:
        raise SystemExit(f"error: pin {path}: " + "; ".join(bad))
    return pin


def check_input(raw: bytes, summary: dict, pin: dict) -> list:
    """Why this summary.json is not the pinned artifact; empty when it is."""
    errs = []
    got = hashlib.sha256(raw).hexdigest()
    if got != pin["summary_sha256"]:
        errs.append(f"summary.json has SHA-256 {got}, the pin is {pin['summary_sha256']}")
    snap = summary.get("snapshot")
    if not isinstance(snap, dict):
        return errs + ["no snapshot block"]
    camp = snap.get("campaign")
    camp = camp if isinstance(camp, dict) else {}
    run_id = str(camp.get("run_id") or "").strip()
    if not run_id:
        errs.append("the snapshot states no campaign run_id")
    elif run_id != pin["run_id"]:
        errs.append(f"campaign run_id {run_id!r}, the pin is {pin['run_id']!r}")
    attempt = str(camp.get("run_attempt") or "").strip()
    if attempt != str(pin["run_attempt"]):
        errs.append(f"campaign run_attempt {attempt!r}, the pin is {pin['run_attempt']!r}")
    rev = snap.get("evaluated_revision")
    commit = rev.get("commit_sha") if isinstance(rev, dict) else None
    if not isinstance(rev, dict):
        errs.append(f"evaluated_revision is {rev!r}, not a commit record")
    elif str(commit or "").strip().lower() in UNPINNED:
        errs.append(f"evaluated_revision.commit_sha is {commit!r}; a revision is pinned by commit (D3)")
    elif commit != pin["commit"]:
        errs.append(f"evaluated_revision.commit_sha {commit!r}, the pin is {pin['commit']!r}")
    tag = snap.get("release_tag")
    if tag != pin["release_tag"]:
        errs.append(f"release_tag {tag!r}, the pin is {pin['release_tag']!r}")
    if camp.get("status") != "complete":
        errs.append(f"campaign status {camp.get('status')!r}, want 'complete'")
    totals = summary.get("totals")
    if not (isinstance(totals, dict) and isinstance(totals.get(ESTIMATOR_BLOCK), dict)):
        errs.append(f"totals has no {ESTIMATOR_BLOCK!r} block: the simulator's figures are "
                    "missing or under another name")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="summary.json from the e2e-accuracy-web artifact")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--pin", default=str(PIN_PATH),
                    help="the pinned artifact run (default: aisimulate_e2e.pin.json beside this file)")
    ap.add_argument("--catalog", default=str(CATALOG_MODELS),
                    help="the catalog's models/ directory (default: this repository's)")
    args = ap.parse_args()

    pin = load_pin(args.pin)
    with open(args.summary, "rb") as fh:
        raw = fh.read()
    try:
        summary = json.loads(raw)
    except ValueError as exc:
        print(f"error: {args.summary} is not JSON: {exc}", file=sys.stderr)
        return 1
    errs = check_input(raw, summary, pin)
    if errs:
        print(f"error: {args.summary} is not the pinned e2e-accuracy-web artifact "
              f"(run {pin['run_id']}):", file=sys.stderr)
        for e in errs:
            print(f"  {e}", file=sys.stderr)
        return 1
    corpus = extract(summary, catalog_models(Path(args.catalog)))

    scenarios = {s["scenario"] for s in corpus["sweeps"]}
    points = sum(len(s["points"]) for s in corpus["sweeps"])
    if points == 0:
        # A corpus of nothing is not a corpus (X22). The artifact's schema drifts: an extractor
        # whose slug maps no longer match reads 0 points and must say so, not exit cleanly.
        print(f"error: 0 sweeps, 0 points from {args.summary}; the artifact's checkpoints or "
              "chip slugs no longer match the catalog and CHIPS", file=sys.stderr)
        for reason, n in sorted(corpus["selection"]["skipped"].items()):
            print(f"  skipped {n:4d}  {reason}", file=sys.stderr)
        return 1
    want = pin["expected"]
    if (len(corpus["sweeps"]), points) != (want["sweeps"], want["points"]):
        print(f"error: {len(corpus['sweeps'])} sweeps, {points} points from the pinned artifact; "
              f"the pin records {want['sweeps']} and {want['points']}", file=sys.stderr)
        return 1
    wantd = want.get("disaggregated")
    if wantd is not None:
        gotd = (len(corpus["disaggregated"]), sum(len(d["points"]) for d in corpus["disaggregated"]))
        if gotd != (wantd["topologies"], wantd["points"]):
            print(f"error: {gotd[0]} disaggregated topologies, {gotd[1]} points from the pinned "
                  f"artifact; the pin records {wantd['topologies']} and {wantd['points']}", file=sys.stderr)
            return 1
    with open(args.out, "w") as fh:
        json.dump(corpus, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"{len(corpus['sweeps'])} sweeps, {points} points, "
          f"{len(scenarios)} distinct deployments -> {args.out}")
    for reason, n in sorted(corpus["selection"]["skipped"].items()):
        print(f"  skipped {n:4d}  {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
