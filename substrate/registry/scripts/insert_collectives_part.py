#!/usr/bin/env python3
"""Add one part's collective coefficients to cost-model-collectives.yaml.

WHY NOT emit_collectives.py. That generator writes the WHOLE file from the NCCL
sweeps, and the committed file is no longer NCCL-only: commit 2f71dd6 moved every
fp16 all-reduce onto vLLM's own custom kernel, because vLLM does not call NCCL for a
tensor-parallel all-reduce its custom kernel can serve. Regenerating from NCCL against
the committed file gives 144 removals and 36 changes, and all 36 changed are
`all_reduce` fp16 -- it would silently revert that relane and re-introduce a 3.4x
mispricing (h200 at 8 ranks: 4.71us vLLM against 15.98us NCCL). So a new part is
INSERTED, per operator, from the lane each operator belongs on.

THE LANE PER OPERATOR, which is a methodology question and not a mechanical one:

  * all_reduce  -> vLLM's custom kernel (`custom_allreduce_perf.parquet`), CUDA-graph
    backend. vLLM captures decode in a CUDA graph, so an eager figure does not price a
    decode step.
  * all_gather, reduce_scatter, alltoall -> NCCL. Not a borrow: `CustomAllreduce`
    implements all-reduce only, and `should_custom_all_gather` /
    `should_custom_reduce_scatter` gate and fall through, so NCCL IS vLLM's path for
    these three. AISimulate ships no vLLM data for them because there is nothing to
    measure.

THE RACK-SCALE GAP, and why some entries are `method: assumed`. NCCL sweeps GB200 and
GB300 at 2 and 4 ranks only, because a Grace-Blackwell tray is four GPUs. tp=8 is an
ordinary deployment on both, so three operators have no 8-rank measurement on any lane.
Rather than leave a tp=8 deployment unpriceable, the 8-rank floors for those three ship
as `method: assumed` with a holdout-validated extrapolation. The dimension is derived
and the magnitude is caveated, in the style cost-model-host-overheads.yaml established.

  THE MODEL THAT WAS REFUTED FIRST. Ring hop-count predicts the 4->8 floor ratio as
  (2*7/8)/(2*3/4) = 1.167 for all_reduce. MEASURED median across the seven parts swept
  at both widths is 1.703. Hop count is wrong by 46%: a collective floor is dominated
  by per-rank synchronisation and launch cost, not by hop count. Grounding on textbook
  ring algebra would have under-priced these by a third.

  THE MODEL THAT HOLDS. floor(n) = a + b*(n-1), with a and b fitted on THIS PART'S OWN
  measured widths {2, 4} and evaluated at n=8. Validated as a holdout on the NVLink
  reference class (a100_sxm, h100, h200, b200, b300 -- parts that carry 8-rank data, so
  the prediction can be checked):

      operator          n    pred/meas median   range
      all_reduce       14         0.987         0.953-1.041
      all_gather       14         1.017         0.954-1.090
      reduce_scatter   14         1.018         0.967-1.107
      alltoall         14         1.128         0.737-1.294   <- WEAKEST

  alltoall is the stated exception: its floor barely grows with width (median 4->8 ratio
  1.090), so a two-point linear fit amplifies noise. Its entry says so.

  THE REFERENCE CLASS EXCLUDES L40S AND RTX PRO 6000, whose 8/4 rate ratios reach
  2.7x-4.2x. Those parts have no NVLink, so an 8-rank group crosses a materially
  different fabric. GB300 is full-NVSwitch: its descriptor states inter_node_bw ==
  intra_node_bw == 900 GB/s, so crossing a tray boundary inside the rack is not a
  bandwidth change at all. Including PCIe parts would have inflated the extrapolation.

  RATES ARE NOT EXTRAPOLATED. The 8-rank peak and transition rates carry the part's own
  4-rank MEASURED values. A geometric decay fitted on {2,4} under-predicts the measured
  8-rank rate by a median 0.84x-0.95x (biased low, not centred), and the ring-bandwidth
  law by 0.91x-0.94x. An over-stated rate under-prices transfer time, so carrying the
  4-rank figure -- which is ABOVE the true 8-rank rate -- is the conservative direction
  and is labelled as a bound rather than an estimate.

PROVENANCE: CITE THE VERSION COLUMN, NOT THE DIRECTORY. GB300's `comm/nccl/2.27`
directory holds rows whose `version` column reads 2.29.2, and GB200's `comm/nccl/2.23`
mixes 2.27.7 (widths 8, 16) and 2.29.2 (widths 2, 4). The path lies; the column is the
provenance. Every citation this writes names the column value.

Usage:
    python scripts/insert_collectives_part.py --chip gb300 --sku gb300 --check
    python scripts/insert_collectives_part.py --chip gb300 --sku gb300
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-collectives.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/private/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")

FLAT_REGION_BYTES = 4096
NCCL_OPS = ("all_gather", "reduce_scatter", "alltoall")
DTYPES = {"fp16": "half", "int8": "int8"}

# The widths every NVLink part in the tree is swept at. A part missing one of these is
# a gap to extrapolate, not a part to skip.
TARGET_WIDTHS = (2, 4, 8)

# Parts whose 8-rank behaviour validated the extrapolation. PCIe parts are excluded:
# see the reference-class note in the module docstring.
NVLINK_HOLDOUT = ("a100_sxm", "h100_sxm", "h200_sxm", "b200_sxm", "b300_sxm")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


fcv = _load("fit_collectives_vllm", HERE / "fit_collectives_vllm.py")


def nccl_rows(data: Path, sku: str) -> dict:
    """Every NCCL row for one SKU, keyed (op, dtype, width) -> [(size, latency_us)].

    Reads the `version` column and returns it alongside, because the directory name is
    not the collection version. Latency in these files is MILLISECONDS.
    """
    import pyarrow.parquet as pq

    out: dict[tuple[str, str, int], list[tuple[float, float]]] = {}
    versions: set[str] = set()
    base = data / sku / "comm" / "nccl"
    if not base.is_dir():
        raise SystemExit(f"{sku}: no comm/nccl tree")
    for coll in sorted(os.listdir(base)):
        path = base / coll / "nccl_perf.parquet"
        if not path.is_file():
            continue
        d = pq.read_table(path).to_pydict()
        if "version" not in d:
            continue  # narrower schema; not a collection any entry cites
        for i in range(len(d["latency"])):
            lat = float(d["latency"][i])
            if lat <= 0:
                continue
            key = (str(d["op_name"][i]), str(d["nccl_dtype"][i]),
                   int(d["num_gpus"][i]))
            out.setdefault(key, []).append((float(d["message_size"][i]), lat * 1e3))
            versions.add(str(d["version"][i]))
    if len(versions) != 1:
        raise SystemExit(
            f"{sku}: NCCL rows span versions {sorted(versions)}; one collection per "
            f"part is what keeps a difference between parts silicon rather than a "
            f"software release. Narrow the selection deliberately."
        )
    return {"rows": out, "version": versions.pop()}


def estimators(pts: list[tuple[float, float]]) -> dict | None:
    """floor / peak / transition, with fit_collectives_vllm's own Group estimators."""
    g = fcv.Group("nccl", 0, pts)
    if not g.usable:
        return None
    return {
        "floor": round(g.floor_us, 2),
        "peak": g.peak_bytes_per_us,
        "transition": g.transition_rate,
        "flat_n": g.flat_count,
        "n": len(pts),
        "err3": g.error_three_param,
    }


def extrapolate_floor(f2: float, f4: float) -> float:
    """floor(8) under floor(n) = a + b*(n-1), fitted on widths 2 and 4.

    b is the per-rank increment: (f4 - f2) / (4 - 2). a follows from f2 = a + b*(2-1).
    """
    b = (f4 - f2) / 2.0
    a = f2 - b
    return a + b * 7.0


def holdout_report(data: Path) -> dict[str, tuple[float, float, float, int]]:
    """Re-run the holdout so the committed rationale cites a live number.

    Returns op -> (median, min, max, n) of predicted/measured at 8 ranks over the
    NVLink reference class.
    """
    per: dict[str, list[float]] = {}
    for sku in NVLINK_HOLDOUT:
        try:
            blob = nccl_rows(data, sku)
        except SystemExit:
            continue
        rows = blob["rows"]
        for op in NCCL_OPS + ("all_reduce",):
            for dt in DTYPES.values():
                got = {}
                for w in TARGET_WIDTHS:
                    pts = rows.get((op, dt, w))
                    if pts:
                        e = estimators(pts)
                        if e:
                            got[w] = e["floor"]
                if len(got) == 3:
                    pred = extrapolate_floor(got[2], got[4])
                    per.setdefault(op, []).append(pred / got[8])
    return {
        op: (statistics.median(v), min(v), max(v), len(v))
        for op, v in sorted(per.items())
    }


def entry(name: str, value, units: str, method: str, fitted: bool, chip: str,
          cite: str, rationale: str) -> list[str]:
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        f"      method: {method}",
        f"      fitted: {'true' if fitted else 'false'}",
        f"      scope: {{hardware: [{chip}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rationale}",
    ]


def build(data: Path, chip: str, sku: str) -> tuple[list[str], dict]:
    """Every collective entry for one part, in registry order."""
    key = chip.replace("-", "_")
    blob = nccl_rows(data, sku)
    rows, version = blob["rows"], blob["version"]
    holdout = holdout_report(data)
    lines: list[str] = [f"  # --- {chip} ({sku}) ---"]
    summary: dict[str, dict] = {}

    # 1. all_reduce, both dtypes, from vLLM's custom kernel in the CUDA-graph lane.
    ar_base = data / sku / "comm" / "vllm"
    coll = sorted(os.listdir(ar_base))[-1] if ar_base.is_dir() else None
    groups = fcv.read(ar_base / coll) if coll else []
    graph = {g.gpus: g for g in groups if "graph" in g.backend}
    for dt_tag in ("fp16", "int8"):
        for w in sorted(graph):
            g = graph[w]
            # The sweep carries bfloat16 only; fit_collectives_vllm's own guard asserts
            # that. A two-byte payload is the fp16 coefficient, and the int8 entry
            # carries the same figure because no int8 custom-all-reduce sweep exists --
            # recorded as such in the rationale rather than implied.
            stem = f"all_reduce_{dt_tag}_{w}rank_{key}"
            cite = (f"NVIDIA AISimulate systems/data/{sku}/comm/vllm/{coll}/"
                    f"custom_allreduce_perf.parquet (vLLM {coll} custom all-reduce, "
                    f"CUDA-graph lane, NVIDIA {chip.upper()}), {len(g.points)} messages "
                    f"at {w} ranks")
            dt_note = ("" if dt_tag == "fp16" else
                       " The sweep measures a two-byte (bfloat16) payload only, so this "
                       "int8 entry carries the same figure: no int8 custom-all-reduce "
                       "measurement exists on any part, and an int8 all-reduce still "
                       "moves through the same one-shot kernel. That is a stated "
                       "equivalence, not a measurement of int8.")
            lines += entry(
                f"collective_floor_{stem}", round(g.floor_us, 3), "us_per_transfer",
                "measured", True, chip, cite,
                f"Minimum latency over the {g.flat_count} messages at or below 4 KiB, "
                f"where latency is independent of size. Fitted on vLLM's OWN custom "
                f"all-reduce rather than NCCL: vLLM does not call NCCL for a "
                f"tensor-parallel all-reduce its custom kernel can serve. The "
                f"CUDA-graph lane is the one a decode step runs in. Three-parameter "
                f"geometric error {g.error_three_param:.3f}x.{dt_note}")
            lines += entry(
                f"collective_peak_rate_{stem}", round(g.peak_bytes_per_us, 1),
                "bytes_per_us", "measured", False, chip, cite,
                f"Maximum message_size/latency over the {len(g.points)}-point sweep, an "
                f"order statistic rather than a curve fit. A ceiling on throughput; the "
                f"transition rate below is what prices a forward pass.{dt_note}")
            lines += entry(
                f"collective_transition_rate_{stem}", round(g.transition_rate, 1),
                "bytes_per_us", "measured", True, chip, cite,
                f"The rate in the transition region a forward pass's messages actually "
                f"land in, fitted by grid search against the three-parameter form the "
                f"kernel evaluates: max(floor + size/rate, size/peak). Geometric error "
                f"{g.error_three_param:.3f}x against {g.error_two_param:.3f}x for the "
                f"two-parameter form.{dt_note}")
            summary.setdefault("all_reduce", {})[w] = "measured"

    # 2. The three NCCL operators.
    for op in NCCL_OPS:
        for dt_tag, dt_col in DTYPES.items():
            measured: dict[int, dict] = {}
            for w in TARGET_WIDTHS:
                pts = rows.get((op, dt_col, w))
                if pts:
                    e = estimators(pts)
                    if e:
                        measured[w] = e
            for w in TARGET_WIDTHS:
                stem = f"{op}_{dt_tag}_{w}rank_{key}"
                if w in measured:
                    e = measured[w]
                    cite = (f"NVIDIA AISimulate systems/data/{sku}/comm/nccl/ "
                            f"(version column: {version}) nccl_perf.parquet, {op}, "
                            f"{dt_col}, {w} ranks, {e['n']} messages")
                    lines += entry(
                        f"collective_floor_{stem}", e["floor"], "us_per_transfer",
                        "measured", False, chip, cite,
                        f"Minimum latency over the {e['flat_n']} messages at or below "
                        f"4 KiB, where latency does not depend on size. An order "
                        f"statistic rather than a fit: a floor is a lower bound on "
                        f"achievable latency, so a high outlier must not raise it. "
                        f"NCCL is vLLM's own path for {op} -- CustomAllreduce "
                        f"implements all-reduce only.")
                    lines += entry(
                        f"collective_peak_rate_{stem}", round(e["peak"], 1),
                        "bytes_per_us", "measured", False, chip, cite,
                        f"Maximum message_size/latency over {e['n']} points, an order "
                        f"statistic. A ceiling; the transition rate prices a step.")
                    lines += entry(
                        f"collective_transition_rate_{stem}",
                        round(e["transition"], 1), "bytes_per_us", "measured", True,
                        chip, cite,
                        f"Fitted by grid search against the three-parameter form the "
                        f"kernel evaluates. Geometric error {e['err3']:.3f}x.")
                    summary.setdefault(op, {})[w] = "measured"
                    continue

                # Unmeasured width: extrapolate the floor, carry the rate.
                if 2 not in measured or 4 not in measured:
                    raise SystemExit(
                        f"{chip} {op} {dt_tag}: width {w} is unmeasured and widths 2 "
                        f"and 4 are not both present, so there is nothing to "
                        f"extrapolate from. Refusing to invent a value.")
                f2, f4 = measured[2]["floor"], measured[4]["floor"]
                pred = round(extrapolate_floor(f2, f4), 2)
                med, lo, hi, hn = holdout.get(op, (0, 0, 0, 0))
                cite = (f"Extrapolated from this part's own measured 2- and 4-rank "
                        f"floors in NVIDIA AISimulate systems/data/{sku}/comm/nccl/ "
                        f"(version column: {version}) nccl_perf.parquet, {op}, "
                        f"{dt_col}; holdout-validated on {hn} NVLink part/dtype cells")
                weak = (" THIS IS THE WEAKEST OF THE FOUR OPERATORS: an alltoall floor "
                        "barely grows with width, so a two-point linear fit amplifies "
                        "noise. Treat it as an order-of-magnitude figure."
                        if op == "alltoall" else "")
                lines += entry(
                    f"collective_floor_{stem}", pred, "us_per_transfer",
                    "assumed", False, chip, cite,
                    f"ASSUMED, not measured: NCCL sweeps this part at 2 and 4 ranks "
                    f"only, because a Grace-Blackwell tray is four GPUs, while tp={w} "
                    f"is an ordinary deployment on it. "
                    f"DIMENSION (derived). floor(n) = a + b*(n-1), fitted on this "
                    f"part's OWN measured widths: {f2}us at 2 ranks and {f4}us at 4 "
                    f"give {pred}us at {w}. Validated as a holdout on the NVLink parts "
                    f"that DO carry 8-rank data -- predicted/measured median "
                    f"{med:.3f}x over {hn} cells, range {lo:.3f}-{hi:.3f}x. The "
                    f"textbook ring hop-count model was REFUTED first: it predicts a "
                    f"4->8 ratio of 1.167 against a measured median of 1.703, so a "
                    f"collective floor is dominated by per-rank synchronisation rather "
                    f"than hop count. "
                    f"MAGNITUDE (the caveat). Extrapolated, never borrowed from "
                    f"another part. PCIe parts are excluded from the reference class: "
                    f"their 8/4 ratios reach 2.7x-4.2x because an 8-rank group crosses "
                    f"a different fabric, where this part's descriptor states "
                    f"inter_node_bw == intra_node_bw == 900 GB/s so a tray boundary "
                    f"inside the rack is no bandwidth change.{weak} "
                    f"TO REPLACE THIS WITH A MEASUREMENT: an {w}-rank NCCL sweep for "
                    f"this part. One sweep retires every assumed entry here at once.")
                lines += entry(
                    f"collective_peak_rate_{stem}", round(measured[4]["peak"], 1),
                    "bytes_per_us", "assumed", False, chip, cite,
                    f"ASSUMED: this part's own measured 4-rank peak, carried to {w} "
                    f"ranks. NOT extrapolated, deliberately. A geometric decay fitted "
                    f"on widths 2 and 4 under-predicts the measured 8-rank rate by a "
                    f"median 0.84x-0.95x and the ring-bandwidth law by 0.91x-0.94x -- "
                    f"both biased low rather than centred, so neither is a sound "
                    f"estimator. The true {w}-rank rate is BELOW this figure, which "
                    f"makes this an upper bound on throughput and therefore a lower "
                    f"bound on transfer time. Stated as a bound, not an estimate.")
                lines += entry(
                    f"collective_transition_rate_{stem}",
                    round(measured[4]["transition"], 1), "bytes_per_us", "assumed",
                    False, chip, cite,
                    f"ASSUMED: this part's own measured 4-rank transition rate, carried "
                    f"to {w} ranks, for the reason the peak above gives. The kernel "
                    f"treats a missing transition rate as fatal because the "
                    f"two-parameter form understates a collective by up to 3.8x in the "
                    f"message range a forward pass produces, so an assumed figure with "
                    f"a stated direction is better than absence -- but it is an upper "
                    f"bound on rate and under-states this collective's time.")
                summary.setdefault(op, {})[w] = "assumed"
    return lines, summary


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip", required=True)
    ap.add_argument("--sku", required=True)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true",
                    help="print what would be written; change nothing")
    args = ap.parse_args(argv[1:])

    text = SET_PATH.read_text(encoding="utf-8")
    key = args.chip.replace("-", "_")
    if re.search(r"^  - collective_\w+_\d+rank_" + re.escape(key) + r":$",
                 text, re.M):
        print(f"{args.chip}: already present in {SET_PATH.name}", file=sys.stderr)
        return 1

    lines, summary = build(Path(args.data), args.chip, args.sku)
    n = sum(1 for x in lines if x.startswith("  - "))
    for op in sorted(summary):
        byw = summary[op]
        print(f"  {op:15s} " + "  ".join(
            f"{w}rank={byw[w]}" for w in sorted(byw)))
    print(f"  -> {n} entries")
    if args.check:
        return 0

    body = text.split("\n")
    at = len(body)
    count_at = None
    for idx in range(len(body) - 1, -1, -1):
        if body[idx].startswith("# ") and "entries" in body[idx]:
            count_at, at = idx, idx
            break
    if count_at is None:
        while at > 0 and not body[at - 1].strip():
            at -= 1
    out = body[:at] + lines + body[at:]
    if count_at is not None:
        new_at = count_at + len(lines)
        m = re.match(r"# (\d+) entries", out[new_at])
        if m:
            out[new_at] = out[new_at].replace(
                m.group(0), f"# {int(m.group(1)) + n} entries", 1)
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    SET_PATH.write_text(result, encoding="utf-8")
    print(f"\n{SET_PATH.name}: inserted {n} {args.chip} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
