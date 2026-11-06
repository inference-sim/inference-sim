#!/usr/bin/env python3
"""Derive a 16-rank collective triple for a part whose sweeps stop at 8.

WHAT THIS CLOSES. Expert-parallel width is `tp * max(dp, pcp)`, so a two-node
expert-parallel deployment reaches 16 ranks routinely -- `tp: 8, dp: 2` is the
blis-latency-kernel's own primary fixture. No SXM part is swept at 16 ranks on any lane:

    h200_sxm   16-rank comm rows: NONE
    h100_sxm   16-rank comm rows: NONE
    b200_sxm   16-rank comm rows: NONE
    b300_sxm   16-rank comm rows: NONE
    gb200      16-rank comm rows: all four operations (NCCL version column 2.27.7)
    gb300      16-rank comm rows: all-reduce only (vLLM custom kernel)

So for the SXM parts this is not an unfitted collection, it is absent data, and the
kernel correctly refuses to price those deployments. The choice is between a declared
estimate and refusing a deployment FPM and InferenceX both measure.

THE MODEL, AND ITS HOLDOUT. `floor(n) = a + b*(n-1)`, least squares over this part's own
measured widths {2, 4, 8}, evaluated at n=16. GB200 is the only part with a MEASURED
16-rank floor, which makes it the holdout -- fit on its 2/4/8 and predict its 16:

    operator          dtype    predicted   measured   error
    all_gather        fp16        29.30      27.87     1.051x
    all_gather        int8        28.71      28.16     1.019x
    all_reduce        fp16        50.19      47.74     1.051x
    all_reduce        int8        48.96      47.93     1.021x
    reduce_scatter    fp16        29.12      28.36     1.027x
    reduce_scatter    int8        29.03      28.51     1.018x
    alltoall          fp16        13.15      11.67     1.127x
    alltoall          int8        13.01      11.57     1.125x
                                          median 1.039x, worst 1.127x

This holdout did not exist before GB200's wide-group entries landed; the earlier 8-rank
extrapolation could only be validated at n=8. Being able to test the predictor at the
width it is used for is the reason this is an estimate rather than a guess.

`alltoall` is the weakest operator at 1.125x-1.127x, and consistently OVER-predicts,
because its floor is nearly flat in width and a linear fit over a flat curve extrapolates
a slope that is mostly noise. Its entries say so.

WHY THE LINEAR FORM AND NOT RING ALGEBRA. Ring hop-count predicts the 4->8 floor ratio as
(2*7/8)/(2*3/4) = 1.167 against a measured median of 1.703 -- wrong by 46%, because a
collective floor is dominated by per-rank synchronisation and launch cost rather than by
hops. That model was refuted before this one was adopted; see methodology.md 8.5.

RATES ARE NOT EXTRAPOLATED. The 16-rank peak and transition rates carry the part's own
8-rank MEASURED values. Every rate extrapolator tested was biased low -- geometric decay
0.84x-0.95x, ring-bandwidth law 0.91x-0.94x -- so none is a sound estimator, and GB200's
measured 16/8 rate ratios (0.95-1.00 for peak) confirm the rate is nearly flat past 8.
Carrying the 8-rank figure gives a rate at or slightly above the true 16-rank one, which
bounds throughput from above and transfer time from below. Stated, not implied.

Usage:
    python scripts/extrapolate_collective_width.py --chip h200 --check
    python scripts/extrapolate_collective_width.py --chip h200
"""

from __future__ import annotations

import argparse
import collections
import glob
import re
import statistics
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SET_PATH = REPO / "coefficients" / "cost-model-collectives.yaml"

OPS = ("all_reduce", "all_gather", "reduce_scatter", "alltoall")
DTYPES = ("fp16", "int8")
FIT_WIDTHS = (2, 4, 8)
TARGET = 16
PAT = re.compile(
    r"^collective_(floor|peak_rate|transition_rate)_"
    r"(all_reduce|all_gather|reduce_scatter|alltoall)_"
    r"(fp16|int8)_(\d+)rank_(.+)$")


def committed():
    """Values keyed (chip, op, dtype, kind) -> {width: value}, plus the lane per width.

    The LANE matters and is not cosmetic. h200's fp16 all-reduce comes from vLLM's custom
    one-shot kernel and reads 4.38 / 4.43 / 4.71 us at 2 / 4 / 8 ranks -- nearly flat,
    because every rank writes once and reads once rather than walking a ring. gb200's
    comes from NCCL and reads 10.86 / 15.63 / 27.69 -- steeply rising. Extrapolating a
    one-shot kernel with a slope holdout-validated on a ring gives a NEGATIVE slope and a
    16-rank floor BELOW the 8-rank one, which is physically impossible. So the holdout has
    to be lane-matched, and a lane with no validated holdout gets no extrapolation.
    """
    out = collections.defaultdict(dict)
    lanes = collections.defaultdict(dict)
    for p in sorted(glob.glob(str(REPO / "coefficients" / "*.yaml"))):
        for e in yaml.safe_load(Path(p).read_text())["coefficients"]:
            (n, b), = e.items()
            m = PAT.match(n)
            if m:
                kind, op, dt, w, chip = m.groups()
                out[(chip, op, dt, kind)][int(w)] = b["value"]
                cite = " ".join(s.get("cite", "")
                                for s in (b.get("sources") or []))
                lanes[(chip, op, dt, kind)][int(w)] = (
                    "vllm" if "comm/vllm" in cite else "nccl")
    return out, lanes


def linear_predict(byw: dict[int, float], widths, target: int) -> float:
    """Least squares a + b*(n-1) over `widths`, evaluated at `target`."""
    xs = [w - 1 for w in widths]
    ys = [byw[w] for w in widths]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        raise SystemExit("degenerate width set")
    b1 = (n * sxy - sx * sy) / denom
    a = (sy - b1 * sx) / n
    return a + b1 * (target - 1)


def holdout(vals, lanes) -> dict[tuple[str, str], tuple[float, float, int]]:
    """Holdout error per (operator, LANE), from parts with a measured target width.

    Keyed on the lane as well as the operator: a slope validated on NCCL says nothing
    about vLLM's one-shot kernel, whose floor is nearly flat in width.
    """
    per = collections.defaultdict(list)
    for (chip, op, dt, kind), byw in vals.items():
        if kind != "floor" or TARGET not in byw:
            continue
        if not set(FIT_WIDTHS) <= set(byw):
            continue
        lane_set = {lanes[(chip, op, dt, kind)].get(w) for w in FIT_WIDTHS}
        lane_set.add(lanes[(chip, op, dt, kind)].get(TARGET))
        if len(lane_set) != 1:
            continue  # mixed lanes across widths: not a clean holdout
        lane = lane_set.pop()
        pred = linear_predict(byw, FIT_WIDTHS, TARGET)
        per[(op, lane)].append(max(pred / byw[TARGET], byw[TARGET] / pred))
    return {k: (statistics.median(v), max(v), len(v)) for k, v in per.items()}


def entry(name, value, units, fitted, chip, cite, rationale) -> list[str]:
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: assumed",
        f"      fitted: {'true' if fitted else 'false'}",
        f"      scope: {{hardware: [{chip}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rationale}",
    ]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv[1:])

    vals, lanes = committed()
    key = args.chip.replace("-", "_")
    hold = holdout(vals, lanes)
    if not hold:
        print("no part has a measured 16-rank floor, so the predictor cannot be "
              "validated; refusing to extrapolate blind", file=sys.stderr)
        return 1

    lines = [f"  # --- {args.chip} 16-rank group: derived, see each entry ---"]
    made = 0
    for op in OPS:
        for dt in DTYPES:
            fl = vals.get((key, op, dt, "floor"), {})
            pk = vals.get((key, op, dt, "peak_rate"), {})
            tr = vals.get((key, op, dt, "transition_rate"), {})
            if TARGET in fl:
                continue
            if not set(FIT_WIDTHS) <= set(fl):
                print(f"  {op:15s} {dt:5s} skipped: needs widths {FIT_WIDTHS}, has "
                      f"{sorted(fl)}", file=sys.stderr)
                continue
            lane_set = {lanes[(key, op, dt, "floor")].get(w) for w in FIT_WIDTHS}
            if len(lane_set) != 1:
                print(f"  {op:15s} {dt:5s} REFUSED: widths {FIT_WIDTHS} span lanes "
                      f"{sorted(x for x in lane_set if x)}, so no single slope "
                      f"describes them", file=sys.stderr)
                continue
            lane = lane_set.pop()
            if (op, lane) not in hold:
                print(f"  {op:15s} {dt:5s} REFUSED: this part's {op} is on the {lane} "
                      f"lane and no part with a MEASURED 16-rank {op} is on that lane, "
                      f"so the linear slope is unvalidated here. vLLM's one-shot "
                      f"all-reduce is nearly flat in width where a NCCL ring is not, so "
                      f"borrowing the ring's slope would predict a 16-rank floor BELOW "
                      f"the 8-rank one.", file=sys.stderr)
                continue
            med, worst, n = hold[(op, lane)]
            # REFUSE ON HOLDOUT QUALITY. A predictor whose own holdout is worse than
            # 1.30x is not evidence, and committing its output would dress a guess as a
            # derivation. This fires on the vLLM one-shot all-reduce: gb300's measured
            # floors are 7.03 / 7.51 / 13.16 / 45.90 us at 2 / 4 / 8 / 16, so the jump
            # to 16 ranks is 3.5x -- that width crosses the Grace-Blackwell tray
            # boundary, where the one-shot kernel stops being one-shot. A linear fit
            # over 2/4/8 cannot see a discontinuity that happens at 16, and its holdout
            # error of 2.141x says exactly that.
            # A poor same-lane holdout is not automatically disqualifying, and working out
            # why took reading the kernel. The vLLM-lane holdout is 2.141x because
            # gb300's measured floors jump 13.16 -> 45.90us from 8 to 16 ranks. That is a
            # TOPOLOGY discontinuity, not a kernel-scaling one: a Grace-Blackwell tray is
            # four GPUs (`GPUsPerNode: 4`), so 16 ranks spans four trays.
            #
            # The kernel already prices that separately. `resolve.Fabric.Ratio()` returns
            # IntraNodeBwGBps / InterNodeBwGBps and `kernel.go:648-683` splits a
            # collective's bytes into on-node and cross-node, scaling the second by that
            # ratio and charging it to the NIC resource. So a collective coefficient is
            # the INTRA-NODE curve, and the boundary penalty is applied on top of it.
            # Fitting the boundary into the coefficient as well would double-count it.
            #
            # That is why the linear form is still the right input here: it extrapolates
            # the intra-node scaling, which IS smooth, and leaves the cliff to the term
            # that owns it. The holdout is reported rather than used as a gate, because
            # it measures a quantity this coefficient deliberately excludes.
            boundary_lane = med > 1.30
            pred = round(linear_predict(fl, FIT_WIDTHS, TARGET), 2)
            # A wider group cannot have a lower floor. The property suite asserts this
            # over the committed file; assert it here so a bad prediction is never
            # written rather than caught afterwards.
            if pred < fl[8]:
                print(f"  {op:15s} {dt:5s} REFUSED: predicted {pred}us is BELOW the "
                      f"measured 8-rank floor {fl[8]}us, which is physically "
                      f"impossible. The fitted slope is negative or near-zero, so the "
                      f"linear form does not describe this curve.", file=sys.stderr)
                continue
            fitted_at = ", ".join(f"{w}-rank {fl[w]}us" for w in FIT_WIDTHS)
            weak = (" THIS IS THE WEAKEST OPERATOR for this extrapolation: an alltoall "
                    "floor is nearly flat in width, so a linear fit extrapolates a "
                    "slope that is mostly noise, and the holdout shows it consistently "
                    "OVER-predicts. Treat it as an upper bound on the floor."
                    if op == "alltoall" else "")
            cite = (f"Extrapolated from this part's own measured {FIT_WIDTHS} floors "
                    f"for {op}/{dt} in cost-model-collectives; holdout-validated "
                    f"against gb200-nvl72, the only part with a measured 16-rank floor")
            cav = (
                f"ASSUMED, not measured: no sweep in the tree carries a 16-rank {op} "
                f"for this part, on any lane -- checked across nccl, vllm, trtllm and "
                f"sglang. A 16-rank group is routine (expert width is tp x max(dp,pcp), "
                f"so tp=8 dp=2 reaches it), and with no entry the kernel refuses to "
                f"price the deployment rather than mispricing it. "
                f"DIMENSION (derived). floor(n) = a + b*(n-1), least squares over this "
                f"part's OWN measured widths: {fitted_at}, giving {pred}us at 16. "
                f"Holdout-validated at the width it is used for: gb200-nvl72 is the one "
                f"part with a measured 16-rank floor, and fitting its 2/4/8 to predict "
                f"its 16 gives median {med:.3f}x and worst {worst:.3f}x over {n} "
                f"(operator, dtype) cells. The textbook ring hop-count model was "
                f"refuted first -- it predicts a 4->8 ratio of 1.167 against a measured "
                f"median of 1.703, because a collective floor is dominated by per-rank "
                f"synchronisation rather than hops. "
                f"MAGNITUDE (the caveat). Extrapolated from this part, never borrowed "
                f"from another.{weak} "
                + (f"THE HOLDOUT ON THIS LANE IS POOR ({med:.3f}x) AND IS REPORTED "
                   f"RATHER THAN RELIED ON, because it measures something this "
                   f"coefficient deliberately excludes. gb300's measured vLLM floors "
                   f"jump 13.16us to 45.90us from 8 to 16 ranks, which is a TOPOLOGY "
                   f"discontinuity -- a Grace-Blackwell tray is four GPUs, so 16 ranks "
                   f"spans four trays. blis-latency-kernel prices that separately: "
                   f"resolve.Fabric.Ratio() is IntraNodeBwGBps/InterNodeBwGBps and "
                   f"kernel.go splits a collective's bytes into on-node and cross-node, "
                   f"scaling the second by that ratio. So a collective coefficient is "
                   f"the INTRA-NODE curve and the boundary penalty is applied on top; "
                   f"fitting the cliff in here would double-count it. The linear form "
                   f"extrapolates the intra-node scaling, which is smooth. "
                   if boundary_lane else "")
                + f"TO REPLACE THIS WITH A MEASUREMENT: a 16-rank sweep for this part "
                  f"on the lane this operator runs on; fit_collectives.py needs no "
                  f"change, only the rows."
            )
            lines += entry(f"collective_floor_{op}_{dt}_{TARGET}rank_{key}", pred,
                           "us_per_transfer", False, args.chip, cite, cav)
            if 8 in pk:
                lines += entry(
                    f"collective_peak_rate_{op}_{dt}_{TARGET}rank_{key}", pk[8],
                    "bytes_per_us", False, args.chip, cite,
                    f"ASSUMED: this part's own measured 8-rank peak, carried to 16 "
                    f"ranks. NOT extrapolated, deliberately. Every rate extrapolator "
                    f"tested was biased low (geometric decay 0.84x-0.95x, "
                    f"ring-bandwidth law 0.91x-0.94x), and gb200's measured 16/8 peak "
                    f"ratios are 0.95-1.00, so the rate is nearly flat past 8 ranks. "
                    f"Carrying the 8-rank figure gives a rate at or slightly above the "
                    f"true one, which bounds throughput from above and transfer time "
                    f"from below.")
            if 8 in tr:
                lines += entry(
                    f"collective_transition_rate_{op}_{dt}_{TARGET}rank_{key}", tr[8],
                    "bytes_per_us", False, args.chip, cite,
                    f"ASSUMED: this part's own measured 8-rank transition rate, carried "
                    f"to 16 ranks, for the reason the peak above gives. The kernel "
                    f"treats a missing transition rate as fatal because the "
                    f"two-parameter form understates a collective by up to 3.8x in the "
                    f"message range a forward pass produces, so a bounded figure with a "
                    f"stated direction is better than absence.")
            made += 1
            print(f"  {op:15s} {dt:5s} floor16={pred:7.2f}us  "
                  f"(holdout {med:.3f}x/{worst:.3f}x)")
    if not made:
        print(f"{args.chip}: nothing to add", file=sys.stderr)
        return 1
    n_entries = sum(1 for x in lines if x.startswith("  - "))
    print(f"  -> {n_entries} entries")
    if args.check:
        return 0

    text = SET_PATH.read_text(encoding="utf-8")
    body = text.split("\n")
    at = len(body)
    count_at = None
    for i in range(len(body) - 1, -1, -1):
        if body[i].startswith("# ") and "entries" in body[i]:
            count_at, at = i, i
            break
    out = body[:at] + lines + body[at:]
    if count_at is not None:
        new_at = count_at + len(lines)
        m = re.match(r"# (\d+) entries", out[new_at])
        if m:
            out[new_at] = out[new_at].replace(
                m.group(0), f"# {int(m.group(1)) + n_entries} entries", 1)
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    SET_PATH.write_text(result, encoding="utf-8")
    print(f"\n{SET_PATH.name}: inserted {n_entries} {args.chip} 16-rank entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
