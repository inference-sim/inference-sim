#!/usr/bin/env python3
"""Derive a coefficient for a part the sweeps do not measure, from parts they do.

WHEN THIS IS THE RIGHT TOOL. Some (part, family) pairs have no data on any lane -- not a
collection this project has not got round to fitting, but rows that do not exist. A100's
attention sweeps carry `window_size: 0` on every row in all three lanes, so a sliding-window
coefficient for A100 cannot be fitted at all. Leaving it absent is not neutral: the
resolver treats an unmatched optional coefficient as absent, the kernel then prices a
windowed layer with the FULL-attention rate, and every windowed layer comes out too cheap.
On the parts where both are measured, the windowed rate is 0.18 to 0.50 of the
full-attention one, so the silent fallback understates a windowed layer by 2x to 5.5x.

So the choice is between a declared estimate and a silent 2x-5.5x error, and this script
makes the first one auditable.

THE PREDICTOR, AND HOW IT WAS CHOSEN. Three hypotheses about what a coefficient transfers
along were scored by LEAVE-ONE-PART-OUT holdout over every part that IS fitted -- hold a
part out, predict it from the others, compare against its measured value:

    family                        global    same-generation    scaled-by-bandwidth
    recurrent_decode_rate_kda     1.936x         1.009x                --
    recurrent_decode_rate_gdn     1.494x         1.003x                --
    attention_decode_rate         6.271x         1.075x              1.556x
    attention_decode_rate_swa    16.166x         1.181x              2.611x
    attention_decode_floor        1.727x         1.158x                --
    attention_prefill_floor       1.420x         1.065x                --
    attention_prefill_work_scale  1.250x         1.188x                --
                                                (worst case shown)

**Same-generation transfer wins on every family, by a wide margin.** The global median is
unusable (up to 16x on windowed rates) and scaling by datasheet bandwidth is worse than
the generation median everywhere it applies -- which is itself a finding: these
coefficients are set by the kernel and the memory subsystem's architecture, not by the
headline bandwidth number. H100 and H200 share the GH100 die and their fitted recurrent
rates agree to 1.009x; B200/B300/GB200/GB300 share Blackwell and agree to 1.011x.

So a prediction is the median over the OTHER parts of the same architecture generation,
and a part whose generation has no fitted sibling gets NO prediction -- the script refuses
rather than falling back to the global median, because that is the predictor the holdout
rejected.

WHAT IT WRITES. `method: assumed`, `fitted: false`, with a rationale that states the
predictor, the holdout error of that predictor on that family, the parts it was derived
from, and the measurement that would replace it. The house style is
cost-model-host-overheads.yaml: dimension derived, magnitude caveated.

Usage:
    python scripts/extrapolate_by_generation.py --family attention_decode_rate_swa \
        --chip a100-sxm --check
    python scripts/extrapolate_by_generation.py --family attention_decode_rate_swa \
        --chip a100-sxm --scope a100-80,a100-sxm
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import statistics
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CATALOG = Path(os.environ.get(
    "BLIS_CATALOG", str(Path(__file__).resolve().parents[2] / "catalog")))

# Architecture generation, which is what the holdout says these coefficients transfer
# along. Not a marketing grouping: each set below shares a die family and a memory
# subsystem, which is why their fitted values agree to ~1.01x.
GENERATION = {
    "h100": "Hopper", "h200": "Hopper",
    "b200": "Blackwell", "b300": "Blackwell",
    "gb200-nvl72": "Blackwell", "gb300": "Blackwell",
    "a100-sxm": "Ampere", "a100-80": "Ampere",
    "l40s": "Ada",
}

# The attention kernel vLLM actually dispatches on each part, read from the sweeps'
# `kernel_source` column rather than inferred. This is a SECOND transfer axis, and for the
# windowed family it is the better one -- the windowed/full rate ratio clusters by kernel,
# not by die:
#
#   FA3   (h100, h200)                         ratio 0.333-0.361, median 0.347
#   FIDecode (b200, b300, gb200, gb300)        ratio 0.500-0.595, median 0.581
#   FA2   (l40s)                               ratio 0.310
#
# Blackwell dispatches vllm_flashinfer_flashinfertrtllmapidecode where Hopper dispatches
# vllm_flash_attn_fa3 and L40S falls back to vllm_flash_attn_fa2 -- three different
# implementations, which is why the ratio has three levels. A100 runs `vllm_flash_attn`,
# the FA2-class path, because Ampere lacks the hardware FA3 requires. So L40S is A100's
# kernel sibling even though they are different generations, and that is the defensible
# predictor where no same-generation sibling exists.
KERNEL_FAMILY = {
    "h100": "fa3", "h200": "fa3",
    "b200": "fi_decode", "b300": "fi_decode",
    "gb200-nvl72": "fi_decode", "gb300": "fi_decode",
    "l40s": "fa2", "a100-sxm": "fa2", "a100-80": "fa2",
}

# Which set each family lives in, and its units.
FAMILY = {
    "attention_decode_rate_swa": ("cost-model-attention.yaml", "bytes_per_us"),
    "attention_decode_floor_swa": ("cost-model-attention.yaml", "us_per_transfer"),
    "attention_decode_rate": ("cost-model-attention.yaml", "bytes_per_us"),
    "attention_decode_floor": ("cost-model-attention.yaml", "us_per_transfer"),
    "attention_prefill_floor": ("cost-model-attention.yaml", "us_per_transfer"),
    "attention_prefill_work_scale": ("cost-model-attention.yaml", "dimensionless"),
    "recurrent_decode_rate_kda": ("cost-model-recurrent.yaml", "tokens"),
    "recurrent_decode_floor_kda": ("cost-model-recurrent.yaml", "us_per_transfer"),
    "recurrent_decode_rate_gdn": ("cost-model-recurrent.yaml", "tokens"),
    "recurrent_decode_floor_gdn": ("cost-model-recurrent.yaml", "us_per_transfer"),
    "recurrent_decode_rate_mamba2": ("cost-model-recurrent.yaml", "tokens"),
    "recurrent_decode_floor_mamba2": ("cost-model-recurrent.yaml", "us_per_transfer"),
}


def committed() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for p in sorted(glob.glob(str(REPO / "coefficients" / "*.yaml"))):
        for e in yaml.safe_load(Path(p).read_text())["coefficients"]:
            (n, b), = e.items()
            for c in (b.get("scope") or {}).get("hardware", []):
                out[n][c] = b["value"]
    return out


def holdout_error(known: dict[str, float]) -> tuple[float, float, int]:
    """Leave-one-part-out error of the same-generation median, over fitted parts.

    Returns (median, worst, n). This is the figure the written rationale quotes, so it
    is computed from the committed file at write time rather than hardcoded -- a later
    refit moves it, and the entry should say what the predictor is worth now.
    """
    errs = []
    for held, truth in sorted(known.items()):
        same = [v for c, v in known.items()
                if c != held and GENERATION.get(c) == GENERATION.get(held)]
        if not same:
            continue
        pred = statistics.median(same)
        errs.append(max(pred / truth, truth / pred))
    if not errs:
        return (float("nan"), float("nan"), 0)
    return (statistics.median(errs), max(errs), len(errs))


# A windowed family's companion full-attention family. Where a part has no windowed
# measurement but DOES have a fitted full-attention one, the windowed value can be
# predicted from that part's OWN measured value times the windowed/full ratio of its
# KERNEL siblings. That is a within-part relation anchored on a measurement of the target
# part, which is stronger than any cross-part median -- and it is the only predictor
# available for A100, whose generation has no windowed fit at all.
COMPANION = {
    "attention_decode_rate_swa": "attention_decode_rate",
    "attention_decode_floor_swa": "attention_decode_floor",
}


def predict(family: str, chip: str, known: dict[str, float], all_vals: dict):
    """Return (value, siblings, method, holdout) for the best available predictor.

    Same-generation median first: it wins the holdout on every family that has a fitted
    sibling. Kernel-matched within-part ratio second, for a part whose generation carries
    no fit. No third fallback -- the global median was scored and rejected.
    """
    gen = GENERATION.get(chip)
    if gen is None:
        raise SystemExit(f"{chip}: no architecture generation recorded; add it to "
                         f"GENERATION with the die family it shares")
    siblings = {c: v for c, v in known.items()
                if GENERATION.get(c) == gen and c != chip}
    if siblings:
        return (statistics.median(siblings.values()), siblings, "generation",
                holdout_error(known))

    companion = COMPANION.get(family)
    if companion:
        comp = all_vals.get(companion, {})
        kf = KERNEL_FAMILY.get(chip)
        ratios = {c: known[c] / comp[c] for c in known
                  if c in comp and KERNEL_FAMILY.get(c) == kf and c != chip}
        if ratios and chip in comp:
            ratio = statistics.median(ratios.values())
            return (ratio * comp[chip], ratios, "kernel-ratio",
                    ratio_holdout(known, comp))

    raise SystemExit(
        f"{chip}: no fitted {gen} sibling for {family}, and no kernel-matched "
        f"within-part ratio available either. The remaining option is the global median "
        f"across all generations, which the leave-one-part-out holdout rejects (up to "
        f"16x error on windowed rates). Refusing to write a value this project has "
        f"measured to be unreliable.")


def ratio_holdout(known: dict[str, float],
                  comp: dict[str, float]) -> tuple[float, float, int]:
    """Leave-one-out error of the kernel-matched within-part ratio predictor."""
    both = [c for c in known if c in comp]
    errs = []
    for held in both:
        kf = KERNEL_FAMILY.get(held)
        others = [known[c] / comp[c] for c in both
                  if c != held and KERNEL_FAMILY.get(c) == kf]
        if not others:
            continue
        pred = statistics.median(others) * comp[held]
        truth = known[held]
        errs.append(max(pred / truth, truth / pred))
    if not errs:
        return (float("nan"), float("nan"), 0)
    return (statistics.median(errs), max(errs), len(errs))


def entry_lines(family: str, chip: str, scope: list[str], value, units: str,
                siblings: dict[str, float], err: tuple[float, float, int],
                method: str) -> list[str]:
    gen = GENERATION[chip]
    med, worst, n = err
    if method == "generation":
        sibtxt = ", ".join(f"{c} {v:,.6g}" for c, v in sorted(siblings.items()))
        derivation = (
            f"The median of the fitted {gen} parts, which share a die family and a "
            f"memory subsystem: {sibtxt}.")
    else:
        companion = COMPANION[family]
        sibtxt = ", ".join(f"{c} {v:.4f}" for c, v in sorted(siblings.items()))
        single = len(siblings) == 1
        derivation = (
            f"This part's OWN measured {companion} times the windowed/full ratio of its "
            f"KERNEL siblings ({sibtxt}). {gen} carries no windowed fit at all, so no "
            f"same-generation median exists -- but the ratio clusters by the kernel vLLM "
            f"dispatches rather than by the die, and that is readable from the sweeps' "
            f"own kernel_source column: Hopper runs vllm_flash_attn_fa3 at ratio "
            f"0.333-0.361, Blackwell runs vllm_flashinfer_flashinfertrtllmapidecode at "
            f"0.500-0.595, and L40S falls back to vllm_flash_attn_fa2 at 0.310. {chip} "
            f"runs vllm_flash_attn, the FA2-class path, because Ampere lacks the "
            f"hardware FA3 requires -- so L40S is its kernel sibling even across "
            f"generations. Anchoring on a measurement OF THIS PART makes this stronger "
            f"than any cross-part median."
            + (f" THE WEAKNESS, STATED: there is exactly ONE fa2-class sibling, so the "
               f"holdout below validates the fa3 and flashinfer groups and cannot test "
               f"this one -- l40s is necessarily excluded from its own prediction. The "
               f"ratio across all seven fitted parts spans 0.310 to 0.595, a 1.92x "
               f"range, and the fa2 figure sits at the BOTTOM of it. That is the "
               f"conservative end: a lower windowed rate means a longer windowed layer, "
               f"so this entry over-states windowed latency rather than under-stating "
               f"it, which is the safe direction for a term whose absence would "
               f"under-state it by 2x-5.5x." if single else ""))
    rat = (
        f"ASSUMED, not measured: no sweep in the AISimulate tree carries this "
        f"measurement for {chip} -- its attention collections report window_size 0 on "
        f"every row in all three lanes, so there is nothing to fit rather than nothing "
        f"fitted yet. "
        f"DIMENSION (derived). {derivation} The predictor was chosen by "
        f"leave-one-part-out holdout over every part this family IS fitted on -- hold a "
        f"part out, predict it, compare against its measured value -- giving median "
        f"{med:.3f}x and worst {worst:.3f}x over {n} held-out parts. Alternatives were "
        f"scored and rejected: the global median across all generations reaches 16x on "
        f"windowed rates, and scaling by the part's datasheet HBM figure is worse than "
        f"the generation median on every family tested. That is itself the finding -- "
        f"these coefficients are set by the kernel and the memory architecture rather "
        f"than by the headline bandwidth number. "
        f"MAGNITUDE (the caveat). This is an inference, not a measurement of {chip}. It "
        f"is committed because the alternative is worse rather than neutral: an absent "
        f"coefficient resolves to nothing and the kernel prices a windowed layer with "
        f"the FULL-attention rate, which on the parts where both are measured is wrong "
        f"by 2x to 5.5x -- larger than this estimate's holdout error. "
        f"TO REPLACE THIS WITH A MEASUREMENT: a {chip} attention sweep carrying non-zero "
        f"window_size rows; the fitter already handles the part and needs only the data."
    )
    return [
        f"  - {family}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: assumed",
        "      fitted: false",
        f"      scope: {{hardware: [{', '.join(scope)}]}}",
        "      rationale: >",
        f"        {rat}",
    ]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(FAMILY))
    ap.add_argument("--chip", required=True)
    ap.add_argument("--scope", help="comma-separated scope; defaults to --chip")
    ap.add_argument("--check", action="store_true",
                    help="print the prediction and its holdout error; write nothing")
    args = ap.parse_args(argv[1:])

    all_vals = committed()
    known = all_vals.get(args.family, {})
    if not known:
        print(f"{args.family}: nothing committed to transfer from", file=sys.stderr)
        return 1
    if args.chip in known:
        print(f"{args.chip}: {args.family} is already committed "
              f"({known[args.chip]}); refusing to overwrite a value with an estimate",
              file=sys.stderr)
        return 1

    value, siblings, method, err = predict(args.family, args.chip, known, all_vals)
    # Round to the precision the family's measured entries use, so an estimate is not
    # distinguishable by spurious digits.
    sample = next(iter(known.values()))
    value = round(value, 3) if isinstance(sample, float) and sample < 10 else round(value)
    label = {"generation": "same-generation median",
             "kernel-ratio": "kernel-matched within-part ratio"}[method]
    print(f"{args.family} on {args.chip} ({GENERATION[args.chip]}, "
          f"kernel {KERNEL_FAMILY.get(args.chip, '?')}):")
    print(f"  predictor  {label}")
    print(f"  prediction {value:,.6g} from {len(siblings)} sibling(s): "
          f"{', '.join(f'{c}={v:,.6g}' for c, v in sorted(siblings.items()))}")
    # The holdout belongs to the predictor actually used, not to whichever one is
    # cheapest to compute. Reporting the generation holdout beside a kernel-ratio
    # prediction would overstate the evidence for the number being written.
    print(f"  holdout of THAT predictor: median {err[0]:.3f}x  worst {err[1]:.3f}x  "
          f"n={err[2]}")
    if args.check:
        return 0

    setname, units = FAMILY[args.family]
    path = REPO / "coefficients" / setname
    text = path.read_text(encoding="utf-8")
    scope = args.scope.split(",") if args.scope else [args.chip]
    lines = text.split("\n")
    # Anchor after the family's last entry so it stays grouped.
    last = None
    for i, ln in enumerate(lines):
        if ln == f"  - {args.family}:":
            j = i + 1
            while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
                j += 1
            last = j
    if last is None:
        print(f"{setname}: no existing {args.family} entry to anchor to",
              file=sys.stderr)
        return 1
    while last > 0 and (not lines[last - 1].strip()
                        or lines[last - 1].lstrip().startswith("#")):
        last -= 1
    block = entry_lines(args.family, args.chip, scope, value, units, siblings, err,
                        method)
    out = lines[:last] + block + lines[last:]
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    path.write_text(result, encoding="utf-8")
    print(f"\n{setname}: inserted {args.family} for {args.chip} "
          f"(scope {', '.join(scope)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
