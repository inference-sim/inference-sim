#!/usr/bin/env python3
"""Write the `transferred` ramp of the dense Marlin W8A16 MXFP8 GEMM on Hopper.

    python scripts/emit_marlin_dense_transfer.py            # rewrite the entries
    python scripts/emit_marlin_dense_transfer.py --check    # exit 1 if a file would change

WHY. On SM90 vLLM v0.31.0 serves every dense MXFP8 linear with MarlinMxfp8LinearKernel
(W8A16: FP8 weights dequantised in registers, BF16 multiply, mma.sync). The registry carried
that GEMM at the BF16 envelope of cuBLAS (`gemm_eps_max_w8a16_mxfp8` 1.0 / 0.992, knee 170 /
110, written by fit_deepseek_v41_h200.py as `assumed`, role upper_bound), which its own
rationale calls optimistic for compute-bound GEMMs: no sweep measures dense Marlin on any part.
The same MMA core in its grouped (MoE) form IS measured on the same die
(`gemm_*_w4a16_mxfp4_grouped`, fit_marlin_grouped_ramp.py), at 0.282 of the BF16 peak with a
knee of 15 tokens on h100 and 0.245 with 26 on h200.

WHAT. Four entries, an asymptote and a half-max on each of h100 and h200, with method
`transferred`. Only the h200 pair is ACTIVE, in coefficients/cost-model-primitives.yaml, where the
lookup reads it. The h100 pair is a CANDIDATE, written to candidates/ where the
loader, the embed patterns and the validators do not look: the landing criteria
(docs/kernel/concepts/hopper-dense-mxfp8-ramp.md) give h100 no held-out unit, so it has no
verdict and the assumed bound keeps pricing it. To select it, a held-out verdict for h100 must
exist; then move h100 from CANDIDATE_PARTS to ACTIVE_PARTS and re-run this script.

The value is the part's own committed grouped pair, moved to the dense kernel by a stated rule (below). It is NOT a fit to any end-to-end residual: no dashboard point, TTFT or
TPOT figure enters this script, which reads only the committed grouped entries.

THE RULE. Same library (vLLM), same kernel family (Marlin), same MMA core and dequantisation
path, same die; the weight format differs (MXFP8 E4M3 with E8M0 scales against MXFP4), and the
grouped kernel adds the routing. The ramp is a fraction of the BF16 peak, so the transfer keeps
the fraction and the knee of the SAME part, never a number from the other part: the h200 pair
is not borrowed for the h100.

WHAT IT CAN CLAIM. No kernel hash exists for either kernel (the engine reports no
implementation for a GEMM yet), so the entries are unkeyed: they state no `kernel_impls` and
no `measured_on`, and a lookup does not count them as an exact match (G1's hash condition is
not met by them). Method `transferred` is rank 3 and wins over the `assumed` rank-7 bound
(design VAL-4); the bound stays in the registry, is the entry the lookup displaces and names,
and is what a price falls back to if these are removed.

WHAT IT DOES NOT CLAIM. Dense Marlin was never timed. The grouped fit sums its compute and
memory halves and evaluates the ramp at the step's token count; whether the dense kernel
shares that composition is not verified. The grouped fit covers 27 token counts of one MoE
shape; a dense GEMM of other shapes is outside any range a sweep has shown.

TO REPLACE IT. A MarlinMxfp8LinearKernel sweep on one H100 and one H200 over the GEMM shapes
the registry already uses, fitted like fit_marlin_grouped_ramp.py, written as `measured` with
kernel_impls; rank 1 then wins over these.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402

SET_PATH = HERE.parent / "coefficients" / "cost-model-primitives.yaml"
# Not matched by any "coefficients/*.yaml" pattern (the loader's embed, the validators' globs).
CANDIDATE_PATH = HERE.parent / "candidates" / "dense-marlin-mxfp8-transferred.yaml"
ACTIVE_PARTS = ("h200",)
CANDIDATE_PARTS = ("h100",)

# (dense name, grouped source name, units)
PAIR = (
    ("gemm_eps_max_w8a16_mxfp8", "gemm_eps_max_w4a16_mxfp4_grouped", "dimensionless"),
    ("gemm_m_half_w8a16_mxfp8", "gemm_m_half_w4a16_mxfp4_grouped", "tokens"),
)

DENSE_NAMES = tuple(d for d, _, _ in PAIR)

CANDIDATE_HEADER = """\
# CANDIDATE entries, NOT READ by the lookup: this directory is outside the registry's data
# directories (acceptance, coefficients, routing), so the loader and the validators skip it.
# The h100 dense Marlin W8A16 MXFP8 ramp is recorded here because it has no held-out unit and
# so no verdict (docs/kernel/concepts/hopper-dense-mxfp8-ramp.md); the `assumed` bound in cost-model-primitives.yaml keeps pricing h100.
# WRITTEN by scripts/emit_marlin_dense_transfer.py, which checks it. Do not edit by hand.
coefficients:
"""

RULE = (
    "The part's own measured Marlin grouped pair (gemm_eps_max_w4a16_mxfp4_grouped and "
    "gemm_m_half_w4a16_mxfp4_grouped, one ramp) is used unchanged as the ramp of the dense "
    "Marlin W8A16 MXFP8 GEMM on the same part: same library, same Marlin kernel family and MMA "
    "core (FP8 or FP4 weights dequantised in registers, BF16 multiply with mma.sync), same die. "
    "The fraction of the BF16 peak and the knee are kept as one pair, never mixed with a "
    "number from the other part. No kernel hash is stated for either kernel."
)


def committed(entries: list[dict], name: str, chip: str) -> dict:
    hits = [b for e in entries for n, b in e.items()
            if n == name and (b.get("scope") or {}).get("hardware") == [chip]
            and b["method"] == "measured"]
    if len(hits) != 1:
        raise SystemExit(f"{name} on {chip}: expected one measured entry, found {len(hits)}")
    return hits[0]


def fmt(x: float) -> str:
    return repr(float(x)) if float(x) != int(x) else str(int(x))


def entry_lines(chip: str, dense: str, src_name: str, units: str, src: dict,
                eps: float, mh: float) -> list[str]:
    value = src["value"]
    bound = "gemm_eps_max_w8a16_mxfp8" if dense.startswith("gemm_eps") else "gemm_m_half_w8a16_mxfp8"
    cite = src["sources"][0]["cite"]
    if units == "dimensionless":
        what = (f"The asymptotic fraction ({fmt(value)}) of this part's dense BF16 peak that the "
                f"dense Marlin W8A16 MXFP8 GEMM is taken to reach")
    else:
        what = (f"The token count ({fmt(value)}) at which the dense Marlin W8A16 MXFP8 GEMM "
                f"ramp is taken to reach half its asymptote")
    pair = f"{fmt(eps)} of the BF16 peak with a knee of {fmt(mh)} tokens"
    text = (
        f"{what}, moved from the same part's measured Marlin grouped GEMM ({src_name}; the "
        f"measured pair is {pair}). "
        "TRANSFERRED, NOT MEASURED: no sweep has timed MarlinMxfp8LinearKernel on any part. "
        "The rule: same Marlin kernel family and MMA core on the same die, so the grouped "
        "form's fraction of the BF16 peak and knee are used for the dense form. This replaces "
        f"the assumed {bound} (the BF16 envelope, kept in the registry as the named upper "
        "bound this entry displaces: a lookup that finds only that entry prices the GEMM at "
        "the BF16 peak, as before). The pair is one ramp: the asymptote alone, with the dense "
        "knee kept, over-prices decode. VALIDITY: scope is this chip and the dense GEMM of "
        "the MXFP8 weight format on SM90 (the name carries the format; the grouped routed "
        "term reads the w4a16 names). The entry is unkeyed, because the engine reports no "
        "kernel implementation for a GEMM, so no kernel_impls or measured_on can be stated "
        "and it does not count as an exact kernel match. The grouped fit covers 27 token "
        "counts of one MoE shape and sums its compute and memory halves; whether the dense "
        "kernel shares that composition, and whether its ramp holds at other GEMM shapes, is "
        "not shown. FITTED NOTHING: no end-to-end result entered this value, and the "
        "committed sibling entries are its only input. TO REPLACE IT: a "
        "MarlinMxfp8LinearKernel sweep on this part, fitted like fit_marlin_grouped_ramp.py "
        "and written as measured with kernel_impls. Reproduce with "
        "scripts/emit_marlin_dense_transfer.py."
    )
    return [
        f"  - {dense}:",
        f"      value: {fmt(value)}",
        f"      units: {units}",
        "      method: transferred",
        "      fitted: false",
        f"      scope: {{hardware: [{chip}]}}",
        f'      transferred_from: "hardware={chip}"',
        f'      transfer_rule: "{RULE}"',
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "        - {kind: model, cite: \"vLLM v0.31.0 model_executor/kernels/linear/mxfp8/marlin.py "
        "MarlinMxfp8LinearKernel\", role: supporting}",
        "      rationale: >",
        f"        {text}",
    ]


def candidate_note(chip: str) -> str:
    return (" CANDIDATE, NOT SELECTED: this part has no held-out unit, so the landing criteria "
            "give it no verdict; this entry sits outside the registry the lookup reads and the "
            "assumed bound prices the GEMM." if chip in CANDIDATE_PARTS else "")


def entries_for(entries: list[dict], chip: str) -> list[list[str]]:
    out = []
    srcs = {g: committed(entries, g, chip) for _, g, _ in PAIR}
    eps, mh = (srcs[g]["value"] for _, g, _ in PAIR)
    for dense, g, units in PAIR:
        lines = entry_lines(chip, dense, g, units, srcs[g], eps, mh)
        if chip in CANDIDATE_PARTS:
            lines[-1] = lines[-1].replace("Reproduce with", candidate_note(chip).strip()
                                          + " Reproduce with")
        out.append(lines)
    return out


def drop_dense_transferred(blocks: list[list[str]], chips: tuple) -> list[list[str]]:
    """Remove the transferred dense entries of `chips` (a part moved out of the active set)."""
    keys = {cb.scope_key(f"{{hardware: [{c}]}}") for c in chips}
    return [b for b in blocks
            if not (cb.name_of(b) in DENSE_NAMES and cb.method_of(b) == "transferred"
                    and cb.scope_of(b) is not None and cb.scope_key(cb.scope_of(b)) in keys)]


def render(text: str) -> str:
    """The active registry file with the ACTIVE_PARTS pairs written and no other part's."""
    entries = yaml.safe_load(text)["coefficients"]
    blocks = drop_dense_transferred(cb.split(text), CANDIDATE_PARTS)
    for chip in ACTIVE_PARTS:
        key = cb.scope_key(f"{{hardware: [{chip}]}}")
        for (dense, _, _), lines in zip(PAIR, entries_for(entries, chip)):
            cb.upsert(blocks, lines,
                      after=lambda b, d=dense, k=key: cb.name_of(b) == d
                      and cb.scope_key(cb.scope_of(b) or "{}") == k)
    return cb.join(blocks)


def render_candidates(text: str) -> str:
    """The candidate file: the CANDIDATE_PARTS pairs, derived from the committed grouped entries."""
    entries = yaml.safe_load(text)["coefficients"]
    body = [line for chip in CANDIDATE_PARTS for lines in entries_for(entries, chip)
            for line in lines]
    return CANDIDATE_HEADER + "\n".join(body) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file would change; write nothing")
    args = ap.parse_args(argv[1:])
    before = SET_PATH.read_text(encoding="utf-8")
    after = render(before)
    cand_before = CANDIDATE_PATH.read_text(encoding="utf-8") if CANDIDATE_PATH.exists() else None
    cand_after = render_candidates(before)
    if args.check:
        stale = [p.name for p, b, a in ((SET_PATH, before, after), (CANDIDATE_PATH, cand_before, cand_after))
                 if a != b]
        if stale:
            print(f"{', '.join(stale)} would change", file=sys.stderr)
            return 1
        print(f"{SET_PATH.name} and {CANDIDATE_PATH.name} already match the transfer")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    CANDIDATE_PATH.parent.mkdir(exist_ok=True)
    CANDIDATE_PATH.write_text(cand_after, encoding="utf-8")
    print(f"dense Marlin MXFP8 transfer: active for {', '.join(ACTIVE_PARTS)} in {SET_PATH.name}; "
          f"candidate for {', '.join(CANDIDATE_PARTS)} in candidates/{CANDIDATE_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
