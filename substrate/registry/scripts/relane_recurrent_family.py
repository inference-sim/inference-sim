#!/usr/bin/env python3
"""Maintain the mamba2 and GDN recurrent pairs in place, from fit_recurrent.py.

WHY THIS EXISTS. `relane_recurrent_kda.py` owns the KDA pair. The other two recurrent
families had no writer at all:

  * MAMBA2 was committed for h100 only, while the sweep covers EIGHT parts. A single
    hand-authored pair could not be extended without hand-authoring seven more.
  * GDN (gated delta rule) was not committed for ANY part, though vLLM-lane sweeps exist
    for seven. That is a whole model class the registry cannot price: the sweep's own
    `model_name` column names Qwen/Qwen3.5-* and Qwen/Qwen3.8-*, which resolve no
    recurrent entry today and therefore fall through to a cheaper path silently -- the
    exact failure mode the KDA h200 commit (640a27e) was written to fix for one part.

Both families take the same form as KDA, and for the same physical reason: vLLM runs a
depthwise convolution and then a recurrent scan for the same step, so a layer costs the
SUM -- floors add, reciprocal rates add.

    latency = floor + tokens / rate

WHAT DIFFERS BETWEEN THE THREE, and why they are not one table:

  KDA     conv + fused_recurrent_kda_packed_decode.               COMPLETE.
  GDN     conv + fused_recurrent_gated_delta_rule_packed_decode.  COMPLETE.
  MAMBA2  conv ONLY. `causal_conv1d_fn` and `causal_conv1d_update` are the only kernels
          in the sweep on every one of the eight parts -- checked, not assumed. The
          selective-scan kernel that does the rest of a Mamba2 layer appears in no
          collection in the tree, so every mamba2 entry is a documented LOWER BOUND and
          says so. Its `method` is `measured` because the convolution measurement is
          real; the incompleteness lives in the rationale.

THE MODEL FILTER IS LOAD-BEARING for mamba2. Its sweep carries six geometries
(MAMBA2_GENERIC_1K/4K and four Nemotron checkpoints) and pooling them fits none of them:
pooled gives floor 4.2 and rate 23.25, a figure that appears in no version of the
committed file. `--model` is required, and the committed entries cite the model they were
fitted on.

GDN pools deliberately, and that is a different decision rather than an inconsistency:
its rows are nine Qwen checkpoints that share one architecture family, the fit is taken
over all of them, and the citation says so. A per-checkpoint GDN fit would be the right
move if a deployment named one; none does yet.

Usage:
    python scripts/relane_recurrent_family.py --family mamba2 --check
    python scripts/relane_recurrent_family.py --family gdn --insert h200 h200_sxm:h200
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-recurrent.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/private/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")

# One collection per part per family, pinned by name. Globbing mixes engine versions and
# silently changes the row set, which is what makes a cross-part difference a software
# release rather than silicon.
FAMILIES = {
    "mamba2": {
        "path": "linear_attention/{coll}/mamba2_perf.parquet",
        "coll": "trtllm/1.3.0rc20",
        # mamba2_perf exists on the TRT-LLM lane and nowhere else -- checked across all
        # eight SKUs. vLLM's linear_attention collections carry gdn_perf only. A strict
        # vLLM-only registry could not price a Nemotron-3 hybrid at all.
        "chain": ["causal_conv1d_update"],
        "model": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        "complete": False,
        "parts": {
            "h100_sxm": "h100", "h200_sxm": "h200", "b200_sxm": "b200",
            "b300_sxm": "b300", "gb200": "gb200-nvl72", "gb300": "gb300",
            "l40s": "l40s",
        },
    },
    "gdn": {
        "path": "linear_attention/{coll}/gdn_perf.parquet",
        "coll": "vllm/0.25.0",
        "chain": ["causal_conv1d_update",
                  "fused_recurrent_gated_delta_rule_packed_decode"],
        # Qwen3.5-397B-A17B is the GDN checkpoint blis-catalog carries
        # (models/qwen3.5-397b-a17b/graph.yaml), and the sweep names it.
        "model": "Qwen/Qwen3.5-397B-A17B",
        # PINNED GEOMETRY, and this is the whole accuracy story for this family. The
        # sweep varies num_v_heads from 2 to 128 across nine Qwen checkpoints, and the
        # rate scales INVERSELY with it -- 99.75 tok/us at 2 v-heads down to 11.25 at
        # 128 -- because bytes per token scale with the head count. Pooling those
        # geometries fits none of them: it lands at 1.27x-1.80x geometric error where a
        # single geometry reaches 1.01x-1.06x. The kernel's coefficient carries no head
        # term (kernel.go prices a recurrent layer as floor + tokens/rate keyed on kind
        # alone), so the fitted geometry MUST be the model's. The catalog declares
        # n_heads: 64 for this checkpoint's RecurrentUpdate and the sweep's matching rows
        # are (num_k_heads 16, num_v_heads 64) -- so that is the pin. This is the same
        # correction relane_recurrent_kda.py made when it moved KDA off num_k_heads=12
        # to Kimi-K3's declared 96.
        "geometry": {"num_k_heads": 16, "num_v_heads": 64},
        "complete": True,
        "parts": {
            "h100_sxm": "h100", "h200_sxm": "h200", "b200_sxm": "b200",
            "b300_sxm": "b300", "gb200": "gb200-nvl72", "gb300": "gb300",
        },
    },
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


fr = _load("fit_recurrent")


def fit_chain(path: Path, spec: dict) -> dict:
    """Fit each kernel in the chain, then compose them in series."""
    rows = fr.read(path)
    gen = [r for r in rows if r.get("phase") == "generation" and r["latency"] > 0]
    if spec["model"]:
        key = "model_name" if gen and "model_name" in gen[0] else "model"
        gen = [r for r in gen if r.get(key) == spec["model"]]
    for col, want in (spec.get("geometry") or {}).items():
        gen = [r for r in gen if r.get(col) == want]
    if not gen:
        raise SystemExit(f"{path}: no generation rows"
                         + (f" for {spec['model']}" if spec["model"] else "")
                         + (f" at {spec['geometry']}" if spec.get("geometry") else ""))
    parts = {}
    for ks in spec["chain"]:
        pts = [(r["num_tokens"], r["latency"] * 1000) for r in gen
               if r["kernel_source"] == ks]
        # A single pinned geometry gives ~10-11 points per kernel, which is the same
        # count the committed KDA and mamba2 pairs are fitted on. Eight is the floor
        # below which a two-parameter fit is not worth stating.
        if len(pts) < 8:
            raise SystemExit(f"{path}: only {len(pts)} points for {ks}")
        floor, rate, err = fr.fit(pts)
        parts[ks] = {"floor": floor, "rate": rate, "err": round(err, 3), "n": len(pts)}
    floor = sum(p["floor"] for p in parts.values())
    rate = 1.0 / sum(1.0 / p["rate"] for p in parts.values())
    return {"parts": parts, "floor": round(floor, 1), "rate": round(rate, 3)}


def entry_lines(kind: str, chip: str, sku: str, family: str, fit: dict) -> list[str]:
    spec = FAMILIES[family]
    chain = " + ".join(spec["chain"])
    errs = " and ".join(f"{p['err']}x" for p in fit["parts"].values())
    ns = " and ".join(str(p["n"]) for p in fit["parts"].values())
    model = spec["model"] or "the sweep's models pooled"
    geom = spec.get("geometry")
    geomtxt = (", " + ", ".join(f"{k}={v}" for k, v in geom.items())) if geom else ""
    cite = (f"NVIDIA AISimulate systems/data/{sku}/"
            f"{spec['path'].format(coll=spec['coll'])}, kernel"
            f"{'s' if len(spec['chain']) > 1 else ''} {chain}"
            f"{' summed' if len(spec['chain']) > 1 else ''}, generation phase"
            f"{geomtxt}, model {model}, {ns} points")
    value = fit["floor"] if kind == "floor" else fit["rate"]
    if family == "mamba2":
        if kind == "floor":
            rat = (f"A LOWER BOUND on one Mamba2 layer's decode-phase cost on {chip}, "
                   f"not the cost. This is the convolution alone: the sweep carries "
                   f"`causal_conv1d_fn` and `causal_conv1d_update` and nothing else, and "
                   f"the selective-scan kernel that does the rest of a Mamba2 layer is in "
                   f"no collection in this tree -- checked on every part, not assumed "
                   f"from one. Geometric error {errs} against the convolution's own "
                   f"measurements, which says the fit is good and says nothing about the "
                   f"missing kernel. Fitted on the rows naming this checkpoint rather "
                   f"than pooling the sweep's six geometries, because pooling fits none "
                   f"of them. TO COMPLETE THIS: measure a selective-scan kernel, or take "
                   f"the difference between a full Mamba2 layer's latency and this "
                   f"convolution figure if a layer-level measurement becomes available.")
        else:
            rat = (f"Tokens per microsecond past the floor, for the convolution alone on "
                   f"{chip}. Higher than a gated delta-rule family's because a depthwise "
                   f"convolution over four taps is far less work -- and because this "
                   f"figure is missing the scan that would lower it. Geometric error "
                   f"{errs}.")
    else:
        if kind == "floor":
            rat = (f"The minimum cost of one gated-delta-rule layer's decode-phase state "
                   f"update on {chip}. vLLM runs a depthwise convolution and then the "
                   f"recurrent scan for the same step, so a layer costs the SUM: "
                   f"{' + '.join(str(p['floor']) + 'us' for p in fit['parts'].values())}. "
                   f"Per-kernel geometric error {errs}. Fitted at num_v_heads=64, which "
                   f"is what Qwen3.5-397B-A17B's graph declares; the kernel's coefficient "
                   f"carries no head term, so the fitted geometry must be the model's. "
                   f"That pin is load-bearing: the sweep spans num_v_heads 2 to 128 and "
                   f"the rate scales inversely with it, so pooling the geometries fits "
                   f"none of them (1.27x-1.80x pooled against 1.01x-1.06x per geometry). "
                   f"Without this entry a GDN model resolves no recurrent coefficient and "
                   f"falls through to a cheaper path silently.")
        else:
            rat = (f"Tokens per microsecond past the {chip} floor, at the "
                   f"num_v_heads=64 geometry Qwen3.5-397B-A17B declares. The two kernels "
                   f"run in series, so their times add and the reciprocal rates add: "
                   f"{' + '.join('1/' + str(p['rate']) for p in fit['parts'].values())} "
                   f"tok/us. The recurrent scan dominates, being sequential in the state "
                   f"dimension -- it cannot spread one sequence's work across the machine "
                   f"the way a matmul spreads a batch's. A model with a different head "
                   f"count needs its own fit: the rate is inversely proportional to "
                   f"num_v_heads across the sweep, which is why this entry names a "
                   f"geometry rather than a family.")
    return [
        f"  - recurrent_decode_{kind}_{family}:",
        f"      value: {value}",
        "      units: " + ("us_per_transfer" if kind == "floor" else "tokens"),
        "      method: measured",
        "      fitted: true",
        f"      scope: {{hardware: [{chip}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rat}",
    ]


def _spans(lines: list[str], family: str):
    pat = re.compile(rf"  - recurrent_decode_(floor|rate)_{family}:$")
    out = []
    i = 0
    while i < len(lines):
        m = pat.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"  - [a-z_0-9]+:$", lines[j]):
            j += 1
        block = lines[i:j]
        tail: list[str] = []
        while block and (not block[-1].strip() or block[-1].lstrip().startswith("#")):
            tail.insert(0, block.pop())
        chip = None
        for b in block:
            sm = re.search(r"scope: \{hardware: \[([^\]]+)\]\}", b)
            if sm:
                sc = [x.strip() for x in sm.group(1).split(",")]
                chip = sc[0] if len(sc) == 1 else None
        out.append((i, j, m.group(1), chip, tail))
        i = j
    return out


def rewrite(text: str, family: str, fits: dict) -> tuple[str, int]:
    lines = text.split("\n")
    out: list[str] = []
    changed, prev = 0, 0
    for start, end, kind, chip, tail in _spans(lines, family):
        out.extend(lines[prev:start])
        prev = end
        if chip is None or chip not in fits:
            out.extend(lines[start:end])
            continue
        sku, fit = fits[chip]
        out += entry_lines(kind, chip, sku, family, fit)
        out.extend(tail)
        changed += 1
    out.extend(lines[prev:])
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, changed


def insert(text: str, family: str, chip: str, fits: dict) -> tuple[str, int]:
    """Append the pair for a part this family does not carry yet.

    Anchors after the family's last entry when it has one, and after the last recurrent
    entry of any family when the family is entirely new, so the file stays grouped.
    """
    if chip not in fits:
        return text, 0
    lines = text.split("\n")
    spans = _spans(lines, family)
    if not spans:
        spans = [s for f in FAMILIES for s in _spans(lines, f)]
        if not spans:
            return text, 0
    at = max(s[1] for s in spans)
    while at > 0 and (not lines[at - 1].strip() or lines[at - 1].lstrip().startswith("#")):
        at -= 1
    sku, fit = fits[chip]
    block: list[str] = []
    for kind in ("floor", "rate"):
        block += entry_lines(kind, chip, sku, family, fit)
    out = lines[:at] + block + lines[at:]
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, 2


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--insert", metavar="CHIP")
    ap.add_argument("parts", nargs="*", metavar="sku:chip")
    args = ap.parse_args(argv[1:])

    spec = FAMILIES[args.family]
    want = ([tuple(p.split(":", 1)) for p in args.parts] if args.parts
            else sorted(spec["parts"].items()))

    fits: dict[str, tuple[str, dict]] = {}
    for sku, chip in want:
        path = Path(args.data) / sku / spec["path"].format(coll=spec["coll"])
        if not path.is_file():
            print(f"{chip:14} no {args.family} sweep at {spec['coll']}", file=sys.stderr)
            continue
        try:
            fit = fit_chain(path, spec)
        except SystemExit as exc:
            print(f"{chip:14} {exc}", file=sys.stderr)
            continue
        fits[chip] = (sku, fit)
        print(f"{chip:14} floor={fit['floor']:5.1f}us rate={fit['rate']:8.3f} tok/us  "
              + "  ".join(f"{k.split('_')[0]}:{p['err']}x"
                          for k, p in fit["parts"].items()))
    if not fits:
        print("no parts fitted", file=sys.stderr)
        return 1

    before = SET_PATH.read_text(encoding="utf-8")

    if args.insert:
        chip = args.insert
        if chip not in fits:
            print(f"{chip}: not fitted above", file=sys.stderr)
            return 1
        if any(c == chip for _, _, _, c, _ in _spans(before.split("\n"), args.family)):
            print(f"{chip}: {args.family} already present; use the rewrite path",
                  file=sys.stderr)
            return 1
        after, n = insert(before, args.family, chip, fits)
        if not n:
            return 1
        SET_PATH.write_text(after, encoding="utf-8")
        print(f"\n{SET_PATH.name}: inserted {n} {chip} {args.family} entries")
        return 0

    after, changed = rewrite(before, args.family, fits)
    if args.check:
        if before != after:
            print(f"\n{SET_PATH.name} would change ({changed} rendered)",
                  file=sys.stderr)
            return 1
        print(f"\n{SET_PATH.name} already matches the {args.family} fit")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"\n{SET_PATH.name}: {changed} {args.family} entries rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
