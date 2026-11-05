#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash on SM90 (H200, H100): the DECODE sparse-MLA form, fitted per part on
the part's own rows of the FlashMLA sparse decode vLLM v0.31.0 runs.

    python scripts/fit_deepseek_v41_decode.py              # print the fits, rewrite the set
    python scripts/fit_deepseek_v41_decode.py --check      # exit 1 if the set would change
    python scripts/fit_deepseek_v41_decode.py --part h100  # one part only

The decode companion of fit_deepseek_v41_prefill.py; it reuses fit_deepseek_v41_h200.py's
geometry check, table reader and entry renderer. Owns, per part, scoped
{hardware: [<part>], model: [deepseek-v4.1-flash]}, in cost-model-attention.yaml:

    attention_decode_launch_floor_sparse_mla      us_per_transfer  measured
    attention_decode_split_floor_sparse_mla       us_per_transfer  measured
    attention_decode_serial_rate_sparse_mla       bytes_per_us     measured
    attention_decode_per_token_sparse_mla         us_per_token     measured
    attention_decode_throughput_rate_sparse_mla   bytes_per_us     measured

THE FORM, per layer, per rank, per decode step (one flash_mla_with_kvcache call):

    t = max( launch_floor + split_floor x [B > 1] + B x 37,376 B / serial_rate ,
             sum over query rows q of ( per_token + blocks_q x 37,376 B / throughput_rate ) )

blocks_q = ceil(min(ctx, window) / 64) + ceil(min(ctx // ratio, index_topk) / 64), the
decode writer's FlashMLA block count (10 for V4.1 past about 1k tokens); B = the largest
blocks_q in the call; 37,376 B = 64 tokens x the 584 B V4 record.

WHY THIS FORM. The committed pair (attention_decode_rate_sparse_mla over the inherited
part-wide floor) is 1.18-1.20x with a batch bias it cannot carry: signed 1.45-1.54 at batch
128 and 0.59-0.80 at 1,024. The rows show two regimes, not one line:
  * Latency-bound, small batch: flat in batch to about 128 rows at 1-2 blocks. One block
    costs about 10.2 (h200) / 10.4 (h100) us; any read of two or more blocks adds about
    3.5 us at once -- the split-KV combine launch FlashMLA adds when it splits a request --
    then about 0.2 us a block on the critical path.
  * Throughput-bound, large batch: linear in rows through the origin, one cost per query row
    (its 64 padded heads in and out) and one per block read.
Which regime binds is the max. The sum of the same terms fits worse and does not hold out.

Fitted on the rows V4.1 can reach (at most v41_blocks per query), as the prefill fit is.
Minimises the mean absolute log ratio by a deterministic coordinate search in log-parameter
space, so --check reproduces byte for byte with numpy alone.

UNITS. `latency` is MILLISECONDS in every AISimulate parquet.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402
import fit_deepseek_v41_h200 as base  # noqa: E402

ATTN_PATH = base.ATTN_PATH
PARTS = base.PARTS
BLOCK = base.BLOCK
NAMES = ("launch_floor", "split_floor", "serial_rate", "per_token", "throughput_rate")
# Held-out splits: (label, train, test) over (rows, blocks).
HOLDOUTS = (
    ("rows", "rows <= 256 -> >= 512", lambda r, b: r <= 256, lambda r, b: r >= 512),
    ("blocks", "blocks <= 4 -> >= 6", lambda r, b: b <= 4, lambda r, b: b >= 6),
    ("unsat", "rows <= 64 -> >= 128", lambda r, b: r <= 64, lambda r, b: r >= 128),
)


def geo(pred, y) -> tuple[float, float]:
    lg = np.log(np.asarray(pred, float) / np.asarray(y, float))
    return float(math.exp(np.mean(np.abs(lg)))), float(math.exp(np.mean(lg)))


def predict(th, rows, blocks):
    f, s, e, a, c = th
    latency = f + s * (blocks > 1) + e * blocks
    throughput = rows * (a + c * blocks)
    return np.maximum(latency, throughput)


# Starting points of the search. The objective is a sum of absolute log errors over a
# max of two branches, so it is piecewise flat and a single start stops on a ridge; a fixed
# grid of starts, best kept (first on a tie), makes the answer the data's and repeatable.
STARTS = [(f, s, e, a, c) for f in (8.0, 10.0, 12.0) for s in (1.0, 3.0, 5.0)
          for e in (0.1, 0.3, 0.6) for a in (0.02, 0.06) for c in (0.01, 0.03, 0.06)]


def fit(rows, blocks, y) -> np.ndarray:
    """(floor, split, us per serial block, us per row, us per row-block) minimising the mean
    |log ratio|; deterministic coordinate search over log(theta) with halving steps, from
    every point of STARTS."""

    def obj(v):
        return float(np.mean(np.abs(np.log(predict(np.exp(v), rows, blocks) / y))))

    best_all, best_lt = math.inf, None
    for x0 in STARTS:
        lt = np.log(np.array(x0))
        best, step = obj(lt), 1.0
        while step > 1e-6:
            improved = False
            for j in range(len(lt)):
                for d in (step, -step):
                    trial = lt.copy()
                    trial[j] += d
                    v = obj(trial)
                    if v < best - 1e-15:
                        lt, best, improved = trial, v, True
            if not improved:
                step /= 2
        if best < best_all - 1e-12:
            best_all, best_lt = best, lt
    return np.exp(best_lt)


def decode_rows(data: Path, sku: str, g: dict) -> dict:
    t = base.read(data, sku, base.FLASHMLA_TABLE)
    rows, blocks, y = [], [], []
    for i in range(len(t["latency"])):
        if t["isl"][i] != 1 or t["model"][i] != base.FLASHMLA_FIT_MODEL:
            continue
        if (t["kernel_source"][i] != base.FLASHMLA_LANE or t["compress_ratio"][i] != 128
                or t["kv_cache_dtype"][i] != "fp8" or t["local_num_heads"][i] != 64
                or not (math.isfinite(t["latency"][i]) and t["latency"][i] > 0)):
            raise SystemExit(f"unexpected FlashMLA row {i}")
        ctx = t["step"][i] + 1
        rows.append(t["batch_size"][i])
        blocks.append(math.ceil(min(ctx, g["window"]) / BLOCK)
                      + math.ceil((ctx // 128) / BLOCK))
        y.append(t["latency"][i] * 1000.0)
    out = {"rows": np.asarray(rows, float), "blocks": np.asarray(blocks, float),
           "y": np.asarray(y, float)}
    if len(y) < 150:
        raise SystemExit(f"{sku}: only {len(y)} FlashMLA decode rows")
    return out


def fit_part(data: Path, sku: str, g: dict, floor: float, rate: float) -> dict:
    allr = decode_rows(data, sku, g)
    keep = allr["blocks"] <= g["v41_blocks"]
    r, b, y = allr["rows"][keep], allr["blocks"][keep], allr["y"][keep]
    th = fit(r, b, y)
    pred = predict(th, r, b)
    err, signed = geo(pred, y)
    committed = floor + r * b * BLOCK * g["record"] / rate
    holds = {}
    for key, label, tr_f, te_f in HOLDOUTS:
        tr, te = tr_f(r, b), te_f(r, b)
        th_tr = fit(r[tr], b[tr], y[tr])
        holds[key] = (label, int(te.sum()), *geo(predict(th_tr, r[te], b[te]), y[te]),
                      *geo(committed[te], y[te]))
    by_batch = {int(k): geo(pred[r == k], y[r == k])[1] for k in sorted(set(r))}
    by_batch_old = {int(k): geo(committed[r == k], y[r == k])[1] for k in sorted(set(r))}
    at10 = b == g["v41_blocks"]
    if not at10.any():
        raise SystemExit(f"{sku}: no row at V4.1's {g['v41_blocks']} blocks")
    rec = BLOCK * g["record"]
    # Rows past which the throughput branch binds at V4.1's read.
    v41_lat = th[0] + th[1] + th[2] * g["v41_blocks"]
    knee = v41_lat / (th[3] + th[4] * g["v41_blocks"])
    sat = r >= 256
    return {
        "n": len(y), "n_all": len(allr["y"]), "err": err, "signed": signed,
        "old": geo(committed, y), "holds": holds, "by_batch": by_batch,
        "by_batch_old": by_batch_old,
        "at10": (int(at10.sum()), *geo(pred[at10], y[at10])),
        "at10_rows": int(r[at10].max()),
        "floor": round(float(th[0]), 1), "split": round(float(th[1]), 1),
        "serial_rate": int(round(rec / th[2])), "per_token": float(f"{th[3]:.4g}"),
        "throughput_rate": int(round(rec / th[4])), "knee": knee, "v41_lat": v41_lat,
        "sat_blocks": sorted({int(x) for x in b[sat]}),
        "b1": {int(k): float(np.median(y[(r == 1) & (b == k)])) for k in (1, 2, 10)},
    }


def fmt_signed(d: dict) -> str:
    return ", ".join(f"{k}: {v:.2f}" for k, v in d.items())


def entries(chip: str, g: dict, f: dict, peak: float, partwide_floor: float,
            rate: float) -> list:
    part = PARTS[chip]
    sku, name = part["sku"], part["name"]
    scope = base.model_scope(chip)
    rec = BLOCK * g["record"]
    src = [("model", f"NVIDIA AISimulate systems/data/{sku}/sparse_attention/vllm/0.25.0/"
            f"{base.FLASHMLA_TABLE} (NVIDIA {name}), {base.FLASHMLA_LANE}, "
            f"{base.FLASHMLA_FIT_MODEL} (64 query heads, the h64 kernel), {f['n']} decode "
            f"points at <= {g['v41_blocks']} blocks", "primary"),
           ("model", "vLLM v0.31.0 models/deepseek_v41/nvidia/flashmla.py "
            "DeepseekV4FlashMLAAttention._forward_decode", "supporting")]
    h = f["holds"]
    form = (
        f"THE FORM, per layer, per rank, per decode step (one flash_mla_with_kvcache call): "
        f"t = max(attention_decode_launch_floor_sparse_mla + "
        f"attention_decode_split_floor_sparse_mla x [B > 1] + B x {rec:,} B / "
        f"attention_decode_serial_rate_sparse_mla, sum over decode query rows q of "
        f"(attention_decode_per_token_sparse_mla + blocks_q x {rec:,} B / "
        f"attention_decode_throughput_rate_sparse_mla)), blocks_q = ceil(min(ctx, window) / "
        f"64) + ceil(min(ctx // ratio, index_topk) / 64) -- {g['v41_blocks']} for V4.1 past "
        f"about 1k tokens -- and B the largest blocks_q in the call. A query row is a query "
        f"token: under speculative decoding a request contributes 1 + num_spec rows, each "
        f"with its own top-k (the sweep is isl 1; that each row costs one row is the "
        f"consumer's assumption, as for the committed rate). ALL-OR-NONE: the kernel uses "
        f"this form only when all five entries resolve, otherwise the committed "
        f"attention_decode_rate_sparse_mla over the part-wide floor. ")
    fit_txt = (
        f"FIT: {f['err']:.3f}x (signed {f['signed']:.3f}, n={f['n']}, rows of at most "
        f"{g['v41_blocks']} blocks) against {f['old'][0]:.3f}x (signed {f['old'][1]:.3f}) for "
        f"the committed rate over the {partwide_floor:g} us floor on the same rows. Signed "
        f"by batch: {fmt_signed(f['by_batch'])} (committed: {fmt_signed(f['by_batch_old'])}). "
        f"HELD OUT, refitted on the training side only: {h['rows'][0]} {h['rows'][2]:.3f}x "
        f"(signed {h['rows'][3]:.3f}, n={h['rows'][1]}; committed {h['rows'][4]:.3f}x); "
        f"{h['blocks'][0]} {h['blocks'][2]:.3f}x (signed {h['blocks'][3]:.3f}, "
        f"n={h['blocks'][1]}; committed {h['blocks'][4]:.3f}x). NOT IDENTIFIED WITHOUT "
        f"SATURATED ROWS: {h['unsat'][0]} gives {h['unsat'][2]:.3f}x (signed "
        f"{h['unsat'][3]:.3f}, n={h['unsat'][1]}; committed {h['unsat'][4]:.3f}x), because "
        f"no training row reaches the throughput branch; the two throughput terms rest on "
        f"the batch >= 256 rows, which are at {'/'.join(map(str, f['sat_blocks']))} blocks. "
        f"AT V4.1'S READ: the {f['at10'][0]} rows at {g['v41_blocks']} blocks (batch <= "
        f"{f['at10_rows']}) fit {f['at10'][1]:.3f}x (signed {f['at10'][2]:.3f}). At "
        f"{g['v41_blocks']} blocks the throughput branch binds above about {f['knee']:.0f} "
        f"query rows per rank (a latency branch of {f['v41_lat']:.1f} us); that region, "
        f"{g['v41_blocks']} blocks at many rows, is not in the table -- its batch x "
        f"context <= 1M grid pairs {g['v41_blocks']} blocks only with batch <= "
        f"{f['at10_rows']} -- so there the form extrapolates the per-block throughput cost "
        f"from 1-3 blocks to {g['v41_blocks']}, linearly, as the committed rate does. ")
    tail = (f"Covers the FlashMLA call alone: compute_global_topk_indices_and_lens (one "
            f"Triton launch per compressed layer) is not in the timed op. {part['e2e']} "
            f"Reproduce with scripts/fit_deepseek_v41_decode.py.")
    lines = base.lines
    return [
        lines("attention_decode_launch_floor_sparse_mla", f"{f['floor']}", "us_per_transfer",
              "measured", True, scope, src,
              f"The launch floor of V4.1's FlashMLA sparse DECODE on SM90: {f['floor']} us, "
              f"what a one-block read costs at small batch (batch-1 rows: 1 block "
              f"{f['b1'][1]:.1f} us, 2 blocks {f['b1'][2]:.1f} us, {g['v41_blocks']} blocks "
              f"{f['b1'][10]:.1f} us). Fitted jointly with the four other "
              f"attention_decode_*_sparse_mla terms of this form. NOT named "
              f"attention_decode_floor_sparse_mla on purpose: the kernel reads that name "
              f"today as the floor of the committed single-rate pair, which was fitted over "
              f"the {partwide_floor:g} us part-wide floor, and would silently re-price it. "
              + form + fit_txt + tail),
        lines("attention_decode_split_floor_sparse_mla", f"{f['split']}", "us_per_transfer",
              "measured", True, scope, src,
              f"{f['split']} us added to the latency branch when any query in the call reads "
              f"more than one block: the step the rows show between one block and two (the "
              f"split-KV combine launch FlashMLA adds once it splits a request across SMs), "
              f"after which a block costs little. Every V4.1 compressed layer past about 64 "
              f"tokens reads more than one block; the window-only layers read one only up to "
              f"64 tokens. See attention_decode_launch_floor_sparse_mla for the form and the "
              f"errors. {part['e2e']} Reproduce with scripts/fit_deepseek_v41_decode.py."),
        lines("attention_decode_serial_rate_sparse_mla", f"{f['serial_rate']}.0",
              "bytes_per_us", "measured", True, scope, src,
              f"{f['serial_rate']:,} B/us over the largest request's {rec:,}-byte blocks: the "
              f"cost of each further block on the latency branch's critical path, "
              f"{rec / f['serial_rate']:.3f} us a block. NOT A BANDWIDTH -- "
              f"{f['serial_rate'] / peak:.3f} of this part's {peak / 1e6:g} TB/s -- but a "
              f"per-block latency at a batch too small to fill the part, expressed in the "
              f"registry's rate unit; B is the largest blocks_q, a choice the sweep's uniform "
              f"rows cannot test. See attention_decode_launch_floor_sparse_mla for the form "
              f"and the errors. {part['e2e']} Reproduce with "
              f"scripts/fit_deepseek_v41_decode.py."),
        lines("attention_decode_per_token_sparse_mla", f"{f['per_token']}", "us_per_token",
              "measured", True, scope, src,
              f"{f['per_token']} us per decode query row on the throughput branch: each row's "
              f"own traffic, its 64 padded heads of query in and output out, independent of "
              f"how many blocks it reads. It is the decode counterpart of "
              f"attention_prefill_per_token_sparse_mla and lands near it. Charged per query "
              f"ROW (a request under speculation contributes 1 + num_spec). See "
              f"attention_decode_launch_floor_sparse_mla for the form and the errors. "
              f"{part['e2e']} Reproduce with scripts/fit_deepseek_v41_decode.py."),
        lines("attention_decode_throughput_rate_sparse_mla", f"{f['throughput_rate']}.0",
              "bytes_per_us", "measured", True, scope, src,
              f"{f['throughput_rate'] / 1e6:.2f} TB/s, {f['throughput_rate'] / peak:.3f} of "
              f"this part's {peak / 1e6:g} TB/s datasheet bandwidth, over every query row's "
              f"blocks of {rec:,} B on the throughput branch, i.e. once the call fills the "
              f"part. Above the committed single rate ({rate / 1e6:.2f} TB/s) because the "
              f"per-row cost now carries what that rate had to absorb. Identified by the "
              f"batch >= 256 rows at 1-3 blocks only; charged at {g['v41_blocks']} blocks it "
              f"is extrapolated on the block axis. See "
              f"attention_decode_launch_floor_sparse_mla for the form and the errors. "
              f"{part['e2e']} Reproduce with scripts/fit_deepseek_v41_decode.py."),
    ]


def render(text: str, attn: list) -> str:
    blocks = cb.split(text)
    for e in attn:
        cb.upsert(blocks, e, after=lambda b: cb.name_of(b) is not None)
    return cb.join(blocks)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=base.DEFAULT_DATA)
    ap.add_argument("--catalog", default=base.DEFAULT_CATALOG)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the set would change; write nothing")
    ap.add_argument("--part", choices=sorted(PARTS),
                    help="fit one part only (default: every part)")
    args = ap.parse_args(argv[1:])
    data, catalog = Path(args.data), Path(args.catalog)
    g = base.check_geometry(data)
    chips = list(PARTS) if args.part is None else [args.part]
    attn = []
    for chip in chips:
        peak, _ = base.catalog_facts(catalog, chip)
        floor = base.committed(ATTN_PATH, "attention_decode_floor", chip)
        rate = model_rate(chip)
        f = fit_part(data, PARTS[chip]["sku"], g, floor, rate)
        h = f["holds"]
        print(f"[{chip}] floor={f['floor']} split={f['split']} serial={f['serial_rate']:,} "
              f"per_token={f['per_token']} throughput={f['throughput_rate']:,} B/us  "
              f"{f['err']:.3f}x signed {f['signed']:.3f} (n={f['n']}/{f['n_all']}); "
              f"committed pair {f['old'][0]:.3f}x")
        for k, v in h.items():
            print(f"    hold {v[0]:<22} n={v[1]:3d} {v[2]:.3f}x signed {v[3]:.3f}  "
                  f"(committed {v[4]:.3f}x)")
        print(f"    by batch {fmt_signed(f['by_batch'])}")
        print(f"    at {g['v41_blocks']} blocks {f['at10'][1]:.3f}x; knee {f['knee']:.0f} rows")
        attn += entries(chip, g, f, peak, floor, rate)
    a0 = ATTN_PATH.read_text(encoding="utf-8")
    a1 = render(a0, attn)
    if args.check:
        if a0 != a1:
            print(f"would change: {ATTN_PATH.name}", file=sys.stderr)
            return 1
        print("the set already matches")
        return 0
    ATTN_PATH.write_text(a1, encoding="utf-8")
    print("the set written")
    return 0


def model_rate(chip: str) -> float:
    """The committed single-rate entry, scoped to this part and model."""
    import yaml
    doc = yaml.safe_load(ATTN_PATH.read_text())
    for e in doc["coefficients"]:
        ((n, body),) = e.items()
        if (n == "attention_decode_rate_sparse_mla"
                and body["scope"] == {"hardware": [chip], "model": [base.MODEL]}):
            return float(body["value"])
    raise SystemExit(f"{chip}: no committed attention_decode_rate_sparse_mla for V4.1")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
