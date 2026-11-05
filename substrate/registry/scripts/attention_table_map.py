#!/usr/bin/env python3
"""The mapping from this project's attention kinds to NVIDIA's measurement families.

Shared by scripts/fit_attention_by_kind.py and
scripts/validate_against_aisimulate_tables.py so the two cannot disagree about which rows
describe which kind -- a disagreement there would fit one thing and validate another.

# Why a mapping is needed at all

`blis-schemas` models four attention kinds (gqa, swa, mla, sparse_mla) and prices them with
one law. NVIDIA collects them in DIFFERENT families, with different schemas:

    attention/           full and windowed attention; `window_size` separates them
    mla/                 multi-head latent attention
    sparse_attention/    sparse MLA (an index selects which KV to read)
    linear_attention/    recurrent mixers, by `model_name`
    kda/                 Kimi delta attention

So a kind is a (family, filter) pair, not a family. Getting that pair wrong is silent: the fit
succeeds and describes the wrong kernel.

# The lane rule

Every family carries `kernel_source`. Latencies differ by up to 5x between lanes for the same
shape -- measured on the b200 MoE sweep, where pooling the two lanes gave a
prediction-over-measurement ratio of 0.309 against 0.625 for the lane a latency-sensitive
engine actually picks. Lanes are therefore never pooled: a caller states the lane, or takes
the fastest, and the choice is recorded with the fit.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pyarrow.parquet as pq

DEFAULT_DATA = os.environ.get("AISIMULATE_DATA", "/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")

# AISimulate SKU directory -> catalog chip name. Same mapping the other fitters use.
SKUS = {
    "h100_sxm": "h100",
    "h200_sxm": "h200",
    "a100_sxm": "a100-sxm",
    "l40s": "l40s",
    "gb200": "gb200-nvl72",
    "b200_sxm": "b200",
    "b300_sxm": "b300",
    "gb300": "gb300",
}

# One entry per attention kind this project models.
#
#   family     the parquet directory under <data>/<sku>/
#   file       the parquet basename within it
#   where      row filter, as (column, predicate) pairs; every one must hold
#   context    the column carrying the KV context length for a decode row
#   heads      the column carrying the query-head count
#
# `step` rather than `isl` is the context for a generation row: `isl` is 1 on every
# generation row in these sweeps and `step` runs to 131071. Reading `isl` as the context
# makes the predicted KV read about 2000x too small and collapses the prediction onto the
# floor -- it produced 0.754 where the correct reading gives 1.294.
KINDS = {
    "gqa": {
        "family": "attention",
        "file": "generation_attention_perf.parquet",
        "where": [("window_size", lambda v: v == 0)],
        "context": "step",
        "heads": "num_heads",
        "kv_heads": "num_key_value_heads",
        "head_dim": "head_dim",
    },
    "swa": {
        "family": "attention",
        "file": "generation_attention_perf.parquet",
        "where": [("window_size", lambda v: v > 0)],
        "context": "step",
        "heads": "num_heads",
        "kv_heads": "num_key_value_heads",
        "head_dim": "head_dim",
    },
    "mla": {
        "family": "mla",
        "file": None,  # resolved by glob: the family's generation file varies by collection
        "where": [],
        "context": "step",
        "heads": "num_heads",
        "kv_heads": None,   # MLA has one latent KV head by construction
        "head_dim": None,   # carried by the model/architecture columns, not a width
    },
    "sparse_mla": {
        "family": "sparse_attention",
        "file": None,
        "where": [],
        "context": "step",
        "heads": "num_heads",
        "kv_heads": None,
        "head_dim": None,
    },
}


# The collection each family is PINNED to, per the same rule the registry's other fitters
# follow: one framework and one version, so a difference between two parts is the silicon
# rather than a software change. Globbing every collection mixes frameworks and versions and
# silently changes the row set -- it produced 66,148 rows where the committed fit used 40,367,
# which is how this pin came to be written.
COLLECTIONS = {
    "attention": "trtllm/1.3.0rc20",
    "mla": "trtllm/1.3.0rc20",
    "sparse_attention": "trtllm/1.3.0rc20",
    "linear_attention": "trtllm/1.3.0rc20",
    "kda": "trtllm/1.3.0rc20",
}


def parquets(data: str, sku: str, family: str, name: str | None,
             collection: str | None = None) -> list[str]:
    """Every parquet for one family, within the pinned collection.

    `collection` overrides the pin, for a caller deliberately comparing collections. Passing
    None uses the pin; passing "" globs every collection, which no fit should do.
    """
    if collection is None:
        collection = COLLECTIONS.get(family, "")
    parts = [data, sku, family]
    if collection:
        parts.append(collection)
    else:
        parts.append("**")
    parts.append(name or "*.parquet")
    return sorted(glob.glob(os.path.join(*parts), recursive=True))


def load(path: str) -> dict:
    return pq.read_table(path).to_pydict()


def rows_for_kind(data: str, sku: str, kind: str, lane: str | None = None,
                  collection: str | None = None) -> list[dict]:
    """Return the measured rows describing one attention kind on one part.

    Each row is a plain dict of the columns the fit needs, plus `latency` in MILLISECONDS as
    the parquet stores it. Filtering happens here so the fitter and the validator cannot
    disagree about which rows belong to a kind.
    """
    spec = KINDS[kind]
    out: list[dict] = []
    for path in parquets(data, sku, spec["family"], spec["file"], collection):
        t = load(path)
        n = len(t.get("latency", []))
        for i in range(n):
            if t["latency"][i] <= 0:
                continue
            if lane is not None and t.get("kernel_source", [None] * n)[i] != lane:
                continue
            ok = True
            for col, pred in spec["where"]:
                if col not in t or not pred(t[col][i]):
                    ok = False
                    break
            if not ok:
                continue
            row = {
                "latency_ms": t["latency"][i],
                "lane": t.get("kernel_source", [""] * n)[i],
                "batch": t["batch_size"][i],
                "context": t[spec["context"]][i],
                "heads": t[spec["heads"]][i],
                "kv_cache_dtype": t.get("kv_cache_dtype", [""] * n)[i],
                "window": t.get("window_size", [0] * n)[i],
                "source": os.path.basename(os.path.dirname(os.path.dirname(path))),
            }
            for key in ("kv_heads", "head_dim"):
                col = spec[key]
                row[key] = t[col][i] if col and col in t else None
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--sku", default="h200_sxm")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(args.data, args.sku)):
        print(f"error: no data for SKU {args.sku} under {args.data}", file=sys.stderr)
        return 1

    failures = []
    print(f"attention-kind coverage for {args.sku} ({SKUS.get(args.sku, '?')})")
    print(f"{'kind':12s} {'family':18s} {'rows':>9s}  lanes")
    for kind, spec in KINDS.items():
        files = parquets(args.data, args.sku, spec["family"], spec["file"])
        if not files:
            print(f"{kind:12s} {spec['family']:18s} {'ABSENT':>9s}")
            failures.append(f"{kind}: family {spec['family']} has no parquet")
            continue
        # Every filter column must EXIST, or the filter silently admits nothing.
        missing = []
        t = load(files[0])
        for col, _ in spec["where"]:
            if col not in t:
                missing.append(col)
        for key in ("context", "heads"):
            if spec[key] not in t:
                missing.append(spec[key])
        if missing:
            print(f"{kind:12s} {spec['family']:18s} {'SCHEMA':>9s}  missing {missing}")
            failures.append(f"{kind}: columns absent: {missing}")
            continue
        rows = rows_for_kind(args.data, args.sku, kind)
        lanes = sorted({r["lane"] for r in rows})
        print(f"{kind:12s} {spec['family']:18s} {len(rows):9,d}  {lanes}")
        if not rows:
            failures.append(f"{kind}: mapping yields zero rows")

    print()
    if failures:
        print(f"FAIL: {len(failures)} mapping problem(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("PASS: every modelled kind maps to a present family with rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
