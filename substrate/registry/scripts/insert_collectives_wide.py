#!/usr/bin/env python3
"""Add the 8- and 16-rank collective coefficients GB200's NCCL tree measures.

WHAT THIS CLOSES. GB200-NVL72 shipped with widths 2 and 4 only, so a tp=8 or EP=16
deployment on it could not be priced at all -- the kernel refuses an unmeasured width
rather than substituting a narrower floor, which is correct and which left a routine
rack-scale deployment unpriceable. FPM's own `nvidia--GLM-5.2-NVFP4/gb200/.../dep16`
configuration is exactly that case: expert width 16 on gb200.

The data was there the whole time. `gb200/comm/nccl/2.23/nccl_perf.parquet` carries all
four operations at 2, 4, 8 AND 16 ranks, and `fit_collectives.py` fits them unmodified.
An earlier reading of this gap concluded that "NCCL sweeps carry only 2, 4 and 8 ranks"
and that a 16-rank figure needed new measurement; that was wrong, and it was wrong
because the DIRECTORY NAME is not the collection version -- see the caveat below.

THE VERSION MIX, WHICH IS THE WHOLE CAVEAT.

    gb200/comm/nccl/2.23/nccl_perf.parquet, by its own `version` column:
        version 2.29.2  ->  widths 2, 4       (840 + 771 rows)
        version 2.27.7  ->  widths 8, 16      (168 + 168 rows)

So the file named `2.23` contains neither 2.23 nor one single version. The committed 2-
and 4-rank gb200 entries cite 2.29.2, and the 8- and 16-rank rows this script fits are
2.27.7. That breaks the one-collection-per-part rule this registry otherwise holds, and
the rule exists for a good reason: globbing collections mixes engine versions and makes a
difference between parts a software release rather than silicon.

It is accepted here deliberately, with the mix stated on every entry, because:

  * The alternative is no coefficient at all, and the kernel then refuses to price a
    deployment FPM has measured. A stated version mix is better evidence than absence.
  * The cross-version disagreement is MEASURED, not assumed. B200 is the one part whose
    same shapes were swept at three NCCL minor versions (2.27.5, 2.28.9, 2.29.2), and on
    the SHIPPED ESTIMATORS -- a min over the flat region and a max rate, both order
    statistics rather than individual points -- the versions agree to a median 1.01x-1.06x
    and p10-p90 within +/-8%. Raw per-point ratios scatter 0.50-1.46x; the estimators do
    not, which is the estimators doing their job.
  * The direction is systematic and known: 2.29.2 floors are the lowest in all 24 (op,
    dtype, width) cells on B200, so an older-version floor runs slightly HIGH. These
    entries therefore over-state a wide collective's floor marginally rather than
    under-stating it.
  * Width-2 floors are the worst case for cross-version agreement (up to 1.205x on B200)
    and no width-2 entry is touched here. The widths this adds are 8 and 16, where
    agreement is 1.033x-1.090x.

PROVENANCE RULE, learned the hard way: cite the `version` COLUMN, never the directory.
Every citation this writes names the column value.

Usage:
    python scripts/insert_collectives_wide.py --chip gb200-nvl72 --sku gb200 --check
    python scripts/insert_collectives_wide.py --chip gb200-nvl72 --sku gb200
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-collectives.yaml"
DEFAULT_DATA = os.environ.get(
    "AISIMULATE_DATA",
    "/private/tmp/aisim/python/aisimulate/src/aisimulate_core/systems/data")

FLAT_REGION_BYTES = 4096
OPS = ("all_reduce", "all_gather", "reduce_scatter", "alltoall")
DTYPES = {"fp16": "half", "int8": "int8"}
# The widths this script adds. 2 and 4 are already committed from 2.29.2 and are left
# alone: re-fitting them from a different version would change a shipped value for no
# gain, and width 2 is where cross-version agreement is weakest.
WIDTHS = (8, 16)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


fcv = _load("fit_collectives_vllm", HERE / "fit_collectives_vllm.py")


def rows_by_key(data: Path, sku: str, coll: str):
    """NCCL rows keyed (op, dtype, width) -> (points, version), latency in us."""
    import pyarrow.parquet as pq

    path = data / sku / "comm" / "nccl" / coll / "nccl_perf.parquet"
    if not path.is_file():
        raise SystemExit(f"no nccl_perf.parquet under {path.parent}")
    d = pq.read_table(path).to_pydict()
    if "version" not in d:
        raise SystemExit(f"{path}: no version column; cannot cite a collection")
    pts: dict[tuple[str, str, int], list[tuple[float, float]]] = (
        collections.defaultdict(list))
    vers: dict[tuple[str, str, int], set[str]] = collections.defaultdict(set)
    for i in range(len(d["latency"])):
        lat = float(d["latency"][i])
        if lat <= 0:
            continue
        key = (str(d["op_name"][i]), str(d["nccl_dtype"][i]), int(d["num_gpus"][i]))
        pts[key].append((float(d["message_size"][i]), lat * 1e3))
        vers[key].add(str(d["version"][i]))
    return pts, vers


def entry(name: str, value, units: str, fitted: bool, chip: str, cite: str,
          rationale: str) -> list[str]:
    return [
        f"  - {name}:",
        f"      value: {value}",
        f"      units: {units}",
        "      method: measured",
        f"      fitted: {'true' if fitted else 'false'}",
        f"      scope: {{hardware: [{chip}]}}",
        "      sources:",
        f'        - {{kind: model, cite: "{cite}", role: primary}}',
        "      rationale: >",
        f"        {rationale}",
    ]


CAVEAT = (
    "VERSION MIX, STATED: this part's committed 2- and 4-rank entries cite NCCL 2.29.2, "
    "and this entry is NCCL {ver}. Both live in the directory named `2.23`, whose rows "
    "self-report two different versions -- the path is not the provenance, the column "
    "is. One collection per part is the rule this registry otherwise holds, and it is "
    "broken here deliberately because the alternative is no coefficient and a "
    "deployment the kernel then refuses to price. The disagreement is measured rather "
    "than assumed: on B200, where the same shapes exist at three NCCL minor versions, "
    "the shipped estimators agree to a median 1.01x-1.06x with p10-p90 inside +/-8%, "
    "and 2.29.2 is the fastest in all 24 cells -- so an older-version floor runs "
    "slightly HIGH and this entry over-states rather than under-states. Width-2 floors "
    "are the worst case for cross-version agreement (to 1.205x) and are not touched."
)


def build(data: Path, chip: str, sku: str, coll: str) -> tuple[list[str], dict]:
    key = chip.replace("-", "_")
    pts, vers = rows_by_key(data, sku, coll)
    lines = [f"  # --- {chip} wide-group collectives ({sku}, NCCL version column "
             f"2.27.7; see each entry's version-mix caveat) ---"]
    summary: dict = collections.defaultdict(list)
    for op in OPS:
        for dt_tag, dt_col in DTYPES.items():
            for w in WIDTHS:
                k = (op, dt_col, w)
                if k not in pts:
                    continue
                g = fcv.Group("nccl", w, pts[k])
                if not g.usable:
                    continue
                ver = "/".join(sorted(vers[k]))
                stem = f"{op}_{dt_tag}_{w}rank_{key}"
                cite = (f"NVIDIA AISimulate systems/data/{sku}/comm/nccl/{coll}/"
                        f"nccl_perf.parquet (version column: {ver}, NVIDIA "
                        f"{chip.upper()}), {op}, {dt_col}, {w} ranks, "
                        f"{len(g.points)} messages")
                cav = CAVEAT.format(ver=ver)
                lines += entry(
                    f"collective_floor_{stem}", round(g.floor_us, 2),
                    "us_per_transfer", False, chip, cite,
                    f"Minimum latency over the {g.flat_count} messages at or below 4 "
                    f"KiB, where latency does not depend on size. An order statistic "
                    f"rather than a fit: a floor is a lower bound on achievable "
                    f"latency, so a high outlier must not raise it. NCCL is vLLM's own "
                    f"path for {op} -- CustomAllreduce implements all-reduce only. "
                    f"{cav}")
                lines += entry(
                    f"collective_peak_rate_{stem}", round(g.peak_bytes_per_us, 1),
                    "bytes_per_us", False, chip, cite,
                    f"Maximum message_size/latency over {len(g.points)} points, an "
                    f"order statistic. A ceiling on throughput; the transition rate "
                    f"below is what prices a forward pass. {cav}")
                lines += entry(
                    f"collective_transition_rate_{stem}",
                    round(g.transition_rate, 1), "bytes_per_us", True, chip, cite,
                    f"The rate in the transition region a forward pass's messages land "
                    f"in, fitted by grid search against the three-parameter form the "
                    f"kernel evaluates. Geometric error "
                    f"{g.error_three_param:.3f}x against "
                    f"{g.error_two_param:.3f}x for the two-parameter form. {cav}")
                summary[op].append((dt_tag, w, round(g.floor_us, 2),
                                    round(g.error_three_param, 3)))
    return lines, summary


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip", required=True)
    ap.add_argument("--sku", required=True)
    ap.add_argument("--collection", default="2.23",
                    help="directory under comm/nccl; its version COLUMN is what gets "
                         "cited (default 2.23, which holds the 8- and 16-rank rows)")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv[1:])

    text = SET_PATH.read_text(encoding="utf-8")
    key = args.chip.replace("-", "_")
    present = set()
    for m in re.finditer(r"^  - collective_\w+_(\d+)rank_" + re.escape(key) + r":$",
                         text, re.M):
        present.add(int(m.group(1)))
    clash = present & set(WIDTHS)
    if clash:
        print(f"{args.chip}: widths {sorted(clash)} already committed; refusing to "
              f"overwrite", file=sys.stderr)
        return 1

    lines, summary = build(Path(args.data), args.chip, args.sku, args.collection)
    n = sum(1 for x in lines if x.startswith("  - "))
    for op in OPS:
        for dt, w, floor, err in summary.get(op, []):
            print(f"  {op:15s} {dt:5s} {w:2d}rank  floor={floor:7.2f}us  "
                  f"3-param err={err}x")
    print(f"  -> {n} entries")
    if args.check:
        return 0
    if not n:
        print("nothing to insert", file=sys.stderr)
        return 1

    body = text.split("\n")
    at = len(body)
    count_at = None
    for i in range(len(body) - 1, -1, -1):
        if body[i].startswith("# ") and "entries" in body[i]:
            count_at, at = i, i
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
