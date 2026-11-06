#!/usr/bin/env python3
"""Re-derive the GEMM efficiency envelope and MoE imbalance from AISimulate data.

Every fitted coefficient in cost-model-primitives-h200.yaml comes from here, so
each one is reproducible: point this at the collection directory the entry cites
and it prints the entry's value, interval and row count. A fitted constant whose
fit cannot be re-run is not a measurement, it is a number someone wrote down.

Reads both layouts AISimulate has shipped: the newer one stores a collection as
`gemm_perf.parquet` under `<sku>/gemm/<framework>/<version>/`, the older one as
`gemm_perf.txt` (CSV) under `<sku>/<framework>/<version>/`. The registry's H200
entries cite the parquet collection; the CSV path is kept because the older tree
is still what some checkouts have, and a reader with only that tree should get a
number rather than an error.

Usage:
    python scripts/fit_gemm_envelope.py <data>/h200_sxm/gemm/trtllm/1.3.0rc20
    python scripts/fit_gemm_envelope.py <data>/h200_sxm/gemm/trtllm/1.3.0rc20 \
        --moe <data>/h200_sxm/moe/trtllm/1.3.0rc20
"""

from __future__ import annotations

import collections
import csv
import math
import os
import statistics
import sys
from pathlib import Path

import yaml

# Which catalog field holds the dense peak for a dtype. The fit is a fraction of
# peak, so the peak it is a fraction of has to come from the same place the cost
# model will read it from — the catalog — rather than being restated here. A dtype
# with no native peak on a part has no envelope to fit: the hardware reaches it
# only through a dequantize path, and a fraction of a rate the silicon does not
# have is not a meaningful number.
PEAK_FIELD = {
    "float16": "TFlopsPeak",
    "bfloat16": "TFlopsPeak",
    "fp8": "TFlopsFP8",
    "fp8_block": "TFlopsFP8",
    "nvfp4": "TFlopsNVFP4",
}

# Maps an AISimulate SKU directory to the catalog chip file that states its peaks.
# Named explicitly rather than pattern-matched: an SKU whose mapping is a guess
# would be fitted against the wrong peak and the error would be invisible.
SKU_TO_CHIP = {
    "h100_sxm": "h100",
    "h200_sxm": "h200",
    "a100_sxm": "a100-sxm",
    "a100_pcie": "a100-80",
    "l40s": "l40s",
    "gb200": "gb200-nvl72",
    "b200_sxm": "b200",
    "b300_sxm": "b300",
}


def peaks_from_catalog(catalog: Path, chip: str) -> dict[str, float]:
    """Read the dense peaks for one chip, in FLOP/s, from the catalog."""
    facts = yaml.safe_load((catalog / "hardware" / f"{chip}.yaml").read_text())
    peaks = {}
    for dtype, field in PEAK_FIELD.items():
        tflops = facts.get(field, 0) or 0
        if tflops > 0:
            peaks[dtype] = tflops * 1e12
    return peaks


def read_rows(base: Path, stem: str) -> list[dict] | None:
    """Read one measurement collection, in whichever layout is present.

    Returns None when neither file exists, so a caller can report the miss
    against the path it was given rather than raising from inside a reader.
    """
    parquet = base / f"{stem}.parquet"
    if parquet.is_file():
        import pyarrow.parquet as pq  # only needed for the newer layout

        table = pq.read_table(parquet).to_pydict()
        columns = list(table)
        return [
            {c: table[c][i] for c in columns} for i in range(len(table[columns[0]]))
        ]
    csv_path = base / f"{stem}.txt"
    if csv_path.is_file():
        return list(csv.DictReader(csv_path.open()))
    return None


def envelope(rows: list[dict], dtype: str, peak: float) -> dict[int, float]:
    """Return the highest achieved fraction of peak at each M.

    The envelope rather than the mean: a cost model predicts what a well-shaped
    GEMM achieves, and averaging in the badly-shaped (N, K) pairs measured at the
    same M would fit a curve no real layer sits on.
    """
    best: dict[int, float] = collections.defaultdict(float)
    for r in rows:
        if r["gemm_dtype"] != dtype:
            continue
        m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
        latency_s = float(r["latency"]) * 1e-3  # the file records milliseconds
        if latency_s <= 0:
            continue
        best[m] = max(best[m], (2 * m * n * k / latency_s) / peak)
    return dict(best)


def fit_ramp(points: dict[int, float]) -> tuple[float, int, float]:
    """Least-squares fit of eff(M) = eps_max * M / (M + M_half).

    Returns (eps_max, M_half, rms_residual). Searched on a grid rather than
    solved: the model is two-parameter and the grid is exhaustive over the range
    any plausible value lies in, so this is reproducible to the printed precision
    without a solver dependency.
    """
    pts = sorted(points.items())
    best: tuple[float, float, int] | None = None
    # The asymptote is a fraction of peak, so it cannot exceed 1. Leaving the grid
    # unbounded let A100's bf16 fit land at 1.003 — not a slightly-too-high number
    # but a signal, since a curve that extrapolates past peak means the peak the
    # fit divides by does not describe the measured kernels. Bounding it here turns
    # that into a pinned 1.000 with a visible residual, which a reader can see.
    for eps_milli in range(300, 1001):
        eps = eps_milli / 1000
        for m_half in range(20, 500):
            ss = sum((eps * m / (m + m_half) - y) ** 2 for m, y in pts)
            if best is None or ss < best[0]:
                best = (ss, eps, m_half)
    assert best is not None
    ss, eps, m_half = best
    return eps, m_half, math.sqrt(ss / len(pts))


def report_gemm(rows: list[dict], peaks: dict[str, float]) -> None:
    device = rows[0]["device"]
    framework = f"{rows[0]['framework']} {rows[0]['version']}"
    print(f"# gemm_perf: {device}, {framework}, {len(rows)} rows")
    # Both spellings of the 16-bit format appear across collections; report
    # whichever the data uses rather than silently skipping a dtype.
    measured = {r["gemm_dtype"] for r in rows}
    for dtype in ("bfloat16", "float16", "fp8", "fp8_block", "nvfp4"):
        if dtype not in measured:
            continue
        if dtype not in peaks:
            print(
                f"{dtype:10s} measured, but this part states no peak for it — the "
                f"rate is reached through a dequantize path, so no envelope is fitted"
            )
            continue
        env = envelope(rows, dtype, peaks[dtype])
        if not env:
            continue
        eps, m_half, rms = fit_ramp(env)
        n = sum(1 for r in rows if r["gemm_dtype"] == dtype)
        pinned = " ASYMPTOTE PINNED AT PEAK — check the peak" if eps >= 1.0 else ""
        print(
            f"{dtype:10s} rows={n:6d} M_values={len(env):3d} "
            f"eps_max={eps:.3f} M_half={m_half:3d} rms={rms:.3f} "
            f"ci95=[{max(0.0, eps - rms):.3f},{min(1.0, eps + rms):.3f}] "
            f"highest_observed={max(env.values()):.3f}{pinned}"
        )


def report_moe(rows: list[dict]) -> None:
    """Report the imbalance multiplier: skewed latency over balanced, same shape."""

    def shape(r: dict) -> tuple:
        return (
            r["moe_dtype"], r["num_tokens"], r["hidden_size"], r["inter_size"],
            r["topk"], r["num_experts"], r["moe_tp_size"], r["moe_ep_size"],
            r["kernel_source"],
        )

    balanced = {
        shape(r): float(r["latency"])
        for r in rows
        if r["distribution"] == "balanced"
    }
    print(f"# moe_perf: {rows[0]['device']}, {len(rows)} rows")
    for dist in sorted({r["distribution"] for r in rows} - {"balanced"}):
        ratios = sorted(
            float(r["latency"]) / balanced[shape(r)]
            for r in rows
            if r["distribution"] == dist
            and shape(r) in balanced
            and balanced[shape(r)] > 0
        )
        n = len(ratios)
        if n == 0:
            # No shape was measured under both this distribution and the balanced
            # one, so there is nothing to take a ratio of. Saying so beats either
            # crashing or printing a zero that reads as a measurement.
            print(
                f"{dist:16s} pairs=     0 — no shape measured under both this "
                f"distribution and balanced, so no imbalance ratio is derivable"
            )
            continue
        print(
            f"{dist:16s} pairs={n:6d} median={statistics.median(ratios):.3f} "
            f"p10={ratios[int(0.10 * n)]:.3f} p90={ratios[int(0.90 * n)]:.3f} "
            f"max={ratios[-1]:.1f}"
        )


def main(argv: list[str]) -> int:
    args = argv[1:]
    moe_dir: Path | None = None
    catalog = Path(os.environ.get("BLIS_CATALOG",
                                 str(Path(__file__).resolve().parents[2] / "catalog")))
    if "--moe" in args:
        i = args.index("--moe")
        if i + 1 >= len(args):
            print("--moe needs a directory", file=sys.stderr)
            return 2
        moe_dir = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    if "--catalog" in args:
        i = args.index("--catalog")
        if i + 1 >= len(args):
            print("--catalog needs a directory", file=sys.stderr)
            return 2
        catalog = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    base = Path(args[0])

    chip = chip_for(base)
    if chip is None:
        print(
            f"cannot tell which catalog chip {base} measures; add its SKU to "
            f"SKU_TO_CHIP rather than guessing a peak",
            file=sys.stderr,
        )
        return 1
    peaks = peaks_from_catalog(catalog, chip)

    gemm = read_rows(base, "gemm_perf")
    if gemm is None:
        print(f"no gemm_perf.parquet or gemm_perf.txt under {base}", file=sys.stderr)
        return 1
    print(f"# catalog chip: {chip}, peaks from hardware/{chip}.yaml")
    report_gemm(gemm, peaks)

    # The newer layout keeps MoE in a sibling collection; the older one keeps it
    # beside the GEMM data. Try the explicit path, then the sibling, then beside.
    for candidate in (moe_dir, _sibling(base, "moe"), base):
        if candidate is None:
            continue
        moe = read_rows(candidate, "moe_perf")
        if moe is not None:
            report_moe(moe)
            break
    return 0


def chip_for(base: Path) -> str | None:
    """Return the catalog chip name for a measurement path, or None if unmapped."""
    for part in reversed(base.parts):
        if part in SKU_TO_CHIP:
            return SKU_TO_CHIP[part]
    return None


def _sibling(base: Path, collection: str) -> Path | None:
    """Return the same framework/version under a different collection.

    `<sku>/gemm/<framework>/<version>` becomes `<sku>/moe/<framework>/<version>`.
    Returns None when the path is not in that shape.
    """
    parts = list(base.parts)
    if len(parts) < 3 or parts[-3] not in {"gemm", "moe", "mlp"}:
        return None
    parts[-3] = collection
    return Path(*parts)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
