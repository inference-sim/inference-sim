"""Report generation — formatted tables and CSV export from error records.

Functions
---------
format_aggregate_table(records)
    MAPE summary grouped by simulator.
format_per_model_table(records)
    MAPE grouped by model × simulator.
format_per_workload_table(records)
    MAPE grouped by workload × simulator.
format_signed_error_table(records)
    MPE summary grouped by simulator.
generate_report(records, output_dir)
    Print all tables and save a flat CSV of all error records.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

from experiment.metrics import ErrorRecord, RuntimeRecord

# Ordered metric columns for the tables.
_METRIC_COLS = [
    "e2e_mean", "e2e_p90", "e2e_p99",
    "ttft_mean", "ttft_p90", "ttft_p99",
    "itl_mean", "itl_p90", "itl_p99",
]


def _group_and_average(
    records: list[ErrorRecord],
    group_key: str,
    value_attr: str,
) -> dict[str, dict[str, float | None]]:
    """Group records by (group_key, metric_name) and compute mean of value_attr.

    Returns ``{group_value: {metric_name: mean_value}}``.
    """
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        key = getattr(r, group_key)
        sums[key][r.metric_name] += getattr(r, value_attr)
        counts[key][r.metric_name] += 1

    result: dict[str, dict[str, float | None]] = {}
    for key in sorted(sums):
        result[key] = {}
        for m in _METRIC_COLS:
            c = counts[key].get(m, 0)
            result[key][m] = sums[key].get(m, 0) / c if c else None
    return result


def _format_table(
    grouped: dict[str, dict[str, float | None]],
    row_label: str,
    value_fmt: str = "{:.2f}",
) -> str:
    """Format grouped data into an aligned text table."""
    if not grouped:
        return f"(no data for {row_label})"
    col_widths = [max(len(row_label), max(len(k) for k in grouped))]
    for m in _METRIC_COLS:
        col_widths.append(max(len(m), 8))

    header = f"{row_label:<{col_widths[0]}}"
    for i, m in enumerate(_METRIC_COLS):
        header += f"  {m:>{col_widths[i + 1]}}"

    sep = "-" * len(header)
    lines = [header, sep]

    for key, metrics in grouped.items():
        row = f"{key:<{col_widths[0]}}"
        for i, m in enumerate(_METRIC_COLS):
            v = metrics.get(m)
            val = "N/A" if v is None else value_fmt.format(v)
            row += f"  {val:>{col_widths[i + 1]}}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_aggregate_table(records: list[ErrorRecord]) -> str:
    """MAPE summary grouped by simulator."""
    grouped = _group_and_average(records, "simulator", "mape")
    return _format_table(grouped, "Simulator")


def format_per_model_table(records: list[ErrorRecord]) -> str:
    """MAPE grouped by model."""
    grouped = _group_and_average(records, "model", "mape")
    return _format_table(grouped, "Model")


def format_per_workload_table(records: list[ErrorRecord]) -> str:
    """MAPE grouped by workload."""
    grouped = _group_and_average(records, "workload", "mape")
    return _format_table(grouped, "Workload")


def format_signed_error_table(records: list[ErrorRecord]) -> str:
    """MPE summary grouped by simulator (signed errors)."""
    grouped = _group_and_average(records, "simulator", "mpe")
    return _format_table(grouped, "Simulator")


def save_csv(records: list[ErrorRecord], output_path: str) -> str:
    """Save all error records as a flat CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "simulator", "experiment_folder", "model", "workload",
        "stage_index", "metric_name", "predicted", "actual",
        "mape", "mpe", "absolute_error",
        "exp_id", "hardware", "dp", "cpu_offload", "gpu_mem_util", "precision",
        "tp", "max_num_batched_tokens",
    ]
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "simulator": r.simulator,
                "experiment_folder": r.experiment_folder,
                "model": r.model,
                "workload": r.workload,
                "stage_index": r.stage_index,
                "metric_name": r.metric_name,
                "predicted": r.predicted,
                "actual": r.actual,
                "mape": r.mape,
                "mpe": r.mpe,
                "absolute_error": r.absolute_error,
                "exp_id": r.exp_id,
                "hardware": r.hardware,
                "dp": r.dp if r.dp is not None else "",  # None → "" → NaN in pandas
                "cpu_offload": r.cpu_offload,
                "gpu_mem_util": r.gpu_mem_util,
                "precision": r.precision,
                "tp": r.tp,
                "max_num_batched_tokens": r.max_num_batched_tokens,
            })
    return output_path


def format_runtime_table(runtime_records: list[RuntimeRecord]) -> str:
    """Format runtime records into a summary table grouped by simulator.

    Shows Mean, Min, Max, Total seconds and run count per simulator.
    """
    if not runtime_records:
        return "(no runtime data)"

    # Group by simulator
    by_sim: dict[str, list[float]] = defaultdict(list)
    for r in runtime_records:
        by_sim[r.simulator].append(r.wall_clock_seconds)

    # Column headers and widths
    headers = ["Simulator", "Mean(s)", "Min(s)", "Max(s)", "Total(s)", "Runs"]
    col_widths = [max(len(headers[0]), max(len(s) for s in by_sim))]
    for h in headers[1:]:
        col_widths.append(max(len(h), 8))

    header_line = f"{headers[0]:<{col_widths[0]}}"
    for i, h in enumerate(headers[1:], 1):
        header_line += f"  {h:>{col_widths[i]}}"

    sep = "-" * len(header_line)
    lines = [header_line, sep]

    for sim in sorted(by_sim):
        times = by_sim[sim]
        mean_v = sum(times) / len(times)
        min_v = min(times)
        max_v = max(times)
        total_v = sum(times)
        count = len(times)
        row = f"{sim:<{col_widths[0]}}"
        row += f"  {mean_v:>{col_widths[1]}.2f}"
        row += f"  {min_v:>{col_widths[2]}.2f}"
        row += f"  {max_v:>{col_widths[3]}.2f}"
        row += f"  {total_v:>{col_widths[4]}.2f}"
        row += f"  {count:>{col_widths[5]}}"
        lines.append(row)

    return "\n".join(lines)


def save_runtime_csv(runtime_records: list[RuntimeRecord], output_path: str) -> str:
    """Save runtime records as a flat CSV — one row per (adapter, experiment)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "simulator", "experiment_folder", "model", "workload", "wall_clock_seconds",
        "exp_id", "hardware", "dp", "cpu_offload", "gpu_mem_util", "precision",
        "tp", "max_num_batched_tokens",
    ]
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in runtime_records:
            writer.writerow({
                "simulator": r.simulator,
                "experiment_folder": r.experiment_folder,
                "model": r.model,
                "workload": r.workload,
                "wall_clock_seconds": r.wall_clock_seconds,
                "exp_id": r.exp_id,
                "hardware": r.hardware,
                "dp": r.dp if r.dp is not None else "",
                "cpu_offload": r.cpu_offload,
                "gpu_mem_util": r.gpu_mem_util,
                "precision": r.precision,
                "tp": r.tp,
                "max_num_batched_tokens": r.max_num_batched_tokens,
            })
    return output_path


def generate_report(
    records: list[ErrorRecord],
    output_dir: str,
    runtime_records: list[RuntimeRecord] | None = None,
) -> None:
    """Print all tables to stdout and save CSV to output_dir."""
    if not records:
        print("No error records to report.")
        return

    print("\n=== MAPE by Simulator ===\n")
    print(format_aggregate_table(records))

    print("\n=== MAPE by Model ===\n")
    print(format_per_model_table(records))

    print("\n=== MAPE by Workload ===\n")
    print(format_per_workload_table(records))

    print("\n=== MPE by Simulator (signed) ===\n")
    print(format_signed_error_table(records))

    csv_path = os.path.join(output_dir, "error_records.csv")
    save_csv(records, csv_path)
    print(f"\nCSV saved to {csv_path}")

    if runtime_records:
        print("\n=== Simulator Runtime ===\n")
        print(format_runtime_table(runtime_records))
        runtime_csv_path = os.path.join(output_dir, "runtime.csv")
        save_runtime_csv(runtime_records, runtime_csv_path)
        print(f"\nRuntime CSV saved to {runtime_csv_path}")
