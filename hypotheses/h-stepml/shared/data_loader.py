"""StepML ground truth data loader.

Loads OpenTelemetry traces (BATCH_SUMMARY events) and per-request lifecycle
metrics from the eval/ground_truth/ experiment directories into pandas
DataFrames suitable for ML training and evaluation.
"""

import json
import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Default data root: eval/ground_truth/ relative to this file's location
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "eval", "ground_truth"
)

# Regex to split directory names: YYYYMMDD-HHMMSS-<model>-tp<N>-<workload>
# The model name can contain hyphens, so we anchor on the -tp<N>- pattern.
_DIR_PATTERN = re.compile(
    r"^(\d{8}-\d{6})-(.+)-tp(\d+)-(\w+)$"
)


def parse_experiment_metadata(dirname: str) -> dict:
    """Parse experiment metadata from a directory name.

    Directory names follow the convention:
        YYYYMMDD-HHMMSS-<model>-tp<N>-<workload>

    The model name may contain hyphens (e.g., llama-2-7b, mixtral-8x7b-v0-1).
    We locate the *last* occurrence of ``-tp<digits>-`` to split model from
    workload, since tp values are always a simple integer and the workload
    name has no hyphens.

    Returns:
        dict with keys: model, tp, workload, timestamp
    """
    m = _DIR_PATTERN.match(dirname)
    if m:
        return {
            "timestamp": m.group(1),
            "model": m.group(2),
            "tp": int(m.group(3)),
            "workload": m.group(4),
        }

    # Fallback: find the *last* -tp<digits>- boundary
    tp_matches = list(re.finditer(r"-tp(\d+)-", dirname))
    if not tp_matches:
        raise ValueError(
            f"Cannot parse experiment metadata from directory name: {dirname}"
        )
    last_tp = tp_matches[-1]
    tp_value = int(last_tp.group(1))

    # Timestamp is the first 15 characters: YYYYMMDD-HHMMSS
    timestamp = dirname[:15]
    # Model is between timestamp+'-' and the tp boundary
    model = dirname[16 : last_tp.start()]
    # Workload is everything after -tp<N>-
    workload = dirname[last_tp.end() :]

    return {
        "timestamp": timestamp,
        "model": model,
        "tp": tp_value,
        "workload": workload,
    }


def _extract_attribute_value(attr: dict):
    """Extract a scalar value from an OpenTelemetry attribute.

    Attribute values are nested like:
        {"key": "step.id", "value": {"intValue": "29"}}
    or:
        {"key": "kv.usage_gpu_ratio", "value": {"doubleValue": 0.02}}

    Note: intValue is often a string in OTLP JSON encoding.
    """
    value_obj = attr.get("value", {})
    if "intValue" in value_obj:
        return int(value_obj["intValue"])
    if "doubleValue" in value_obj:
        return float(value_obj["doubleValue"])
    if "stringValue" in value_obj:
        return value_obj["stringValue"]
    return None


def load_experiment_steps(experiment_dir: str) -> pd.DataFrame:
    """Load step-level BATCH_SUMMARY data from an experiment's traces.json.

    Parses the OpenTelemetry JSONL file line by line, extracts events named
    ``step.BATCH_SUMMARY``, and flattens their attributes into a DataFrame.

    Args:
        experiment_dir: Path to a single experiment directory containing
            traces.json.

    Returns:
        DataFrame with one row per step. Columns include all BATCH_SUMMARY
        attribute keys plus ``experiment_id`` (the directory basename).
    """
    traces_path = os.path.join(experiment_dir, "traces.json")
    experiment_id = os.path.basename(experiment_dir)
    rows = []

    with open(traces_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for resource_span in data.get("resourceSpans", []):
                for scope_span in resource_span.get("scopeSpans", []):
                    for span in scope_span.get("spans", []):
                        for event in span.get("events", []):
                            if event.get("name") != "step.BATCH_SUMMARY":
                                continue
                            row = {}
                            for attr in event.get("attributes", []):
                                key = attr["key"]
                                row[key] = _extract_attribute_value(attr)
                            rows.append(row)

    df = pd.DataFrame(rows)
    df["experiment_id"] = experiment_id

    # Ensure integer columns are proper int types (not object/float)
    int_columns = [
        "step.id",
        "step.ts_start_ns",
        "step.ts_end_ns",
        "step.duration_us",
        "batch.prefill_tokens",
        "batch.decode_tokens",
        "batch.scheduled_tokens",
        "batch.num_prefill_reqs",
        "batch.num_decode_reqs",
        "batch.num_finished",
        "batch.num_preempted",
        "queue.running_depth",
        "queue.waiting_depth",
        "kv.blocks_free_gpu",
        "kv.blocks_total_gpu",
    ]
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Ensure float columns
    float_columns = ["kv.usage_gpu_ratio"]
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


def load_all_experiments(data_root: str = None) -> pd.DataFrame:
    """Load step data from all experiments and concatenate into one DataFrame.

    Iterates over subdirectories in ``data_root``, calls
    :func:`load_experiment_steps` for each, and adds metadata columns
    (model, tp, workload, timestamp) parsed from each directory name.

    Args:
        data_root: Path to the ground truth data directory. Defaults to
            ``eval/ground_truth/`` relative to this file.

    Returns:
        Concatenated DataFrame with all step data plus metadata columns.
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    frames = []
    for dirname in sorted(os.listdir(data_root)):
        dirpath = os.path.join(data_root, dirname)
        if not os.path.isdir(dirpath):
            continue
        # Skip directories that don't look like experiment dirs
        traces_path = os.path.join(dirpath, "traces.json")
        if not os.path.isfile(traces_path):
            continue

        df = load_experiment_steps(dirpath)
        meta = parse_experiment_metadata(dirname)
        df["model"] = meta["model"]
        df["tp"] = meta["tp"]
        df["workload"] = meta["workload"]
        df["timestamp"] = meta["timestamp"]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    # Ensure tp is integer dtype in the concatenated result
    result["tp"] = result["tp"].astype("Int64")

    return result


def load_lifecycle_data(experiment_dir: str) -> pd.DataFrame:
    """Load per-request lifecycle metrics from an experiment.

    Reads ``results/per_request_lifecycle_metrics.json`` which contains a
    JSON array of request objects.

    Args:
        experiment_dir: Path to a single experiment directory.

    Returns:
        DataFrame indexed by ``request_id`` (0-based) with columns:
        start_time, end_time, input_tokens, output_tokens,
        output_token_times (as list).
    """
    metrics_path = os.path.join(
        experiment_dir, "results", "per_request_lifecycle_metrics.json"
    )

    with open(metrics_path, "r") as f:
        data = json.load(f)

    rows = []
    for i, entry in enumerate(data):
        info = entry.get("info", {})
        rows.append(
            {
                "request_id": i,
                "start_time": entry.get("start_time"),
                "end_time": entry.get("end_time"),
                "input_tokens": info.get("input_tokens"),
                "output_tokens": info.get("output_tokens"),
                "output_token_times": info.get("output_token_times", []),
            }
        )

    df = pd.DataFrame(rows)
    df = df.set_index("request_id")
    return df
