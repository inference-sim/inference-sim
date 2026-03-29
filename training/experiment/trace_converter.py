"""Convert per-request lifecycle metrics to BLIS trace v2 format.

The BLIS simulator expects a two-file trace:
    * A YAML header declaring metadata (time units, mode, etc.)
    * A CSV data file with one row per request (22 columns)

Additionally a workload spec YAML is generated for the ``--workload-spec``
CLI flag.
"""

from __future__ import annotations

import csv
import json
import os

import yaml


_CSV_COLUMNS = [
    "request_id", "client_id", "tenant_id", "slo_class", "session_id",
    "round_index", "prefix_group", "streaming", "input_tokens", "output_tokens",
    "text_tokens", "image_tokens", "audio_tokens", "video_tokens", "reason_ratio",
    "arrival_time_us", "send_time_us", "first_chunk_time_us", "last_chunk_time_us",
    "num_chunks", "status", "error_message",
]


def convert_to_blis_trace(
    per_request_path: str,
    output_dir: str,
) -> tuple[str, str]:
    """Convert ``per_request_lifecycle_metrics.json`` to BLIS trace v2.

    Parameters
    ----------
    per_request_path:
        Path to the ``per_request_lifecycle_metrics.json`` file.
    output_dir:
        Directory where trace files will be written.

    Returns
    -------
    tuple[str, str]
        ``(header_yaml_path, data_csv_path)``
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(per_request_path) as fh:
        requests = json.load(fh)

    # Compute relative timestamps (seconds → microseconds, relative to first request)
    if requests:
        first_start = requests[0]["start_time"]
    else:
        first_start = 0.0

    # --- Header YAML ---
    header_path = os.path.join(output_dir, "trace_header.yaml")
    header = {
        "trace_version": 2,
        "time_unit": "microseconds",
        "mode": "real",
        "warm_up_requests": 0,
    }
    with open(header_path, "w") as fh:
        yaml.dump(header, fh, default_flow_style=False)

    # --- Data CSV ---
    data_path = os.path.join(output_dir, "trace_data.csv")
    with open(data_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_COLUMNS)

        for i, req in enumerate(requests):
            arrival_us = int((req["start_time"] - first_start) * 1_000_000)
            info = req["info"]

            row = [
                i,                        # request_id
                "",                       # client_id
                "",                       # tenant_id
                "",                       # slo_class
                "",                       # session_id
                0,                        # round_index
                "",                       # prefix_group
                "true",                   # streaming
                info["input_tokens"],     # input_tokens
                info["output_tokens"],    # output_tokens
                info["input_tokens"],     # text_tokens (same as input)
                0,                        # image_tokens
                0,                        # audio_tokens
                0,                        # video_tokens
                0.0,                      # reason_ratio
                arrival_us,               # arrival_time_us
                arrival_us,               # send_time_us (same as arrival)
                0,                        # first_chunk_time_us (unknown)
                0,                        # last_chunk_time_us (unknown)
                0,                        # num_chunks
                "ok",                     # status
                "",                       # error_message
            ]
            writer.writerow(row)

    # --- Workload spec YAML ---
    spec_path = os.path.join(output_dir, "workload_spec.yaml")
    spec = {
        "workloads": [
            {
                "name": "trace-replay",
                "trace_header": "trace_header.yaml",
                "trace_data": "trace_data.csv",
            }
        ]
    }
    with open(spec_path, "w") as fh:
        yaml.dump(spec, fh, default_flow_style=False)

    return header_path, data_path
