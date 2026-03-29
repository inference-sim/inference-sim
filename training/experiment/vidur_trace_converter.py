"""Convert per-request lifecycle metrics to Vidur trace CSV format.

Vidur's ``trace_replay`` request generator expects a simple 3-column CSV::

    arrived_at,num_prefill_tokens,num_decode_tokens
    0.0,591,140
    0.125,547,248

Arrival times are in **seconds**, relative to the first request.
"""

from __future__ import annotations

import csv
import json


def convert_to_vidur_trace(
    per_request_path: str,
    output_csv_path: str,
) -> str:
    """Convert ``per_request_lifecycle_metrics.json`` to Vidur CSV.

    Parameters
    ----------
    per_request_path:
        Path to the ``per_request_lifecycle_metrics.json`` file.
    output_csv_path:
        Where to write the output CSV.

    Returns
    -------
    str
        The *output_csv_path* (for convenience in chaining).
    """
    with open(per_request_path) as fh:
        requests = json.load(fh)

    # Sort requests by start_time to ensure chronological order
    # (per_request_lifecycle_metrics.json may not be sorted)
    requests = sorted(requests, key=lambda r: r["start_time"])

    first_start = requests[0]["start_time"] if requests else 0.0

    with open(output_csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
        for req in requests:
            arrived_at = req["start_time"] - first_start
            writer.writerow([
                arrived_at,
                req["info"]["input_tokens"],
                req["info"]["output_tokens"],
            ])

    return output_csv_path
