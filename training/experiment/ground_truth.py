"""Discover and parse ground-truth experiment directories.

Functions
---------
discover_experiments(base_dir, *, safe_only=True)
    Manifest-driven discovery: load ``experiments.json`` and resolve each
    entry to its directory on disk.  Returns ``(manifest_entry, dir_path)``
    pairs sorted by experiment id.
load_manifest(base_dir)
    Load ``experiments.json`` from *base_dir*.
resolve_experiment_dir(base_dir, exp_id)
    Find the directory matching ``<id>-*`` in *base_dir*.
parse_experiment(folder_path)
    Load all config/metrics from a single experiment directory into an
    ``Experiment`` dataclass.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re

import yaml

logger = logging.getLogger(__name__)

from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    StageMetrics,
    ThroughputMetrics,
)
from experiment.kv_cache_extractor import extract_cpu_kv_blocks, extract_total_kv_blocks

_EXPERIMENT_DIR_RE = re.compile(r"^\d{8}-\d{6}-.+-tp\d+-\w+$")


# ---------------------------------------------------------------------------
# Legacy discovery (kept for backward compatibility with parse_experiment)
# ---------------------------------------------------------------------------

def _discover_experiments_legacy(base_dir: str) -> list[str]:
    """Return sorted absolute paths of experiment directories under *base_dir*.

    Matches directories whose name follows the pattern
    ``YYYYMMDD-HHMMSS-*-tp<N>-<workload>``.  Non-directory entries and files
    like ``SCHEMA.md`` are excluded.

    .. deprecated:: Use :func:`discover_experiments` (manifest-driven) instead.
    """
    results: list[str] = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and _EXPERIMENT_DIR_RE.match(entry):
            results.append(os.path.abspath(full_path))
    results.sort()
    return results


# ---------------------------------------------------------------------------
# Manifest-driven discovery
# ---------------------------------------------------------------------------

def load_manifest(base_dir: str) -> list[dict]:
    """Load experiments.json from base_dir."""
    path = os.path.join(base_dir, "experiments.json")
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(f"Manifest not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc


def resolve_experiment_dir(base_dir: str, exp_id: int) -> str | None:
    """Find the directory matching '<id>-*' in base_dir."""
    prefix = f"{exp_id}-"
    matches = [
        entry for entry in os.listdir(base_dir)
        if entry.startswith(prefix) and os.path.isdir(os.path.join(base_dir, entry))
    ]
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple directories for id=%d in %s: %s (using first)",
            exp_id, base_dir, matches,
        )
    return os.path.abspath(os.path.join(base_dir, sorted(matches)[0]))


def resolve_perf_dir(folder_path: str) -> str:
    """Return the performance-data subdirectory for an experiment.

    Prefers ``results/`` (numbered experiments) and falls back to
    ``inference-perf-data/`` (legacy layout).
    """
    perf_dir = os.path.join(folder_path, "results")
    if not os.path.isdir(perf_dir):
        perf_dir = os.path.join(folder_path, "inference-perf-data")
    return perf_dir


_REQUIRED_MANIFEST_KEYS = {"id", "hw", "dp", "cpu_offload", "gpu_mem", "precision", "safe"}


def discover_experiments(
    base_dir: str,
    *,
    safe_only: bool = True,
) -> list[tuple[dict, str]]:
    """Return (manifest_entry, dir_path) pairs for runnable experiments.

    Reads ``experiments.json`` from *base_dir* and resolves each experiment
    to its directory on disk.  Experiments whose directory is missing are
    skipped with a warning.

    Parameters
    ----------
    base_dir:
        Root directory containing ``experiments.json`` and experiment
        sub-directories named ``<id>-<slug>``.
    safe_only:
        When *True* (default), only experiments marked ``"safe": "safe"``
        in the manifest are included.
    """
    manifest = load_manifest(base_dir)
    results: list[tuple[dict, str]] = []
    for entry in manifest:
        if safe_only and entry.get("safe") != "safe":
            continue
        exp_id = entry.get("id")
        if exp_id is None:
            logger.warning("Manifest entry missing 'id' key, skipping: %s", entry)
            continue
        dir_path = resolve_experiment_dir(base_dir, exp_id)
        if dir_path is not None:
            results.append((entry, dir_path))
        else:
            logger.warning(
                "No directory found for experiment id=%d model=%s",
                exp_id, entry.get("model", "?"),
            )
    if manifest and not results:
        logger.warning("No experiment directories resolved from %d manifest entries", len(manifest))
    results.sort(key=lambda x: x[0]["id"])
    return results


def parse_experiment(folder_path: str, manifest_entry: dict | None = None) -> Experiment:
    """Parse a single experiment directory into an :class:`Experiment`."""
    folder_path = os.path.abspath(folder_path)
    folder_name = os.path.basename(folder_path)

    # 0. Validate manifest keys up-front (before any access)
    if manifest_entry is not None:
        required = _REQUIRED_MANIFEST_KEYS | {"workload"}
        missing = required - manifest_entry.keys()
        if missing:
            raise KeyError(
                f"Manifest entry for {folder_path} missing keys: {sorted(missing)}"
            )

    # 1. Parse exp-config.yaml
    with open(os.path.join(folder_path, "exp-config.yaml")) as fh:
        exp_cfg = yaml.safe_load(fh)

    model = exp_cfg["model"]
    tp = exp_cfg["tensor_parallelism"]
    max_model_len = exp_cfg["max_model_len"]
    max_num_batched_tokens = exp_cfg["max_num_batched_tokens"]
    max_num_seqs = exp_cfg["max_num_seqs"]

    # 2. Extract workload — prefer manifest to avoid suffix corruption
    if manifest_entry is not None:
        workload = manifest_entry["workload"]
    else:
        workload = _extract_workload(folder_name)

    # 3. Parse profile.yaml (single-line JSON that YAML can parse)
    with open(os.path.join(folder_path, "profile.yaml")) as fh:
        profile_config = yaml.safe_load(fh)

    stages_config = profile_config["load"]["stages"]

    # 4. Parse stage lifecycle metrics — auto-detect results dir
    perf_dir = resolve_perf_dir(folder_path)
    stage_files = sorted(glob.glob(os.path.join(perf_dir, "stage_*_lifecycle_metrics.json")))
    if len(stage_files) != len(stages_config):
        logger.warning(
            "Stage count mismatch in %s: %d files vs %d in profile.yaml",
            folder_name, len(stage_files), len(stages_config),
        )
    stages: list[StageMetrics] = []
    for i, stage_file in enumerate(stage_files):
        with open(stage_file) as fh:
            stage_data = json.load(fh)
        stage_cfg = stages_config[i] if i < len(stages_config) else {}
        stages.append(_parse_stage_metrics(stage_data, stage_index=i, stage_cfg=stage_cfg))

    # 5. Parse summary lifecycle metrics
    summary_path = os.path.join(perf_dir, "summary_lifecycle_metrics.json")
    with open(summary_path) as fh:
        summary_data = json.load(fh)
    summary = _parse_stage_metrics(summary_data, stage_index=-1, stage_cfg=None)

    # 6. Extract KV blocks
    vllm_log = os.path.join(folder_path, "vllm.log")
    total_kv_blocks = extract_total_kv_blocks(vllm_log)
    cpu_kv_blocks = extract_cpu_kv_blocks(vllm_log)

    # 7. Manifest metadata (new fields from experiments.json)
    kwargs = {}
    if manifest_entry is not None:
        kwargs = dict(
            exp_id=manifest_entry["id"],
            hardware=manifest_entry["hw"],
            dp=manifest_entry["dp"],
            cpu_offload=manifest_entry["cpu_offload"],
            gpu_mem_util=manifest_entry["gpu_mem"],
            precision=manifest_entry["precision"],
            safe=manifest_entry["safe"],
        )

    return Experiment(
        folder=folder_path,
        model=model,
        tp=tp,
        workload=workload,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        total_kv_blocks=total_kv_blocks,
        cpu_kv_blocks=cpu_kv_blocks,
        stages=stages,
        summary=summary,
        profile_config=profile_config,
        **kwargs,
    )


def _extract_workload(folder_name: str) -> str:
    """Extract the workload type from a folder name like ``...-tp1-codegen``."""
    # Find the last occurrence of -tp<digits>- and take everything after it
    match = re.search(r"-tp\d+-(.+)$", folder_name)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract workload from folder name: {folder_name}")


def _parse_stage_metrics(
    data: dict,
    stage_index: int,
    stage_cfg: dict | None,
) -> StageMetrics:
    """Parse a stage or summary lifecycle metrics JSON into a StageMetrics."""
    load = data["load_summary"]
    successes = data["successes"]
    latency = successes["latency"]

    # Rate and duration from stage config (absent in summary)
    if stage_cfg is not None:
        rate = float(stage_cfg.get("rate", load.get("requested_rate", 0.0)))
        duration = float(stage_cfg.get("duration", load.get("send_duration", 0.0)))
    else:
        rate = 0.0
        duration = 0.0

    num_requests = successes["count"]

    # Latencies: seconds → milliseconds
    e2e = _parse_latency_dist(latency["request_latency"])
    ttft = _parse_latency_dist(latency["time_to_first_token"])

    # Calculate standard TPOT (Time Per Output Token) using industry formula:
    # TPOT = (E2E - TTFT) / (output_tokens - 1)
    # This matches the definition from GCP docs and is the standard metric.
    output_tokens = successes["output_len"]["mean"]
    tpot_seconds = (latency["request_latency"]["mean"] - latency["time_to_first_token"]["mean"]) / max(output_tokens - 1, 1)
    tpot_ms = tpot_seconds * 1000

    # TPOT is what simulators predict as ITL (inter-token latency)
    itl = LatencyDistribution(mean=tpot_ms, p90=None, p99=None)

    # Throughput (already in tokens/sec or req/sec)
    tp_data = successes["throughput"]
    throughput = ThroughputMetrics(
        input_tokens_per_sec=tp_data["input_tokens_per_sec"],
        output_tokens_per_sec=tp_data["output_tokens_per_sec"],
        requests_per_sec=tp_data["requests_per_sec"],
    )

    return StageMetrics(
        stage_index=stage_index,
        rate=rate,
        duration=duration,
        num_requests=num_requests,
        e2e=e2e,
        ttft=ttft,
        itl=itl,
        throughput=throughput,
    )


def _parse_latency_dist(d: dict) -> LatencyDistribution:
    """Convert a latency dict (values in seconds) to LatencyDistribution (ms)."""
    return LatencyDistribution(
        mean=d["mean"] * 1000,
        p90=d["p90"] * 1000,
        p99=d["p99"] * 1000,
    )
