"""Extract KV cache block counts from vLLM ground-truth logs.

Two extraction functions:

* ``extract_total_kv_blocks`` — parse *vllm.log* for the GPU KV cache size
  line and convert token count to block count (tokens / 16).
* ``extract_cpu_kv_blocks`` — parse *vllm.log* for the CPU offloading capacity
  from ``cpu_bytes_to_use`` in the KVTransferConfig and convert to blocks.

The CPU block count represents the **configured capacity** (from
``--kv-offloading-size``), not peak usage during the experiment.
"""

from __future__ import annotations

import json
import re

_KV_CACHE_RE = re.compile(r"GPU KV cache size:\s+([\d,]+)\s+tokens")
_CPU_BYTES_RE = re.compile(r"cpu_bytes_to_use['\"]?\s*:\s*([\d.]+)")

BLOCK_SIZE = 16


def extract_total_kv_blocks(vllm_log_path: str) -> int:
    """Return the total GPU KV cache block count from a vllm.log file.

    Searches for a line matching::

        INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 119,408 tokens

    Strips commas from the token count, converts to int, and divides by
    ``BLOCK_SIZE`` (16) using integer division.

    Parameters
    ----------
    vllm_log_path:
        Filesystem path to the ``vllm.log`` file.

    Returns
    -------
    int
        Number of KV cache blocks (tokens // 16).

    Raises
    ------
    ValueError
        If no matching line is found in the log file.
    """
    with open(vllm_log_path, "r") as fh:
        for line in fh:
            match = _KV_CACHE_RE.search(line)
            if match:
                tokens = int(match.group(1).replace(",", ""))
                return tokens // BLOCK_SIZE

    raise ValueError(
        f"No 'GPU KV cache size' line found in {vllm_log_path}"
    )


def extract_cpu_kv_blocks(vllm_log_path: str) -> int:
    """Return the CPU KV cache capacity in blocks from vllm.log.

    Parses the ``cpu_bytes_to_use`` value from the KVTransferConfig line
    and converts it to block count using the model's KV cache bytes-per-token
    ratio inferred from the GPU KV cache size.

    Searches for lines matching::

        ... kv_connector_extra_config={'cpu_bytes_to_use': 8589934592.0} ...
        ... GPU KV cache size: 425,648 tokens

    The conversion uses the formula::

        bytes_per_token = GPU_bytes / GPU_tokens
        cpu_blocks = cpu_bytes_to_use / (bytes_per_token * BLOCK_SIZE)

    where GPU_bytes is estimated from available GPU memory reported by vLLM.

    Parameters
    ----------
    vllm_log_path:
        Filesystem path to the ``vllm.log`` file.

    Returns
    -------
    int
        Number of CPU KV cache blocks based on configured capacity.
        Returns 0 if ``cpu_bytes_to_use`` is not found (no CPU offloading).

    Raises
    ------
    ValueError
        If GPU KV cache size is found but cannot be used for conversion.
    """
    cpu_bytes = None
    gpu_tokens = None
    gpu_bytes_available = None

    with open(vllm_log_path, "r") as fh:
        for line in fh:
            # Parse cpu_bytes_to_use from KVTransferConfig
            if cpu_bytes is None and "cpu_bytes_to_use" in line:
                match = _CPU_BYTES_RE.search(line)
                if match:
                    cpu_bytes = float(match.group(1))

            # Parse GPU KV cache size in tokens
            if gpu_tokens is None:
                match = _KV_CACHE_RE.search(line)
                if match:
                    gpu_tokens = int(match.group(1).replace(",", ""))

            # Parse available KV cache memory in GiB
            if gpu_bytes_available is None and "Available KV cache memory:" in line:
                # Example: "Available KV cache memory: 38.97 GiB"
                mem_match = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", line)
                if mem_match:
                    gpu_bytes_available = float(mem_match.group(1)) * (1024 ** 3)

    # If no CPU offloading configured, return 0
    if cpu_bytes is None:
        return 0

    # Need GPU info to convert bytes to blocks
    if gpu_tokens is None:
        raise ValueError(
            f"Found cpu_bytes_to_use but no 'GPU KV cache size' line in {vllm_log_path}"
        )
    if gpu_bytes_available is None:
        raise ValueError(
            f"Found cpu_bytes_to_use but no 'Available KV cache memory' line in {vllm_log_path}"
        )

    # Calculate bytes per token from GPU cache
    bytes_per_token = gpu_bytes_available / gpu_tokens

    # Convert CPU bytes to blocks
    bytes_per_block = bytes_per_token * BLOCK_SIZE
    cpu_blocks = int(cpu_bytes / bytes_per_block)

    return cpu_blocks
