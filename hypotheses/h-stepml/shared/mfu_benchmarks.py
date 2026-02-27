"""MFU benchmark indexer for InferSim bench_data.

Loads and indexes hardware MFU (Model FLOPs Utilization) benchmarks from
InferSim's bench_data/ directory so that research ideas can query real
hardware performance data by GPU type, matrix shape, and attention
configuration.

Supported benchmark families:
  - GEMM: dense matrix multiply MFU by (m, k, n) shape
  - MHA decode: multi-head attention decode MFU by head config and KV length
  - MHA prefill: multi-head attention prefill MFU by head config and seq length
  - Grouped GEMM: MoE expert MFU (up_proj + down_proj) by expert config

All data is stored as pandas DataFrames and queried with exact-match or
nearest-neighbor interpolation.
"""

import os
import re
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default bench_data root: InferSim/bench_data/ relative to this file
# ---------------------------------------------------------------------------
DEFAULT_BENCH_DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "InferSim", "bench_data"
)

# Regex for MHA decode filenames: {heads}-{kv_heads}-{head_dim}[-tp{N}].csv
_MHA_DECODE_PATTERN = re.compile(
    r"^(\d+)-(\d+)-(\d+)(?:-tp(\d+))?\.csv$"
)

# Regex for MHA prefill filenames: {heads}-{kv_heads}-{head_dim}.csv
_MHA_PREFILL_PATTERN = re.compile(
    r"^(\d+)-(\d+)-(\d+)\.csv$"
)


def _nearest_row(df: pd.DataFrame, columns: list[str], values: list) -> pd.Series:
    """Find the row in *df* whose *columns* are nearest to *values*.

    Uses Euclidean distance across the specified columns after casting
    everything to float.  Returns the first row at minimum distance.
    """
    dist = np.zeros(len(df))
    for col, val in zip(columns, values):
        dist += (df[col].astype(float).values - float(val)) ** 2
    idx = int(np.argmin(dist))
    return df.iloc[idx]


class MFUBenchmarkIndex:
    """Index and query InferSim hardware MFU benchmarks.

    Parameters
    ----------
    bench_data_root : str, optional
        Path to the ``bench_data/`` directory.  Defaults to
        ``InferSim/bench_data/`` relative to this file.
    """

    def __init__(self, bench_data_root: str = None):
        if bench_data_root is None:
            bench_data_root = DEFAULT_BENCH_DATA_ROOT
        self._root = os.path.abspath(bench_data_root)

        # Lazy-loaded caches
        self._gemm: dict[str, pd.DataFrame] = {}
        self._mha_decode: dict[str, pd.DataFrame] = {}
        self._mha_prefill: dict[str, pd.DataFrame] = {}
        self._grouped_gemm: dict[str, pd.DataFrame] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Public loading
    # ------------------------------------------------------------------

    def load(self) -> "MFUBenchmarkIndex":
        """Eagerly load all available benchmark data.

        Returns *self* for chaining.
        """
        self._load_gemm()
        self._load_mha_decode()
        self._load_mha_prefill()
        self._load_grouped_gemm()
        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # GEMM queries
    # ------------------------------------------------------------------

    def query_gemm_mfu(self, gpu: str, m: int, k: int, n: int) -> float:
        """Look up GEMM MFU for a given (m, k, n) matrix shape.

        Parameters
        ----------
        gpu : str
            GPU identifier (e.g. ``"h100"``).
        m, k, n : int
            GEMM dimensions.

        Returns
        -------
        float
            MFU value.  Exact match preferred; falls back to nearest
            neighbor in (m, k, n) space.

        Raises
        ------
        ValueError
            If no GEMM data is available for *gpu*.
        """
        self._ensure_gemm(gpu)
        df = self._gemm[gpu]

        exact = df[(df["m"] == m) & (df["k"] == k) & (df["n"] == n)]
        if len(exact) > 0:
            return float(exact.iloc[0]["mfu"])

        row = _nearest_row(df, ["m", "k", "n"], [m, k, n])
        return float(row["mfu"])

    # ------------------------------------------------------------------
    # MHA decode queries
    # ------------------------------------------------------------------

    def query_mha_decode_mfu(
        self,
        gpu: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        batch_size: int,
        kv_len: int,
        tp: int = 1,
    ) -> float:
        """Look up MHA decode MFU.

        Parameters
        ----------
        gpu : str
            GPU identifier.
        num_heads, num_kv_heads, head_dim : int
            Attention head configuration.
        batch_size : int
            Batch size.
        kv_len : int
            KV cache length.
        tp : int
            Tensor parallelism degree (default 1).

        Returns
        -------
        float
            MFU value.

        Raises
        ------
        ValueError
            If no matching benchmark file exists.
        """
        self._ensure_mha_decode(gpu)
        key = self._mha_decode_key(gpu, num_heads, num_kv_heads, head_dim, tp)
        if key not in self._mha_decode:
            raise ValueError(
                f"No MHA decode benchmark for gpu={gpu}, "
                f"heads={num_heads}-{num_kv_heads}-{head_dim}, tp={tp}"
            )
        df = self._mha_decode[key]

        exact = df[(df["batch_size"] == batch_size) & (df["kv_len"] == kv_len)]
        if len(exact) > 0:
            return float(exact.iloc[0]["mfu"])

        row = _nearest_row(df, ["batch_size", "kv_len"], [batch_size, kv_len])
        return float(row["mfu"])

    # ------------------------------------------------------------------
    # MHA prefill queries
    # ------------------------------------------------------------------

    def query_mha_prefill_mfu(
        self,
        gpu: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
    ) -> float:
        """Look up MHA prefill MFU.

        Parameters
        ----------
        gpu : str
            GPU identifier.
        num_heads, num_kv_heads, head_dim : int
            Attention head configuration.
        seq_len : int
            Sequence length.

        Returns
        -------
        float
            MFU value.

        Raises
        ------
        ValueError
            If no matching benchmark file exists.
        """
        self._ensure_mha_prefill(gpu)
        key = self._mha_prefill_key(gpu, num_heads, num_kv_heads, head_dim)
        if key not in self._mha_prefill:
            raise ValueError(
                f"No MHA prefill benchmark for gpu={gpu}, "
                f"heads={num_heads}-{num_kv_heads}-{head_dim}"
            )
        df = self._mha_prefill[key]

        exact = df[df["seq_len"] == seq_len]
        if len(exact) > 0:
            return float(exact.iloc[0]["mfu"])

        row = _nearest_row(df, ["seq_len"], [seq_len])
        return float(row["mfu"])

    # ------------------------------------------------------------------
    # Grouped GEMM queries (MoE)
    # ------------------------------------------------------------------

    def query_grouped_gemm_mfu(self, gpu: str, phase: str, **kwargs) -> dict:
        """Look up grouped GEMM (MoE) MFU.

        Parameters
        ----------
        gpu : str
            GPU identifier.
        phase : str
            ``"decode"`` or ``"prefill"``.
        **kwargs
            Filter columns to match (e.g. ``batch_size_per_gpu=128``).

        Returns
        -------
        dict
            Dictionary with at least ``up_mfu`` and ``down_mfu`` keys,
            plus all columns from the matched row.

        Raises
        ------
        ValueError
            If no grouped GEMM data is available for *gpu*/*phase*, or
            *phase* is invalid.
        """
        if phase not in ("decode", "prefill"):
            raise ValueError(f"phase must be 'decode' or 'prefill', got {phase!r}")

        self._ensure_grouped_gemm(gpu, phase)
        key = f"{gpu}/{phase}"
        if key not in self._grouped_gemm:
            raise ValueError(
                f"No grouped GEMM benchmark for gpu={gpu}, phase={phase}"
            )
        df = self._grouped_gemm[key]

        # Try exact match on all provided kwargs
        mask = pd.Series(True, index=df.index)
        filter_cols = []
        filter_vals = []
        for col, val in kwargs.items():
            if col in df.columns:
                mask = mask & (df[col] == val)
                filter_cols.append(col)
                filter_vals.append(val)

        exact = df[mask]
        if len(exact) > 0:
            row = exact.iloc[0]
        elif filter_cols:
            row = _nearest_row(df, filter_cols, filter_vals)
        else:
            row = df.iloc[0]

        return row.to_dict()

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def available_gpus(self) -> list[str]:
        """List GPU types with any benchmark data.

        Scans the GEMM and MHA directories for GPU subdirectories.

        Returns
        -------
        list[str]
            Sorted list of GPU identifiers (e.g. ``["h100", "h20", "h800"]``).
        """
        gpus: set[str] = set()
        for subdir in ("gemm", "mha/decode", "mha/prefill",
                        "grouped_gemm/decode", "grouped_gemm/prefill"):
            path = os.path.join(self._root, subdir)
            if os.path.isdir(path):
                for name in os.listdir(path):
                    if os.path.isdir(os.path.join(path, name)):
                        gpus.add(name)
        return sorted(gpus)

    def available_benchmarks(self) -> dict:
        """Summarize what benchmarks are available.

        Returns
        -------
        dict
            Mapping of benchmark family to list of GPU types that have
            data for that family.
        """
        result = {}
        families = {
            "gemm": "gemm",
            "mha_decode": "mha/decode",
            "mha_prefill": "mha/prefill",
            "grouped_gemm_decode": "grouped_gemm/decode",
            "grouped_gemm_prefill": "grouped_gemm/prefill",
        }
        for family_name, subdir in families.items():
            path = os.path.join(self._root, subdir)
            if os.path.isdir(path):
                gpus = sorted(
                    name
                    for name in os.listdir(path)
                    if os.path.isdir(os.path.join(path, name))
                )
                if gpus:
                    result[family_name] = gpus
        return result

    # ------------------------------------------------------------------
    # Internal loading helpers
    # ------------------------------------------------------------------

    def _ensure_gemm(self, gpu: str) -> None:
        """Ensure GEMM data for *gpu* is loaded."""
        if gpu in self._gemm:
            return
        csv_path = os.path.join(self._root, "gemm", gpu, "data.csv")
        if not os.path.isfile(csv_path):
            raise ValueError(
                f"No GEMM benchmark data for gpu={gpu!r} "
                f"(expected {csv_path})"
            )
        df = pd.read_csv(csv_path)
        # Ensure numeric types
        for col in ("m", "k", "n"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["mfu"] = pd.to_numeric(df["mfu"], errors="coerce")
        self._gemm[gpu] = df

    def _load_gemm(self) -> None:
        """Load GEMM data for all available GPUs."""
        gemm_dir = os.path.join(self._root, "gemm")
        if not os.path.isdir(gemm_dir):
            return
        for gpu in os.listdir(gemm_dir):
            gpu_dir = os.path.join(gemm_dir, gpu)
            if os.path.isdir(gpu_dir):
                try:
                    self._ensure_gemm(gpu)
                except ValueError:
                    pass

    @staticmethod
    def _mha_decode_key(
        gpu: str, num_heads: int, num_kv_heads: int, head_dim: int, tp: int
    ) -> str:
        if tp > 1:
            return f"{gpu}/{num_heads}-{num_kv_heads}-{head_dim}-tp{tp}"
        return f"{gpu}/{num_heads}-{num_kv_heads}-{head_dim}"

    def _ensure_mha_decode(self, gpu: str) -> None:
        """Load all MHA decode CSVs for *gpu* if not already loaded."""
        # Check if we already loaded this GPU (use a sentinel)
        sentinel = f"_loaded_mha_decode_{gpu}"
        if hasattr(self, sentinel):
            return

        decode_dir = os.path.join(self._root, "mha", "decode", gpu)
        if not os.path.isdir(decode_dir):
            raise ValueError(
                f"No MHA decode benchmark data for gpu={gpu!r} "
                f"(expected directory {decode_dir})"
            )
        for fname in os.listdir(decode_dir):
            m = _MHA_DECODE_PATTERN.match(fname)
            if not m:
                continue
            num_heads = int(m.group(1))
            num_kv_heads = int(m.group(2))
            head_dim = int(m.group(3))
            tp = int(m.group(4)) if m.group(4) else 1
            key = self._mha_decode_key(gpu, num_heads, num_kv_heads, head_dim, tp)
            csv_path = os.path.join(decode_dir, fname)
            df = pd.read_csv(csv_path)
            for col in ("batch_size", "kv_len"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "mfu" in df.columns:
                df["mfu"] = pd.to_numeric(df["mfu"], errors="coerce")
            self._mha_decode[key] = df

        setattr(self, sentinel, True)

    def _load_mha_decode(self) -> None:
        """Load MHA decode data for all available GPUs."""
        decode_dir = os.path.join(self._root, "mha", "decode")
        if not os.path.isdir(decode_dir):
            return
        for gpu in os.listdir(decode_dir):
            if os.path.isdir(os.path.join(decode_dir, gpu)):
                try:
                    self._ensure_mha_decode(gpu)
                except ValueError:
                    pass

    @staticmethod
    def _mha_prefill_key(
        gpu: str, num_heads: int, num_kv_heads: int, head_dim: int
    ) -> str:
        return f"{gpu}/{num_heads}-{num_kv_heads}-{head_dim}"

    def _ensure_mha_prefill(self, gpu: str) -> None:
        """Load all MHA prefill CSVs for *gpu* if not already loaded."""
        sentinel = f"_loaded_mha_prefill_{gpu}"
        if hasattr(self, sentinel):
            return

        prefill_dir = os.path.join(self._root, "mha", "prefill", gpu)
        if not os.path.isdir(prefill_dir):
            raise ValueError(
                f"No MHA prefill benchmark data for gpu={gpu!r} "
                f"(expected directory {prefill_dir})"
            )
        for fname in os.listdir(prefill_dir):
            m = _MHA_PREFILL_PATTERN.match(fname)
            if not m:
                continue
            num_heads = int(m.group(1))
            num_kv_heads = int(m.group(2))
            head_dim = int(m.group(3))
            key = self._mha_prefill_key(gpu, num_heads, num_kv_heads, head_dim)
            csv_path = os.path.join(prefill_dir, fname)
            df = pd.read_csv(csv_path)
            if "seq_len" in df.columns:
                df["seq_len"] = pd.to_numeric(df["seq_len"], errors="coerce")
            if "mfu" in df.columns:
                df["mfu"] = pd.to_numeric(df["mfu"], errors="coerce")
            self._mha_prefill[key] = df

        setattr(self, sentinel, True)

    def _load_mha_prefill(self) -> None:
        """Load MHA prefill data for all available GPUs."""
        prefill_dir = os.path.join(self._root, "mha", "prefill")
        if not os.path.isdir(prefill_dir):
            return
        for gpu in os.listdir(prefill_dir):
            if os.path.isdir(os.path.join(prefill_dir, gpu)):
                try:
                    self._ensure_mha_prefill(gpu)
                except ValueError:
                    pass

    def _ensure_grouped_gemm(self, gpu: str, phase: str) -> None:
        """Load grouped GEMM data for *gpu*/*phase* if not already loaded."""
        key = f"{gpu}/{phase}"
        if key in self._grouped_gemm:
            return
        csv_path = os.path.join(
            self._root, "grouped_gemm", phase, gpu, "data.csv"
        )
        if not os.path.isfile(csv_path):
            raise ValueError(
                f"No grouped GEMM benchmark data for gpu={gpu!r}, "
                f"phase={phase!r} (expected {csv_path})"
            )
        df = pd.read_csv(csv_path)
        # Coerce numeric columns
        for col in df.columns:
            if col not in ("dtype", "kv_dtype"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        self._grouped_gemm[key] = df

    def _load_grouped_gemm(self) -> None:
        """Load grouped GEMM data for all available GPUs and phases."""
        for phase in ("decode", "prefill"):
            phase_dir = os.path.join(self._root, "grouped_gemm", phase)
            if not os.path.isdir(phase_dir):
                continue
            for gpu in os.listdir(phase_dir):
                if os.path.isdir(os.path.join(phase_dir, gpu)):
                    try:
                        self._ensure_grouped_gemm(gpu, phase)
                    except ValueError:
                        pass
