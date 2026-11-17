"""The DeepSeek-V4.1 writers read the FlashMLA kernel table at its own pin, without the sweeps.

AISimulate retired the FlashMLA sparse decode sidecar (dsv4_hca_attn_module_perf, the kernel
call alone) and deleted its table; the main pin no longer has it. The table that replaced it,
dsv4_hca_generation_module_perf, times the whole attention module, a different boundary
(VAL-9). So `fit_deepseek_v41_h200.read` takes the kernel table from KERNEL_DATA and every other
table from the main data tree. These tests build two small trees and check which one each read
comes from, so a writer that silently fell back to the module table, or read the indexer from
the old pin, fails here rather than in a refit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import fit_deepseek_v41_h200 as v41  # noqa: E402

SKU = "h200_sxm"


def table(root: Path, name: str, latency: float) -> None:
    d = root / SKU / "sparse_attention" / v41.COLLECTION
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"latency": [latency]}), d / name)


@pytest.fixture
def trees(tmp_path, monkeypatch):
    main, kernel = tmp_path / "main" / "data", tmp_path / "kernel" / "data"
    monkeypatch.setattr(v41, "KERNEL_DATA", kernel, raising=False)
    return main, kernel


def test_the_kernel_table_is_read_from_the_kernel_pin_when_the_main_pin_lacks_it(trees):
    main, kernel = trees
    table(kernel, v41.FLASHMLA_TABLE, 0.0102)
    main.mkdir(parents=True)
    assert v41.read(main, SKU, v41.FLASHMLA_TABLE)["latency"] == [0.0102]


def test_the_kernel_pin_wins_over_a_same_named_table_at_the_main_pin(trees):
    main, kernel = trees
    table(kernel, v41.FLASHMLA_TABLE, 0.0102)
    table(main, v41.FLASHMLA_TABLE, 0.0684)
    assert v41.read(main, SKU, v41.FLASHMLA_TABLE)["latency"] == [0.0102]


def test_every_other_table_is_read_at_the_main_pin(trees):
    main, kernel = trees
    table(kernel, v41.INDEXER_TABLE, 1.0)
    table(main, v41.INDEXER_TABLE, 2.0)
    assert v41.read(main, SKU, v41.INDEXER_TABLE)["latency"] == [2.0]


def test_a_missing_kernel_table_refuses_and_names_the_path(trees):
    main, kernel = trees
    table(main, "dsv4_hca_generation_module_perf.parquet", 0.0684)
    with pytest.raises(SystemExit, match="dsv4_hca_attn_module_perf.parquet is absent"):
        v41.read(main, SKU, v41.FLASHMLA_TABLE)
