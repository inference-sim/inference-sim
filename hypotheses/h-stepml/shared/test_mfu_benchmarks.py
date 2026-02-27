"""Tests for MFU benchmark indexer.

All tests use real bench_data from InferSim/bench_data/ -- no synthetic data.
"""

import os

import pytest

from mfu_benchmarks import DEFAULT_BENCH_DATA_ROOT, MFUBenchmarkIndex

# Skip entire module if bench_data is not available
pytestmark = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_BENCH_DATA_ROOT),
    reason="InferSim bench_data directory not found",
)


@pytest.fixture(scope="module")
def index() -> MFUBenchmarkIndex:
    """Create and eagerly load a shared benchmark index."""
    return MFUBenchmarkIndex().load()


# ------------------------------------------------------------------
# GEMM tests
# ------------------------------------------------------------------


class TestGEMM:
    def test_load_gemm_h100(self, index: MFUBenchmarkIndex):
        """GEMM data for h100 should be loadable with many rows."""
        df = index._gemm["h100"]
        assert len(df) > 0
        assert set(df.columns) >= {"m", "k", "n", "latency_us", "mfu"}

    def test_query_gemm_exact_match(self, index: MFUBenchmarkIndex):
        """Exact (m, k, n) match returns the correct MFU value."""
        # Row from h100 data: m=8, k=2048, n=6144 -> mfu=0.011
        mfu = index.query_gemm_mfu("h100", m=8, k=2048, n=6144)
        assert mfu == pytest.approx(0.011, abs=1e-4)

    def test_query_gemm_exact_match_larger(self, index: MFUBenchmarkIndex):
        """Verify another exact-match row."""
        # Row: m=1024, k=2048, n=6144 -> mfu=0.506
        mfu = index.query_gemm_mfu("h100", m=1024, k=2048, n=6144)
        assert mfu == pytest.approx(0.506, abs=1e-4)

    def test_query_gemm_nearest_neighbor(self, index: MFUBenchmarkIndex):
        """Non-exact query falls back to nearest neighbor."""
        # m=10 doesn't exist; nearest is m=8 (mfu=0.011) or m=16 (mfu=0.023)
        mfu = index.query_gemm_mfu("h100", m=10, k=2048, n=6144)
        assert mfu in (
            pytest.approx(0.011, abs=1e-4),
            pytest.approx(0.023, abs=1e-4),
        )

    def test_query_gemm_returns_float(self, index: MFUBenchmarkIndex):
        """Return type is always a Python float."""
        mfu = index.query_gemm_mfu("h100", m=8, k=2048, n=6144)
        assert isinstance(mfu, float)


# ------------------------------------------------------------------
# MHA decode tests
# ------------------------------------------------------------------


class TestMHADecode:
    def test_load_mha_decode_h100(self, index: MFUBenchmarkIndex):
        """MHA decode data for h100 should contain head configurations."""
        # 28-4-128-tp1.csv has tp=1, so the key omits the -tp suffix
        key = "h100/28-4-128"
        assert key in index._mha_decode
        df = index._mha_decode[key]
        assert len(df) > 0
        assert set(df.columns) >= {"batch_size", "kv_len", "mfu"}

    def test_query_mha_decode_exact_match(self, index: MFUBenchmarkIndex):
        """Exact match on batch_size and kv_len returns correct MFU."""
        # 28-4-128-tp1.csv: batch_size=1, kv_len=1024 -> mfu=0.001
        mfu = index.query_mha_decode_mfu(
            "h100",
            num_heads=28,
            num_kv_heads=4,
            head_dim=128,
            batch_size=1,
            kv_len=1024,
            tp=1,
        )
        assert mfu == pytest.approx(0.001, abs=1e-4)

    def test_query_mha_decode_nearest_neighbor(self, index: MFUBenchmarkIndex):
        """Non-exact query falls back to nearest neighbor."""
        # batch_size=2 doesn't exist; should return nearest (bs=1 or bs=16)
        mfu = index.query_mha_decode_mfu(
            "h100",
            num_heads=28,
            num_kv_heads=4,
            head_dim=128,
            batch_size=2,
            kv_len=1024,
            tp=1,
        )
        assert isinstance(mfu, float)
        assert 0.0 <= mfu <= 1.0

    def test_query_mha_decode_tp_variant(self, index: MFUBenchmarkIndex):
        """TP > 1 queries resolve to the correct file."""
        # 28-4-128-tp2.csv should exist for h100
        key = "h100/28-4-128-tp2"
        assert key in index._mha_decode
        mfu = index.query_mha_decode_mfu(
            "h100",
            num_heads=28,
            num_kv_heads=4,
            head_dim=128,
            batch_size=1,
            kv_len=1024,
            tp=2,
        )
        assert isinstance(mfu, float)

    def test_query_mha_decode_no_tp_suffix(self, index: MFUBenchmarkIndex):
        """GPUs without -tp suffix in filenames use tp=1 implicitly."""
        # h20 decode files don't have -tp suffix
        mfu = index.query_mha_decode_mfu(
            "h20",
            num_heads=128,
            num_kv_heads=2,
            head_dim=128,
            batch_size=1,
            kv_len=1024,
            tp=1,
        )
        assert isinstance(mfu, float)
        assert mfu >= 0.0


# ------------------------------------------------------------------
# MHA prefill tests
# ------------------------------------------------------------------


class TestMHAPrefill:
    def test_load_mha_prefill_h100(self, index: MFUBenchmarkIndex):
        """MHA prefill data for h100 should contain head configurations."""
        key = "h100/28-4-128"
        assert key in index._mha_prefill
        df = index._mha_prefill[key]
        assert len(df) > 0
        assert set(df.columns) >= {"seq_len", "mfu"}

    def test_query_mha_prefill_exact_match(self, index: MFUBenchmarkIndex):
        """Exact match on seq_len returns correct MFU."""
        # 28-4-128.csv: seq_len=1024 -> mfu=0.201
        mfu = index.query_mha_prefill_mfu(
            "h100",
            num_heads=28,
            num_kv_heads=4,
            head_dim=128,
            seq_len=1024,
        )
        assert mfu == pytest.approx(0.201, abs=1e-4)

    def test_query_mha_prefill_nearest_neighbor(self, index: MFUBenchmarkIndex):
        """Non-exact seq_len falls back to nearest neighbor."""
        # seq_len=2048 doesn't exist; nearest is 1024 or 4096
        mfu = index.query_mha_prefill_mfu(
            "h100",
            num_heads=28,
            num_kv_heads=4,
            head_dim=128,
            seq_len=2048,
        )
        assert isinstance(mfu, float)
        assert 0.0 <= mfu <= 1.0


# ------------------------------------------------------------------
# Grouped GEMM tests
# ------------------------------------------------------------------


class TestGroupedGEMM:
    def test_load_grouped_gemm(self, index: MFUBenchmarkIndex):
        """Grouped GEMM data should be loadable for available GPUs."""
        benchmarks = index.available_benchmarks()
        if "grouped_gemm_decode" in benchmarks:
            gpu = benchmarks["grouped_gemm_decode"][0]
            key = f"{gpu}/decode"
            assert key in index._grouped_gemm

    def test_query_grouped_gemm_returns_dict(self, index: MFUBenchmarkIndex):
        """Query returns a dict with up_mfu and down_mfu."""
        benchmarks = index.available_benchmarks()
        if "grouped_gemm_decode" not in benchmarks:
            pytest.skip("No grouped GEMM decode data available")
        gpu = benchmarks["grouped_gemm_decode"][0]
        result = index.query_grouped_gemm_mfu(gpu, "decode")
        assert isinstance(result, dict)
        assert "up_mfu" in result
        assert "down_mfu" in result

    def test_query_grouped_gemm_invalid_phase(self, index: MFUBenchmarkIndex):
        """Invalid phase raises ValueError."""
        with pytest.raises(ValueError, match="phase must be"):
            index.query_grouped_gemm_mfu("h100", "invalid_phase")


# ------------------------------------------------------------------
# Discovery tests
# ------------------------------------------------------------------


class TestDiscovery:
    def test_available_gpus(self, index: MFUBenchmarkIndex):
        """available_gpus() returns known GPU types."""
        gpus = index.available_gpus()
        assert isinstance(gpus, list)
        assert len(gpus) > 0
        # h100 should always be present (it has GEMM data)
        assert "h100" in gpus

    def test_available_gpus_sorted(self, index: MFUBenchmarkIndex):
        """GPU list is sorted alphabetically."""
        gpus = index.available_gpus()
        assert gpus == sorted(gpus)

    def test_available_benchmarks(self, index: MFUBenchmarkIndex):
        """available_benchmarks() returns a dict with known families."""
        benchmarks = index.available_benchmarks()
        assert isinstance(benchmarks, dict)
        # GEMM data for h100 exists, so this family must be present
        assert "gemm" in benchmarks
        assert "h100" in benchmarks["gemm"]

    def test_available_benchmarks_contains_mha(self, index: MFUBenchmarkIndex):
        """MHA families should appear in available benchmarks."""
        benchmarks = index.available_benchmarks()
        assert "mha_decode" in benchmarks
        assert "mha_prefill" in benchmarks


# ------------------------------------------------------------------
# Error handling tests
# ------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_gpu_gemm(self, index: MFUBenchmarkIndex):
        """Querying GEMM for a nonexistent GPU raises ValueError."""
        with pytest.raises(ValueError, match="No GEMM benchmark"):
            index.query_gemm_mfu("nonexistent_gpu_xyz", m=8, k=2048, n=6144)

    def test_missing_gpu_mha_decode(self, index: MFUBenchmarkIndex):
        """Querying MHA decode for a nonexistent GPU raises ValueError."""
        with pytest.raises(ValueError, match="No MHA decode benchmark"):
            index.query_mha_decode_mfu(
                "nonexistent_gpu_xyz",
                num_heads=28,
                num_kv_heads=4,
                head_dim=128,
                batch_size=1,
                kv_len=1024,
            )

    def test_missing_gpu_mha_prefill(self, index: MFUBenchmarkIndex):
        """Querying MHA prefill for a nonexistent GPU raises ValueError."""
        with pytest.raises(ValueError, match="No MHA prefill benchmark"):
            index.query_mha_prefill_mfu(
                "nonexistent_gpu_xyz",
                num_heads=28,
                num_kv_heads=4,
                head_dim=128,
                seq_len=1024,
            )

    def test_missing_head_config(self, index: MFUBenchmarkIndex):
        """Querying a nonexistent head configuration raises ValueError."""
        with pytest.raises(ValueError, match="No MHA decode benchmark"):
            index.query_mha_decode_mfu(
                "h100",
                num_heads=999,
                num_kv_heads=999,
                head_dim=999,
                batch_size=1,
                kv_len=1024,
            )

    def test_custom_bench_data_root_missing(self):
        """Constructor with a nonexistent root doesn't fail until query."""
        idx = MFUBenchmarkIndex(bench_data_root="/nonexistent/path")
        with pytest.raises(ValueError):
            idx.query_gemm_mfu("h100", m=8, k=2048, n=6144)
