"""Behavioural tests for the KV/server-state comparison (KV_METRICS.md).
Run: uv run --no-project --with pytest python -I -m pytest -q test_kvm.py"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kvbind  # noqa: E402
import kvcompare as K  # noqa: E402
import kvmeas  # noqa: E402

LINE = ("(APIServer pid=1) INFO 04-27 22:43:40 [loggers.py:259] Engine 000: Avg prompt throughput: 3170.9 tokens/s, "
        "Avg generation throughput: 409.7 tokens/s, Running: 32 reqs, Waiting: 3 reqs, GPU KV cache usage: 96.9%, "
        "Prefix cache hit rate: 12.5%")


def test_stat_line_parses_gauges_as_fractions():
    """vLLM prints KV usage and hit rate in percent; the harness must compare fractions."""
    st = kvmeas.parse_stats(LINE)
    assert (st["engines"], st["running"], st["waiting"]) == (1, [32], [3])
    assert st["kv"] == [pytest.approx(0.969)] and st["hit"] == [pytest.approx(0.125)]


def test_binding_needs_kv_limited_wait_not_just_wait():
    """A queue behind max_num_seqs with a half-empty pool is not KV-binding."""
    m = {"log": {"stat_lines": 3, "kv": [0.5, 0.5, 0.5], "running": [8] * 3, "waiting": [4] * 3, "preemptions": 0}}
    assert kvbind.binding(m)[0] == "nonbinding"
    m["log"]["kv"] = [0.5, 0.95, 0.5]
    assert kvbind.binding(m)[0] == "binding"


def test_binding_uses_measured_window_only():
    """Warmup samples (outside the profiling window) must not make a point binding."""
    s = {"window_s": [10, 20], "kv": [[1, 1.0], [15, 0.2]], "running": [[15, 1]], "waiting": [[1, 9], [15, 0]]}
    assert kvbind.binding({"series": s})[0] == "nonbinding"


def _blis(kv, wait, interval=1.0):
    n = len(kv)
    e = {"bins": n, "kv_cache_usage": kv, "kv_cache_usage_last": kv, "num_requests_running": [1] * n,
         "num_requests_running_last": [1] * n, "num_requests_waiting": wait, "num_requests_waiting_last": wait,
         "prompt_tokens": [10] * n, "generation_tokens": [1] * n, "prefix_cache_queries": [10] * n,
         "prefix_cache_hits": [5] * n, "prompt_tokens_local_compute": [5] * n,
         "prompt_tokens_local_cache_hit": [5] * n, "prompt_tokens_external_kv_transfer": [0] * n}
    return {"server_series": {"interval_s": interval, "engines": [e]}, "measured_window_s": n,
            "requests": [{"arrived_at": 0.0, "phase": "profile"}]}


def test_gross_flag_on_kv_usage_10_vs_90():
    """The user's example: BLIS 10% KV usage vs measured 90% must flag."""
    b = K.blis_series(_blis([0.1] * 10, [0] * 10), 1.0, "agentic_traces")
    m = {"kv": [0.9] * 10, "running": [1] * 10, "waiting": [0] * 10, "prompt_tps": [10] * 10, "gen_tps": [1] * 10,
         "hit_rate_window": 0.5, "src_share": None, "engines": 1, "interval": 1.0, "source": "trace-server-metrics"}
    c = K.compare("p", m, b, 1.0)
    assert any(g.startswith("kv_usage") for g in c["gross"])
    b2 = K.blis_series(_blis([0.85] * 10, [0] * 10), 1.0, "agentic_traces")
    assert not K.compare("p", m, b2, 1.0)["gross"]


def test_interval_mismatch_asserts():
    """A 10 s log comparison against a 1 s BLIS series is not apples to apples."""
    with pytest.raises(AssertionError):
        K.blis_series(_blis([0.1] * 3, [0] * 3, interval=1.0), 10.0, "single_turn")


def test_engine_count_mismatch_asserts():
    """Per-engine handling: a DP run must not be compared against a single-engine simulation."""
    b = K.blis_series(_blis([0.5] * 4, [0] * 4), 1.0, "agentic_traces")
    m = {"kv": [0.5] * 4, "running": [1] * 4, "waiting": [0] * 4, "prompt_tps": [1] * 4, "gen_tps": [1] * 4,
         "hit_rate_window": None, "src_share": None, "engines": 8, "interval": 1.0, "source": "trace-server-metrics"}
    with pytest.raises(AssertionError):
        K.compare("p", m, b, None)
