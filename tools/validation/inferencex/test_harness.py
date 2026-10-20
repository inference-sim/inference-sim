"""Behavioural tests for the coverage harness (run: uv run --no-project --with pytest python -I -m pytest -q).

Each test names the failure it guards against.
"""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cov  # noqa: E402
import metrics as M  # noqa: E402
import spec as S  # noqa: E402


def _blis(reqs, **agg):
    return {"requests": reqs, "measured_completed_requests": len(reqs), **agg}


def _req(t, e2e, ttft, osl, isl=100, phase="profile"):
    return {"arrived_at": t, "e2e_ms": e2e, "ttft_ms": ttft, "output_tokens": osl, "num_prefill_tokens": isl,
            "phase": phase, "agentic": {"outcome": "completed"}}


def test_b2_row_latency_equals_aggregate(tmp_path, monkeypatch):
    """B2: per-request latency minus host_completion must reproduce BLIS's own aiperf-latency aggregate."""
    hc = 0.777
    (tmp_path / "coefficients").mkdir()
    (tmp_path / "coefficients" / "cost-model-host-overheads.yaml").write_text(
        "terms:\n  - host_completion:\n      value: 777.3\n")
    monkeypatch.setenv("IX_REGISTRY", str(tmp_path))
    lat = [100.0, 200.0, 300.0]
    m = _blis([_req(i, x + hc, 10, 11) for i, x in enumerate(lat)],
              e2e_p50_ms=M.pct(lat, 50), e2e_p90_ms=M.pct(lat, 90))
    assert cov.host_completion_offset(m) == pytest.approx(hc)
    rows = M.blis_rows(m, hc, True)
    assert M.pct([r["lat_ms"] for r in rows], 50) == pytest.approx(m["e2e_p50_ms"])
    # without the fix the ITL is biased by hc/(osl-1)
    on = M.request_metrics(rows)["p50_itl_ms"]
    off = M.request_metrics(M.blis_rows(m, hc, False))["p50_itl_ms"]
    assert off - on == pytest.approx(hc / 10)


def test_warmup_rows_excluded():
    """Window: warmup records never enter the profile metrics (aiperf drops them)."""
    m = _blis([_req(0, 50, 5, 3, phase="warmup"), _req(1, 1000, 10, 11)])
    assert M.request_metrics(M.blis_rows(m, 0, False))["requests"] == 1


def test_window_mismatch_asserts():
    """B4: a whole-run measured value is never compared with a profile-window prediction."""
    meas = {"hit": {"value": 0.97, "def": "prefix hit/query tokens", "window": "whole server run (warmup + profiling)"}}
    pred = {"hit": {"value": 0.98, "def": "prefix hit/query tokens", "window": "aiperf profiling phase"}}
    with pytest.raises(AssertionError):
        M.assert_same(meas, pred)


def test_interactivity_is_reciprocal_of_itl_percentile():
    """InferenceX p90_intvty = 1/p90_itl (slow tail), not the p90 of per-request reciprocals."""
    rows = [{"start_ms": i, "lat_ms": 10 + 10 * (k + 1), "ttft_ms": 10, "isl": 1, "osl": 11} for i, k in enumerate(range(10))]
    r = M.request_metrics(rows)
    assert r["p90_intvty"] == pytest.approx(1e3 / r["p90_itl_ms"])
    assert r["p90_intvty"] < r["p50_intvty"]


def test_logged_dp_run_keeps_tp_default():
    """A logged DP8 run states data_parallel_size but not tensor_parallel_size: TP is vLLM's default 1,
    not InferenceX's decode_tp (8), which would make an 8-node layout."""
    inv = {"gpu": "h200", "parallel": {"decode": {"tp": 8, "ep": 8, "gpus": 8, "dp_attention": True}, "disagg": False},
           "hf_repo": "x/M", "engine_image": "vllm/vllm-openai:v0.25.1"}
    log = {"non_default": {"data_parallel_size": 8, "enable_expert_parallel": True, "max_model_len": 2304,
                           "model": "x/M"}, "vllm_version": "0.25.1"}
    d = S.resolve(inv, log, None, 4, {"m": "m-cat"})
    assert (d["tp"], d["dp"], d["nodes"]) == (1, 8, 1)


def test_version_defaults_not_v031():
    """A v0.15.1 run takes v0.15.1's gpu_memory_utilization default (0.9), not v0.31.0's (0.92)."""
    inv = {"gpu": "h100", "parallel": {"decode": {"tp": 2, "ep": 1, "gpus": 2}, "disagg": False},
           "hf_repo": "openai/gpt-oss-120b", "engine_image": "vllm/vllm-openai:v0.15.1"}
    log = {"non_default": {"tensor_parallel_size": 2, "max_model_len": 10240, "model": "openai/gpt-oss-120b"},
           "vllm_version": "0.15.1"}
    d = S.resolve(inv, log, None, 4, {"gpt-oss-120b": "gpt-oss-120b"})
    assert d["engine"]["gpu_memory_utilization"] == 0.9
    assert any(m["item"].startswith("model runner") for m in d["mismatches"])


def test_simple_cpu_offload_is_blocked_not_dropped():
    """A connector BLIS does not simulate is a named block, never silently replayed without offload."""
    inv = {"gpu": "h100", "parallel": {"decode": {"tp": 8, "ep": 1, "gpus": 8}, "disagg": False}, "offload": "on",
           "hf_repo": "x/M", "engine_image": "vllm/vllm-openai:v0.27.1", "kv_connector": "SimpleCPUOffloadConnector"}
    log = {"non_default": {"tensor_parallel_size": 8, "max_model_len": 1000, "model": "x/M"}, "vllm_version": "0.27.1",
           "kv_connector": "SimpleCPUOffloadConnector", "kv_connector_extra": "{'cpu_bytes_to_use': 1000}"}
    with pytest.raises(S.Unexpressible, match="not simulated"):
        S.resolve(inv, log, None, 4, {"m": "m"})


def test_single_turn_lengths_match_bench_serving():
    """vllm bench random: lengths uniform on [int(L*0.8), L], num_prompts = 10*CONC."""
    w = cov.single_turn_workload(1024, 1024, 4)
    assert "num_requests: 40" in w and '"819":' in w and '"1024":' in w and '"818":' not in w


def test_r6_anchor_reproduces(tmp_path):
    """The harness scorer reproduces the DSV4.1 replay's compare_ix BLIS value on an anchor level (B2 off).
    Needs that replay's local outputs ($IX_R6_DIR); skipped without them."""
    import os
    if not os.environ.get("IX_R6_DIR"):
        pytest.skip("IX_R6_DIR (DSV4.1 InferenceX replay outputs) not set")
    r6 = Path(os.environ["IX_R6_DIR"])
    cmp = json.loads((r6 / "compare_h100-tp8-0918.json").read_text())
    lv = next(x for x in cmp["levels"] if x["concurrency"] == 1)
    m = json.loads((r6 / lv["blis_metrics"]["path"]).read_text())
    got = M.request_metrics(M.blis_rows(m, 0.0, False))
    assert got["p90_intvty"] == lv["blis"]["p90_intvty"]
    assert got["output_tput_tps"] == lv["blis"]["output_tput_tps"]


def test_unresolvable_build_flagged():
    """A build outside vllm-project/vllm history (179dd0fa9) is flagged, not resolved to a release."""
    inv = {"gpu": "h100", "parallel": {"decode": {"tp": 8, "ep": 1, "gpus": 8}, "disagg": False},
           "hf_repo": "x/M", "engine_image": "vllm/vllm-openai:deepseekv41-flash-0909"}
    log = {"non_default": {"tensor_parallel_size": 8, "max_model_len": 1000, "model": "x/M"},
           "vllm_version": "0.1.dev20904+g179dd0fa9"}
    d = S.resolve(inv, log, None, 1, {"m": "m"})
    assert d["build"].get("flag") == "unresolvable build"


def test_legacy_cuda_graph_sizes_is_a_ceiling_on_v0_10():
    """v0.10.2 --cuda-graph-sizes 2048 is a ceiling on the 8-step grid, capped at max_num_batched_tokens."""
    inv = {"gpu": "h100", "parallel": {"decode": {"tp": 2, "ep": 1, "gpus": 2}, "disagg": False},
           "hf_repo": "openai/gpt-oss-120b", "engine_image": "vllm/vllm-openai:v0.10.2"}
    rec = {"tensor-parallel-size": "2", "max-model-len": "2068", "cuda-graph-sizes": "2048",
           "max-num-batched-tokens": "8192", "_provenance": "x"}
    d = S.resolve(inv, None, rec, 4, {"gpt-oss-120b": "gpt-oss-120b"})
    caps = d["engine"]["cudagraph_capture_sizes"]
    assert caps[:4] == [1, 2, 4, 8] and caps[-1] == 2048 and 12 not in caps


def _st_inv():
    return {"gpu": "b200", "parallel": {"decode": {"tp": 4, "ep": 1, "gpus": 4}, "disagg": False},
            "hf_repo": "MiniMaxAI/MiniMax-M2.5", "engine_image": "vllm/vllm-openai:v0.19.0"}


def test_stream_interval_named_and_chunk_itl_not_scored():
    """RUN_DESCRIPTION_AUDIT S1: a run with stream_interval 20 streams 20-token chunks, so vllm bench's pooled ITL
    is ~20x BLIS's per-token step ITL (-95% "error"). It must be a named mismatch and its chunk ITL unscored."""
    log = {"non_default": {"tensor_parallel_size": 4, "max_model_len": 9416, "model": "MiniMaxAI/MiniMax-M2.5",
                           "stream_interval": 20}, "vllm_version": "0.19.0"}
    d = S.resolve(_st_inv(), log, None, 64, {"minimax-m2.5": "minimax-m2.5"})
    assert d["stream_interval"] == 20
    assert any(m["item"] == "stream_interval" for m in d["mismatches"])
    km, un = M.drop_chunk_itl(M.IX_SINGLE, M.SINGLE_UNEXPRESSED, d.get("stream_interval"))
    assert "mean_itl" not in km and "p90_itl" not in km and "median_tpot" in km
    assert M.drop_chunk_itl(M.IX_SINGLE, {}, 1)[0] is M.IX_SINGLE


def test_long_prefill_threshold_stated_and_unstateable_args_named():
    """RUN_DESCRIPTION_AUDIT S3: long_prefill_token_threshold has a schema field and must reach the scenario;
    prefill_schedule_interval / api_server_count have none and must be named, not dropped."""
    log = {"non_default": {"tensor_parallel_size": 4, "max_model_len": 9416, "model": "MiniMaxAI/MiniMax-M2.5",
                           "long_prefill_token_threshold": 4096, "prefill_schedule_interval": 8,
                           "api_server_count": 8}, "vllm_version": "0.19.0"}
    d = S.resolve(_st_inv(), log, None, 64, {"minimax-m2.5": "minimax-m2.5"})
    assert d["engine"]["long_prefill_token_threshold"] == 4096
    assert "long_prefill_token_threshold: 4096" in S.scenario_yaml("x", d)
    items = {m["item"] for m in d["mismatches"]}
    assert {"prefill_schedule_interval", "api_server_count"} <= items


def test_single_turn_pooled_itl_never_shares_agentic_key():
    """RUN_DESCRIPTION_AUDIT S4: single-turn p90_itl is vllm bench's pooled per-chunk ITL; agentic p90_itl is the
    p90 of per-request (lat-ttft)/(osl-1). The aggregate pools errors by key, so the keys must differ."""
    single = {hk for hk, _ in M.IX_SINGLE.values()}
    agentic = {hk for hk, _ in M.IX_AGENTIC.values()}
    assert not ({k for k in single if "itl" in k} & {k for k in agentic if "itl" in k})


# ---------------------------------------------------------------- distribution distance (distance.py, dist_agg.py)
import dist_agg  # noqa: E402
import distance as DIST  # noqa: E402


def test_w1_ks_known_values():
    """W1 of a shifted copy is the shift; KS of disjoint supports is 1; identical samples give 0 for both."""
    a = [1.0, 2.0, 3.0, 4.0]
    assert DIST.w1(a, [x + 0.5 for x in a]) == pytest.approx(0.5)
    assert DIST.ks(a, [x + 10 for x in a]) == 1.0
    assert DIST.w1(a, a) == 0.0 and DIST.ks(a, a) == 0.0
    d = DIST.distances([2.0, 4.0, 6.0], [1.0, 2.0, 3.0])
    assert d["w1_norm"] == pytest.approx(2.0 / 2.0) and d["q50_err_pct"] == pytest.approx(100.0)


def _timeline(tmp_path, reqs, row_id=7, series=None):
    (tmp_path / "request-timeline").mkdir(exist_ok=True)
    (tmp_path / "request-timeline" / f"rt_{row_id}.body").write_text(json.dumps({"startNs": 0, "requests": reqs}))
    if series is not None:
        (tmp_path / "trace-server-metrics").mkdir(exist_ok=True)
        (tmp_path / "trace-server-metrics" / f"tsm_{row_id}.body").write_text(json.dumps({"startNs": 0, "queueDepth": series}))


def _ixr(start_s, lat_ms, ttft, osl, phase="profiling", cancelled=False):
    return {"start": int(start_s * 1e9), "end": int(start_s * 1e9 + lat_ms * 1e6), "ttftMs": ttft, "osl": osl,
            "phase": phase, "cancelled": cancelled}


def test_distance_excludes_warmup_and_cancelled(tmp_path):
    """The measured population is profiling, not cancelled: warmup and cancelled records never enter a distance."""
    reqs = [_ixr(i, 100.0, 10.0, 11) for i in range(5)] + [_ixr(9, 9e6, 9e5, 11, phase="warmup"),
                                                            _ixr(9, 9e6, 9e5, 11, cancelled=True)]
    _timeline(tmp_path, reqs)
    m = _blis([_req(i, 100.0, 10.0, 11) for i in range(5)] + [_req(9, 9e6, 9e5, 11, phase="warmup")])
    d = DIST.point_distances(m, 0.0, 7, {"total_requests_completed": 5}, 1, base=tmp_path)
    assert d["ix_census"] == {"warmup": 1, "cancelled": 1, "incomplete": 0}
    assert d["metrics"]["ttft_ms"]["w1"] == 0.0 and d["metrics"]["e2el_ms"]["n_meas"] == 5


def test_distance_population_mismatch_is_named_not_computed(tmp_path):
    """If the timeline's profiled set is not the set the DB aggregate counts, record a MISMATCH and no distance."""
    _timeline(tmp_path, [_ixr(i, 100.0, 10.0, 11) for i in range(5)])
    m = _blis([_req(i, 100.0, 10.0, 11) for i in range(5)])
    d = DIST.point_distances(m, 0.0, 7, {"total_requests_completed": 6}, 1, base=tmp_path)
    assert d["metrics"] == {} and d["mismatches"][0]["item"] == "request set"


def test_distance_population_assert():
    """Like assert_same: two different request-set definitions are never paired."""
    with pytest.raises(AssertionError):
        DIST.assert_same_population(DIST.REQ_SET, {"def": "all records incl. warmup", "window": "profiling phase"})


def test_blis_queue_series_counts_waiting_then_running():
    """BLIS num_waiting / num_running come from arrival, first schedule and end of every request."""
    reqs = [dict(_req(0.0, 3000.0, 10, 11), scheduling_delay_ms=1500.0), dict(_req(0.0, 3000.0, 10, 11))]
    s = DIST.blis_series(_blis(reqs), 0.0, 3.0)
    assert s["num_waiting"] == [1.0, 1.0, 0.0, 0.0] and s["num_running"] == [1.0, 1.0, 2.0, 0.0]


def test_dp_series_not_compared(tmp_path):
    """dp > 1: InferenceX queue depth sums engines, so the series distance is a named mismatch."""
    _timeline(tmp_path, [_ixr(i, 100.0, 10.0, 11) for i in range(3)], series=[{"t": 0, "running": 1, "waiting": 0}])
    m = _blis([_req(i, 100.0, 10.0, 11) for i in range(3)])
    d = DIST.point_distances(m, 0.0, 7, None, 8, base=tmp_path)
    assert "num_running" not in d["metrics"] and d["mismatches"][-1]["item"] == "server series"


def test_dist_agg_post_stratified_and_abs_quantiles():
    """Strata are weighted by planned size, not by scored count; quantile errors aggregate as |err|."""
    pts = [{"stratum": "a", "run": "r1", "f": {}, "dist": {"metrics": {"ttft_ms": {"w1_norm": 1.0, "q50_err_pct": -10}}}},
           {"stratum": "a", "run": "r2", "f": {}, "dist": {"metrics": {"ttft_ms": {"w1_norm": 1.0, "q50_err_pct": 10}}}},
           {"stratum": "b", "run": "r3", "f": {}, "dist": {"metrics": {"ttft_ms": {"w1_norm": 3.0, "q50_err_pct": 30}}}}]
    s = dist_agg.summarize(pts, {"a": 1, "b": 3}, b=50)["overall"]
    assert s["ttft_ms.w1_norm"]["mean"] == pytest.approx((1 * 1.0 + 3 * 3.0) / 4)
    assert s["ttft_ms.q50_abs_err_pct"]["mean"] == pytest.approx((1 * 10 + 3 * 30) / 4)


# ---------------------------------------------------------------- per-request join (perreq.py)
import perreq as PR  # noqa: E402


def _rt(reqs):
    return {"requests": reqs}


def _mreq(cid, ti, start, ttft, lat, isl, osl, phase="profiling", cancelled=False):
    return {"cid": cid, "ti": ti, "start": start * 1e6, "ack": (start + 1) * 1e6, "end": (start + lat) * 1e6,
            "ttftMs": ttft, "isl": isl, "osl": osl, "phase": phase, "cancelled": cancelled, "srcKind": "weka_main"}


def _breq(stream, turn, start, ttft, e2e, isl, osl, phase="profile", outcome="completed"):
    return {"arrived_at": start / 1e3, "ttft_ms": ttft, "e2e_ms": e2e, "num_prefill_tokens": isl, "output_tokens": osl,
            "phase": phase, "scheduling_delay_ms": 1.0,
            "agentic": {"stream": stream, "turn": turn, "kind": "root", "outcome": outcome}}


def test_perreq_join_pairs_by_conversation_turn_and_occurrence():
    """A recycled trace replays the same (cid, turn) twice: the k-th occurrence must pair with the k-th,
    not with whichever comes first, or late-run requests are compared against early-run ones."""
    meas = PR.measured_rows(_rt([_mreq("a", 0, 0, 10, 100, 1000, 11), _mreq("a", 0, 5000, 20, 200, 1000, 11)]))
    blis = PR.blis_rows({"requests": [_breq("a", 0, 0, 11, 101, 1000, 11), _breq("a", 0, 6000, 21, 201, 1000, 11)]}, 0.0)
    j = PR.join(meas, blis)
    got = sorted((m["ttft_ms"], b["ttft_ms"]) for m, b in j["pairs"])
    assert got == [(10, 11), (20, 21)]


def test_perreq_join_drops_a_request_cancelled_on_either_side_from_both():
    """Cancelled handling must be identical: a request cancelled in BLIS must not stay in the measured set."""
    meas = PR.measured_rows(_rt([_mreq("a", 0, 0, 10, 100, 1000, 11), _mreq("b", 0, 0, 10, 100, 1000, 11)]))
    blis = PR.blis_rows({"requests": [_breq("a", 0, 0, 11, 101, 1000, 11),
                                      _breq("b", 0, 0, 11, 101, 1000, 11, outcome="cancelled")]}, 0.0)
    j = PR.join(meas, blis)
    assert [m["cid"] for m, _ in j["pairs"]] == ["a"]
    assert j["verify"]["cancelled_either_excluded"] == 1


def test_perreq_join_pairs_only_the_profile_phase():
    """A warmup request on one side must never pair with a profile request on the other."""
    meas = PR.measured_rows(_rt([_mreq("a", 0, 0, 10, 100, 1000, 11, phase="warmup")]))
    blis = PR.blis_rows({"requests": [_breq("a", 0, 0, 11, 101, 1000, 11)]}, 0.0)
    j = PR.join(meas, blis)
    assert j["pairs"] == [] and j["verify"]["phase_crossed_keys"] == 1


def test_perreq_join_rejects_unequal_output_length():
    """ignore_eos: both sides generate the trace's OSL; a pair with different OSL is a mis-join."""
    meas = PR.measured_rows(_rt([_mreq("a", 0, 0, 10, 100, 1000, 11)]))
    blis = PR.blis_rows({"requests": [_breq("a", 0, 0, 11, 101, 1000, 12)]}, 0.0)
    assert PR.join(meas, blis)["pairs"] == []


def test_perreq_inflight_counts_warmup_requests_still_running():
    """A warmup request still running when a profile request arrives loads the server; leaving it out
    labels loaded requests as solo."""
    meas = PR.measured_rows(_rt([_mreq("w", 0, 0, 10, 10000, 1000, 11, phase="warmup"),
                                 _mreq("a", 0, 100, 10, 100, 1000, 11)]))
    blis = PR.blis_rows({"requests": [_breq("a", 0, 100, 11, 101, 1000, 11)]}, 0.0)
    PR.join(meas, blis)
    assert meas["profile"][0]["inflight"] == 1


def test_perreq_wasserstein_matches_known_values():
    """W1 of two equal-size samples is the mean gap of order statistics; unequal sizes use the CDF form."""
    assert PR.wasserstein1([0, 1, 2], [1, 2, 3]) == pytest.approx(1.0)
    assert PR.wasserstein1([0.0, 1.0], [0.0, 0.5, 1.0]) == pytest.approx(1 / 6)
    assert PR.ks([0, 1], [0, 1]) == 0


def test_hybrid_attention_block_size_overrides_the_kernel_page():
    """A hybrid Mamba/KDA model (Kimi-K3 with prefix caching) logs the backend's kernel page first and
    then vLLM's raised attention block (vllm/platforms/interface.py:990 at v0.31.0); prefix hits land on
    the raised block, so the replay must run at it, not at the kernel page."""
    log = ("INFO [interface.py:635] Setting kv cache block size to 32 for FLASHINFER_MLA backend.\n"
           "INFO [interface.py:911] Setting attention block size to 1536 tokens to ensure that attention "
           "page size is >= mamba page size.\n")
    assert S.parse_log(log)["block_size"] == "1536"
    assert S.parse_log("Setting kv cache block size to 256 for FLASHMLA backend(s).\n")["block_size"] == "256"
