"""Behavioural tests for the parity rules (blockers.py; repos/cov/PARITY_FIXES.md).
Run: uv run --no-project --with pytest python -I -m pytest -q test_parity.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blockers as BLK  # noqa: E402
import cov  # noqa: E402
import logfacts as LF  # noqa: E402


def _pt(m, conc=512, model="gpt-oss-120b", served="catalog repo match (openai/gpt-oss-120b)"):
    return {"scenario": "single_turn", "conc": conc, "measured_rows": [{"measured": m}],
            "resolved": {"model": model, "sources": {"model": served}}}


def _row(tput_per_gpu, e2e_s, ttft_s=0.1, osl=1024):
    tpot = (e2e_s - ttft_s) * 1e3 / (osl - 1)
    return {"output_tput_per_gpu": tput_per_gpu, "mean_e2el_ms": e2e_s * 1e3, "mean_ttft_ms": ttft_s * 1e3,
            "mean_tpot_ms": tpot}


def _names(pb):
    return {b["name"] for b in pb}


def test_offered_load_not_reached_blocks_client_limited_point():
    """gpt-oss H100 c512 row 74839: 187.9 tok/s/GPU x 4 GPUs x 26.5 s / 1024 = 19.5 in flight, not 512."""
    x = _pt(_row(187.9, 26.5), conc=512)
    assert BLK.offered_load_ratio(x["measured_rows"][0]["measured"], 512, 4) < 0.05
    assert "offered load not reached" in _names(BLK.parity_blockers(x, None, None, None, 4))


def test_offered_load_reached_is_not_blocked():
    """A closed loop holding ~0.97 x C in flight (the corpus mode) must stay scoreable."""
    e2e, osl, conc, gpus = 10.0, 1024, 64, 8
    tput = 0.97 * conc * osl / e2e / gpus
    x = _pt(_row(tput, e2e, osl=osl), conc=conc)
    assert not BLK.parity_blockers(x, None, None, osl * 10 // 9, gpus)


def test_osl_identity_gate_at_5_percent():
    """Replayed mean OSL is (int(0.8 L) + L) / 2; measured 30% short is a different workload, 3% is not."""
    want = BLK.replayed_mean_osl(1024)
    m = lambda osl: _row(0.97 * 8 * osl / 10.0, 10.0, osl=round(osl))  # noqa: E731
    assert "workload identity: OSL" in _names(BLK.parity_blockers(_pt(m(want * 0.7), conc=8), None, None, 1024, 1))
    assert not BLK.parity_blockers(_pt(m(want * 0.97), conc=8), None, None, 1024, 1)


def test_nvfp4_served_checkpoint_blocked_by_quant_or_backend_not_only_name():
    """Kimi-K2.5-NVFP4 from a local path, and DSV4-Pro-NVFP4 whose quantization reads deepseek_v4_fp8 but whose MoE
    backend is NvFp4: both priced as a non-NVFP4 catalog entry must be blocked."""
    base = {"quant_method": "compressed-tensors", "repo": "moonshotai/Kimi-K2.5"}
    kimi = {"model": "/lustre/fsw/models/Kimi-K2.5-NVFP4", "quantization": "modelopt_fp4"}
    by_backend = {"model": "/scratch/models/dsv4", "quantization": "deepseek_v4_fp8",
                  "moe_backend": "Using 'FLASHINFER_CUTEDSL' NvFp4 MoE backend"}
    x = _pt(_row(100, 10.0), conc=1, model="kimi-k2.5")
    opaque = {"model": "/data/models/4dcbe117c837f0376137d295d69424c15201a899", "quantization": "modelopt_fp4"}
    for f in (kimi, by_backend, opaque):
        assert "served checkpoint not in catalog" in _names(BLK.parity_blockers(x, f, base, None, 8))
    plain = {"model": "moonshotai/Kimi-K2.5", "quantization": "compressed-tensors",
             "moe_backend": "Using 'MARLIN' WNA16 MoE backend"}
    assert not BLK.parity_blockers(x, plain, base, None, 8)
    assert not BLK.parity_blockers(x, kimi, {"quant_method": "modelopt", "repo": "nvidia/Kimi-K2.5-NVFP4"}, None, 8)


def test_cudagraph_profiling_override_is_not_a_pool_override():
    """vLLM logs "Overriding num_gpu_blocks=0 with num_gpu_blocks_override=2048" while it profiles CUDA-graph memory;
    the real pool is sized after. Only an override in the run's own args blocks the point."""
    log = ("non-default args: {'model': 'deepseek-ai/DeepSeek-V4-Pro', 'max_model_len': 4096}\n"
           "Overriding num_gpu_blocks=0 with num_gpu_blocks_override=2048\n")
    f = LF.facts(log)
    assert f["override_lines"] == ["0>2048"] and BLK.override_user_set(f) is None
    f2 = LF.facts(log.replace("'max_model_len': 4096", "'max_model_len': 4096, 'num_gpu_blocks_override': 512"))
    x = _pt(_row(100, 10.0), conc=1)
    assert "num_gpu_blocks_override set" in _names(BLK.parity_blockers(x, f2, None, None, 1))


def test_dsv4_pro_logged_size_is_never_compared_as_pool_tokens():
    """The log's size and concurrency lines agree (same function), so the self-consistency check passes; the guard must
    still refuse the token comparison for DeepSeek-V4-Pro."""
    assert cov.kv_units_differ("no-such-log", "deepseek-v4-pro")
    assert not cov.kv_units_differ("no-such-log", "gpt-oss-120b")


def test_dsv4_pro_pool_blocks_reproduce_log_13691():
    """H200 TP8 v0.25.1, mml 2304, mnbt 512, 27.24 GiB: vLLM logs 90.63x. The derived pool is ~19.8k blocks and is
    accepted; a log whose concurrency line disagrees is refused (layout not this one)."""
    f = {"avail_gib": [27.24], "max_conc": [90.63, 2304], "block_size": "256"}
    nb = BLK.dsv4_pro_pool_blocks(f, 2304, 512)
    assert nb is not None and 19800 <= nb <= 19900
    assert BLK.dsv4_pro_pool_blocks({**f, "max_conc": [45.0, 2304]}, 2304, 512) is None
