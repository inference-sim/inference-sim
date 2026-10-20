"""Parity blockers (repos/cov/PARITY_FIXES.md). Each rule decides from the run's own description and measured rows
alone, never from BLIS's error, whether a point is compared apples to apples. A point that fails one gets a NAMED
blocker and its reason.

  offered load not reached     single-turn closed-loop client: Little's-law in-flight / nominal concurrency < 0.8
  served checkpoint not in catalog   the log's served checkpoint (id, quantization, MoE backend) is NVFP4 and the
                               catalog entry the harness priced is not
  num_gpu_blocks_override set  the run's own args set the pool size (vLLM's CUDA-graph profiling override is not)
  workload identity: OSL       implied measured mean OSL differs from the replayed mean OSL by more than 5%

It also replaces the DSV4-Pro pool comparison: vLLM logs get_kv_cache_capacity (max_concurrency x max_model_len,
kv_cache_utils.py:2444-2472@v0.31.0), which is not pool tokens. dsv4_pro_pool_blocks() derives vLLM's pool block
count from the log, and the pool is compared block for block where the derivation reproduces the logged
max concurrency.
"""

from __future__ import annotations

import re

OFFERED_LOAD_MIN = 0.8
OSL_TOL = 0.05
RANGE_RATIO = 0.8

NVFP4_QUANT = {"modelopt_fp4", "modelopt"}


def _v(d: dict, k: str):
    x = d.get(k)
    return x["value"] if isinstance(x, dict) else x


def implied_osl(m: dict) -> float | None:
    """vllm bench: mean_tpot = mean of (e2el - ttft)/(osl - 1), so (mean_e2el - mean_ttft)/mean_tpot + 1 is the mean
    output length (to the spread of osl)."""
    try:
        return (_v(m, "mean_e2el_ms") - _v(m, "mean_ttft_ms")) / _v(m, "mean_tpot_ms") + 1
    except (TypeError, ZeroDivisionError):
        return None


def offered_load_ratio(m: dict, conc: int, gpus: int) -> float | None:
    """Little's law on the measured row: in flight = completion rate x mean E2E latency, where completion rate =
    output_tput_per_gpu x GPUs / mean OSL (InferenceX divides throughput by its TP env, process_result.py)."""
    osl = implied_osl(m)
    tput, e2e = _v(m, "output_tput_per_gpu"), _v(m, "mean_e2el_ms")
    if not osl or tput is None or e2e is None or not conc:
        return None
    return tput * gpus / osl * e2e / 1e3 / conc


def replayed_mean_osl(osl: int) -> float:
    """The harness draws uniform integer lengths in [int(L * 0.8), L] (cov.single_turn_workload)."""
    return (int(osl * RANGE_RATIO) + osl) / 2


def served_is_nvfp4(facts: dict | None, served: str | None) -> bool:
    f = facts or {}
    if (f.get("quantization") or "") in NVFP4_QUANT:
        return True
    if re.search(r"\bNvFp4 MoE backend", f.get("moe_backend") or ""):
        return True
    return "nvfp4" in (f.get("model") or served or "").lower()


def catalog_is_nvfp4(catalog_quant: dict | None) -> bool:
    q = catalog_quant or {}
    return q.get("quant_method") == "modelopt" or "nvfp4" in str(q.get("repo", "")).lower()


def override_user_set(facts: dict | None) -> int | None:
    """vLLM sets num_gpu_blocks_override itself to max_cudagraph_capture_size while it profiles CUDA-graph memory and
    restores it before sizing the real pool (gpu_model_runner.py _init_minimal_kv_cache_for_profiling, :6484-6505
    @v0.31.0); that logs "Overriding num_gpu_blocks=0 with ...". Only an override in the run's own args sizes the
    pool."""
    f = facts or {}
    if not f.get("override_in_args"):
        return None
    for ln in f.get("override_lines") or []:
        return int(ln.split(">")[1])
    return -1


def parity_blockers(x: dict, facts: dict | None, catalog_quant: dict | None, nominal_osl: int | None,
                    gpus: int) -> list[dict]:
    """x: a scored result (cov.run_point). Returns [] or the named blockers, each {name, reason}."""
    out = []
    m = x["measured_rows"][0]["measured"] if x.get("measured_rows") else {}
    if x["scenario"] == "single_turn":
        r = offered_load_ratio(m, x["conc"], gpus)
        if r is not None and r < OFFERED_LOAD_MIN:
            out.append({"name": "offered load not reached",
                        "reason": f"measured in flight {r * x['conc']:.1f} = {r:.2f} x nominal {x['conc']} (< "
                                  f"{OFFERED_LOAD_MIN}): the client did not sustain its concurrency; the excess "
                                  "waited client-side, which a closed-loop replay at any concurrency does not state"})
        osl = implied_osl(m)
        if osl and nominal_osl:
            want = replayed_mean_osl(nominal_osl)
            if abs(osl / want - 1) > OSL_TOL:
                out.append({"name": "workload identity: OSL",
                            "reason": f"implied measured mean OSL {osl:.0f} vs replayed {want:.0f} "
                                      f"({(osl / want - 1) * 100:+.1f}%, tolerance {OSL_TOL * 100:.0f}%)"})
    served = ((x.get("resolved") or {}).get("sources") or {}).get("model", "")
    if served_is_nvfp4(facts, served) and not catalog_is_nvfp4(catalog_quant):
        f = facts or {}
        out.append({"name": "served checkpoint not in catalog",
                    "reason": f"served {f.get('model') or served} (quantization {f.get('quantization')}, "
                              f"{f.get('moe_backend')}) is NVFP4; priced as catalog {x['resolved']['model']} "
                              f"({(catalog_quant or {}).get('quant_method')})"})
    ov = override_user_set(facts)
    if ov is not None:
        out.append({"name": "num_gpu_blocks_override set",
                    "reason": f"the run's args set num_gpu_blocks_override={ov}; blis-schemas states no pool override"})
    return out


# ---------------------------------------------------------------- DSV4-Pro pool, block for block
def _cd(a, b):
    return -(-a // b)


def dsv4_pro_layout(mml: int, mnbt: int, bs: int = 256, compress_ratios: list[int] | None = None) -> tuple[int, int]:
    """(pool blocks one max_model_len request reserves, packed pool bytes per block) for DeepSeek-V4-Pro, fp8_ds_mla,
    block 256 (scripts/dsv4pro_kvpool/repro.py; vLLM kv_cache_utils.py get_max_concurrency_for_kv_cache_config
    :1090-1122@v0.31.0, _get_kv_cache_config_packed; specs models/deepseek_v4/attention.py, compressor.py)."""
    def al(x):
        return _cd(x, 576) * 576
    r = compress_ratios or ([128, 128] + [4, 128] * 29 + [4])
    n4, n128, nl = r.count(4), r.count(128), len(r)
    c4a, c4i, c128a = al(bs // 4 * 584), al(bs // 4 * 132), al(bs // 128 * 584)
    swa, st4, st128, ist = al(64 * 584), al(4 * 2048 * 4), al(8 * 1024 * 4), al(4 * 512 * 4)

    def pad(p):
        return min(x for x in (c4a, c4i, c128a) if x >= p)

    def win(w, b):
        return _cd(min(w - 1 + mnbt, mml), b) + 1
    full = n4 * c4a + n4 * c4i + n128 * c128a
    groups = [(_cd(mml, bs), full), (win(128, 64), 31 * pad(swa)), (win(128, 64), (nl - 31) * pad(swa)),
              (win(8, 4), n4 * pad(st4) + n4 * pad(ist)), (win(128, 8), n128 * pad(st128))]
    return _cd(sum(n * p for n, p in groups), full), 31 * c4a + n4 * c4i + 31 * c128a


def dsv4_pro_pool_blocks(facts: dict | None, mml: int, mnbt: int) -> int | None:
    """vLLM's pool block count from the log's "Available KV cache memory", accepted only when it reproduces the log's
    own "Maximum concurrency" line to 1% at the same max_model_len (else the image's layout is not this one)."""
    f = facts or {}
    if not f.get("avail_gib") or not f.get("max_conc") or str(f.get("block_size")) != "256":
        return None
    per_req, per_block = dsv4_pro_layout(mml, mnbt)
    nb = int(f["avail_gib"][0] * 2 ** 30 // per_block)
    conc, at = f["max_conc"]
    if at != mml or abs(nb / per_req / conc - 1) > 0.01:
        return None
    return nb
