# Original to blis-vllm.
"""Oracle for core/kv's start-up KV pool figures: vLLM's own sizing, run unmodified.

For each case the layers' KV cache specs are built with vLLM's own spec classes
(FullAttentionSpec, SlidingWindowSpec), handed to vLLM's own get_kv_cache_configs (one dict of
specs per worker, so a pipeline-parallel case is each stage's layers and memory), and the
resulting KVCacheConfig is read back: the KV cache groups and their layers, num_blocks, and the
figure the engine logs at start-up, "GPU KV cache size: N tokens", from get_kv_cache_capacity.

The cases are synthetic. They reproduce the shape of a hybrid sliding-window model with a
sliding-window-only drafter (Muse Spark Open's layer_types, 46 sliding-window layers of 2048
tokens and 16 full-attention layers, with its DSpark drafter's five sliding-window layers
after them), at the per-rank KV heads of tensor-parallel widths 4 and 8, at fp8 and bf16, and
under four pipeline stages. The memory each worker has for KV is a stated input, not a
measurement.

Replacements (design FID-1): the VllmConfig is a SimpleNamespace that holds only what these
functions read (model, scheduler, parallel, device, attention and speculative settings,
max_in_flight_tokens and num_prefill_lookahead_tokens); the CacheConfig is vLLM's own, and its
KV cache layout is resolved by vLLM's own resolve_kv_cache_layout over its default preference.
No decision is replaced.

The release is the one internal/upstream/upstream.go names (Commit); the functions are found where
that release keeps them (vllm/v1/core/kv_cache_utils.py, or vllm/v1/core/kv_cache_planning.py
in a release that has it), so the same cases run on another release's source.

Usage:
    python -I oracle/kvcapacity/extract.py <vllm source root> print            the cases and vLLM's answers, as JSON
    python -I oracle/kvcapacity/extract.py <vllm source root> check <file>     fail unless <file> is what print gives

<vllm source root> is a directory holding vllm/ at internal/upstream's Commit, run with a Python that has
torch installed (the oracle environment, oracle/requirements.lock).
"""
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 1 << 30
MAX_MODEL_LEN = 1 << 20


def upstream_commit():
    """The modelled release's commit, from internal/upstream/upstream.go (Commit = "...")."""
    with open(os.path.join(HERE, "..", "..", "internal", "upstream", "upstream.go"), encoding="utf-8") as f:
        m = re.search(r'Commit\s*=\s*"([0-9a-f]{40})"', f.read())
    assert m, "internal/upstream/upstream.go names no Commit"
    return m.group(1)


def target_layers():
    """Muse Spark Open's text layer_types in model order: one sliding-window layer, then fifteen
    times one full-attention layer and three sliding-window layers, then one full-attention
    layer (catalog muse-spark-open-816b-a42b-nvfp4 config.json)."""
    return ["S"] + ["F", "S", "S", "S"] * 15 + ["F"]


DRAFTER = ["S"] * 5  # DSpark's five 2048-token windows (global_attn_cfg), after the target's


def pp_indices(layers, pp, stage):
    """get_pp_indices (vllm/distributed/utils.py) without VLLM_PP_LAYER_PARTITION."""
    parts = [layers // pp] * pp
    for i in range(2, layers % pp + 2):
        parts[pp - i] += 1
    start = sum(parts[:stage])
    return start, start + parts[stage]


def cases():
    out = []

    def case(name, kv_heads, dtype, mnbt, batches, lookahead, drafter, avail_gib, pp=1):
        layers = target_layers() + (DRAFTER if drafter else [])
        target = len(target_layers())
        stages = []
        for s in range(pp):
            lo, hi = pp_indices(target, pp, s)
            if s == pp - 1:
                hi = len(layers)  # the drafter runs on the last stage
            stages.append([lo, hi])
        out.append({"name": name, "block_size": 16, "num_kv_heads": kv_heads, "head_size": 64, "dtype": dtype,
                    "sliding_window": 2048, "layers": "".join(layers), "stages": stages,
                    "max_model_len": MAX_MODEL_LEN, "max_num_batched_tokens": mnbt,
                    "max_concurrent_batches": batches, "num_prefill_lookahead_tokens": lookahead,
                    "available_bytes": [int(g * GIB) for g in avail_gib]})

    case("tp8-bf16", 2, "bf16", 16384, 2, 0, False, [90])
    case("tp8-bf16-sync", 2, "bf16", 16384, 1, 0, False, [90])
    case("tp8-bf16-mnbt32768", 2, "bf16", 32768, 2, 0, False, [88])
    case("tp4-pp4-bf16", 4, "bf16", 8192, 4, 0, False, [130, 128, 128, 131], pp=4)
    case("tp4-pp4-bf16-async-v2", 4, "bf16", 8192, 5, 0, False, [130, 128, 128, 131], pp=4)
    case("tp4-fp8", 4, "fp8", 16384, 2, 0, False, [25])
    case("tp4-fp8-dspark", 4, "fp8", 16384, 2, 1, True, [25])
    case("tp4-fp8-dspark-mnbt8192", 4, "fp8", 8192, 2, 1, True, [34])
    case("tp4-pp2-fp8-dspark", 4, "fp8", 8192, 2, 1, True, [60, 58], pp=2)
    return out


def load(root):
    if root not in sys.path:
        sys.path.append(root)
    import torch
    from vllm.config.cache import CacheConfig
    from vllm.v1.attention.backends.utils import _DEFAULT_LAYOUT_PREFERENCE, resolve_kv_cache_layout
    import vllm.v1.core.kv_cache_utils as utils
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    try:
        import vllm.v1.core.kv_cache_planning as planning
    except ImportError:
        planning = None
    return types.SimpleNamespace(torch=torch, CacheConfig=CacheConfig, layouts=_DEFAULT_LAYOUT_PREFERENCE,
                                 resolve_layout=resolve_kv_cache_layout, utils=utils, planning=planning,
                                 register=register_all_kvcache_specs, Full=FullAttentionSpec,
                                 Sliding=SlidingWindowSpec)


def run(v, c):
    dtype = {"fp8": v.torch.uint8, "bf16": v.torch.bfloat16}[c["dtype"]]
    common = dict(block_size=c["block_size"], num_kv_heads=c["num_kv_heads"], head_size=c["head_size"], dtype=dtype)
    specs = []
    for lo, hi in c["stages"]:
        w = {}
        for i in range(lo, hi):
            name = f"model.layers.{i}.self_attn.attn"
            if c["layers"][i] == "S":
                w[name] = v.Sliding(sliding_window=c["sliding_window"], **common)
            else:
                w[name] = v.Full(**common)
        specs.append(w)
    pp = len(c["stages"])
    vc = types.SimpleNamespace(
        cache_config=v.CacheConfig(block_size=c["block_size"], cache_dtype="fp8" if c["dtype"] == "fp8" else "auto"),
        model_config=types.SimpleNamespace(max_model_len=c["max_model_len"], original_max_model_len=c["max_model_len"],
                                           supports_kv_cache_grouping=False),
        scheduler_config=types.SimpleNamespace(disable_hybrid_kv_cache_manager=False,
                                               max_num_batched_tokens=c["max_num_batched_tokens"]),
        parallel_config=types.SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1,
                                              tensor_parallel_size=1, pipeline_parallel_size=pp),
        device_config=types.SimpleNamespace(device_type="cuda"),
        attention_config=types.SimpleNamespace(hisparse_config=None),
        kv_transfer_config=None, speculative_config=None,
        max_in_flight_tokens=c["max_concurrent_batches"] * c["max_num_batched_tokens"],
        num_prefill_lookahead_tokens=c["num_prefill_lookahead_tokens"])
    v.register(vc)
    every = [s for w in specs for s in w.values()]
    v.resolve_layout(vc, [[m.name for m in v.layouts]], every)
    if v.planning is not None and hasattr(v.planning, "get_global_kv_cache_groups"):
        groups = v.planning.get_global_kv_cache_groups(vc, specs)
        configs = v.planning.get_kv_cache_configs(vc, specs, c["available_bytes"], groups)
    else:
        configs = v.utils.get_kv_cache_configs(vc, specs, c["available_bytes"])
    cfg = v.utils.generate_scheduler_kv_cache_config(configs)
    tokens, concurrency = v.utils.get_kv_cache_capacity(vc, cfg)
    per_request = sum(v.utils.cdiv(g.kv_cache_spec.max_memory_usage_bytes(vc), g.kv_cache_spec.page_size_bytes)
                      for g in cfg.kv_cache_groups)
    # The group layers of the whole model: each worker's config holds its own layers of every group.
    sizes = [0] * len(cfg.kv_cache_groups)
    for w in configs:
        for i, g in enumerate(w.kv_cache_groups):
            sizes[i] += len(g.layer_names)
    kinds = ["sliding_window" if isinstance(g.kv_cache_spec, v.Sliding) else "full" for g in cfg.kv_cache_groups]
    return {"groups": [{"kind": k, "layers": n} for k, n in zip(kinds, sizes)],
            "padding_layers": max(sizes) * len(sizes) - len(c["layers"]),
            "num_blocks": cfg.num_blocks, "blocks_per_request": per_request,
            "kv_cache_size_tokens": tokens, "max_concurrency": concurrency}


def build(root):
    v = load(root)
    return {"commit": upstream_commit(), "cases": [dict(c, vllm=run(v, c)) for c in cases()]}


def main(argv):
    if len(argv) < 3 or argv[2] not in ("print", "check"):
        raise SystemExit(__doc__)
    out = json.dumps(build(argv[1]), indent=1, sort_keys=True) + "\n"
    if argv[2] == "print":
        sys.stdout.write(out)
        return
    with open(argv[3], encoding="utf-8") as f:
        if f.read() != out:
            raise SystemExit(f"{argv[3]} is not what vLLM at {upstream_commit()} gives; regenerate it with `print`")
    print("ok:", argv[3])


if __name__ == "__main__":
    main(sys.argv)
