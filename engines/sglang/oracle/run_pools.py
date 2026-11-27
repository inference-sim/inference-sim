# Original to blis-sglang.
"""Direct traces at SGLang's allocators and pools, and their sizing (design FID-5, §7.6; SGL-6).

A sequence (oracle/scenarios-pools/*.json) builds SGLang's own pools and allocators with their
own constructors on CPU, makes calls at them in orderings no scheduler makes (frees of pages
handed out out of order, free groups holding real slots, a sorted free list, generation clipped
at the pool's edge, extend and decode allocation across page boundaries), and records the calls
as oracle/cache_interfaces.py records every call at the cache interfaces. It also records each
pool's buffer bytes (get_kv_size_bytes) and, for each `cells` entry, the per-token cell that
SGLang's DefaultPoolConfigurator charges (pool_configurator.py:340-536), which divides the KV
budget into the token capacity.

Nothing is replaced: the pools and allocators run unmodified on CPU, and the cell size is
SGLang's own _compute_cell_size called on an object that holds only that entry's inputs (the
model shape, the KV dtype, the layer range). SGLang's own model-config resolution needs CUDA on
this platform for the DSA architectures, so the inputs are given directly; the head count is
the per-rank count at tensor parallelism 1.

A sequence:
    {"description": "...", "model": "...", "seed": 0,
     "pools": [{"as": name, "class": cls, "args": {...}, "kvcache": name}],
     "calls": [{"on": name, "m": method, "args": {...}, "as": name}],
     "cells": [{"name": ..., "architecture": ..., "use_mla_backend": bool, "dtype": "bfloat16",
                "layers": n, "head_num": n, "head_dim": n, "v_head_dim": n, "kv_lora_rank": n,
                "qk_rope_head_dim": n, "index_head_dim": n, "index_topk": n}]}
An argument {"value": name} is a result bound by "as", {"value": [name, start, end]} a slice of
it, {"tensor": [...]} an int64 tensor, {"dtype": "bfloat16"} a torch dtype.

Usage: python -I oracle/run_pools.py direct <sglang checkout> <sequence.json> <out.json>
Environment as for oracle/run_cache_iface.py (ORACLE_LOCKFILE, ORACLE_MODEL_REVISION).
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import traces  # noqa: E402

TIER = 2
INPUTS = ("oracle/run_pools.py", "oracle/cache_interfaces.py", "oracle/traces.py", "oracle/mocks.json",
          "oracle/harness.json", "oracle/requirements.lock")
CLASSES = {
    "MHATokenToKVPool": "sglang.srt.mem_cache.memory_pool",
    "MLATokenToKVPool": "sglang.srt.mem_cache.memory_pool",
    "DSATokenToKVPool": "sglang.srt.mem_cache.memory_pool",
    "ReqToTokenPool": "sglang.srt.mem_cache.memory_pool",
    "TokenToKVPoolAllocator": "sglang.srt.mem_cache.allocator.token",
    "PagedTokenToKVPoolAllocator": "sglang.srt.mem_cache.allocator.paged",
}


def decode(v, bound):
    import torch

    if isinstance(v, dict) and "value" in v:
        x = v["value"]
        return bound[x[0]][x[1]:x[2]] if isinstance(x, list) else bound[x]
    if isinstance(v, dict) and "tensor" in v:
        return torch.tensor(v["tensor"], dtype=torch.int64)
    if isinstance(v, dict) and "dtype" in v:
        return getattr(torch, v["dtype"])
    if isinstance(v, list):
        return [decode(x, bound) for x in v]
    return v


def kv_size_bytes(pool):
    got = pool.get_kv_size_bytes()
    return [int(x) for x in got] if isinstance(got, tuple) else int(got)


def cell_size(entry: dict) -> int:
    """SGLang's DefaultPoolConfigurator._compute_cell_size for one entry's inputs."""
    import torch
    from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
    from transformers import PretrainedConfig

    hf_fields = {k: entry[k] for k in ("index_head_dim", "index_topk") if k in entry}
    hf = PretrainedConfig(architectures=[entry["architecture"]], **hf_fields)
    heads = entry.get("head_num")
    model_config = types.SimpleNamespace(
        hf_config=hf, head_dim=entry.get("head_dim"), v_head_dim=entry.get("v_head_dim"),
        kv_lora_rank=entry.get("kv_lora_rank"), qk_rope_head_dim=entry.get("qk_rope_head_dim"),
        get_num_kv_heads=lambda tp_size, dcp_size=1: heads)
    layers = entry["layers"]
    kvc = types.SimpleNamespace(
        model_config=model_config, kv_cache_dtype=getattr(torch, entry["dtype"]),
        use_mla_backend=entry["use_mla_backend"], server_args=types.SimpleNamespace(enable_hisparse=False),
        is_draft_worker=False, mambaish_config=None, kv_cache_dtype_str="auto", modelslim_cache_layout=None, kv_shard_size=1,
        layer_info=types.SimpleNamespace(start_layer=0, end_layer=layers, num_effective_layers=layers))

    class Configurator:
        # The configurator's own methods, on an object holding only the KV dtype string.
        kv_cache_dtype_str = "auto"
        _compute_cell_size = DefaultPoolConfigurator._compute_cell_size
        _compute_dsa_indexer_cell_size = DefaultPoolConfigurator._compute_dsa_indexer_cell_size
        _compute_qsa_cell_size = staticmethod(DefaultPoolConfigurator._compute_qsa_cell_size)

    return int(Configurator()._compute_cell_size(kvc, layers))


def drive(sglang_root: str, path: str) -> dict:
    import importlib

    import run_scheduler

    for name in [n for n in os.environ if n.startswith("SGLANG_")]:
        os.environ.pop(name)
    os.environ.update(run_scheduler.KEPT_SGLANG_ENV)
    from cache_interfaces import Monitor
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    with open(path) as f:
        sq = json.load(f)
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=sq.get("page_size", 1)))
    mon = Monitor()
    bound: dict = {}
    built = []
    with mon:
        for p in sq["pools"]:
            cls = getattr(importlib.import_module(CLASSES[p["class"]]), p["class"])
            kwargs = {k: decode(v, bound) for k, v in p["args"].items()}
            if "kvcache" in p:
                kwargs["kvcache"] = bound[p["kvcache"]]
            bound[p["as"]] = cls(device="cpu", **kwargs)
            built.append(p)
        for c in sq["calls"]:
            out = getattr(bound[c["on"]], c["m"])(**{k: decode(v, bound) for k, v in c.get("args", {}).items()})
            if "as" in c:
                bound[c["as"]] = out
    sizing = [{"pool": p["as"], "class": p["class"], "kv_size_bytes": kv_size_bytes(bound[p["as"]])}
              for p in built if hasattr(bound[p["as"]], "get_kv_size_bytes")]
    cells = [{"name": e["name"], "cell_size": cell_size(e)} for e in sq.get("cells", [])]
    return {"provenance": traces.provenance(TIER, sglang_root, path, sq, INPUTS), "kind": "direct",
            "calls": mon.calls, "sizing": sizing, "cells": cells}


def main(argv) -> int:
    if len(argv) != 4 or argv[0] != "direct":
        print(__doc__.split("Usage:", 1)[1].split("Environment", 1)[0].strip(), file=sys.stderr)
        return 2
    _, root, path, out = argv
    if traces.refuses_write(out):
        print(traces.REFUSAL, file=sys.stderr)
        return 2
    sys.path.insert(0, os.path.join(os.path.abspath(root), "python"))
    trace = drive(root, path)
    if not any(c["i"] == "allocator" for c in trace["calls"]) and not trace["sizing"]:
        print(f"{path}: no allocator call and no pool was recorded", file=sys.stderr)
        return 1
    traces.write(out, trace)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
