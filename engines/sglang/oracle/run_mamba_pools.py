# Original to blis-sglang.
"""Direct traces at SGLang's hybrid (mamba / linear-attention) pools and their sizing (design
FID-5, §7.6; SGL-6; inference-sim/inference-sim#2366).

A sequence (oracle/scenarios-mamba-pools/*.json) builds SGLang's own MambaPool,
HybridReqToTokenPool and HybridLinearKVPool with their own constructors on CPU, from the state
shape SGLang's own model configuration computes (Qwen3_5MoeTextConfig.mamba2_cache_params, built
from the model's published linear-attention fields), and makes calls at them in orderings no
scheduler makes. Each call records its arguments, its answer, the requests it touched and the
state of the request pool, the mamba slot allocator and the ping-pong maps after it. Each pool's
buffer bytes (MambaPool.mem_usage_bytes, the physical bytes) are recorded, and each `budgets`
entry calls SGLang's own KVCacheConfigurator._handle_max_mamba_cache and resolve_max_num_reqs on
an object that holds only that entry's inputs (the model configuration, the speculative
algorithm, the server arguments).

Nothing is replaced: pools, allocators and the configurator's methods run unmodified on CPU.

A sequence:
    {"description": "...", "model": "...", "seed": 0,
     "server_args": {field: value},            # the context SGLang's code reads; a budget's own override it
     "config": {"class": "Qwen3_5MoeTextConfig", "fields": {...}},
     "pools": [{"as": name, "class": cls, "args": {...}}],
     "calls": [{"op": "req", "as": r, "set": {kv field: value}}
               | {"on": name, "m": method, "args": {...}, "as": name}
               | {"op": "set", "req": r, "set": {kv field: value}}],
     "budgets": [{"name": ..., "total_rest_memory_gib": x, "server_args": {...},
                  "spec_algorithm": "EAGLE"|null, "token_capacity": n, "context_len": n}]}
An argument {"req": name} is a request, {"reqs": [names]} a list of them, {"tensor": [...]} an
int64 tensor, {"value": name} a result bound by "as", {"pool": name.attr} an attribute of a pool.

Usage: python -I oracle/run_mamba_pools.py direct <sglang checkout> <sequence.json> <out.json>
Environment as for oracle/run_cache_iface.py (ORACLE_LOCKFILE, ORACLE_MODEL_REVISION).
"""
import dataclasses
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import traces  # noqa: E402

TIER = 2
INPUTS = ("oracle/run_mamba_pools.py", "oracle/traces.py", "oracle/mocks.json", "oracle/harness.json",
          "oracle/requirements.lock")
KV_FIELDS = ("req_pool_idx", "kv_allocated_len", "mamba_pool_idx", "mamba_ping_pong_track_buffer",
             "mamba_ping_pong_empty", "mamba_next_track_idx", "mamba_last_track_idx",
             "mamba_last_track_seqlen", "mamba_prev_track_seqlen", "mamba_needs_clear",
             "mamba_cow_src_index")


class Req:
    """A request as the pool code reads one: a row of ReqKvInfo (managers/schedule_batch.py)."""

    def __init__(self, rid, fields):
        from sglang.srt.managers.schedule_batch import ReqKvInfo

        global KV_ALL
        KV_ALL = {f.name for f in dataclasses.fields(ReqKvInfo)}
        self.rid = rid
        self.mamba_branching_seqlen = None
        self.origin_input_ids = []
        self.kv = ReqKvInfo(**{k: v for k, v in fields.items() if k in KV_ALL})
        self.update({k: v for k, v in fields.items() if k not in KV_ALL})

    def update(self, fields):
        import dataclasses

        import torch

        for k in [k for k in fields if k not in KV_ALL]:
            setattr(self, k, fields[k])
        self.kv = dataclasses.replace(
            self.kv, **{k: torch.tensor(v, dtype=torch.int64) if isinstance(v, list) else v
                        for k, v in fields.items() if k in KV_ALL})


def enc(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    return x


def kv_state(req):
    return {f: enc(getattr(req.kv, f)) for f in KV_FIELDS}


def decode(v, bound, reqs):
    import torch

    if isinstance(v, dict) and "req" in v:
        return reqs[v["req"]]
    if isinstance(v, dict) and "reqs" in v:
        return [reqs[n] for n in v["reqs"]]
    if isinstance(v, dict) and "tensor" in v:
        return torch.tensor(v["tensor"], dtype=torch.int64)
    if isinstance(v, dict) and "value" in v:
        return bound[v["value"]]
    if isinstance(v, dict) and "pool" in v:
        obj = bound[v["pool"].split(".")[0]]
        for a in v["pool"].split(".")[1:]:
            obj = getattr(obj, a)
        return obj
    if isinstance(v, dict) and "dtype" in v:
        return getattr(torch, v["dtype"])
    return v


def physical_bytes(mp):
    """MambaPool.mem_usage_bytes as the pool itself computes it for its log line: the speculative
    conv-window view over-reports, so the physical buffers are charged instead
    (memory_pool.py:1086-1099@297e54fd)."""
    from sglang.srt.mem_cache.memory_pool import get_tensor_size_bytes

    total = mp.mamba_cache.mem_usage_bytes()
    if isinstance(mp.mamba_cache, mp.SpeculativeState):
        total -= get_tensor_size_bytes(mp.mamba_cache.intermediate_conv_window)
        total += get_tensor_size_bytes(mp._intermediate_conv_window_phys)
    return int(total)


def pool_state(hybrid):
    """What the request pool and its mamba pieces hold after a call."""
    out = {"req_free": list(hybrid.free_slots), "mamba_free": enc(hybrid.mamba_allocator.free_slots)}
    out["req_to_mamba"] = enc(hybrid.req_index_to_mamba_index_mapping)
    if hybrid.enable_mamba_extra_buffer:
        out["ping_pong_map"] = enc(hybrid.req_index_to_mamba_ping_pong_track_buffer_mapping)
    return out


def lifecycle_call(target, method, args):
    """A scheduler-side request-pool call. At v0.5.23 the hybrid request table bound a request's
    row and state slots in `alloc` and gave the state back in `free_mamba_cache`; at c4af57ff those
    are the state storage's `acquire` (bound to rows by static_kv_storage.acquire_rows, as
    allocation.py:331 reaches them) and `release`. The sequence names keep the v0.5.23 calls."""
    if hasattr(target, "state") and method == "alloc":
        from sglang.srt.mem_cache.static_kv_storage import acquire_rows

        return acquire_rows(target, target.state, **args)
    if hasattr(target, "state") and method == "free_mamba_cache":
        return target.state.release(target, args["req"],
                                    mamba_ping_pong_track_buffer_to_keep=args.get("mamba_ping_pong_track_buffer_to_keep"))
    return getattr(target, method)(**args)


def build_config(sq):
    import importlib

    c = sq["config"]
    mod = importlib.import_module("sglang.srt.configs.qwen3_5")
    return getattr(mod, c["class"])(**c["fields"])


def ssm_dtype_env(server_args):
    """--mamba-ssm-dtype reaches the pools through SGLANG_MAMBA_SSM_DTYPE, which SGLang's start-up
    sets from the flag (arg_groups/serving_hook.py:439@297e54fd); the same setting, for this scope."""
    import contextlib

    from sglang.srt.environ import envs

    value = server_args.get("mamba_ssm_dtype")
    return envs.SGLANG_MAMBA_SSM_DTYPE.override(value) if value is not None else contextlib.nullcontext()


def mamba_budget(entry, cfg, K, scenario_args):
    from sglang.srt.runtime_context import get_context, get_schedule
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = entry.get("spec_algorithm")
    spec = SpeculativeAlgorithm.from_string(algo) if algo else SpeculativeAlgorithm.NONE
    class Configurator:
        """Holds only this entry's inputs; the configurator's own methods run on it."""

        _calculate_mamba_ratio = K._calculate_mamba_ratio

        def __init__(self):
            self.mambaish_config = self.hybrid_gdn_config = cfg
            self.hybrid_kda_config = None
            self.pp_size = 1
            self.attn_dp_size = entry.get("attn_dp_size", 1)
            self.model_config = types.SimpleNamespace(num_hidden_layers=cfg.num_hidden_layers,
                                                      context_len=entry.get("context_len", 4096))
            self.spec_algorithm = spec
            self.extra_mamba_cache_bytes_per_req = 0
            self.extra_mamba_cache_bytes_per_request_row = 0
            self.allocate_intermediate_ssm = entry.get("allocate_intermediate_ssm", True)

        def fused_entry_bytes(self, name):
            return None

    kvc = Configurator()
    out = {"name": entry["name"]}
    args = {**scenario_args, **entry.get("server_args", {})}
    with get_context().override_server_args(model_path="dummy", **args), ssm_dtype_env(args):
        try:
            out["rest_gib"] = K._handle_max_mamba_cache(kvc, entry["total_rest_memory_gib"])
            out["max_mamba_cache_size"] = get_schedule().max_mamba_cache_size
            out["ratio"] = K._calculate_mamba_ratio(kvc)
            if "token_capacity" in entry:
                out["max_num_reqs"] = K.resolve_max_num_reqs(kvc, entry["token_capacity"])
        except (RuntimeError, AssertionError) as e:
            out["raises"] = type(e).__name__
            out["message"] = str(e).split(".")[0].split(" Computed")[0]
    return out


def drive(sglang_root: str, path: str) -> dict:
    import importlib

    import run_scheduler

    for name in [n for n in os.environ if n.startswith("SGLANG_")]:
        os.environ.pop(name)
    os.environ.update(run_scheduler.KEPT_SGLANG_ENV)
    import torch
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator as K
    from sglang.srt.mem_cache.kv.state_storage import StateSlotsExhausted
    from sglang.srt.runtime_context import get_context

    with open(path) as f:
        sq = json.load(f)
    calls, sizing, budgets = [], [], []
    with get_context().override_server_args(model_path="dummy", page_size=sq.get("page_size", 1),
                                            **sq.get("server_args", {})), ssm_dtype_env(sq.get("server_args", {})):
        cfg = build_config(sq)
        cache_params = cfg.mamba2_cache_params
        full_ids = list(cfg.full_attention_layer_ids)
        facts = {"mamba_cache_per_req": int(cache_params.mamba_cache_per_req), "layers": list(cache_params.layers),
                 "full_attention_layer_ids": full_ids, "conv": [list(c) for c in cache_params.shape.conv],
                 "temporal": list(cache_params.shape.temporal), "conv_dtype": str(cache_params.dtype.conv),
                 "temporal_dtype": str(cache_params.dtype.temporal),
                 "num_k_heads_per_tp": cache_params.shape.num_k_heads_per_tp,
                 "conv_kernel": cache_params.shape.conv_kernel,
                 "replayssm_ring_bytes_per_req_16": int(cache_params.replayssm_ring_bytes_per_req(16))}
        bound, reqs, built = {}, {}, []
        mod = importlib.import_module("sglang.srt.mem_cache.memory_pool")
        for p in sq.get("pools", []):
            kwargs = {k: decode(v, bound, reqs) for k, v in p["args"].items()}
            cls = getattr(mod, p["class"])
            if p["class"] == "MambaPool":
                kwargs.update(cache_params=cache_params, mamba_layer_ids=list(cache_params.layers))
            elif p["class"] == "HybridReqToTokenPool":
                kwargs.update(cache_params=cache_params, mamba_layer_ids=list(cache_params.layers),
                              enable_memory_saver=False)
            elif p["class"] == "HybridLinearKVPool":
                kwargs.update(full_attention_layer_ids=full_ids)
            bound[p["as"]] = cls(device="cpu", **kwargs)
            built.append(p)
        for c in sq.get("calls", []):
            if c.get("op") == "req":
                reqs[c["as"]] = Req(c["as"], c.get("set", {}))
                continue
            if c.get("op") == "set":
                reqs[c["req"]].update(c["set"])
                continue
            target = bound[c["on"]]
            args = {k: decode(v, bound, reqs) for k, v in c.get("args", {}).items()}
            touched = [n for n in sorted(reqs) if any(isinstance(a, Req) and a is reqs[n]
                       or isinstance(a, list) and reqs[n] in a for a in args.values())]
            try:
                out = lifecycle_call(target, c["m"], args)
                rec = {"on": c["on"], "m": c["m"], "ret": enc(out)}
            except (AssertionError, StateSlotsExhausted) as e:
                rec = {"on": c["on"], "m": c["m"], "raises": type(e).__name__, "message": str(e).split(".")[0]}
                out = None
            if "as" in c:
                bound[c["as"]] = out
            rec["args"] = c.get("args", {})
            rec["reqs"] = {n: kv_state(reqs[n]) for n in touched}
            hybrid = [b for b in built if b["class"] == "HybridReqToTokenPool"]
            if hybrid:
                rec["state"] = pool_state(bound[hybrid[0]["as"]])
            calls.append(rec)
        for p in built:
            pool = bound[p["as"]]
            if p["class"] == "MambaPool":
                sizing.append({"pool": p["as"], "class": "MambaPool", "bytes": physical_bytes(pool)})
            elif p["class"] == "HybridReqToTokenPool":
                sizing.append({"pool": p["as"], "class": "HybridReqToTokenPool",
                               "bytes": physical_bytes(pool.mamba_pool), "mamba_slots": int(pool.mamba_pool.size),
                               "tracking_slots": int(pool.mamba_ping_pong_track_buffer_size),
                               "initial_tracking_slots": int(pool.mamba_initial_tracking_slots)})
            elif p["class"] == "HybridLinearKVPool":
                sizing.append({"pool": p["as"], "class": "HybridLinearKVPool",
                               "kv_size_bytes": [int(x) for x in pool.get_kv_size_bytes()]})
        for b in sq.get("budgets", []):
            budgets.append(mamba_budget(b, cfg, K, sq.get("server_args", {})))
    return {"provenance": traces.provenance(TIER, sglang_root, path, sq, INPUTS), "kind": "direct",
            "facts": facts, "calls": calls, "sizing": sizing, "budgets": budgets}


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
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(trace, f, indent=1, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
