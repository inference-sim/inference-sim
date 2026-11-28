# Original to blis-sglang.
"""Direct traces at SGLang's unified radix cache with the mamba state component (design FID-5,
§7.6; inference-sim/inference-sim#2366, #2442): the tree side of the hybrid cache.

A sequence (oracle/scenarios-mamba-tree/*.json) builds SGLang's own HybridReqToTokenPool, a token
pool and allocator, and a UnifiedRadixCache with the slots FULL and MAMBA, by their own
constructors on CPU (the allocator's state storage is the request pool's, as the KV cache
configurator binds it), and makes calls at the cache as a scheduler would and in orderings it does
not make. The same recorder as oracle/run_cache_iface.py records every call at the cache
framework's interfaces, now with MambaComponent among the components. Nothing is replaced.

A sequence is that of oracle/run_mamba_component.py ("model", "seed", "page_size", "server_args",
"config") with a "cache" block
    {"kv_size": n, "mamba_size": n, "max_num_reqs": n, "max_context_len": n,
     "extra_buffer": bool, "lazy": bool, "overlap": bool, "checkpoint_interval": n}
and calls
    {"op": "req", "as": r, "set": {...}} | {"op": "set", "req": r, "set": {...}}
    | {"on": "cache", "m": "match_prefix", "tokens": [...], "cow": bool, "defer_slot": bool, "req": r, "as": name}
    | {"on": "cache", "m": "insert", "tokens": [...], "value": name | [name, a, b], "mamba": name,
       "prev_prefix_len": n, "keep_idx": n, "allow_missing_mamba": bool, "as": name}
    | {"on": "cache", "m": "evict", "num_tokens": n, "mamba_num": n}
    | {"op": "write_row", "req": r, "slots": [...]}   (the KV slots of the request's row)
    | {"op": "lock", "req": r, "node": name}          (the request takes the tree lock and last node)
    | {"on": "cache", "m": "checkpoint", "req": r, "up_to": n}
    | {"on": "cache", "m": "on_release", "req": r, "checkpointed": bool}
    | {"on": "cache", "m": "evict_for_alloc", "full_tokens": n, "state_slots": n}
    | {"on": "cache", "m": "inc_lock_ref", "node": name, "as": name} | {"on": "cache", "m": "dec_lock_ref", "node": name, "lock": name}
    | {"on": "cache", "m": reset | evictable_size | protected_size | full_evictable_size | full_protected_size |
       mamba_evictable_size | mamba_protected_size | total_size}
    | {"on": "allocator", "m": "alloc", "n": n, "as": name}
    | {"on": "mamba", "m": "alloc", "n": n, "as": name}      (the pool's mamba slot allocator)
    | {"on": "pool", "m": "alloc", "reqs": [names]}          (the request table's acquire)
The trace carries "driver": the mamba allocator calls, each with the seq of the last recorded call before it.
A key is "tokens" with an optional "extra_key" and "cache_salt".

Usage: python -I oracle/run_mamba_tree.py direct <sglang checkout> <sequence.json> <out.json>
Environment as for oracle/run_cache_iface.py (ORACLE_LOCKFILE, ORACLE_MODEL_REVISION).
"""
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_mamba_pools as P  # noqa: E402
import traces  # noqa: E402

TIER = 2
INPUTS = ("oracle/run_mamba_tree.py", "oracle/run_mamba_pools.py", "oracle/cache_interfaces.py",
          "oracle/run_scheduler.py", "oracle/traces.py", "oracle/mocks.json", "oracle/harness.json",
          "oracle/requirements.lock")
EXTRA_IMPLEMENTATIONS = ("sglang.srt.mem_cache.unified_cache.components.mamba", "sglang.srt.mem_cache.memory_pool")


class Req(P.Req):
    """run_mamba_pools.Req with what the cache reads of a request beside its KV record: no session,
    no tree lock, no extra key."""

    def __init__(self, rid, fields):
        self.session = None
        self.lock = None
        self.last_node = None
        self.extra_key = None
        self.cache_salt = None
        self.priority = 0
        self.session_id = None
        self.output_ids = []
        self.full_untruncated_fill_ids = []
        self.prefix_len = 0
        self.skip_radix_cache_insert = False
        self.kv_rotation_base = None
        self.finished_reason = None
        self.is_finished = False
        super().__init__(rid, fields)

    def update(self, fields):
        from array import array

        super().update(fields)
        for k in ("origin_input_ids", "output_ids", "full_untruncated_fill_ids"):
            v = getattr(self, k)
            if isinstance(v, list):
                setattr(self, k, array("q", v))

    def finished(self):
        return self.is_finished


def drive(sglang_root: str, path: str) -> dict:
    import run_scheduler

    for name in [n for n in os.environ if n.startswith("SGLANG_")]:
        os.environ.pop(name)
    os.environ.update(run_scheduler.KEPT_SGLANG_ENV)
    from array import array

    import torch
    from cache_interfaces import IMPLEMENTATIONS, Monitor
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
    from sglang.srt.mem_cache.kv.plan import TREE_STATE, TREE_TOKEN, TreeSlot
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.static_kv_storage import acquire_rows
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.runtime_context import get_context

    with open(path) as f:
        sq = json.load(f)
    cb = sq["cache"]
    page = sq.get("page_size", 1)
    bound, reqs = {}, {}
    driver = []   # the driver's own calls at objects no interface records (the mamba slot allocator)

    def key(c):
        return RadixKey(array("q", c["tokens"]), c.get("extra_key"), cache_salt=c.get("cache_salt"))

    def value(v):
        if isinstance(v, list):
            return bound[v[0]][v[1]:v[2]]
        return bound[v]

    mon = Monitor(implementations=tuple(IMPLEMENTATIONS) + EXTRA_IMPLEMENTATIONS)
    with get_context().override_server_args(model_path="dummy", page_size=page, **sq.get("server_args", {})), \
            P.ssm_dtype_env(sq.get("server_args", {})), mon:
        cfg = P.build_config(sq)
        cp = cfg.mamba2_cache_params
        mod = importlib.import_module("sglang.srt.mem_cache.memory_pool")
        pool = mod.HybridReqToTokenPool(
            device="cpu", size=cb["max_num_reqs"], mamba_size=cb["mamba_size"],
            mamba_spec_state_size=cb["max_num_reqs"], max_context_len=cb["max_context_len"],
            enable_mamba_extra_buffer=cb["extra_buffer"], enable_mamba_extra_buffer_lazy=cb.get("lazy", False),
            enable_overlap_schedule=cb.get("overlap", True), cache_params=cp, mamba_layer_ids=list(cp.layers),
            enable_memory_saver=False)
        kv = mod.MHATokenToKVPool(size=cb["kv_size"], page_size=page, dtype=torch.bfloat16, head_num=1, head_dim=2,
                                  layer_num=1, device="cpu", enable_memory_saver=False)
        if page == 1:
            alloc = TokenToKVPoolAllocator(size=cb["kv_size"], dtype=torch.bfloat16, device="cpu", kvcache=kv,
                                           need_sort=False)
        else:
            from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

            alloc = PagedTokenToKVPoolAllocator(size=cb["kv_size"], page_size=page, dtype=torch.bfloat16,
                                                device="cpu", kvcache=kv, need_sort=False)
        alloc.state = pool.state
        slots = (TreeSlot(component="FULL", family=TREE_TOKEN),
                 TreeSlot(component="MAMBA", family=TREE_STATE, checkpoint_interval=cb.get("checkpoint_interval", 64)))
        params = CacheInitParams(disable=False, req_to_token_pool=pool, token_to_kv_pool_allocator=alloc,
                                 page_size=page, tree_slots=slots, enable_mamba_extra_buffer=cb["extra_buffer"],
                                 enable_mamba_extra_buffer_lazy=cb.get("lazy", False),
                                 eviction_policy=cb.get("eviction_policy", "lru"), tree_core_backend="python")
        cache = UnifiedRadixCache(params=params)
        targets = {"cache": cache, "allocator": alloc, "mamba": pool.mamba_allocator, "pool": pool}
        for c in sq["calls"]:
            op = c.get("op")
            if op == "req":
                reqs[c["as"]] = Req(c["as"], c.get("set", {}))
                continue
            if op == "set":
                reqs[c["req"]].update(c["set"])
                continue
            if op == "write_row":
                q = reqs[c["req"]]
                row = q.kv.req_pool_idx
                pool.write((row, slice(0, len(c["slots"]))), torch.tensor(c["slots"], dtype=torch.int64))
                continue
            if op == "lock":
                q = reqs[c["req"]]
                n = bound[c["node"]]
                last = getattr(n, "last_device_node", n)
                q.update({"last_node": last, "lock": cache.lock(last)})
                continue
            t, m = targets[c["on"]], c["m"]
            if c["on"] == "pool" and m == "alloc":
                out = acquire_rows(pool, pool.state, [reqs[n] for n in c["reqs"]])
            elif m == "match_prefix":
                out = t.match_prefix(MatchPrefixParams(key=key(c), cow_mamba=c.get("cow", False),
                                                       defer_cow_slot=c.get("defer_slot", False),
                                                       req=reqs[c["req"]] if "req" in c else None))
            elif m == "insert":
                out = t.insert(InsertParams(
                    key=key(c), value=value(c["value"]), prev_prefix_len=c.get("prev_prefix_len", 0),
                    mamba_value=bound[c["mamba"]] if c.get("mamba") else None, mamba_keep_idx=c.get("keep_idx"),
                    allow_missing_mamba=c.get("allow_missing_mamba", False), priority=c.get("priority", 0)))
            elif m == "checkpoint":
                out = t.checkpoint(reqs[c["req"]], up_to=c["up_to"])
            elif m == "on_release":
                out = t.on_release(reqs[c["req"]], checkpointed=c["checkpointed"])
            elif m == "evict":
                out = t.evict(EvictParams(num_tokens=c.get("num_tokens", 0), mamba_num=c.get("mamba_num", 0)))
            elif m == "evict_for_alloc":
                out = t.evict_for_alloc(KVDemand(full_tokens=c.get("full_tokens", 0), state_slots=c.get("state_slots", 0)))
            elif m == "inc_lock_ref":
                n = bound[c["node"]]
                out = t.inc_lock_ref(getattr(n, "last_device_node", n))
            elif m == "dec_lock_ref":
                n = bound[c["node"]]
                out = t.dec_lock_ref(getattr(n, "last_device_node", n), bound[c["lock"]].to_dec_params())
            elif m == "alloc":
                out = t.alloc(c["n"])
            elif m in ("reset", "evictable_size", "protected_size", "full_evictable_size", "full_protected_size",
                       "mamba_evictable_size", "mamba_protected_size", "total_size", "take_events"):
                out = getattr(t, m)()
            else:
                raise SystemExit(f"{path}: the driver has no call {c['on']}.{m}")
            if c["on"] == "mamba":
                driver.append({"after": mon.calls[-1]["seq"] if mon.calls else -1, "m": m, "n": c.get("n"),
                               "ret": None if out is None else out.tolist()})
            if "as" in c:
                bound[c["as"]] = out
    return {"provenance": traces.provenance(TIER, sglang_root, path, sq, INPUTS), "kind": "direct", "calls": mon.calls,
            "driver": driver}


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
    if not any(c["i"] == "prefix_cache" for c in trace["calls"]):
        print(f"{path}: no call at the prefix cache was recorded", file=sys.stderr)
        return 1
    traces.write(out, trace)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
