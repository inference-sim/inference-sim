# Original to blis-sglang.
"""Direct traces at SGLang's MambaComponent request-side lifecycle (design FID-5, §7.6;
inference-sim/blis-sglang#94, #128 piece A1).

A sequence (oracle/scenarios-mamba-component/*.json) builds SGLang's own HybridReqToTokenPool as
oracle/run_mamba_pools.py does, and a MambaComponent without its tree: the component object is
made without UnifiedRadixCache.__init__ and its `cache` is a stand-in that holds what the
lifecycle methods read of the cache, namely req_to_token_pool, enable_mamba_extra_buffer, and
evict_for_alloc / evict / dec_lock_ref. The stand-in's evictions free mamba slots the sequence
names in `tree_mamba_slots` (slots the tree is taken to hold), oldest first, one per `mamba_num`,
and record the call; the methods under test (_alloc_mamba_slot, _free_mamba_value,
_select_finished_checkpoint, prepare_for_caching_req, cleanup_after_caching_req) are SGLang's own,
unmodified, on the real pools.

A sequence is that of oracle/run_mamba_pools.py with
    "component": {"extra_buffer": bool, "tree_mamba_slots": n}   # slots taken from the mamba allocator at start
and calls
    {"op": "req", "as": r, "set": {...}} | {"op": "set", "req": r, "set": {...}}
    | {"op": "insert_params", "as": p, "set": {...}}  (an InsertParams)  | {"op": "insert_result", "as": q, "set": {...}}
    | {"on": "h", "m": method, "args": {...}, "as": name}                 (a pool call)
    | {"on": "component", "m": method, "args": {...}, "as": name}
An argument {"insert_params": name} / {"insert_result": name} is such an object.

Usage: python -I oracle/run_mamba_component.py direct <sglang checkout> <sequence.json> <out.json>
Environment as for oracle/run_cache_iface.py (ORACLE_LOCKFILE, ORACLE_MODEL_REVISION).
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_mamba_pools as P  # noqa: E402
import traces  # noqa: E402

TIER = 2
INPUTS = ("oracle/run_mamba_component.py", "oracle/run_mamba_pools.py", "oracle/traces.py", "oracle/mocks.json",
          "oracle/harness.json", "oracle/requirements.lock")


class Req(P.Req):
    """run_mamba_pools.Req with the one attribute the component reads of a request's tree lock: none
    (a request here holds no tree lock; the extra_buffer_lazy branch that drops one is the cache's)."""

    lock = None


def decode(v, bound, reqs, objs):
    if isinstance(v, dict) and ("insert_params" in v or "insert_result" in v):
        return objs[v.get("insert_params") or v.get("insert_result")]
    return P.decode(v, bound, reqs)


def obj_state(o):
    return {k: P.enc(getattr(o, k)) for k in ("mamba_value", "mamba_keep_idx", "mamba_exist")
            if hasattr(o, k)}


def drive(sglang_root: str, path: str) -> dict:
    import importlib

    import run_scheduler

    for name in [n for n in os.environ if n.startswith("SGLANG_")]:
        os.environ.pop(name)
    os.environ.update(run_scheduler.KEPT_SGLANG_ENV)
    import torch
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, InsertResult
    from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
    from sglang.srt.runtime_context import get_context

    with open(path) as f:
        sq = json.load(f)
    calls, evictions = [], []
    with get_context().override_server_args(model_path="dummy", page_size=sq.get("page_size", 1),
                                            **sq.get("server_args", {})), P.ssm_dtype_env(sq.get("server_args", {})):
        cfg = P.build_config(sq)
        cache_params = cfg.mamba2_cache_params
        mod = importlib.import_module("sglang.srt.mem_cache.memory_pool")
        bound, reqs, objs = {}, {}, {}
        for p in sq["pools"]:
            kwargs = {k: P.decode(v, bound, reqs) for k, v in p["args"].items()}
            kwargs.update(cache_params=cache_params, mamba_layer_ids=list(cache_params.layers), enable_memory_saver=False)
            bound[p["as"]] = getattr(mod, p["class"])(device="cpu", **kwargs)
        pool = bound[sq["pools"][0]["as"]]
        spec = sq["component"]
        held = []
        if spec.get("tree_mamba_slots"):
            held = [int(s) for s in pool.mamba_allocator.alloc(spec["tree_mamba_slots"]).tolist()]

        class Cache:
            """What the lifecycle methods read of the unified radix cache."""

            req_to_token_pool = pool
            enable_mamba_extra_buffer = spec["extra_buffer"]
            tree_components = ()

            def evict_for_alloc(self, params):
                evictions.append({"m": "evict_for_alloc", "mamba_num": params.mamba_num, "num_tokens": params.num_tokens})
                for _ in range(params.mamba_num):
                    if held:
                        pool.mamba_allocator.free(torch.tensor([held.pop(0)], dtype=torch.int64))

            def evict(self, params):
                evictions.append({"m": "evict", "mamba_num": params.mamba_num, "num_tokens": params.num_tokens})
                self.evict_for_alloc(params)

            def dec_lock_ref(self, node, receipt):
                evictions.append({"m": "dec_lock_ref"})

        class Component(MambaComponent):
            """MambaComponent without UnifiedRadixCache.__init__: its methods are SGLang's own, and
            its cache is the stand-in above."""

            def __init__(self, cache):  # noqa: D107 - deliberately not TreeComponent.__init__
                self.cache = cache

        bound["component"] = Component(Cache())
        for c in sq["calls"]:
            op = c.get("op")
            if op == "req":
                reqs[c["as"]] = Req(c["as"], c.get("set", {}))
                continue
            if op == "set":
                reqs[c["req"]].update(c["set"])
                continue
            if op == "insert_params":
                s = {k: (torch.tensor(v, dtype=torch.int64) if isinstance(v, list) else v) for k, v in c.get("set", {}).items()}
                objs[c["as"]] = InsertParams(**s)
                continue
            if op == "insert_result":
                objs[c["as"]] = InsertResult(**c.get("set", {}))
                continue
            target = bound[c["on"]]
            args = {k: decode(v, bound, reqs, objs) for k, v in c.get("args", {}).items()}
            touched = [n for n in sorted(reqs) if any(a is reqs[n] for a in args.values())]
            before = len(evictions)
            try:
                out = getattr(target, c["m"])(**args)
                rec = {"on": c["on"], "m": c["m"], "ret": P.enc(out)}
            except AssertionError as e:
                rec = {"on": c["on"], "m": c["m"], "raises": "AssertionError", "message": str(e).split(".")[0]}
                out = None
            if "as" in c:
                bound[c["as"]] = out
            rec["args"] = c.get("args", {})
            rec["reqs"] = {n: P.kv_state(reqs[n]) for n in touched}
            rec["objs"] = {k: obj_state(o) for k, o in sorted(objs.items())}
            rec["evictions"] = evictions[before:]
            rec["state"] = P.pool_state(pool)
            rec["tree_mamba_slots"] = list(held)
            calls.append(rec)
    return {"provenance": traces.provenance(TIER, sglang_root, path, sq, INPUTS), "kind": "direct", "calls": calls}


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
