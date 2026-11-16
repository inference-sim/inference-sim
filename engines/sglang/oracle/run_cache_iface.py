# Original to blis-sglang.
"""Traces at the cache framework's interfaces (design FID-5, §7.6): recorded and direct.

recorded  A tier-2 scenario (oracle/scenarios/*.json) runs exactly as oracle/run_scheduler.py runs
          it, through SGLang's own Scheduler, while oracle/cache_interfaces.py records every call
          made at the six cache interfaces. Nothing is replaced beyond what that runner replaces
          (oracle/mocks.json); the recording only observes. The run must be a trace run_scheduler.py
          would accept.

direct    A sequence (oracle/scenarios-cache-iface/*.json) is a list of calls this driver makes
          itself, as a scheduler would but in orderings the scheduler does not make: an insert
          over a cached prefix with new slots, an eviction larger than what is evictable, a lock
          held across an eviction, a release of a lock taken on a node a split has since cut, a
          free of slots the tree still holds after it has evicted them. The cache, its tree core
          and FullComponent, the allocator, the request pool and the KV pool are SGLang's own
          classes, built by their own constructors as Scheduler.__init__ builds them for the
          sequence's `cache` block. The same recorder records every call at the interfaces.

core/cache/iface_trace_test.go reads both kinds; core/cache replays them against the port.

A sequence:
    {"description": "...", "model": "...", "seed": 0,
     "cache": {"page_size": n, "kv_size": n, "max_num_reqs": n, "max_context_len": n,
               "layer_num": n, "head_num": n, "head_dim": n, "kv_cache_events": bool},
     "calls": [{"on": "cache"|"allocator"|"req_to_token", "m": method, ...arguments, "as": name}]}
A call's arguments: "tokens" (a key, namespaced by "extra_key" and "cache_salt" when given), "value" (a name bound by "as", or [name, start, end] for a
slice of it), "node" (a name bound to a match or insert result, whose last device node it is),
"lock" (a name bound by inc_lock_ref), "n", "num_tokens", "prev_prefix_len".

Usage:
    python -I oracle/run_cache_iface.py recorded <sglang checkout> <scenario.json> <out.json>
    python -I oracle/run_cache_iface.py direct <sglang checkout> <sequence.json> <out.json>
Environment as for oracle/run_scheduler.py (ORACLE_LOCKFILE, ORACLE_MODEL_REVISION; ORACLE_MODEL
for recorded). BLIS_SGLANG_ALLOW_GOLDEN_UPDATE=1 lets it write a golden.
"""
import json
import multiprocessing
import os
import signal
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import traces  # noqa: E402

TIER = 2
INPUTS = ("oracle/run_cache_iface.py", "oracle/cache_interfaces.py", "oracle/run_scheduler.py", "oracle/harness.py",
          "oracle/stubs.py", "oracle/traces.py", "oracle/mocks.json", "oracle/harness.json", "oracle/requirements.lock")


def recorded_child(scenario: dict, sglang_root: str, out_path: str) -> None:
    import run_scheduler
    from cache_interfaces import Monitor

    mon = Monitor()
    with tempfile.TemporaryDirectory() as tmp:
        result_path = os.path.join(tmp, "result.json")
        with mon:
            run_scheduler.child(scenario, sglang_root, result_path)
        with open(result_path) as f:
            result = json.load(f)
    with open(out_path, "w") as f:
        json.dump({"result": result, "calls": mon.calls}, f)


def record_scenario(sglang_root: str, path: str) -> dict:
    import run_scheduler

    with open(path) as f:
        scenario = json.load(f)
    quit_seen = []
    signal.signal(signal.SIGQUIT, lambda *a: quit_seen.append(True))
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.json")
        proc = multiprocessing.get_context("fork").Process(target=recorded_child, args=(scenario, sglang_root, out))
        proc.start()
        proc.join(run_scheduler.JOIN_TIMEOUT_S)
        if proc.is_alive():
            proc.terminate()
            proc.join(10)
            raise SystemExit(f"{path}: the scheduler process did not finish in {run_scheduler.JOIN_TIMEOUT_S} s")
        if quit_seen or proc.exitcode != 0 or not os.path.exists(out):
            raise SystemExit(f"{path}: the scheduler process failed (exit {proc.exitcode}, SIGQUIT {bool(quit_seen)})")
        with open(out) as f:
            got = json.load(f)
    problems = run_scheduler.trace_problems(got["result"], scenario)
    if problems:
        raise SystemExit(f"{path}: not a run run_scheduler.py would record: " + "; ".join(problems))
    return {"provenance": traces.provenance(TIER, sglang_root, path, scenario, INPUTS),
            "kind": "recorded", "classes": got["result"]["classes"], "calls": got["calls"]}


def build_cache(cfg: dict):
    """SGLang's own pools, allocator and radix cache for a sequence's cache block, built by their
    constructors with the arguments Scheduler.__init__ passes for a full-attention MHA model
    (python/sglang/srt/mem_cache/kv_cache_builder.py, registry.py)."""
    import torch
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    page = cfg["page_size"]
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=page))
    req_to_token = ReqToTokenPool(size=cfg["max_num_reqs"], max_context_len=cfg["max_context_len"], device="cpu",
                                  enable_memory_saver=False)
    kv = MHATokenToKVPool(size=cfg["kv_size"], page_size=page, dtype=torch.bfloat16, head_num=cfg["head_num"],
                          head_dim=cfg["head_dim"], layer_num=cfg["layer_num"], device="cpu", enable_memory_saver=False)
    if page == 1:
        alloc = TokenToKVPoolAllocator(size=cfg["kv_size"], dtype=torch.bfloat16, device="cpu", kvcache=kv, need_sort=False)
    else:
        alloc = PagedTokenToKVPoolAllocator(size=cfg["kv_size"], page_size=page, dtype=torch.bfloat16, device="cpu",
                                            kvcache=kv, need_sort=False)
    params = CacheInitParams(disable=False, req_to_token_pool=req_to_token, token_to_kv_pool_allocator=alloc,
                             page_size=page, tree_components=(ComponentType.FULL,),
                             enable_kv_cache_events=bool(cfg.get("kv_cache_events", False)),
                             dynamo_kv_event_format=bool(cfg.get("dynamo_kv_event_format", False)))
    return UnifiedRadixCache(params=params), alloc, req_to_token


def drive_sequence(sglang_root: str, path: str) -> dict:
    import run_scheduler

    for name in [n for n in os.environ if n.startswith("SGLANG_")]:
        os.environ.pop(name)
    os.environ.update(run_scheduler.KEPT_SGLANG_ENV)
    from array import array

    from cache_interfaces import Monitor
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    with open(path) as f:
        sq = json.load(f)
    mon = Monitor()
    bound: dict = {}

    def value(c):
        v = c["value"]
        if isinstance(v, list):
            return bound[v[0]][v[1]:v[2]]
        return bound[v]

    def node(c):
        r = bound[c["node"]]
        return getattr(r, "last_device_node", r)

    def key(c):
        return RadixKey(array("q", c["tokens"]), c.get("extra_key"), cache_salt=c.get("cache_salt"))

    with mon:
        cache, alloc, req_to_token = build_cache(sq["cache"])
        targets = {"cache": cache, "allocator": alloc, "req_to_token": req_to_token}
        for c in sq["calls"]:
            t, m = targets[c["on"]], c["m"]
            if m == "match_prefix":
                out = t.match_prefix(MatchPrefixParams(key=key(c)))
            elif m == "insert":
                out = t.insert(InsertParams(key=key(c), value=value(c), prev_prefix_len=c.get("prev_prefix_len", 0)))
            elif m == "evict":
                out = t.evict(EvictParams(num_tokens=c["num_tokens"]))
            elif m == "inc_lock_ref":
                out = t.inc_lock_ref(node(c))
            elif m == "dec_lock_ref":
                out = t.dec_lock_ref(node(c), bound[c["lock"]].to_dec_params())
            elif m == "alloc":
                out = t.alloc(c["n"])
            elif m in ("free", "free_segment"):
                out = t.free(value(c)) if m == "free" else t.free_segment(value(c), start_pos=0)
            elif m in ("take_events", "reset", "evictable_size", "protected_size", "available_size", "total_size",
                       "full_evictable_size", "full_protected_size"):
                out = getattr(t, m)()
            else:
                raise SystemExit(f"{path}: the driver has no call {c['on']}.{m}")
            if "as" in c:
                bound[c["as"]] = out
    return {"provenance": traces.provenance(TIER, sglang_root, path, sq, INPUTS), "kind": "direct", "calls": mon.calls}


def main(argv) -> int:
    if len(argv) != 4 or argv[0] not in ("recorded", "direct"):
        print(__doc__.split("Usage:", 1)[1].split("Environment", 1)[0].strip(), file=sys.stderr)
        return 2
    kind, root, path, out = argv
    if traces.refuses_write(out):
        print(traces.REFUSAL, file=sys.stderr)
        return 2
    sys.path.insert(0, os.path.join(os.path.abspath(root), "python"))
    trace = record_scenario(root, path) if kind == "recorded" else drive_sequence(root, path)
    if not any(c["i"] == "prefix_cache" for c in trace["calls"]):
        print(f"{path}: no call at the prefix cache was recorded", file=sys.stderr)
        return 1
    traces.write(out, trace)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
