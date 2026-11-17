# Original to blis-sglang.
"""Traces at the HiCache framework's interfaces (design FID-5, §7.6): recorded and direct.

recorded  A scheduler scenario with a "hicache" block (oracle/scenarios/hicache/recorded/*.json)
          runs exactly as oracle/run_scheduler.py runs it, through SGLang's own Scheduler, with
          HiCache on, while oracle/hicache_interfaces.py records every call at the HiCache
          interfaces and the cache framework's. The controller's threads and device events run by
          the step-driven runner (oracle/step_runner.py), whose boundary k is the first receive of
          loop iteration k.

direct    A sequence (oracle/scenarios/hicache/direct/*.json) is a list of calls this driver makes
          itself, at the radix cache, at its controller, or at a step boundary, in orderings the
          scheduler does not make: a write the host pool cannot hold, two writes merged into one
          transfer, a load the device cannot hold, a load-back whose ack is polled before its
          event completes, more loads than the layer counter has events, a backup and a prefetch
          across several boundaries, a prefetch terminated in flight. The cache, its tree core,
          the pools, the host pool, the controller and the storage backend are SGLang's own
          classes, built by their own constructors: the pools and the cache as
          oracle/run_cache_iface.py builds them, then UnifiedRadixCache.init_hicache
          (mem_cache/unified_radix_cache.py:455@297e54fd) with the sequence's HiCache settings,
          as Scheduler.__init__ calls it (mem_cache/registry.py:192-196).

core/hicache's trace test reads both kinds; core/hicache replays them against the port. Each trace
lists, in boundary_at, how many calls were recorded before each step boundary, so a replay applies
boundary k at the same place; a direct trace also records, in states, the controller's queues,
acks, layer counter and host free list after each entry of the sequence.

A sequence:
    {"description": "...", "model": "...", "seed": 0,
     "cache": {...as oracle/run_cache_iface.py...},
     "hicache": {"write_policy": "write_through"|"write_back"|"write_through_selective",
                 "io_backend": "direct", "mem_layout": "layer_first", "ratio": r,
                 "storage_backend": null|"file", "prefetch_policy": "best_effort"|"wait_complete"|"timeout",
                 "prefetch_threshold": n},
     "calls": [{"on": "cache"|"controller"|"allocator", "m": method, ...arguments, "as": name}
               | {"boundary": true}]}
Arguments as oracle/run_cache_iface.py's, and "values" (names whose values are concatenated), "host" (a name bound to host indices, or
[name, start, end]), "device" (likewise for device indices), "rid", "node_id", "priority",
"flush", "op" (a name bound to a prefetch operation), "write_back".

Usage:
    python -I oracle/run_hicache.py recorded <sglang checkout> <scenario.json> <out.json>
    python -I oracle/run_hicache.py direct <sglang checkout> <sequence.json> <out.json>
Environment as for oracle/run_cache_iface.py. BLIS_SGLANG_ALLOW_GOLDEN_UPDATE=1 lets it write a golden.
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
INPUTS = ("oracle/run_hicache.py", "oracle/hicache_interfaces.py", "oracle/step_runner.py",
          "oracle/cache_interfaces.py", "oracle/run_cache_iface.py", "oracle/run_scheduler.py", "oracle/harness.py",
          "oracle/stubs.py", "oracle/traces.py", "oracle/mocks.json", "oracle/harness.json", "oracle/requirements.lock")
HICACHE_INTERFACES = ("controller", "layer_counter", "l2_transfer", "host_pool", "storage")


def copy_no_kv(src_layers, dst_layers, src_indices, dst_indices, page_size) -> None:
    """sgl_kernel.kvcacheio.transfer_kv_direct on a device with no kernels: it copies nothing,
    since the oracle runs no forward pass and compares no KV value."""
    return None


def install_host_register_noop() -> None:
    """The two GPU steps of an L2 transfer (gpu, oracle/mocks.json). HiCache page-locks its host
    pool for DMA with cudaHostRegister (mem_cache/pool_host/common.py:147@297e54fd), and copies
    KV between host and device with sgl_kernel's transfer_kv_direct, which pool_host/mha.py
    imports only on CUDA and ROCm (:60-88). The oracle has neither; page-locking and the bytes a
    copy moves change no decision."""
    import sglang.srt.mem_cache.pool_host.common as host_common
    import sglang.srt.mem_cache.pool_host.mha as host_mha

    host_common._cuda_host_register = lambda buffer, granularity=None: None
    host_mha.transfer_kv_direct = copy_no_kv


def hicache_server_args(page: int, h: dict, storage_dir: str):
    """The ServerArgs a sequence's HiCache block selects, by SGLang's own field names."""
    from sglang.srt.server_args import ServerArgs

    extra = None
    if h.get("storage_backend") == "file":
        os.environ.update({"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": storage_dir})
        if "prefetch_threshold" in h:
            extra = json.dumps({"prefetch_threshold": h["prefetch_threshold"]})
    return ServerArgs(model_path="dummy", page_size=page, enable_hierarchical_cache=True,
                      hicache_ratio=h["ratio"], hicache_io_backend=h["io_backend"],
                      hicache_mem_layout=h["mem_layout"], hicache_write_policy=h["write_policy"],
                      hicache_storage_backend=h.get("storage_backend"),
                      hicache_storage_backend_extra_config=extra,
                      hicache_storage_prefetch_policy=h.get("prefetch_policy", "best_effort"))


def build_hicache(cfg: dict, h: dict, storage_dir: str):
    """SGLang's own pools, allocator, radix cache and HiCache stack for a sequence."""
    import torch
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.server_args import set_global_server_args_for_scheduler

    if (h["io_backend"], h["mem_layout"]) != ("direct", "layer_first"):
        raise SystemExit("the oracle runs HiCache's direct io backend on the layer_first layout only: the "
                         "kernel backend and the other layouts need CUDA kernels it cannot run")
    page = cfg["page_size"]
    args = hicache_server_args(page, h, storage_dir)
    set_global_server_args_for_scheduler(args)
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
                             eviction_policy=cfg.get("eviction_policy", "lru"))
    cache = UnifiedRadixCache(params=params)
    # One rank: the storage config reads the parallel topology, which a scheduler would publish.
    with get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1):
        cache.init_hicache(args, params)
    return cache, alloc, req_to_token


def controller_state(cc, mon) -> dict:
    """What the controller holds between calls: its operation and ack queues, the layer counter's
    indices and events, the host pool's free slots and, with storage, the storage queues. Read
    from SGLang's own objects, after a driver call or a boundary."""
    def acks(q):
        return [{"node_ids": list(a.node_ids), "num_tokens": a.num_tokens, "done": bool(a.finish_event.query())}
                for a in q]

    def ops(q):
        return [{"node_ids": list(o.node_ids), "host_indices": o.host_indices.tolist(),
                 "device_indices": o.device_indices.tolist(), "priority": o.priority} for o in q]

    counter = cc.layer_done_counter
    host = cc.mem_pool_host.anchor_entry.host_pool
    out = {"write_queue": ops(cc.write_queue), "load_queue": ops(cc.load_queue),
           "ack_write_queue": acks(cc.ack_write_queue), "ack_load_queue": acks(cc.ack_load_queue),
           "layer_counter": {"producer_index": counter.producer_index, "consumer_index": counter.consumer_index,
                             "finished": [bool(e.finish_event.query()) for e in counter.events]},
           # The host pool's free list, read without a call the recorder would record.
           "host_free": len(host.free_slots) + host.num_release_slots}
    if cc.enable_storage:
        out["storage"] = {name: getattr(cc, name).qsize() for name in
                          ("prefetch_queue", "backup_queue", "prefetch_buffer", "prefetch_sync_queue",
                           "prefetch_hit_queue", "ack_prefetch_queue", "ack_backup_queue", "host_mem_release_queue")}
        out["storage"]["prefetch_tokens_occupied"] = cc.prefetch_tokens_occupied
    return mon.enc(out)


def drive_sequence(sglang_root: str, path: str) -> dict:
    import atexit

    import run_scheduler

    for name in [n for n in os.environ if n.startswith("SGLANG_")]:
        os.environ.pop(name)
    os.environ.update(run_scheduler.KEPT_SGLANG_ENV)
    from array import array

    import step_runner
    from hicache_interfaces import HiCacheMonitor

    runner = step_runner.install()
    install_host_register_noop()
    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, EvictParams, InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.runtime_context import get_parallel

    with open(path) as f:
        sq = json.load(f)
    mon = HiCacheMonitor()
    bound: dict = {}

    def ref(c, k):
        if k == "value" and "values" in c:
            import torch

            return torch.cat([bound[n] for n in c["values"]])
        v = c[k]
        if isinstance(v, list):
            return bound[v[0]][v[1]:v[2]]
        return bound[v]

    def node(c):
        r = bound[c["node"]]
        return getattr(r, "last_device_node", r)

    def key(c):
        return RadixKey(array("q", c["tokens"]), c.get("extra_key"), cache_salt=c.get("cache_salt"))

    with tempfile.TemporaryDirectory() as storage_dir, \
            get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1), mon:
        cache, alloc, _ = build_hicache(sq["cache"], sq["hicache"], storage_dir)
        atexit.unregister(cache.shutdown)
        cc = cache.cache_controller
        states = []

        def snapshot(at):
            states.append({"after": len(mon.calls) - 1, "at": at, "boundaries": runner.boundaries,
                           "controller": controller_state(cc, mon)})

        snapshot("built")
        call = make_call(path, cache, cc, alloc, bound, ref, node, key)
        boundary_at = []
        for i, c in enumerate(sq["calls"]):
            if c.get("boundary"):
                boundary_at.append(len(mon.calls))
                runner.boundary()
                snapshot(i)
                continue
            try:
                out = call(c)
            except Exception as e:  # a call the sequence expects SGLang to refuse
                if type(e).__name__ != c.get("raises"):
                    raise
                out = None
            else:
                if "raises" in c:
                    raise SystemExit(f"{path}: call {i} ({c['on']}.{c['m']}) did not raise {c['raises']}")
            if "as" in c:
                bound[c["as"]] = out
            snapshot(i)
    return {"provenance": traces.provenance(TIER, sglang_root, path, sq, INPUTS), "kind": "direct",
            "boundaries": runner.boundaries, "boundary_at": boundary_at, "bodies_run": [list(r) for r in runner.ran],
            "states": states, "calls": mon.calls}


def make_call(path, cache, cc, alloc, bound, ref, node, key):
    """The driver's dispatch: one sequence entry, made on SGLang's own objects."""
    import torch  # noqa: F401
    from array import array

    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, EvictParams, InsertParams, MatchPrefixParams

    def call(c):
        on, m = c["on"], c["m"]
        if on == "cache":
            if m == "match_prefix":
                out = cache.match_prefix(MatchPrefixParams(key=key(c)))
            elif m == "insert":
                out = cache.insert(InsertParams(key=key(c), value=ref(c, "value"),
                                                prev_prefix_len=c.get("prev_prefix_len", 0)))
            elif m == "evict":
                out = cache.evict(EvictParams(num_tokens=c["num_tokens"]))
            elif m == "inc_lock_ref":
                out = cache.inc_lock_ref(node(c))
            elif m == "dec_lock_ref":
                out = cache.dec_lock_ref(node(c), bound[c["lock"]].to_dec_params())
            elif m == "load_back":
                # A load-back starts at the match's best node, which may be on host only.
                r = bound[c["node"]]
                out = cache.load_back(getattr(r, "best_match_node", r), c.get("mem_quota"))
            elif m == "evict_host":
                out = cache.evict_host(c["num_tokens"])
            elif m == "write_backup_storage":
                out = cache.write_backup_storage(node(c))
            elif m == "prefetch_from_storage":
                # From the match's last host node, the tokens past it, as the scheduler's
                # admission calls it (managers/scheduler.py).
                r = bound[c["node"]]
                out = cache.prefetch_from_storage(CacheRequestHandle(c["rid"], 0), r.last_host_node, array("q", c["tokens"]),
                                              cache.get_last_hash_value(r.last_host_node))
            elif m == "check_prefetch_progress":
                out = cache.check_prefetch_progress(CacheRequestHandle(c["rid"], 0))
            elif m == "writing_check":
                out = cache.writing_check(write_back=c.get("write_back", False))
            elif m in ("loading_check", "check_hicache_events", "ready_to_load_host_cache",
                       "flush_pending_backups", "evictable_size", "protected_size"):
                out = getattr(cache, m)()
            else:
                raise SystemExit(f"{path}: the driver has no call cache.{m}")
        elif on == "controller":
            if m == "write":
                out = cc.write(ref(c, "device"), priority=c.get("priority"), node_id=c.get("node_id", -1),
                               flush=c.get("flush", True))
            elif m == "load":
                out = cc.load(ref(c, "host"), priority=c.get("priority"), node_id=c.get("node_id", -1))
            elif m == "evict_host":
                out = cc.evict_host(ref(c, "host"))
            elif m == "write_storage":
                out = cc.write_storage(ref(c, "host"), c["token_ids"], hash_value=c.get("hash_value"))
            elif m == "terminate_prefetch":
                out = cc.terminate_prefetch(bound[c["op"]])
            elif m in ("start_writing", "start_loading", "prefetch_rate_limited", "has_inflight_device_transfers"):
                out = getattr(cc, m)()
            else:
                raise SystemExit(f"{path}: the driver has no call controller.{m}")
        elif on == "allocator":
            if m == "alloc":
                out = alloc.alloc(c["n"])
            elif m == "free":
                out = alloc.free(ref(c, "value"))
            else:
                raise SystemExit(f"{path}: the driver has no call allocator.{m}")
        else:
            raise SystemExit(f"{path}: no target {on!r}")
        return out
    return call


def recorded_child(scenario: dict, sglang_root: str, out_path: str) -> None:
    import run_scheduler
    import step_runner
    from hicache_interfaces import HiCacheMonitor

    runner = step_runner.RUNNER
    monitors = []
    boundary_at = []

    def on_step():
        # The calls recorded before boundary k, so a replay applies it at the same place.
        boundary_at.append(len(monitors[0].calls))
        runner.boundary()

    def setup():
        # After the harness's replacements, before Scheduler.__init__ builds the controller.
        step_runner.install()
        install_host_register_noop()
        mon = HiCacheMonitor()
        mon.__enter__()
        monitors.append(mon)

    with tempfile.TemporaryDirectory() as tmp:
        result_path = os.path.join(tmp, "result.json")
        try:
            run_scheduler.child(scenario, sglang_root, result_path, on_step=on_step, setup=setup)
        finally:
            for m in monitors:
                m.__exit__(None, None, None)
        mon = monitors[0]
        with open(result_path) as f:
            result = json.load(f)
    with open(out_path, "w") as f:
        json.dump({"result": result, "calls": mon.calls, "boundaries": runner.boundaries, "boundary_at": boundary_at,
                   "bodies_run": [list(r) for r in runner.ran]}, f)


def record_scenario(sglang_root: str, path: str) -> dict:
    import run_scheduler

    with open(path) as f:
        scenario = json.load(f)
    if "hicache" not in scenario:
        raise SystemExit(f"{path}: a HiCache scenario names its HiCache settings in a \"hicache\" block")
    if scenario["hicache"].get("storage_backend") not in (None, "file"):
        raise SystemExit(f"{path}: the oracle runs the file storage backend only")
    if scenario["hicache"].get("storage_backend"):
        # The scheduler process keeps no SGLANG_* variable, so the file backend's directory is
        # its extra configuration's: one fixed per scenario, emptied first, so the trace names
        # the same directory every run and no run reads pages another stored.
        import shutil

        store = os.path.join(tempfile.gettempdir(), "blis-sglang-hicache", os.path.basename(path).removesuffix(".json"))
        shutil.rmtree(store, ignore_errors=True)
        scenario = {**scenario, "hicache": {**scenario["hicache"], "file_storage_path": store}}
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
    return {"provenance": traces.provenance(TIER, sglang_root, path, scenario, INPUTS), "kind": "recorded",
            "classes": got["result"]["classes"], "boundaries": got["boundaries"], "boundary_at": got["boundary_at"],
            "bodies_run": got["bodies_run"],
            "calls": got["calls"]}


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
    if not any(c["i"] == "controller" for c in trace["calls"]):
        print(f"{path}: no call at the HiCache controller was recorded", file=sys.stderr)
        return 1
    traces.write(out, trace)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
