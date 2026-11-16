# Original to blis-vllm.
"""Direct traces at the KV offload interfaces (design FID-5, §7.6): calls no feature makes.

A sequence (oracle/scenarios-fid5/*.json) is a list of calls this driver makes itself on vLLM's
own classes: on an OffloadingManager that a spec built (CPUOffloadingSpec, or TieringOffloadingSpec
with no secondary tiers, whose manager forwards to its CPU primary tier), or on an LRUCachePolicy
alone. The orderings are ones the connector never makes: a failed store, a touch, a reset in the
middle of a request, a request finished twice or never seen, a store with every chunk pinned, a
store miss reported for a resident key. oracle/kv_interfaces.py records the calls at the interface
and vLLM's answers, in the format the recorded traces use, and core/connector/offloading/
fid5_test.go replays them against the port.

A sequence:
    {"description": "...",
     "spec": {"name": "CPUOffloadingSpec", "cpu_bytes_to_use": n, "worker_kv_bytes_per_block": n,
              "world_size": n, "blocks_per_chunk": n, "extra": {...}},   (or "policy": {"capacity": n})
     "calls": [{"m": "<method>", "ctx": "<request>", "key": "<label>", "keys": [...], ...}, ...]}
A key label "name" or "name/<group>" is the OffloadKey sha256(name) + the group as 4 big-endian
bytes (make_offload_key, vllm/v1/kv_offload/base.py:29-31@5df63a72). Besides the interface methods a
sequence may "pos" (set a key's end position on a request, as the scheduler does), and, for a policy
alone, "ref" (set a chunk's ref_cnt, as the manager does).

Usage:
    python -I oracle/drive_kv_interfaces.py <vllm checkout> <sequence dir> <out dir> [--check <golden dir>]
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "direct-"


def key_of(label: str) -> bytes:
    name, _, group = label.partition("/")
    return hashlib.sha256(name.encode()).digest() + int(group or 0).to_bytes(4, "big")


def offloading_config(spec: dict, name: str):
    from vllm.v1.kv_offload.config import (
        OffloadingCacheConfig,
        OffloadingConfig,
        OffloadingGroupConfig,
        OffloadingModelConfig,
        OffloadingParallelConfig,
    )
    extra = dict(spec.get("extra") or {})
    extra["spec_name"] = spec["name"]
    extra["cpu_bytes_to_use"] = spec["cpu_bytes_to_use"]
    world = spec.get("world_size", 1)
    return OffloadingConfig(
        groups=(OffloadingGroupConfig(tokens_per_block=16, layer_names=("layer.0",), group_id=0),),
        worker_kv_bytes_per_block=spec["worker_kv_bytes_per_block"],
        enable_kv_cache_events=bool(spec.get("enable_kv_cache_events", False)),
        extra_config=extra,
        engine_id="blis-fid5-" + name,
        model=OffloadingModelConfig(name="blis-fid5", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=spec.get("blocks_per_chunk", 1)),
        parallel=OffloadingParallelConfig(rank=0, world_size=world, tp_size=world, pp_size=1, pcp_size=1, dcp_size=1,
                                          data_parallel_index=0, data_parallel_size=1, data_parallel_rank_local=None,
                                          is_parallelism_agnostic=False),
    )


def run_sequence(vllm_root: str, path: str) -> dict:
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    sys.path.insert(0, HERE)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    from kv_interfaces import Monitor  # noqa: E402
    from run_scheduler import provenance  # noqa: E402
    from vllm.v1.kv_offload.base import ReqContext, ScheduleEndContext  # noqa: E402
    from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus  # noqa: E402
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy  # noqa: E402
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory  # noqa: E402

    with open(path) as f:
        sq = json.load(f)
    name = os.path.basename(path)[:-5]
    ctxs: dict = {}
    chunks: dict = {}

    def ctx(c: dict):
        return ctxs.setdefault(c["ctx"], ReqContext(req_id=c["ctx"]))

    def ks(c: dict) -> list:
        return [key_of(k) for k in c["keys"]]

    with Monitor() as mon:
        if "spec" in sq:
            spec = OffloadingSpecFactory.create_spec(offloading_config(sq["spec"], name))
            target = spec.get_manager()
        else:
            target = LRUCachePolicy(cache_capacity=sq["policy"]["capacity"])
        try:
            for c in sq["calls"]:
                m = c["m"]
                if m == "pos":
                    ctx(c).set_offload_key_position(key_of(c["key"]), c["pos"])
                elif m == "ref":
                    chunks[c["key"]].ref_cnt = c["ref"]
                elif "spec" in sq:
                    call_manager(target, m, c, ctx, ks, ScheduleEndContext)
                else:
                    call_policy(target, m, c, ctx, ks, chunks, ChunkStatus)
        finally:
            if "spec" in sq:
                target.shutdown()
    prov = provenance(vllm_root, path)
    prov["kind"] = "direct"
    return {"provenance": prov, **mon.trace}


def call_manager(mgr, m: str, c: dict, ctx, ks, ScheduleEndContext) -> None:
    if m in ("lookup", "get_load_source"):
        getattr(mgr, m)(key_of(c["key"]), ctx(c))
    elif m in ("prepare_load", "touch", "complete_load", "prepare_store"):
        getattr(mgr, m)(ks(c), ctx(c))
    elif m == "complete_store":
        mgr.complete_store(ks(c), ctx(c), success=c.get("success", True))
    elif m in ("on_new_request", "on_request_finished"):
        getattr(mgr, m)(ctx(c))
    elif m == "on_schedule_end":
        mgr.on_schedule_end(ScheduleEndContext(new_req_ids=c.get("new", []), preempted_req_ids=c.get("preempted", [])))
    elif m == "take_events":
        list(mgr.take_events())
    elif m in ("has_pending_work", "reset_cache", "get_stats"):
        getattr(mgr, m)()
    else:
        raise SystemExit("unknown manager call %r" % m)


def call_policy(pol, m: str, c: dict, ctx, ks, chunks: dict, ChunkStatus) -> None:
    if m == "insert":
        ch = ChunkStatus(c["chunk"]["id"])
        ch.ref_cnt = c["chunk"].get("ref", -1)
        chunks[c["key"]] = ch
        pol.insert(key_of(c["key"]), ch)
    elif m in ("get", "remove", "mark_evictable", "mark_non_evictable"):
        getattr(pol, m)(key_of(c["key"]))
    elif m in ("touch", "on_store_miss"):
        getattr(pol, m)(ks(c), ctx(c))
    elif m == "on_request_finished":
        groups = tuple(tuple(key_of(k) for k in g) for g in c["groups"])
        pol.on_request_finished(groups, set(), set(), ctx(c))
    elif m == "evict":
        pol.evict(c["n"], {key_of(k) for k in c.get("protected", [])})
    elif m == "clear":
        pol.clear()
    else:
        raise SystemExit("unknown policy call %r" % m)


def main(argv) -> int:
    check = None
    if "--check" in argv:
        i = argv.index("--check")
        check, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    vllm_root, seq_dir, out_dir = argv
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for f in sorted(x for x in os.listdir(seq_dir) if x.endswith(".json")):
        trace = run_sequence(vllm_root, os.path.join(seq_dir, f))
        name = PREFIX + f[:-5] + ".fid5.json"
        with open(os.path.join(out_dir, name), "w") as fh:
            json.dump(trace, fh, indent=1, sort_keys=True)
        names.append(name)
    if check:
        sys.path.insert(0, HERE)
        import run_kv_interfaces  # noqa: E402
        bad = run_kv_interfaces.compare(out_dir, check, names, PREFIX)
        if bad:
            print("%d direct interface trace(s) differ" % bad)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
