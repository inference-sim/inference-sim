# Original to blis-sglang.
"""Direct traces of the radix cache's depth-first weight order (design FID-5, §7.6).

dfs_weight_order (python/sglang/srt/mem_cache/base_prefix_cache.py:347-398@297e54fd) orders a
queue of requests by their match nodes: heavier subtrees first, and children of equal weight in
the order they entered their parent's children dict. A sequence (oracle/scenarios-dfs-order/
*.json) builds a tree through SGLang's own UnifiedRadixCache, as oracle/run_cache_iface.py builds
it, binds match results, and asks for the order of lists of their nodes, so that ties, splits and
deletions decide it. Every call at the cache interfaces is recorded by oracle/cache_interfaces.py.

A sequence: {"description", "model", "seed", "cache": {as run_cache_iface.py's},
 "calls": [{"m": "insert", "tokens": [...]} | {"m": "match_prefix", "tokens": [...], "as": name}
           | {"m": "evict", "num_tokens": n} | {"m": "dfs_weight_order", "nodes": [name, ...]}]}
An insert allocates its own slots first.

Usage: python -I oracle/run_dfs_order.py <sglang checkout> <sequence.json> <out.json>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import traces  # noqa: E402

INPUTS = ("oracle/run_dfs_order.py", "oracle/run_cache_iface.py", "oracle/cache_interfaces.py",
          "oracle/run_scheduler.py", "oracle/harness.py", "oracle/stubs.py", "oracle/traces.py", "oracle/mocks.json",
          "oracle/harness.json", "oracle/requirements.lock")


def drive_order(sglang_root: str, path: str) -> dict:
    import run_cache_iface
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
    bound = {}
    with mon:
        cache, alloc, _ = run_cache_iface.build_cache(sq["cache"])
        for c in sq["calls"]:
            key = RadixKey(array("q", c["tokens"])) if "tokens" in c else None
            if c["m"] == "insert":
                cache.insert(InsertParams(key=key, value=alloc.alloc(len(c["tokens"]))))
            elif c["m"] == "match_prefix":
                bound[c["as"]] = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
            elif c["m"] == "evict":
                cache.evict(EvictParams(num_tokens=c["num_tokens"]))
            elif c["m"] == "dfs_weight_order":
                cache.dfs_weight_order([bound[n] for n in c["nodes"]])
            else:
                raise SystemExit(f"{path}: no call {c['m']}")
    return {"provenance": traces.provenance(2, sglang_root, path, sq, INPUTS), "kind": "direct", "calls": mon.calls}


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__.split("Usage:", 1)[1].strip(), file=sys.stderr)
        return 2
    root, path, out = argv
    if traces.refuses_write(out):
        print(traces.REFUSAL, file=sys.stderr)
        return 2
    sys.path.insert(0, os.path.join(os.path.abspath(root), "python"))
    trace = drive_order(root, path)
    if not any(c["i"] == "prefix_cache" and c["m"] == "dfs_weight_order" for c in trace["calls"]):
        print(f"{path}: no dfs_weight_order call was recorded", file=sys.stderr)
        return 1
    traces.write(out, trace)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
