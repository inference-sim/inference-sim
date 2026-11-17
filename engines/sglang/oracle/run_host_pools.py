# Original to blis-sglang.
"""Direct traces of HiCache's host pools' sizing (design FID-5, §7.6): MHA, MLA and DSA.

A sequence (oracle/scenarios-host-pools/*.json) builds SGLang's own device pools with their own
constructors on CPU, then each host pool as HiCache's assembler builds it: the MHA and MLA KV host
pools by _build_mha_mla_host_pool (mem_cache/hybrid_cache/hybrid_pool_assembler.py:1576-1620
@297e54fd), and a DSA pool's indexer host pool by make_dsa_indexer_pool_decl and
DSAIndexerPoolHost (pool_host/dsa.py:113-260), anchored on its KV host pool. The calls at the
host pools are recorded as oracle/hicache_interfaces.py records them (each constructor's answer is
the pool's sizing), and each host pool's size, pages, bytes per token and layers are listed in
"hosts".

The one replacement is run_hicache.py's: cudaHostRegister, which page-locks the host buffer for
DMA, is a no-op (gpu, oracle/mocks.json).

A sequence:
    {"description": "...", "model": "none: ...", "seed": 0,
     "pools": [{"as": name, "class": cls, "args": {...}}],
     "hosts": [{"as": name, "pool": name, "layout": l, "ratio": r, "page_size": p, "raises": exc?}],
     "indexers": [{"as": name, "pool": name, "anchor": host name, "raises": exc?}]}
An entry with "raises" names the exception SGLang's constructor must raise: a layout the pool does
not support; "hosts" then lists the exception's type and message for it.
Arguments as oracle/run_pools.py's: {"dtype": "bfloat16"} is a torch dtype.

Usage: python -I oracle/run_host_pools.py direct <sglang checkout> <sequence.json> <out.json>
Environment as for oracle/run_cache_iface.py (ORACLE_LOCKFILE, ORACLE_MODEL_REVISION).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import traces  # noqa: E402

TIER = 2
INPUTS = ("oracle/run_host_pools.py", "oracle/run_hicache.py", "oracle/hicache_interfaces.py",
          "oracle/cache_interfaces.py", "oracle/run_scheduler.py", "oracle/traces.py", "oracle/mocks.json",
          "oracle/harness.json", "oracle/requirements.lock")
DEVICE_POOLS = ("MHATokenToKVPool", "MLATokenToKVPool", "DSATokenToKVPool")


def decode(v):
    import torch

    if isinstance(v, dict) and "dtype" in v:
        return getattr(torch, v["dtype"])
    if isinstance(v, list):
        return [decode(x) for x in v]
    return v


def host_summary(name, host) -> dict:
    return {"as": name, "class": type(host).__name__, "layout": host.layout, "size": int(host.size),
            "page_num": int(host.page_num), "page_size": int(host.page_size),
            "size_per_token": int(host.size_per_token), "layer_num": int(host.layer_num)}


def drive(sglang_root: str, path: str) -> dict:
    import run_scheduler

    for name in [n for n in os.environ if n.startswith("SGLANG_")]:
        os.environ.pop(name)
    os.environ.update(run_scheduler.KEPT_SGLANG_ENV)
    import run_hicache
    from hicache_interfaces import HiCacheMonitor
    from sglang.srt.mem_cache import memory_pool
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _build_mha_mla_host_pool
    from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost, make_dsa_indexer_pool_decl
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    run_hicache.install_host_register_noop()
    with open(path) as f:
        sq = json.load(f)
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    mon = HiCacheMonitor()
    pools, hosts, summary = {}, {}, []
    with mon:
        for p in sq["pools"]:
            if p["class"] not in DEVICE_POOLS:
                raise SystemExit(f"{path}: no device pool {p['class']}")
            pools[p["as"]] = getattr(memory_pool, p["class"])(device="cpu", **{k: decode(v) for k, v in p["args"].items()})
        def build(entry, make):
            try:
                host = make()
            except Exception as e:  # a layout the sequence expects SGLang to refuse
                if type(e).__name__ != entry.get("raises"):
                    raise
                summary.append({"as": entry["as"], "raises": type(e).__name__, "message": str(e)})
                return
            if "raises" in entry:
                raise SystemExit(f"{path}: {entry['as']} did not raise {entry['raises']}")
            hosts[entry["as"]] = host
            summary.append(host_summary(entry["as"], host))

        for h in sq["hosts"]:
            build(h, lambda h=h: _build_mha_mla_host_pool(pool=pools[h["pool"]], host_to_device_ratio=h["ratio"],
                                                          page_size=h["page_size"], layout=h["layout"],
                                                          allocator_type="default", pool_label="kv"))
        for x in sq.get("indexers", []):
            build(x, lambda x=x: DSAIndexerPoolHost(decl=make_dsa_indexer_pool_decl(pools[x["pool"]]),
                                                    anchor_host=hosts[x["anchor"]], allocator_type="default"))
    for h in hosts.values():
        h.destroy()
    return {"provenance": traces.provenance(TIER, sglang_root, path, sq, INPUTS), "kind": "direct",
            "calls": mon.calls, "hosts": summary}


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
    if not trace["hosts"]:
        print(f"{path}: no host pool was built", file=sys.stderr)
        return 1
    traces.write(out, trace)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
