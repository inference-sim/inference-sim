# Original to blis-sglang.
"""Traces at the HiCache framework's interfaces (design FID-5, §7.6).

The HiCache framework of SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90) is five
classes, each traced as an interface here, besides the six of the cache framework
(oracle/cache_interfaces.py), whose calls the controller makes in turn:

  controller     HiCacheController         managers/cache_controller.py:295, with the public
                                           methods its subclass HybridCacheController adds
                                           (mem_cache/hybrid_cache/hybrid_cache_controller.py:206),
                                           which UnifiedRadixCache calls, and __init__
  layer_counter  LayerDoneCounter          managers/cache_controller.py:83
  l2_transfer    L2TransferEngine          mem_cache/l2_transfer.py:50
  host_pool      HostKVCache               mem_cache/pool_host/base.py:157, and __init__ (its sizing)
  storage        HiCacheStorage            mem_cache/hicache_storage.py:293, and __init__

The controller's thread bodies (prefetch_thread_func, prefetch_io_aux_func,
prefetch_sync_thread_func, backup_thread_func) are its public methods: oracle/step_runner.py calls
them at each step boundary, so each run of one is a controller call the trace records, with the
storage calls it made inside it.

The format and the recording rules are oracle/cache_interfaces.py's. Three encodings differ:
- a floating-point tensor (KV data: a page read from or written to storage) is its shape and
  dtype, not its values: the oracle runs no forward pass, so the values are not SGLang's decisions;
- an integer that is the id() of an object the trace has met (HiCache keys an allocator's
  transfer event on (id(controller), "write"), cache_controller.py:906) is {"_id_of": its class}:
  a trace never records a memory address;
- a storage or prefetch operation is its fields, and a HiCache ack its counts and node ids;
- a path the runner made for one run (the file backend's directory in a recorded scenario) is
  its placeholder, wherever a string holds it, so a trace names no run's own directory.
"""
import inspect

from cache_interfaces import IMPLEMENTATIONS, INTERFACES, SIZING, Monitor

HICACHE_INTERFACES = (
    ("controller", "sglang.srt.managers.cache_controller", "HiCacheController", ("__init__",)),
    ("layer_counter", "sglang.srt.managers.cache_controller", "LayerDoneCounter", ()),
    ("l2_transfer", "sglang.srt.mem_cache.l2_transfer", "L2TransferEngine", ()),
    ("host_pool", "sglang.srt.mem_cache.pool_host.base", "HostKVCache", ("__init__",)),
    ("storage", "sglang.srt.mem_cache.hicache_storage", "HiCacheStorage", ("__init__",)),
)
HYBRID_CONTROLLER = ("sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller", "HybridCacheController")
HICACHE_IMPLEMENTATIONS = (
    "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
    "sglang.srt.mem_cache.pool_host.mha",
    "sglang.srt.mem_cache.hicache_storage",
)
HOST_SIZING = ("size", "page_size", "page_num", "size_per_token", "layout", "dtype", "layer_num", "head_num",
               "head_dim", "device_capacity_tokens", "start_layer", "end_layer")
STORAGE_OPERATION_FIELDS = ("id", "request_id", "token_ids", "last_hash", "hash_value", "completed_tokens",
                            "host_indices", "storage_hit_count", "prefix_keys")


def public_functions(cls) -> tuple:
    return tuple(sorted(n for n, v in vars(cls).items() if not n.startswith("_")
                        and (inspect.isfunction(v) or isinstance(v, (staticmethod, classmethod)))))


class HiCacheMonitor(Monitor):
    """Records the calls at the HiCache interfaces and the cache framework's, while entered."""

    sizing_names = tuple(dict.fromkeys(SIZING + HOST_SIZING))

    def __init__(self, placeholders=None):
        import importlib

        # A path made for this run -> the placeholder the trace records in its place.
        self.placeholders = dict(placeholders or {})

        hybrid = getattr(importlib.import_module(HYBRID_CONTROLLER[0]), HYBRID_CONTROLLER[1])
        ifaces = []
        for iface, mod, cls_name, extra in HICACHE_INTERFACES:
            if iface == "controller":
                extra = tuple(extra) + public_functions(hybrid)
            ifaces.append((iface, mod, cls_name, extra))
        self.addresses = {}  # id() of every object an interface call was made on -> its class name
        super().__init__(interfaces=INTERFACES + tuple(ifaces),
                         implementations=IMPLEMENTATIONS + HICACHE_IMPLEMENTATIONS)

    def _start(self, code, offset, f=None):
        import sys

        f = f if f is not None else sys._getframe(2)
        slf = f.f_locals.get("self")
        if slf is not None:
            self.addresses.setdefault(id(slf), type(slf).__name__)
        super()._start(code, offset, f)

    def enc(self, x, depth=0):
        if type(x) is str:  # a str enum is the base class's to encode, by its name
            for path, placeholder in self.placeholders.items():
                x = x.replace(path, placeholder)
            return x
        if isinstance(x, int) and not isinstance(x, bool) and x in self.addresses:
            return {"_id_of": self.addresses[x]}
        tname = type(x).__name__
        mod = type(x).__module__ or ""
        if tname == "Tensor" and mod.startswith("torch") and x.is_floating_point():
            return {"_t": "Tensor", "shape": list(x.shape), "dtype": str(x.dtype)}
        if tname in ("StorageOperation", "PrefetchOperation"):
            out = {"_t": tname}
            for k in STORAGE_OPERATION_FIELDS:
                if hasattr(x, k):
                    out[k] = self.enc(getattr(x, k), depth + 1)
            if hasattr(x, "is_terminated"):
                out["terminated"] = bool(x._terminated_flag)
            return out
        if tname == "HiCacheAck":
            return {"_t": tname, "node_ids": self.enc(list(x.node_ids), depth + 1), "num_tokens": x.num_tokens,
                    "num_bytes": x.num_bytes, "num_tokens_by_pool": self.enc(x.num_tokens_by_pool, depth + 1)}
        if tname == "CacheOperation":
            return {"_t": tname, "id": x.id, "node_ids": list(x.node_ids), "priority": x.priority,
                    "host_indices": self.enc(x.host_indices, depth + 1),
                    "device_indices": self.enc(x.device_indices, depth + 1)}
        return super().enc(x, depth)
