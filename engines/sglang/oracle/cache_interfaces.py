# Original to blis-sglang.
"""Traces at the cache framework's interfaces (design FID-5, §7.6).

The cache framework is six abstract classes of SGLang v0.5.23 (python/sglang/srt/mem_cache/ at
297e54fd105e974cdf401e5b4d20253a22c16f90):

  prefix_cache   BasePrefixCache            base_prefix_cache.py:390, with insert (every tree
                                            cache defines it; InsertParams, :77, is its argument)
                                            and __init__ (its CacheInitParams)
  tree_core      UnifiedTreeCoreInterface   unified_cache/unified_tree_core_interface.py:167
  component      TreeComponent              unified_cache/components/base.py:173
  allocator      BaseTokenToKVPoolAllocator allocator/base.py:41
  req_to_token   ReqToTokenPool             memory_pool.py:376
  kv_pool        KVCache                    memory_pool.py:2005

An interface's methods are the functions its abstract class defines (public ones, and __init__ for
the three pools and the allocator, whose construction is their sizing), as each loaded subclass
defines or inherits them. A trace is the sequence of calls made into an interface from outside it,
each with its arguments and SGLang's answer. A call an implementation makes into its own interface
(UnifiedRadixCache.checkpoint calling its own insert, TokenToKVPoolAllocator.__init__ calling the
base __init__) is its business and is not recorded; a call it makes into another interface (the
radix cache freeing slots at the allocator) is recorded there, with `in` naming the enclosing
recorded call, and so is a call that comes back through another interface (the allocator's
evict_to_free_tokens calling the cache's evict_for_alloc, which frees slots at the allocator).

Calls are observed with sys.monitoring (PEP 669), as events on the code objects of the interface
methods. No SGLang object, class or module is written to, so this replaces nothing and needs no
entry in oracle/mocks.json. A method wrapped by a decorator (rank_consensus) is observed on the
function it wraps.

The format, which core/cache/trace.go reads:
    {"calls": [call...]}
    call = {"seq": n, "i": interface, "cls": class, "m": method, "args": {...},
            "ret": ..., "raise": exception class (instead of ret), "in": seq of the enclosing call}
Values are encoded by enc(): a tensor or array is its list, a RadixKey its tokens and namespace,
a tree node {"node": id}, a request {"req": rid, ...its KV record}, a dataclass, named tuple or
msgspec struct its fields, an enum its name. Each request is also encoded on return
("req_after"), since checkpoint and the pools change it.
"""

import dataclasses
import enum
import inspect
import sys

# (interface, module, abstract class, extra methods)
INTERFACES = (
    ("prefix_cache", "sglang.srt.mem_cache.base_prefix_cache", "BasePrefixCache", ("insert", "__init__")),
    ("tree_core", "sglang.srt.mem_cache.unified_cache.unified_tree_core_interface", "UnifiedTreeCoreInterface", ()),
    ("component", "sglang.srt.mem_cache.unified_cache.components.base", "TreeComponent", ()),
    ("allocator", "sglang.srt.mem_cache.allocator.base", "BaseTokenToKVPoolAllocator", ("__init__",)),
    ("req_to_token", "sglang.srt.mem_cache.memory_pool", "ReqToTokenPool", ("__init__",)),
    ("kv_pool", "sglang.srt.mem_cache.memory_pool", "KVCache", ("__init__",)),
)
# The modules whose classes implement the interfaces on the paths the scenarios take; importing
# them before the monitor starts registers their classes.
IMPLEMENTATIONS = (
    "sglang.srt.mem_cache.unified_radix_cache",
    "sglang.srt.mem_cache.unified_cache.unified_tree_core",
    "sglang.srt.mem_cache.unified_cache.components.full",
    "sglang.srt.mem_cache.allocator.token",
    "sglang.srt.mem_cache.allocator.paged",
)
CO_GENERATOR = 0x20
MAX_DEPTH = 8

# What a pool or allocator is, once built: the attributes its sizing sets.
SIZING = ("disable", "size", "page_size", "dtype", "store_dtype", "layer_num", "head_num", "head_dim", "v_head_dim",
          "kv_lora_rank", "qk_rope_head_dim", "index_head_dim", "max_context_len", "need_sort")


def interface_methods(base) -> set:
    names = set()
    for name, value in vars(base).items():
        if name.startswith("_") or isinstance(value, property):
            continue
        if inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod)):
            names.add(name)
    return names


def subclasses(cls) -> list:
    out, todo = [], [cls]
    while todo:
        c = todo.pop()
        out.append(c)
        todo.extend(c.__subclasses__())
    return out


def code_of(value):
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    if not callable(value):
        return None
    value = inspect.unwrap(value)
    return getattr(value, "__code__", None)


class Monitor:
    """Records the calls at the six interfaces while it is entered."""

    def __init__(self):
        import importlib

        for mod in IMPLEMENTATIONS:
            importlib.import_module(mod)
        self.codes = {}  # code object -> (interface, method)
        for iface, mod, cls_name, extra in INTERFACES:
            base = getattr(importlib.import_module(mod), cls_name)
            methods = interface_methods(base) | set(extra)
            for cls in subclasses(base):
                for name in methods:
                    code = code_of(cls.__dict__.get(name))
                    if code is not None:
                        prev = self.codes.setdefault(code, (iface, name))
                        if prev != (iface, name):
                            raise RuntimeError(f"cache_interfaces: {code.co_qualname} is both {prev} and {(iface, name)}")
        self.calls = []
        self.stack = []  # (code, record or None, interface, frame) of the calls in progress
        self.tool = None
        self.reqs = []  # every request seen, kept alive so encoding stays stable
        self.req_to_token_pool = None  # the pool whose __init__ was recorded, to read a request's row

    # --- encoding ---

    def enc(self, x, depth=0):
        if depth > MAX_DEPTH:
            return {"_t": type(x).__name__}
        if isinstance(x, enum.Enum):
            return x.name
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, slice):
            return {"_t": "slice", "start": self.enc(x.start), "stop": self.enc(x.stop), "step": self.enc(x.step)}
        tname = type(x).__name__
        mod = type(x).__module__ or ""
        if tname == "Tensor" and mod.startswith("torch"):
            return x.tolist()
        if mod.startswith("torch"):
            return str(x)  # a dtype or device
        if tname == "array":
            return list(x)
        if tname == "RadixKey":
            return {"tokens": list(x.raw_token_ids()), "extra_key": x.extra_key, "cache_salt": x.cache_salt,
                    "bigram": bool(x.is_bigram)}
        if tname == "UnifiedTreeNode":
            return {"node": x.id}
        if tname == "Req" and hasattr(x, "rid"):
            return self.req(x, depth)
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            return {"_t": tname, **{f.name: self.enc(getattr(x, f.name), depth + 1) for f in dataclasses.fields(x)}}
        if isinstance(x, tuple) and hasattr(x, "_fields"):
            return {"_t": tname, **{f: self.enc(getattr(x, f), depth + 1) for f in x._fields}}
        if hasattr(type(x), "__struct_fields__"):
            return {"_t": tname, **{f: self.enc(getattr(x, f), depth + 1) for f in type(x).__struct_fields__}}
        if isinstance(x, dict):
            return {str(self.enc(k, depth + 1)): self.enc(v, depth + 1) for k, v in x.items()}
        if isinstance(x, (list, tuple, set, frozenset)):
            items = [self.enc(v, depth + 1) for v in x]
            return sorted(items, key=repr) if isinstance(x, (set, frozenset)) else items
        if callable(x):
            return {"_t": "callable"}
        return {"_t": tname}

    def req(self, r, depth):
        if not any(r is q for q in self.reqs):
            self.reqs.append(r)
        kv = getattr(r, "kv", None)
        out = {"req": r.rid}
        if kv is not None:
            out["kv"] = {k: self.enc(getattr(kv, k, None), depth + 1) for k in
                         ("req_pool_idx", "cache_protected_len", "cache_inserted_len", "kv_committed_len",
                          "kv_allocated_len")}
            pool = self.req_to_token_pool
            if pool is not None and kv.req_pool_idx is not None:
                # The request's row of the pool: the slots its KV holds, as checkpoint reads them.
                out["row"] = pool.req_to_token[kv.req_pool_idx, :kv.kv_allocated_len].tolist()
        for k in ("origin_input_ids", "output_ids", "full_untruncated_fill_ids", "extra_key", "cache_salt", "priority",
                  "session_id"):
            if hasattr(r, k):
                out[k] = self.enc(getattr(r, k), depth + 1)
        if hasattr(r, "last_node"):
            out["last_node"] = self.enc(r.last_node, depth + 1)
        # The tree lock the request holds and its matched prefix, which checkpoint, unlock and
        # prefix_device_indices read.
        if hasattr(r, "lock"):
            out["lock"] = self.enc(r.lock, depth + 1)
        if hasattr(r, "prefix_len"):
            out["prefix_len"] = self.enc(r.prefix_len, depth + 1)
        out["finished"] = bool(r.finished()) if hasattr(r, "finished") else None
        return out

    def args(self, frame, code):
        n = code.co_argcount + code.co_kwonlyargcount
        names = code.co_varnames[:n]
        if code.co_flags & inspect.CO_VARARGS:
            names += (code.co_varnames[n],)
            n += 1
        if code.co_flags & inspect.CO_VARKEYWORDS:
            names += (code.co_varnames[n],)
        loc = frame.f_locals
        return {k: self.enc(loc[k]) for k in names if k not in ("self", "cls") and k in loc}

    def sizing(self, obj):
        return {k: self.enc(getattr(obj, k)) for k in SIZING if hasattr(obj, k)
                and not callable(getattr(type(obj), k, None))}

    # --- events ---

    def _top(self, iface) -> bool:
        # A call is at the interface unless the innermost interface call in progress is one of
        # the same interface: a call an implementation makes into itself. A call that re-enters
        # an interface through another one (the allocator's evict_to_free_tokens calling the
        # cache's evict_for_alloc, which frees slots at the allocator) is at the interface.
        return not self.stack or self.stack[-1][2] != iface

    def _enclosing(self):
        for _, rec, _, _ in reversed(self.stack):
            if rec is not None:
                return rec["seq"]
        return None

    def _start(self, code, offset):
        iface, m = self.codes[code]
        f = sys._getframe(2)
        rec = None
        if self._top(iface):
            rec = {"seq": len(self.calls), "i": iface, "m": m}
            slf = f.f_locals.get("self")
            if slf is not None:
                rec["cls"] = type(slf).__name__
            rec["args"] = self.args(f, code)
            enclosing = self._enclosing()
            if enclosing is not None:
                rec["in"] = enclosing
            self.calls.append(rec)
        self.stack.append((code, rec, iface, f))

    def _finish(self, code, frame):
        if not self.stack or self.stack[-1][0] is not code or self.stack[-1][3] is not frame:
            raise RuntimeError("cache_interfaces: unpaired return from %s" % code.co_qualname)
        return self.stack.pop()

    def _return(self, code, offset, value):
        f = sys._getframe(2)
        if code.co_flags & CO_GENERATOR:
            raise RuntimeError("cache_interfaces: generator interface method %s" % code.co_qualname)
        _, rec, iface, frame = self._finish(code, f)
        if rec is None:
            return
        if rec["m"] == "__init__":
            built = frame.f_locals["self"]
            rec["ret"] = self.sizing(built)
            if iface == "req_to_token":
                self.req_to_token_pool = built
        else:
            rec["ret"] = self.enc(value)
        reqs = [r for r in self.reqs if any(isinstance(v, dict) and v.get("req") == r.rid for v in rec["args"].values())]
        if reqs:
            rec["req_after"] = [self.req(r, 0) for r in reqs]

    def _unwind(self, code, offset, exc):
        f = sys._getframe(2)
        _, rec, _, _ = self._finish(code, f)
        if rec is not None:
            rec["raise"] = type(exc).__name__

    def __enter__(self):
        mon = sys.monitoring
        for tid in (mon.PROFILER_ID, 3, 4):
            if mon.get_tool(tid) is None:
                self.tool = tid
                break
        if self.tool is None:
            raise RuntimeError("cache_interfaces: no free sys.monitoring tool id")
        mon.use_tool_id(self.tool, "blis-sglang FID-5 cache interface trace")
        E = mon.events
        mon.register_callback(self.tool, E.PY_START, lambda c, o: self._start(c, o) if c in self.codes else None)
        mon.register_callback(self.tool, E.PY_RETURN, lambda c, o, v: self._return(c, o, v) if c in self.codes else None)
        mon.register_callback(self.tool, E.PY_UNWIND, lambda c, o, e: self._unwind(c, o, e) if c in self.codes else None)
        # PY_UNWIND is a global event only.
        mon.set_events(self.tool, E.PY_UNWIND)
        for code in self.codes:
            mon.set_local_events(self.tool, code, E.PY_START | E.PY_RETURN)
        return self

    def __exit__(self, *exc):
        mon = sys.monitoring
        mon.set_events(self.tool, 0)
        for code in self.codes:
            mon.set_local_events(self.tool, code, 0)
        for e in (mon.events.PY_START, mon.events.PY_RETURN, mon.events.PY_UNWIND):
            mon.register_callback(self.tool, e, None)
        mon.free_tool_id(self.tool)
        return False
