# Original to blis-vllm.
"""Traces at the KV offload framework's interfaces (design FID-5, §7.6).

A framework's trace is the sequence of calls made at its interface, each with its arguments and
vLLM's answer. Two interfaces are traced here: OffloadingManager (vllm/v1/kv_offload/base.py:
235-421@5df63a72), as the connector's scheduler side calls it, and CachePolicy
(vllm/v1/kv_offload/cpu/policies/base.py:48-140@5df63a72), as the manager calls it. Each spec's
get_manager is recorded with what the spec derived from its config (OffloadingSpec, base.py:595-640).

Interface calls are observed with sys.monitoring (PEP 669): events on the code objects of the
interface methods, nothing more. No vLLM object, class or module is written to, so this replaces
nothing and needs no entry in oracle/mocks.json. A call made from inside another call of the same
interface (TieringOffloadingManager forwarding to its CPU primary tier, the policy's touch inside
on_store_miss) is the callee's business, not a call at the interface, and is not recorded.

No traced method reads the clock: the CPU manager, the LRU policy and a tiering manager with no
secondary tiers decide on keys, counts and ranks alone. The scheduler that calls them runs with the
clock replacement oracle/mocks.json lists (FID-1).

The trace format, which core/connector/offloading/fid5_test.go replays against the port:
    {"spec": [call...], "manager": [call...], "policy": [call...]}
    call = {"m": method, "args": {...}, "ret": ...}
An OffloadKey is its bytes in hex, a ReqContext is "c<n>" in order of first use, with its req_id.
"""

import sys

MANAGER_METHODS = ("on_new_request", "lookup", "get_load_source", "prepare_load", "touch", "complete_load",
                   "prepare_store", "complete_store", "on_request_finished", "take_events", "on_schedule_end",
                   "has_pending_work", "reset_cache", "get_stats", "shutdown")
POLICY_METHODS = ("get", "insert", "remove", "touch", "on_store_miss", "on_request_finished", "evict", "clear",
                  "mark_evictable", "mark_non_evictable")
CO_GENERATOR = 0x20


class Monitor:
    """Records the calls at the two interfaces while it is entered."""

    def __init__(self):
        from vllm.v1.kv_offload.base import OffloadingManager
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
        from vllm.v1.kv_offload.cpu.policies.base import CachePolicy
        from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
        from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager
        from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

        self.trace = {"spec": [], "manager": [], "policy": []}
        self.codes = {}  # code object -> (interface, method)
        for cls in (OffloadingManager, CPUOffloadingManager, CPUPrimaryTierOffloadingManager, TieringOffloadingManager):
            for name in MANAGER_METHODS:
                f = cls.__dict__.get(name)
                if callable(f) and hasattr(f, "__code__"):
                    self.codes[f.__code__] = ("manager", name)
        for cls in (CachePolicy, LRUCachePolicy):
            for name in POLICY_METHODS:
                f = cls.__dict__.get(name)
                if callable(f) and hasattr(f, "__code__"):
                    self.codes[f.__code__] = ("policy", name)
        for cls in (CPUOffloadingSpec, TieringOffloadingSpec):
            f = cls.__dict__.get("get_manager")
            if f is not None:
                self.codes[f.__code__] = ("spec", "get_manager")
        self.ctx = {}  # id(ReqContext) -> (name, the context, kept alive)
        self.stack = []  # (code, record or None) of the interface calls in progress
        self.gens = {}  # generator frame -> its record
        self.tool = None

    # --- serialisation ---

    def key(self, k) -> str:
        return bytes(k).hex()

    def keys(self, ks) -> list:
        return [self.key(k) for k in ks]

    def context(self, c) -> dict:
        # Each context is held for the whole trace, so a freed one's id() is never reused for another.
        if id(c) not in self.ctx:
            self.ctx[id(c)] = ("c%d" % (len(self.ctx) + 1), c)
        return {"ctx": self.ctx[id(c)][0], "req": c.req_id}

    def positions(self, c, ks) -> dict:
        return {self.key(k): c.get_offload_key_position(k) for k in ks if c.get_offload_key_position(k) is not None}

    def chunk(self, ch):
        return None if ch is None else {"id": int(ch.chunk_id), "ref": int(ch.ref_cnt)}

    def args(self, iface, m, f):
        a = f.f_locals
        out = {}
        if "req_context" in a:
            out.update(self.context(a["req_context"]))
        if iface == "manager":
            if m in ("lookup", "get_load_source"):
                out["key"] = self.key(a["key"])
            elif m in ("prepare_load", "touch", "complete_load", "prepare_store"):
                out["keys"] = self.keys(a["keys"])
            elif m == "complete_store":
                out["keys"], out["success"] = self.keys(a["keys"]), bool(a.get("success", True))
            elif m == "on_request_finished":
                c = a["req_context"]
                out["positions"] = {self.key(k): v for k, v in c._offload_key_positions.items()}
            elif m == "on_schedule_end":
                ctx = a["context"]
                out["new"], out["preempted"] = sorted(ctx.new_req_ids), sorted(ctx.preempted_req_ids)
        elif iface == "policy":
            if m in ("get", "remove", "mark_evictable", "mark_non_evictable"):
                out["key"] = self.key(a["key"])
            elif m == "insert":
                out["key"], out["chunk"] = self.key(a["key"]), self.chunk(a["chunk"])
            elif m in ("touch", "on_store_miss"):
                ks = list(a["keys"])
                out["keys"] = self.keys(ks)
            elif m == "on_request_finished":
                groups = [list(g) for g in a["key_groups"]]
                out["groups"] = [self.keys(g) for g in groups]
                out["positions"] = self.positions(a["req_context"], [k for g in groups for k in g])
            elif m == "evict":
                out["n"], out["protected"] = int(a["n"]), sorted(self.keys(a["protected"]))
        else:
            s = a["self"]
            c = s.config
            out = {"spec": type(s).__name__, "cpu_bytes_to_use": int(s.extra_config.get("cpu_bytes_to_use") or 0),
                   "worker_kv_bytes_per_block": int(c.worker_kv_bytes_per_block),
                   "world_size": int(c.parallel.world_size), "blocks_per_chunk": int(s.blocks_per_chunk),
                   "replicated_layout": bool(s.replicated_layout), "alignment": int(s.BLOCK_SIZE_ALIGNMENT),
                   "eviction_policy": s.eviction_policy, "offload_prompt_only": bool(s.offload_prompt_only),
                   "secondary_tiers": len(s.extra_config.get("secondary_tiers") or [])}
        return out

    def ret(self, iface, m, v, f):
        if iface == "spec":
            s = f.f_locals["self"]
            return {"manager": type(v).__name__, "num_chunks": int(s.num_chunks),
                    "kv_bytes_per_chunk": int(s.kv_bytes_per_chunk)}
        if v is None:
            return None
        if iface == "manager":
            if m == "lookup":
                return v.name
            if m == "get_load_source":
                return v.name
            if m == "on_new_request":
                return {"policy": v.policy.name}
            if m == "prepare_load":
                return [int(i) for i in v.block_ids]
            if m == "prepare_store":
                return {"keys_to_store": self.keys(v.keys_to_store), "ids": [int(i) for i in v.store_spec.block_ids],
                        "evicted": self.keys(v.evicted_keys)}
            if m == "has_pending_work":
                return bool(v)
            if m == "get_stats":
                return {name: {repr(lv): val for lv, val in sorted(vals.items(), key=lambda kv: repr(kv[0]))}
                        for name, vals in sorted(v._values.items())}
            return repr(v)
        if m == "get":
            return self.chunk(v)
        if m == "evict":
            return [{"key": self.key(k), "id": int(ch.chunk_id)} for k, ch in v]
        return repr(v)

    def event(self, e) -> dict:
        return {"keys": self.keys(e.keys), "medium": e.medium.name, "removed": bool(e.removed)}

    # --- events ---
    # Each callback runs inside the lambda registered for it, so the interface method's own frame is
    # two above the callback's: sys._getframe(2).

    def _top(self, iface) -> bool:
        return not any(r[2] == iface for r in self.stack)

    def _start(self, code, offset):
        iface, m = self.codes[code]
        f = sys._getframe(2)
        rec = None
        if self._top(iface):
            rec = {"m": m, "args": self.args(iface, m, f)}
            self.trace[iface].append(rec)
        if code.co_flags & CO_GENERATOR:
            if rec is not None:
                rec["ret"] = []
            self.gens[f] = None if rec is None else (iface, rec)
        else:
            self.stack.append((code, rec, iface, f))

    def _yield(self, code, offset, value):
        r = self.gens.get(sys._getframe(2))
        if r is not None:
            r[1]["ret"].append(self.event(value))

    def _return(self, code, offset, value):
        f = sys._getframe(2)
        if code.co_flags & CO_GENERATOR:
            self.gens.pop(f, None)
            return
        if not self.stack or self.stack[-1][0] is not code:
            raise RuntimeError("kv_interfaces: unpaired return from %s" % code.co_qualname)
        _, rec, iface, frame = self.stack.pop()
        if rec is not None:
            rec["ret"] = self.ret(iface, rec["m"], value, frame)

    def __enter__(self):
        mon = sys.monitoring
        for tid in (mon.PROFILER_ID, 3, 4):
            if mon.get_tool(tid) is None:
                self.tool = tid
                break
        if self.tool is None:
            raise RuntimeError("kv_interfaces: no free sys.monitoring tool id")
        mon.use_tool_id(self.tool, "blis-vllm FID-5 interface trace")
        E = mon.events
        mon.register_callback(self.tool, E.PY_START, lambda c, o: self._start(c, o) if c in self.codes else None)
        mon.register_callback(self.tool, E.PY_RETURN, lambda c, o, v: self._return(c, o, v) if c in self.codes else None)
        mon.register_callback(self.tool, E.PY_YIELD, lambda c, o, v: self._yield(c, o, v) if c in self.codes else None)
        for code in self.codes:
            ev = E.PY_START | E.PY_RETURN
            if code.co_flags & CO_GENERATOR:
                ev |= E.PY_YIELD
            mon.set_local_events(self.tool, code, ev)
        return self

    def __exit__(self, *exc):
        mon = sys.monitoring
        for code in self.codes:
            mon.set_local_events(self.tool, code, 0)
        for e in (mon.events.PY_START, mon.events.PY_RETURN, mon.events.PY_YIELD):
            mon.register_callback(self.tool, e, None)
        mon.free_tool_id(self.tool)
        return False
