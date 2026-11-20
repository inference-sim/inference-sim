# Original to blis-vllm.
"""An in-memory NIXL agent: the byte transport of vLLM's obj and p2p secondary tiers.

vLLM v0.33.0's ObjectStoreSecondaryTierManager (vllm/v1/kv_offload/tiering/obj/manager.py@4ba787ec)
moves blocks between host memory and an S3 bucket through NIXL's OBJ plugin. Neither NIXL nor a
bucket is in the oracle's CPU environment, so oracle/run_tiering.py puts this package first on
sys.path and vllm.distributed.nixl_utils imports it as nixl._api (nixl_utils.py:40-78). It is the
transport (oracle/mocks.json): it stores and reads bytes and answers existence probes. Every
decision stays in vLLM's tier.

The bucket follows S3's object semantics, as the OBJ plugin uses them:
- query_memory answers None for a key with no object and a dict for one that exists;
- a WRITE puts each object whole (an existing object is replaced); a READ of a key with no object
  fails, and NIXL reports the transfer as a whole (state "ERR");
- buckets are shared by name across agents in the process, as two instances share one bucket.

The p2p tier's NixlTransport (tiering/p2p/data/nixl.py@4ba787ec) uses the UCX backend: an agent's
metadata names it, add_remote_agent makes a peer's blocks addressable, and a WRITE copies the
local blocks to the peer's (both instances' host memory is in this process). A peer that is
down() fails the WRITEs to it, as UCX reports a lost endpoint.

Timing is the driver's: transfer() leaves a transfer in flight ("PROC") until the driver calls
complete_all() after each call, so every transfer a call submits ends, in submission order, before
the next call. The one exception is the tier's drain_jobs, which blocks until NIXL finishes: a
check_xfer_state made inside it finishes every transfer first. The driver can hold an agent's
transfers in flight across calls (hold, as a slow transfer spans steps), make the bucket
refuse writes (deny) or make an agent's next transfer() fail at submission (refuse_next).
"""

import ctypes
import sys
import threading

BUCKETS: dict = {}
DENIED: set = set()
_INFLIGHT: list = []
_DRAIN = {("vllm.v1.kv_offload.tiering.obj.manager", "drain_jobs"),
          ("vllm.v1.kv_offload.tiering.p2p.manager", "drain_jobs")}


class nixl_agent_config:  # noqa: N801 (NIXL's name)
    def __init__(self, backends=None, capture_telemetry=False, **kw):
        self.backends = list(backends or [])
        self.capture_telemetry = capture_telemetry


class _RegDescs:
    def __init__(self, descs, mem):
        self.descs, self.mem = list(descs), mem

    def trim(self):
        return list(self.descs)


class _Dlist:
    def __init__(self, descs, agent=None):
        self.descs, self.agent = list(descs), agent


AGENTS: dict = {}  # agent name -> agent
# NixlTransport registers a peer on a worker thread (data/nixl.py add_remote_peer_async). Whether
# that finishes before the session next checks it would be the thread scheduler's choice, so a
# registration made off the main thread waits here until the driver opens the gate, after the
# call that started it: every registration resolves between calls.
REG_GATE = threading.Event()
DOWN: set = set()  # names of agents whose instance is down


class _Telemetry:
    xferDuration = 0  # noqa: N815 (NIXL's name)


class _Xfer:
    def __init__(self, agent, op, pairs, peer=None):
        self.agent, self.op, self.pairs, self.state, self.peer = agent, op, pairs, "PROC", peer

    def run(self):
        if self.peer is not None:
            if self.peer in DOWN or self.peer not in AGENTS:
                self.state = "ERR"
                return
            for (src, n), (dst, m) in self.pairs:
                ctypes.memmove(dst, src, min(n, m))
            self.state = "DONE"
            return
        bucket = self.agent.bucket
        if self.op == "WRITE":
            if self.agent.bucket_name in DENIED:
                self.state = "ERR"
                return
            for (addr, n), key in self.pairs:
                bucket[key] = ctypes.string_at(addr, n)
            self.state = "DONE"
        else:
            if any(key not in bucket for _, key in self.pairs):
                self.state = "ERR"
                return
            for (addr, n), key in self.pairs:
                ctypes.memmove(addr, bucket[key], min(n, len(bucket[key])))
            self.state = "DONE"


def complete_all(draining: bool = False) -> None:
    """Finish every transfer in flight, in submission order, except a held agent's (unless the
    tier is draining)."""
    for x in [x for x in _INFLIGHT if draining or not x.agent.held]:
        _INFLIGHT.remove(x)
        x.run()


def reset() -> None:
    BUCKETS.clear()
    DENIED.clear()
    AGENTS.clear()
    DOWN.clear()
    REG_GATE.clear()
    _INFLIGHT.clear()


def put(agent, key: str, data: bytes) -> None:
    """Another process stores an object in the agent's bucket."""
    agent.bucket[key] = data


def remove(agent, key: str) -> None:
    """Another process deletes an object from the agent's bucket."""
    del agent.bucket[key]


def deny(agent, on: bool) -> None:
    """The agent's bucket refuses writes (on) or accepts them again."""
    (DENIED.add if on else DENIED.discard)(agent.bucket_name)


def agent_down(name: str) -> None:
    """The instance owning agent `name` died: transfers to it fail."""
    DOWN.add(name)


def hold(agent, on: bool) -> None:
    """The agent's transfers stay in flight across driver calls (on) until released; its tier's
    drain_jobs still finishes them."""
    agent.held = on


def refuse_next(agent) -> None:
    """The agent's next transfer() fails at submission."""
    agent.refuse_next += 1


def _in_drain() -> bool:
    f = sys._getframe(1)
    while f is not None:
        if (f.f_globals.get("__name__"), f.f_code.co_name) in _DRAIN:
            return True
        f = f.f_back
    return False


class nixl_agent:  # noqa: N801 (NIXL's name)
    def __init__(self, name, config=None):
        self.name, self.config = name, config
        self.bucket_name, self.bucket = None, None
        # How many of this agent's next transfer() calls fail at submission (the driver's refuse_next).
        self.refuse_next = 0
        # Whether the driver holds this agent's transfers in flight (hold).
        self.held = False
        self.remotes: set = set()
        AGENTS[name] = self

    def create_backend(self, kind, params):
        if kind == "UCX":
            return
        assert kind == "OBJ", kind
        self.bucket_name = params["bucket"]
        self.bucket = BUCKETS.setdefault(self.bucket_name, {})

    def query_memory(self, descs, backend, mem):
        return [{} if d[3] in self.bucket else None for d in descs]

    def register_memory(self, descs, mem_type=None):
        return _RegDescs(descs, mem_type)

    def deregister_memory(self, _):
        return None

    def get_agent_metadata(self) -> bytes:
        return self.name.encode()

    def add_remote_agent(self, metadata) -> str:
        if threading.current_thread() is not threading.main_thread():
            REG_GATE.wait()
        name = bytes(metadata).decode()
        self.remotes.add(name)
        return name

    def remove_remote_agent(self, name) -> None:
        self.remotes.discard(name)

    def get_xfer_descs(self, descs, mem_type=None):
        return list(descs)

    def prep_xfer_dlist(self, agent, descs, mem_type=None):
        return _Dlist(descs, agent)

    def make_prepped_xfer(self, op, local, local_idx, remote, remote_idx):
        if remote.agent not in ("ObjAgent", "NIXL_INIT_AGENT", None):
            assert op == "WRITE", op
            pairs = [((local.descs[int(li)][0], local.descs[int(li)][1]),
                      (remote.descs[int(ri)][0], remote.descs[int(ri)][1]))
                     for li, ri in zip(list(local_idx), list(remote_idx))]
            return _Xfer(self, op, pairs, peer=remote.agent)
        pairs = []
        for li, ri in zip(list(local_idx), list(remote_idx)):
            addr, n = local.descs[int(li)][0], local.descs[int(li)][1]
            pairs.append(((addr, n), remote.descs[int(ri)][3]))
        return _Xfer(self, op, pairs)

    def transfer(self, x):
        if self.refuse_next:
            self.refuse_next -= 1
            return "ERR"
        _INFLIGHT.append(x)
        return "PROC"

    def check_xfer_state(self, x):
        if x.state == "PROC" and _in_drain():
            complete_all(draining=True)
        return x.state

    def get_xfer_telemetry(self, x):
        return _Telemetry()

    def release_xfer_handle(self, x):
        if x in _INFLIGHT:
            _INFLIGHT.remove(x)

    def release_dlist_handle(self, _):
        return None
