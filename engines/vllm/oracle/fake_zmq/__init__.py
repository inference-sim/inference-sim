# Original to blis-vllm.
"""An in-memory stand-in for the part of pyzmq the p2p tier's control transport uses.

vLLM v0.33.0's ZmqTransport (vllm/v1/kv_offload/tiering/p2p/control/zmq.py@4ba787ec) carries the
p2p session protocol between instances: each instance binds a ROUTER, and opens one DEALER per
peer, whose identity is the instance's own id, connected to that peer's ROUTER. A monitor socket
on each DEALER reports a lost connection (EVENT_DISCONNECTED). Over real sockets the time a
message takes to arrive is the operating system's, so oracle/run_tiering.py gives the control
module this package as its `zmq` instead (the transport, listed in oracle/mocks.json). Messages
and their order are vLLM's; only delivery is the driver's:

- a DEALER's send puts the bytes on the wire, in send order;
- deliver() moves every message on the wire to the ROUTER bound at its address, in order (a
  message to an address nobody has bound stays on the wire, as ZMQ queues it until it connects);
- down(addr) is a peer that died: its ROUTER is unbound, what is on the wire to it is lost, and
  every DEALER connected to it reports EVENT_DISCONNECTED on its monitor.
"""

ROUTER, DEALER, PAIR = 6, 5, 0
NOBLOCK = 1
HEARTBEAT_IVL, HEARTBEAT_TIMEOUT, HEARTBEAT_TTL = 75, 77, 76
SNDHWM, LINGER = 23, 17
EVENT_CLOSED, EVENT_DISCONNECTED = 0x0080, 0x0200


class ZMQError(Exception):
    pass


class Again(ZMQError):
    pass


_ROUTERS: dict = {}  # bound address -> socket
_MONITORS: dict = {}  # inproc address -> list of events
_WIRE: list = []  # (address, identity, data), in send order
_DEALERS: list = []


def reset() -> None:
    _ROUTERS.clear()
    _MONITORS.clear()
    _WIRE.clear()
    _DEALERS.clear()


def deliver() -> None:
    """Move every message on the wire to its bound ROUTER, in send order."""
    keep = []
    for addr, ident, data in _WIRE:
        r = _ROUTERS.get(addr)
        if r is None:
            keep.append((addr, ident, data))
        else:
            r._inbox.append([ident, data])
    _WIRE[:] = keep


def in_flight() -> int:
    return len(_WIRE)


def down(addr: str) -> None:
    """The instance bound at addr died: unbind it, drop what was sent to it, and disconnect
    every DEALER connected to it."""
    _ROUTERS.pop(addr, None)
    _WIRE[:] = [w for w in _WIRE if w[0] != addr]
    for d in _DEALERS:
        if d._target == addr and not d._closed and d._monitor_addr is not None:
            _MONITORS.setdefault(d._monitor_addr, []).append({"event": EVENT_DISCONNECTED})


class _Socket:
    def __init__(self, kind: int) -> None:
        self.kind = kind
        self.identity = b""
        self._inbox: list = []
        self._target = None
        self._monitor_addr = None
        self._pair_addr = None
        self._closed = False
        self._bound = None
        if kind == DEALER:
            _DEALERS.append(self)

    def setsockopt(self, opt, value) -> None:
        return None

    def bind(self, addr: str) -> None:
        if addr in _ROUTERS:
            raise ZMQError("Address already in use")
        _ROUTERS[addr] = self
        self._bound = addr

    def connect(self, addr: str) -> None:
        if self.kind == PAIR:
            self._pair_addr = addr
            _MONITORS.setdefault(addr, [])
        else:
            self._target = addr

    def monitor(self, addr: str, mask: int) -> None:
        self._monitor_addr = addr
        _MONITORS.setdefault(addr, [])

    def send(self, data, flags: int = 0) -> None:
        if self._closed:
            raise ZMQError("Socket operation on non-socket")
        _WIRE.append((self._target, bytes(self.identity), bytes(data)))

    def recv_multipart(self, flags: int = 0):
        if not self._inbox:
            raise Again("Resource temporarily unavailable")
        return self._inbox.pop(0)

    def close(self, linger=None) -> None:
        self._closed = True
        if self._bound is not None and _ROUTERS.get(self._bound) is self:
            del _ROUTERS[self._bound]


class Context:
    def socket(self, kind: int) -> _Socket:
        return _Socket(kind)

    def destroy(self, linger=None) -> None:
        return None


class _Monitor:
    @staticmethod
    def recv_monitor_message(sock: _Socket, flags: int = 0) -> dict:
        q = _MONITORS.get(sock._pair_addr)
        if not q:
            raise Again("Resource temporarily unavailable")
        return q.pop(0)


class _Utils:
    monitor = _Monitor()


utils = _Utils()
Socket = _Socket
