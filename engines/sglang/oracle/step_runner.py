# Original to blis-sglang.
"""The step-driven runner (design FID-1, §7.6): HiCache's controller threads and device events,
run at declared step boundaries instead of concurrently.

SGLang v0.5.23's HiCache controller does its storage work in threads
(python/sglang/srt/managers/cache_controller.py:455-478 and :838-856@297e54fd, and
mem_cache/hybrid_cache/hybrid_cache_controller.py:282-283 and :506-507), and its host<->device
copies on device streams whose events the radix cache polls (cache_controller.py:65-80,
mem_cache/l2_transfer.py:31-34, :186-189). Run as threads, the order of their decisions would be a
race. Here no thread starts. Each thread object records its target; at a step boundary the runner
calls each target, unmodified, so that it runs one iteration of its own loop. Nothing of a thread
body is copied (FID-8).

The scheduling rule is RULE, below; it is stated nowhere else. Runner.boundary and
StepDeviceEvent carry it out, and oracle/mocks.json's entries and core/hicache point to it.

Who calls boundary(): oracle/run_hicache.py's direct sequences, at each {"boundary": true} entry;
in a scheduler scenario, the first receive of each event-loop iteration (oracle/harness.py's
begin_step), so boundary k is the start of loop iteration k.

The replacements, each listed in oracle/mocks.json:
  environment  the `threading` module as cache_controller.py and hybrid_cache_controller.py see it:
               Thread is StepThread and Event is StepEvent; every other name is threading's own.
  transport    `Queue` in those two modules: StepQueue, a queue.Queue whose get never waits.
  gpu          `device_module` in cache_controller.py and l2_transfer.py: Event is StepDeviceEvent;
               every other name is the platform's own device module (torch.cpu on the oracle).
"""
import queue
import threading
import types

RULE = """\
At step boundary k, every thread started and not yet joined runs once, in the order the threads
were started. A run is one iteration of the target's loop: the stop event the loop tests reads
clear at its first test and set from then on, unless the controller really set it. A loop whose
own condition goes on past a set stop event (prefetch_thread_func continues while its queue is
not empty, cache_controller.py:1240) runs as its condition says. A queue get never waits: an
empty queue raises Empty at once, as a timed-out get would.

A device event recorded after boundary k-1 completes at boundary k: its query() is False until
then. synchronize() completes it at once, as a blocking wait does. An event never recorded is
complete, as a CUDA event is. Its elapsed_time() is 0.0 ms: the oracle has no device clock, and
a copy's duration is the simulator's to price.
"""

# Thread targets the oracle refuses to start, each with its open issue (X28).
REFUSED = {"pp_prefetch_command_thread_func": "the PP prefetch ticket thread runs only with buffer_only host memory "
           "and pipeline parallelism, which is not modelled (inference-sim/inference-sim#2456)"}

# The modules whose names the runner replaces, and the names.
THREAD_MODULES = ("sglang.srt.managers.cache_controller", "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller")
DEVICE_MODULES = ("sglang.srt.managers.cache_controller", "sglang.srt.mem_cache.l2_transfer")


class Runner:
    """The threads started, the boundary count, and the stop-event grant of the body running."""

    def __init__(self):
        self.threads = []
        self.boundaries = 0
        self.running = None  # the StepThread whose body is running
        self.grant = 0  # how many more stop-event tests read clear for that body
        self.ran = []  # (boundary, target name) of every body run, for the trace

    def take_grant(self) -> bool:
        """Whether a stop-event test reads clear: outside a body always, inside one only for its
        granted tests."""
        if self.running is None:
            return True
        if self.grant > 0:
            self.grant -= 1
            return True
        return False

    def boundary(self) -> None:
        """Step boundary k, by RULE."""
        self.boundaries += 1
        for t in list(self.threads):
            if t not in self.threads:
                continue  # joined by a body that ran before it
            self.running, self.grant = t, 1
            try:
                t.target(*t.args, **t.kwargs)
            finally:
                self.running, self.grant = None, 0
            self.ran.append((self.boundaries, t.name))


RUNNER = Runner()


class StepThread:
    """threading.Thread whose start registers its target with the runner; no thread runs."""

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        self.target, self.args, self.kwargs = target, tuple(args), dict(kwargs or {})
        self.name = name or getattr(target, "__name__", "thread")
        self.daemon = daemon
        # threading.Thread.ident: None until started, then a number (the controller counts a worker
        # as started by it, cache_controller.py:603@c4af57ff).
        self.ident = None

    def start(self) -> None:
        if self.name in REFUSED:
            raise NotImplementedError(f"step_runner: {self.name} is not modelled: {REFUSED[self.name]}")
        if self in RUNNER.threads:
            raise RuntimeError("step_runner: thread started twice")
        self.ident = len(RUNNER.threads) + 1
        RUNNER.threads.append(self)

    def join(self, timeout=None) -> None:
        if self in RUNNER.threads:
            RUNNER.threads.remove(self)

    def is_alive(self) -> bool:
        return self in RUNNER.threads


class StepEvent:
    """threading.Event as a thread body's stop event, read as RULE says; outside a body it is the
    flag the controller sets and clears."""

    def __init__(self):
        self._flag = False

    def set(self) -> None:
        self._flag = True

    def clear(self) -> None:
        self._flag = False

    def is_set(self) -> bool:
        if self._flag:
            return True
        return not RUNNER.take_grant()

    def wait(self, timeout=None) -> bool:
        return self._flag


class StepQueue(queue.Queue):
    """queue.Queue whose get never waits (RULE)."""

    def get(self, block=True, timeout=None):
        return super().get(block=False)


class StepDeviceEvent:
    """A device event that completes at the first boundary after its record (RULE). It takes
    torch.cuda.Event's keyword arguments (enable_timing, blocking, interprocess), so the L2
    transfer's timing probe (mem_cache/l2_transfer.py:18-28@297e54fd) finds timing supported, as
    it does on CUDA."""

    def __init__(self, **kwargs):
        # An event never recorded is complete, as a CUDA event is (cudaEventQuery returns
        # cudaSuccess): LayerDoneCounter.update_producer asserts this of a fresh counter
        # (managers/cache_controller.py:92-96@297e54fd).
        self.recorded_at = None
        self.done = True

    def record(self, stream=None) -> None:
        self.recorded_at = RUNNER.boundaries
        self.done = False

    def query(self) -> bool:
        if self.done:
            return True
        if self.recorded_at is not None and RUNNER.boundaries > self.recorded_at:
            self.done = True
        return self.done

    def synchronize(self) -> None:
        self.done = True

    def wait(self, stream=None) -> None:
        return None

    def elapsed_time(self, end_event) -> float:
        """The milliseconds between this event and end_event: 0.0, by RULE."""
        return 0.0


class _Namespace(types.SimpleNamespace):
    """A module seen through a few replaced names; every other name is the module's own."""

    def __init__(self, base, **names):
        super().__init__(**names)
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)


def install() -> Runner:
    """Replace the thread, queue and device-event names in the HiCache modules. Call before any
    HiCache controller is built."""
    import importlib

    for name in THREAD_MODULES:
        mod = importlib.import_module(name)
        mod.threading = _Namespace(threading, Thread=StepThread, Event=StepEvent)
        mod.Queue = StepQueue
    for name in DEVICE_MODULES:
        mod = importlib.import_module(name)
        mod.device_module = _Namespace(mod.device_module, Event=StepDeviceEvent)
    return RUNNER
