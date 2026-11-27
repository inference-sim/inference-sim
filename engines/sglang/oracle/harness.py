# Original to blis-sglang.
"""The oracle harness: SGLang's own Scheduler, built by SGLang's own run_scheduler_process on CPU,
with only its environment replaced.

Everything the scheduler decides runs as released at the pinned tag: ServerArgs resolution,
Scheduler.__init__ (pool sizing, tree-cache selection, policy set-up) and the event loop itself,
event_loop_normal or event_loop_overlap, unmodified. The replacements, each listed in
oracle/mocks.json with its sites:

  gpu        ModelRunner.load_model builds no weights (and refuses a model class whose sliding
             window or prefill-aware SWA the empty model could not report); ModelRunner.forward
             returns empty logits and records the batch it was asked to run; ModelRunner.sample
             returns the scenario's token; the attention backends and CUDA graphs are not built.
  platform   get_available_gpu_memory and get_device_memory_capacity report the scenario's device
             memory, so ServerArgs resolution and the KV pool are sized from the scenario and not
             from the host's memory. The run resolves on
             SGLang's base platform; restoring the CUDA platform's values is not done yet.
  outside    the tokenizer: the socket the scheduler reads from it yields the scenario's arrivals
  service    for each loop iteration and, once every request has finished, a ShutdownReq.
  transport  the RPC socket yields nothing; the sockets the scheduler writes to record its outputs.
  clock      every loaded decision module, and SGLang's observability modules, read step-derived
             virtual time; the runner fails if one still binds the real clock after the run.

The loop is stopped by the scheduler's own ShutdownReq handler (Scheduler.handle_shutdown, which
sets gracefully_exit), reached through the replaced socket: the harness never writes
gracefully_exit, and copies no loop body (FID-8).
"""
import ast
import collections
import json
import os
import sys
import time as real_time
import types

HERE = os.path.dirname(os.path.abspath(__file__))

# The real clock functions a module may have bound by name (from time import monotonic).
CLOCK_FUNCTIONS = ("monotonic", "perf_counter", "time", "monotonic_ns", "perf_counter_ns", "time_ns")
# Modules outside the decision prefixes whose clock reads reach recorded values.
CLOCKED_EXTRA = ("sglang.srt.observability.",)


class Run:
    """One scenario's run: what the replaced sockets deliver, and what the scheduler did."""

    def __init__(self, scenario: dict, sglang_root: str, on_step=None):
        self.scenario = scenario
        # Called at the start of every loop iteration, before its arrivals are delivered: the
        # step boundary of oracle/step_runner.py when a HiCache scenario runs (run_hicache.py).
        self.on_step = on_step
        self.sglang_root = os.path.realpath(sglang_root)
        self.problems: list[str] = []
        self.classes: dict = {}
        self.clock = None
        self.step = -1
        self.arrivals = collections.defaultdict(list)
        for r in scenario["requests"]:
            self.arrivals[r["arrival_step"]].append(r)
        self.expected = {r["rid"] for r in scenario["requests"]}
        self.finished: dict[str, dict] = {}
        self.steps: list[dict] = []
        self.shutdown_sent = False
        self.truncated = False

    def begin_step(self) -> list:
        """The first receive of a loop iteration: start the iteration's record, and return what
        arrives in it."""
        self.step += 1
        if self.on_step is not None:
            self.on_step()
        if self.step == 0 and self.clock is not None:
            # Scheduler.__init__ has imported what it imports: replace their clocks too.
            install_clock(self.clock)
        self.steps.append({"step": self.step, "arrivals": [], "forwards": [], "outputs": []})
        if self.step >= self.scenario["max_steps"] and not self.shutdown_sent:
            self.truncated = True
            return [self.shutdown()]
        reqs = [tokenized(r, self.scenario) for r in self.arrivals.get(self.step, [])]
        self.steps[-1]["arrivals"] = [r["rid"] for r in self.arrivals.get(self.step, [])]
        if not reqs and not self.shutdown_sent and self.done():
            return [self.shutdown()]
        return reqs

    def use_clock(self, clock) -> None:
        self.clock = clock

    def note_classes(self, classes: dict) -> None:
        if not self.classes:
            self.classes = classes

    def done(self) -> bool:
        return self.step > max(self.arrivals, default=0) and set(self.finished) >= self.expected

    def shutdown(self):
        from sglang.srt.managers.io_struct import ShutdownReq

        self.shutdown_sent = True
        self.steps[-1]["shutdown"] = True
        return ShutdownReq()

    def record_forward(self, rec: dict) -> None:
        self.steps[-1]["forwards"].append(rec)

    def record_output(self, obj) -> None:
        rec = {"type": type(obj).__name__}
        rids = getattr(obj, "rids", None)
        if rids is not None:
            rec["rids"] = list(rids)
            reasons = getattr(obj, "finished_reasons", None) or [None] * len(rids)
            for rid, reason in zip(rids, reasons):
                if reason is not None:
                    self.finished[rid] = {"step": self.step, "reason": jsonable(reason)}
            rec["finished"] = [rid for rid, reason in zip(rids, reasons) if reason is not None]
            out = getattr(obj, "output_ids", None) or getattr(obj, "decode_ids", None)
            if out is not None:
                rec["output_lens"] = [len(x) if hasattr(x, "__len__") else 1 for x in out]
            else:
                # SGLang main sends the rows' ids in one array and their row boundaries.
                offsets = getattr(obj, "output_ids_offsets", None)
                if offsets is not None:
                    rec["output_lens"] = [b - a for a, b in zip(offsets, offsets[1:])]
        if self.steps:
            self.steps[-1]["outputs"].append(rec)


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "to_json"):
        return jsonable(x.to_json())
    return repr(x)


def tokenized(r: dict, scenario: dict):
    """A scenario request as the tokenizer would send it (skip_tokenizer_init: ids, no text)."""
    from array import array

    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.sampling.sampling_params import SamplingParams

    params = SamplingParams(max_new_tokens=r["max_new_tokens"], ignore_eos=True, temperature=0.0)
    params.normalize(None)
    params.verify(scenario["vocab_size"])
    return TokenizedGenerateReqInput(
        rid=r["rid"], input_text="", input_ids=array("q", r["prompt"]), input_embeds=None,
        mm_inputs=None, token_type_ids=None, sampling_params=params, return_logprob=False,
        logprob_start_len=-1, top_logprobs_num=0, token_ids_logprob=None, stream=False,
    )


class TokenizerSocket:
    """The socket the scheduler reads from the tokenizer. Its first receive in a loop iteration
    starts the iteration; it then yields that iteration's arrivals, and raises zmq.Again when
    they are drained, as an empty non-blocking socket does."""

    def __init__(self, run: Run):
        self.run = run
        self.pending = None
        # Set by the RPC socket, which the scheduler polls after this one in every iteration.
        self.iteration_ended = False

    def _next(self):
        if self.iteration_ended and self.pending:
            # The receive stopped before draining this iteration's arrivals (max_recv_per_poll):
            # steps would no longer be loop iterations.
            self.run.problems.append(f"step {self.run.step}: {len(self.pending)} arrivals left undelivered "
                                     "when the iteration ended")
            self.pending = None
        self.iteration_ended = False
        if self.pending is None:
            self.pending = collections.deque(self.run.begin_step())
        if self.pending:
            return self.pending.popleft()
        self.pending = None
        import zmq

        raise zmq.Again()

    def recv_pyobj(self, flags=0):
        return self._next()

    def recv(self, flags=0, copy=True, track=False):
        from sglang.srt.managers.io_struct import msgpack_encode

        return msgpack_encode(self._next())

    def close(self, *a, **k):
        pass


class SilentSocket:
    """The RPC socket: nothing arrives on it. Its receive ends an iteration's receives."""

    def __init__(self, tokenizer: TokenizerSocket):
        self.tokenizer = tokenizer

    def recv_pyobj(self, flags=0):
        self.tokenizer.iteration_ended = True
        import zmq

        raise zmq.Again()

    recv = recv_pyobj

    def close(self, *a, **k):
        pass


class OutputSocket:
    """A socket the scheduler writes to: each message is recorded."""

    # zmq.Socket.closed: SchedulerKvEventsPublisher.emit_kv_metrics reads it before it sends.
    closed = False

    def __init__(self, run: Run):
        self.run = run

    def send_pyobj(self, obj, flags=0, protocol=None):
        self.run.record_output(obj)

    def send(self, data, flags=0, copy=True, track=False):
        from sglang.srt.managers.io_struct import msgpack_decode

        self.run.record_output(msgpack_decode(data))

    def close(self, *a, **k):
        pass


class VirtualTime:
    """The time module as the scheduler sees it: each loop iteration is step_seconds of virtual
    time; every other attribute is the real module's."""

    def __init__(self, run: Run, step_seconds: float):
        self._run, self._dt = run, step_seconds

    def _now(self) -> float:
        return 1_000_000.0 + max(self._run.step, 0) * self._dt

    def monotonic(self):
        return self._now()

    perf_counter = time = monotonic

    def monotonic_ns(self):
        return int(self._now() * 1e9)

    perf_counter_ns = time_ns = monotonic_ns

    def __getattr__(self, name):
        return getattr(real_time, name)


def clocked_modules() -> list:
    """The loaded modules whose clock is replaced: those under a decision prefix of
    oracle/harness.json, and SGLang's observability modules."""
    with open(os.path.join(HERE, "harness.json")) as f:
        prefixes = tuple(json.load(f)["decision_prefixes"]) + CLOCKED_EXTRA
    return sorted((name, mod) for name, mod in list(sys.modules.items())
                  if mod is not None and (name + ".").startswith(prefixes))


def real_clock_bindings(mod) -> list:
    """The names under which a module holds the real time module or one of its clock functions."""
    out = []
    for attr, value in list(vars(mod).items()):
        if value is real_time or any(value is getattr(real_time, f) for f in CLOCK_FUNCTIONS):
            out.append(attr)
    return out


def install_clock(clock: "VirtualTime") -> None:
    """Bind the virtual clock in every clocked module, in place of the real one."""
    for _, mod in clocked_modules():
        for attr in real_clock_bindings(mod):
            value = getattr(mod, attr)
            setattr(mod, attr, clock if value is real_time else getattr(clock, value.__name__))


def clock_problems() -> list:
    return [f"{name} still reads the real clock through {', '.join(attrs)}"
            for name, mod in clocked_modules() if (attrs := real_clock_bindings(mod))]


def install(run: Run) -> None:
    """Make every replacement, before the scheduler module is imported."""
    import torch
    import zmq

    gib = run.scenario["device_memory_gib"]

    # platform: the device's free memory is the scenario's.
    import sglang.srt.utils as srt_utils
    import sglang.srt.utils.device as utils_device

    def available_memory(device, gpu_id, distributed=False, empty_cache=True, cpu_group=None):
        return float(gib)

    def memory_capacity(device=None):
        return float(gib) * 1024  # MiB, as get_device_memory_capacity reports it

    utils_device.get_available_gpu_memory = available_memory
    srt_utils.get_available_gpu_memory = available_memory
    utils_device.get_device_memory_capacity = memory_capacity
    srt_utils.get_device_memory_capacity = memory_capacity

    # transport: the scheduler's sockets.
    import sglang.srt.managers.scheduler_components.ipc_channels as ipc_channels

    sockets = {}

    def fake_socket(context, socket_type, endpoint, bind=False, **kw):
        if socket_type == zmq.PULL:
            sockets["tokenizer"] = TokenizerSocket(run)
            return sockets["tokenizer"]
        if socket_type == zmq.DEALER:
            if "tokenizer" not in sockets:
                raise RuntimeError("the RPC socket was made before the tokenizer socket")
            return SilentSocket(sockets["tokenizer"])
        if socket_type == zmq.PUSH:
            return OutputSocket(run)
        raise RuntimeError(f"an unexpected socket type {socket_type} for {endpoint}")

    ipc_channels.get_zmq_socket = fake_socket

    # gpu: no weights, no kernels; the batch each forward is asked to run is recorded.
    import sglang.srt.model_executor.model_runner as model_runner_module
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.model_runner_components import attention_backend_setup
    from sglang.srt.model_loader.utils import get_model_architecture

    runner = model_runner_module.ModelRunner
    token = run.scenario["sampled_token"]

    class NoModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return None

    # The ModelRunner each replacement is called on is named `mr`, not `self`, so check_mocks.py
    # reads every attribute it sets as a site of the replacement that sets it.
    def fake_load_model(mr):
        # The empty model reports no sliding window and no prefill-aware SWA; a model class that
        # defines either would make SGLang choose pools from values the empty model cannot give.
        model_class, arch = get_model_architecture(mr.model_config)
        own = [m for m in ("get_attention_sliding_window_size", "is_prefill_aware_swa") if hasattr(model_class, m)]
        if own:
            raise RuntimeError(f"the oracle cannot run {arch}: its model class defines {', '.join(own)}, which the "
                               "GPU replacement's empty model does not reproduce")
        mr.model = NoModel()
        mr.dtype = mr.model_config.dtype
        mr.sliding_window_size = model_runner_module.resolve_sliding_window_size(mr.model, mr.model_config)
        mr.prefill_aware_swa = False
        mr.weight_load_mem_usage = 0
        mr.load_config = None
        mr.loader = types.SimpleNamespace(preloaded_weights_bytes=0)

    def fake_forward(mr, forward_batch, *args, **kwargs):
        run.record_forward(batch_record(forward_batch, mr, run))
        logits = LogitsProcessorOutput(
            next_token_logits=torch.empty((forward_batch.batch_size, mr.model_config.vocab_size)))
        return model_runner_module.ModelRunnerOutput(logits_output=logits, can_run_graph=False)

    def fake_sample(mr, logits_output, forward_batch):
        return torch.full((logits_output.next_token_logits.shape[0],), token, dtype=torch.int64)

    def no_attention_backends(mr):
        # The backend names are SGLang's own resolution; only the backend objects, which build
        # GPU kernels, are not made.
        resolved = attention_backend_setup.resolve_attention_backend_strs(model_runner=mr)
        mr.prefill_attention_backend_str = resolved.prefill
        mr.decode_attention_backend_str = resolved.decode
        mr.attn_backend = None
        mr.decode_attn_backend = None
        mr.decode_attn_backend_group = None

    def no_cuda_graphs(mr, capture_decode_cuda_graph=True):
        mr.graph_mem_usage = 0
        mr.cuda_graph_runner = None
        mr.eager_runner = None
        mr.prefill_cuda_graph_runner = None
        mr.decode_cuda_graph_runner = None

    runner.load_model = fake_load_model
    runner.forward = fake_forward
    runner.sample = fake_sample
    runner.init_attention_backends = no_attention_backends
    runner.init_cuda_graphs = no_cuda_graphs

    # clock: every loaded decision module's, now and again at the first step.
    import sglang.srt.managers.scheduler  # noqa: F401  (loads the scheduler and what it imports)

    run.use_clock(VirtualTime(run, run.scenario["step_seconds"]))
    install_clock(run.clock)


def tolist(t):
    if t is None:
        return None
    return t.tolist() if hasattr(t, "tolist") else list(t)


LOOPS = ("event_loop_normal", "event_loop_overlap")


_LOOP_LINES: dict = {}


def loop_lines(path: str) -> dict:
    """{name: first line} of each Scheduler event loop in the file, as the code object reports it:
    the line of its first decorator, or of its def."""
    if path not in _LOOP_LINES:
        with open(path) as f:
            tree = ast.parse(f.read(), path)
        lines = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Scheduler":
                for fn in node.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name in LOOPS:
                        lines[fn.name] = min([d.lineno for d in fn.decorator_list] + [fn.lineno])
        _LOOP_LINES[path] = lines
    return _LOOP_LINES[path]


def running_loop(run: Run) -> dict:
    """The scheduler event loop this forward pass was launched from, checked against the checkout:
    its code object's file must be the checkout's managers/scheduler.py, and its first line that
    of the method's definition there, so a copy defined anywhere else does not pass. The
    scheduler's tree cache, tree core and allocator classes are recorded from the loop's self."""
    frame = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        if code.co_name in LOOPS:
            path = os.path.realpath(code.co_filename)
            want = os.path.join(run.sglang_root, "python", "sglang", "srt", "managers", "scheduler.py")
            rec = {"function": code.co_name, "file": os.path.relpath(path, run.sglang_root).replace(os.sep, "/"),
                   "line": code.co_firstlineno,
                   "upstream_code": path == want and loop_lines(want).get(code.co_name) == code.co_firstlineno}
            sched = frame.f_locals.get("self")
            if sched is not None:
                cache = getattr(sched, "tree_cache", None)
                core = getattr(cache, "tree_core", None)
                run.note_classes({
                    "tree_cache": type(cache).__name__,
                    "tree_core": type(core).__name__ if core is not None else None,
                    "allocator": type(getattr(sched, "token_to_kv_pool_allocator", None)).__name__,
                })
            return rec
        frame = frame.f_back
    return {"function": None}


def batch_record(fb, model_runner, run: Run) -> dict:
    """What one forward pass was asked to run, the pool's free tokens once it was scheduled, and the
    loop it was launched from."""
    rec = {
        "loop": running_loop(run),
        "mode": fb.forward_mode.name,
        "batch_size": int(fb.batch_size),
        "seq_lens": tolist(fb.seq_lens),
    }
    if fb.forward_mode.is_extend():
        rec["extend_seq_lens"] = tolist(fb.extend_seq_lens)
        rec["extend_prefix_lens"] = tolist(fb.extend_prefix_lens)
    alloc = getattr(model_runner, "token_to_kv_pool_allocator", None)
    if alloc is not None:
        rec["available_tokens"] = int(alloc.available_size())
    return rec
