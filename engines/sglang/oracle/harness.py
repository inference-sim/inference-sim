# Original to blis-sglang.
"""The oracle harness: SGLang's own Scheduler, built by SGLang's own run_scheduler_process on CPU,
with only its environment replaced.

Everything the scheduler decides runs as released at the pinned tag: ServerArgs resolution,
Scheduler.__init__ (pool sizing, tree-cache selection, policy set-up) and the event loop itself,
event_loop_normal or event_loop_overlap, unmodified. The replacements, each listed in
oracle/mocks.json:

  gpu        ModelRunner.load_model builds no weights; ModelRunner.forward returns empty logits and
             records the batch it was asked to run; ModelRunner.sample returns the scenario's
             token; the attention backends and CUDA graphs are not built.
  platform   get_available_gpu_memory reports the scenario's device memory, so the KV pool is
             sized from the scenario and not from the host's free memory.
  transport  the scheduler's ZMQ sockets (ipc_channels.get_zmq_socket): the socket it reads from
             the tokenizer yields the scenario's arrivals for each loop iteration and, once every
             request has finished, a ShutdownReq; the sockets it writes to record its outputs.
  clock      the scheduler module's time reads step-derived virtual time.

The loop is stopped by the scheduler's own ShutdownReq handler (Scheduler.handle_shutdown, which
sets gracefully_exit), reached through the replaced socket: the harness never writes
gracefully_exit, and copies no loop body (FID-8).
"""
import collections
import json
import os
import sys
import time as real_time
import types


class Run:
    """One scenario's run: what the replaced sockets deliver, and what the scheduler did."""

    def __init__(self, scenario: dict):
        self.scenario = scenario
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
        self.steps.append({"step": self.step, "arrivals": [], "forwards": [], "outputs": []})
        if self.step >= self.scenario["max_steps"] and not self.shutdown_sent:
            self.truncated = True
            return [self.shutdown()]
        reqs = [tokenized(r, self.scenario) for r in self.arrivals.get(self.step, [])]
        self.steps[-1]["arrivals"] = [r["rid"] for r in self.arrivals.get(self.step, [])]
        if not reqs and not self.shutdown_sent and self.done():
            return [self.shutdown()]
        return reqs

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

    def _next(self):
        import zmq

        if self.pending is None:
            self.pending = collections.deque(self.run.begin_step())
        if self.pending:
            return self.pending.popleft()
        self.pending = None
        raise zmq.Again()

    def recv_pyobj(self, flags=0):
        return self._next()

    def recv(self, flags=0, copy=True, track=False):
        from sglang.srt.managers.io_struct import msgpack_encode

        return msgpack_encode(self._next())

    def close(self, *a, **k):
        pass


class SilentSocket:
    """The RPC socket: nothing arrives on it."""

    def recv_pyobj(self, flags=0):
        import zmq

        raise zmq.Again()

    recv = recv_pyobj

    def close(self, *a, **k):
        pass


class OutputSocket:
    """A socket the scheduler writes to: each message is recorded."""

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


def install(run: Run) -> None:
    """Make every replacement, before the scheduler module is imported."""
    import torch
    import zmq

    gib = run.scenario["device_memory_gib"]

    # platform: the device's free memory is the scenario's.
    import sglang.srt.utils as srt_utils
    import sglang.srt.utils.common as utils_common

    def available_memory(device, gpu_id, distributed=False, empty_cache=True, cpu_group=None):
        return float(gib)

    utils_common.get_available_gpu_memory = available_memory
    srt_utils.get_available_gpu_memory = available_memory

    # transport: the scheduler's sockets.
    import sglang.srt.managers.scheduler_components.ipc_channels as ipc_channels

    def fake_socket(context, socket_type, endpoint, bind=False, **kw):
        if socket_type == zmq.PULL:
            return TokenizerSocket(run)
        if socket_type == zmq.DEALER:
            return SilentSocket()
        return OutputSocket(run)

    ipc_channels.get_zmq_socket = fake_socket

    # gpu: no weights, no kernels; the batch each forward is asked to run is recorded.
    import sglang.srt.model_executor.model_runner as model_runner_module
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.model_runner_components import attention_backend_setup

    runner = model_runner_module.ModelRunner
    token = run.scenario["sampled_token"]

    class NoModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return None

    # The ModelRunner each replacement is called on is named `mr`, not `self`, so check_mocks.py
    # reads every attribute it sets as a site of the replacement that sets it.
    def fake_load_model(mr):
        mr.model = NoModel()
        mr.dtype = mr.model_config.dtype
        mr.sliding_window_size = model_runner_module.resolve_sliding_window_size(mr.model, mr.model_config)
        mr.prefill_aware_swa = False
        mr.weight_load_mem_usage = 0
        mr.load_config = None
        mr.loader = types.SimpleNamespace(preloaded_weights_bytes=0)

    def fake_forward(mr, forward_batch, *args, **kwargs):
        run.record_forward(batch_record(forward_batch, mr))
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

    # clock: the scheduler's time.
    import sglang.srt.managers.scheduler as scheduler_module

    scheduler_module.time = VirtualTime(run, run.scenario["step_seconds"])


def tolist(t):
    if t is None:
        return None
    return t.tolist() if hasattr(t, "tolist") else list(t)


LOOPS = ("event_loop_normal", "event_loop_overlap")


def running_loop() -> dict:
    """The scheduler event loop this forward pass was launched from: the function name, and the
    file and first line of the code object on the call stack, so a trace shows the loop that ran
    is SGLang's own code at the checkout, not a copy."""
    frame = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        if code.co_name in LOOPS:
            import inspect

            from sglang.srt.managers.scheduler import Scheduler

            path = code.co_filename.replace(os.sep, "/")
            cut = path.rfind("/python/sglang/")
            return {"function": code.co_name, "file": path[cut + 1:] if cut >= 0 else path,
                    "line": code.co_firstlineno,
                    # The frame runs the very code object of SGLang's Scheduler method.
                    "upstream_code": code is inspect.unwrap(getattr(Scheduler, code.co_name)).__code__}
        frame = frame.f_back
    return {"function": None}


def batch_record(fb, model_runner) -> dict:
    """What one forward pass was asked to run, the pool's free tokens once it was scheduled, and the
    loop it was launched from."""
    rec = {
        "loop": running_loop(),
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
