# Original to blis-sglang.
"""Tier-2 scheduler runner (S13): SGLang v0.5.23's own Scheduler on CPU, with the state its
decisions leave behind recorded after every loop iteration.

It drives the same scheduler as oracle/run_scheduler.py (run_scheduler_process, then the event loop
ServerArgs selects, unmodified) and adds to each step of the trace:

- for each forward pass, the ids of the requests in the batch, in batch order, the rows the batch
  holds, the request that is still chunking, and the requests a mixed batch decodes;
- after the step (read when the next iteration starts to receive, so it is the state the loop left),
  the waiting queue in order, the running batch, the chunked request, the pool's free and
  evictable tokens, the free request rows, new_token_ratio, and per running request its output
  length and KV lengths.

It also records the scheduler's resolved settings the Go port is configured from ("config"), so the
port is compared on its decisions and not on SGLang's start-up resolution (T1, S08).

A scenario may add `server_args` (extra ServerArgs fields, each named in SCHED_SERVER_ARGS) and, per
request, `ignore_eos` (default true, as in oracle/run_scheduler.py). Nothing else differs from
run_scheduler.py: the replaced seams are the ones oracle/mocks.json lists.

Usage: python -I oracle/run_sched.py <sglang checkout> <scenario.json> [out.json]
Environment: as oracle/run_scheduler.py.
"""
import json
import multiprocessing
import os
import signal
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_scheduler  # noqa: E402

# ServerArgs fields a scheduler scenario may set beyond run_scheduler.server_args's.
SCHED_SERVER_ARGS = ("enable_mixed_chunk", "max_prefill_tokens", "schedule_conservativeness",
                     "radix_eviction_policy", "enable_priority_scheduling", "prefill_max_requests")

ORACLE_INPUTS = run_scheduler.ORACLE_INPUTS + ("oracle/run_sched.py",)


def sched_server_args(scenario: dict):
    from sglang.srt.server_args import ServerArgs

    extra = scenario.get("server_args", {})
    bad = sorted(set(extra) - set(SCHED_SERVER_ARGS))
    if bad:
        raise SystemExit(f"scenario server_args {bad} are not settable by oracle/run_sched.py")
    return ServerArgs(
        model_path=os.environ["ORACLE_MODEL"], load_format="dummy", skip_tokenizer_init=True, device="cpu",
        attention_backend="torch_native", sampling_backend="pytorch", disable_cuda_graph=True,
        random_seed=scenario["seed"], page_size=scenario["page_size"],
        max_total_tokens=scenario["max_total_tokens"], max_running_requests=scenario["max_running_requests"],
        chunked_prefill_size=scenario["chunked_prefill_size"], schedule_policy=scenario["schedule_policy"],
        disable_overlap_schedule=not scenario["overlap"], disable_radix_cache=scenario["disable_radix_cache"],
        **extra)


def event_loop_scheduler():
    """The Scheduler whose event loop is on the call stack."""
    import harness

    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name in harness.LOOPS:
            return frame.f_locals.get("self")
        frame = frame.f_back
    return None


def frame_local(name: str, var: str):
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == name:
            return frame.f_locals.get(var)
        frame = frame.f_back
    return None


def rid_or_none(r):
    return None if r is None else r.rid


def state_of(s) -> dict:
    """What the scheduler holds between two loop iterations."""
    rb = s.running_batch
    cache = s.tree_cache
    return {
        "waiting": [r.rid for r in s.waiting_queue],
        "running": [{"rid": r.rid, "output_len": len(r.output_ids), "kv_committed": r.kv.kv_committed_len,
                     "kv_allocated": r.kv.kv_allocated_len, "prefix_len": r.prefix_len} for r in rb.reqs],
        "batch_is_full": bool(rb.batch_is_full),
        "chunked": rid_or_none(s.chunked_req),
        "available_tokens": int(s.token_to_kv_pool_allocator.available_size()),
        "evictable": int(cache.evictable_size()),
        "protected": int(cache.protected_size()),
        "req_pool_available": int(s.req_to_token_pool.available_size()),
        "new_token_ratio": float(s.new_token_ratio_tracker.current),
        "forward_ct": int(s.forward_ct),
        "processed_tokens": int(s.processed_tokens_counter),
    }


def config_of(s) -> dict:
    """The settings the Go port is configured from, as the scheduler resolved them."""
    from sglang.srt.environ import envs
    from sglang.srt.managers import schedule_policy as sp

    from sglang.srt.runtime_context import get_parallel

    t = s.new_token_ratio_tracker
    return {
        "pp_max_micro_batch_size": int(get_parallel().pp_max_micro_batch_size),
        "pp_size": int(get_parallel().pp_size),
        "prefill_decode_interval": int(s.prefill_decode_interval),
        "page_size": int(s.page_size),
        "policy": s.policy.policy.value,
        "max_total_num_tokens": int(s.max_total_num_tokens),
        "max_prefill_tokens": int(s.max_prefill_tokens),
        "chunked_prefill_size": None if s.chunked_prefill_size is None else int(s.chunked_prefill_size),
        "max_running_requests": int(s.max_running_requests),
        "req_pool_size": int(s.req_to_token_pool.size),
        "max_context_len": int(s.req_to_token_pool.max_context_len),
        "is_mixed_chunk": bool(s.is_mixed_chunk),
        "radix_cache_disabled": bool(getattr(s.tree_cache, "disable", True)),
        "new_token_ratio": {"init": t.init, "min": t.min, "decay": t.decay},
        "retract_decode_steps": int(envs.SGLANG_RETRACT_DECODE_STEPS.get()),
        "clip_max_new_tokens": int(sp.CLIP_MAX_NEW_TOKENS),
        "in_batch_prefix_check_threshold": int(sp.IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD),
        "in_batch_prefix_deprioritize_threshold": int(sp.IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD),
        "enable_priority_scheduling": bool(s.enable_priority_scheduling),
        "max_req_len": int(s.max_req_len),
        "max_req_input_len": int(s.max_req_input_len),
        "need_sort": bool(getattr(s.token_to_kv_pool_allocator, "need_sort", False)),
        "eos_token_ids": sorted(int(t) for t in (s.model_config.hf_eos_token_id or [])),
        "vocab_size": int(s.model_config.vocab_size),
        "radix_eviction_policy": str(s.tree_cache.eviction_policy) if hasattr(s.tree_cache, "eviction_policy") else None,
        "allocator": type(s.token_to_kv_pool_allocator).__name__,
        "tree_cache": type(s.tree_cache).__name__,
        "disaggregation_mode": str(s.disaggregation_mode.value),
        "enable_overlap": bool(s.enable_overlap),
    }


def make_run_class():
    import harness

    class SchedRun(harness.Run):
        def __init__(self, scenario, sglang_root):
            super().__init__(scenario, sglang_root)
            self.requests = {r["rid"]: r for r in scenario["requests"]}
            self.config = None

        def begin_step(self):
            sched = event_loop_scheduler()
            if sched is not None:
                if self.step == -1:
                    self.config = config_of(sched)
                else:
                    self.steps[-1]["after"] = state_of(sched)
            reqs = super().begin_step()
            # harness.tokenized builds each arrival with ignore_eos set; a scenario request may say
            # otherwise.
            for req in reqs:
                spec = self.requests.get(getattr(req, "rid", None))
                if spec is not None:
                    req.sampling_params.ignore_eos = spec.get("ignore_eos", True)
            return reqs

        def record_forward(self, rec):
            batch = frame_local("run_batch", "batch")
            if batch is not None:
                rec["rids"] = [r.rid for r in batch.reqs]
                rec["req_pool_indices"] = [int(r.kv.req_pool_idx) for r in batch.reqs]
                rec["chunked"] = rid_or_none(getattr(batch, "chunked_req", None))
                rec["decoding"] = None if batch.decoding_reqs is None else [r.rid for r in batch.decoding_reqs]
            super().record_forward(rec)

    return SchedRun


def child(scenario: dict, sglang_root: str, out_path: str) -> None:
    """run_scheduler.child, with this module's run class, ServerArgs and tokenized request."""
    for name in [n for n in os.environ if n.startswith("SGLANG_")]:
        del os.environ[name]
    os.environ.update(run_scheduler.KEPT_SGLANG_ENV)
    import stubs

    stubs.install()
    import harness

    run = make_run_class()(scenario, sglang_root)
    harness.install(run)
    from sglang.srt.environ import envs
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.server_args import PortArgs

    args = sched_server_args(scenario)
    receive = {"max_recv_per_poll": envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get(),
               "scheduler_recv_interval": args.scheduler_recv_interval}
    if receive["max_recv_per_poll"] >= 0 or receive["scheduler_recv_interval"] > 1:
        raise SystemExit(f"the oracle counts a step per loop iteration and needs the default receive settings, not {receive}")
    pipe = run_scheduler.InitInfo()
    run_scheduler_process(args, PortArgs.init_new(args), gpu_id=0, tp_rank=0, pp_rank=0, dp_rank=None, pipe_writer=pipe)
    init = pipe.sent[0] if pipe.sent and isinstance(pipe.sent[0], dict) else {}
    result = {
        "steps": run.steps, "finished": run.finished, "shutdown_sent": run.shutdown_sent, "truncated": run.truncated,
        "pool": {"max_total_num_tokens": init.get("max_total_num_tokens"), "max_req_input_len": init.get("max_req_input_len")},
        "classes": run.classes, "receive": receive, "sglang_env": dict(sorted(run_scheduler.KEPT_SGLANG_ENV.items())),
        "resolved": {k: getattr(args, k) for k in sorted(scenario["record_server_args"])},
        "config": getattr(run, "config", None),
        "harness_problems": run.problems + harness.clock_problems(),
    }
    with open(out_path, "w") as f:
        json.dump(result, f)


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__.split("Usage:", 1)[1].split("\n\n", 1)[0].strip(), file=sys.stderr)
        return 2
    sglang_root, scenario_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) == 4 else None
    import traces

    if traces.refuses_write(out):
        print(traces.REFUSAL, file=sys.stderr)
        return 2
    sys.path.insert(0, os.path.join(os.path.abspath(sglang_root), "python"))
    with open(scenario_path) as f:
        scenario = json.load(f)
    prov = traces.provenance(2, sglang_root, scenario_path, scenario, ORACLE_INPUTS)
    quit_seen = []
    signal.signal(signal.SIGQUIT, lambda *a: quit_seen.append(True))
    with tempfile.TemporaryDirectory() as tmp:
        result_path = os.path.join(tmp, "result.json")
        proc = multiprocessing.get_context("fork").Process(target=child, args=(scenario, sglang_root, result_path))
        proc.start()
        proc.join(run_scheduler.JOIN_TIMEOUT_S)
        if proc.is_alive():
            proc.terminate()
            proc.join(10)
            print(f"the scheduler process did not finish in {run_scheduler.JOIN_TIMEOUT_S} s; terminated", file=sys.stderr)
            return 1
        if quit_seen or proc.exitcode != 0 or not os.path.exists(result_path):
            print(f"the scheduler process failed (exit {proc.exitcode}, SIGQUIT {bool(quit_seen)})", file=sys.stderr)
            return 1
        with open(result_path) as f:
            result = json.load(f)
    problems = run_scheduler.trace_problems(result, scenario)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    traces.write(out, {"provenance": prov, **result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
