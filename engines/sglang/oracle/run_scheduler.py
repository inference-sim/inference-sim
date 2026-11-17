# Original to blis-sglang.
"""Tier-2 oracle spike: run one scenario through SGLang v0.5.23's own Scheduler on CPU and record
what it decided at each step of its event loop.

The scheduler is built and run by SGLang's own entry point, run_scheduler_process
(python/sglang/srt/managers/scheduler.py:6310@297e54fd), in a child process as SGLang's launcher
runs it: it publishes the runtime context, constructs Scheduler(server_args, port_args) through
Scheduler.__init__, and runs the event loop ServerArgs selects (event_loop_normal, or
event_loop_overlap) until a ShutdownReq ends it. oracle/harness.py makes the replacements
oracle/mocks.json lists, in the child, before the scheduler module is imported. The trace records,
for each loop iteration: the requests that arrived, each forward pass the scheduler launched (its
mode, batch size, sequence, extend and prefix lengths, and the pool's free tokens), and each output
it sent.

Usage: python -I oracle/run_scheduler.py <sglang checkout> <scenario.json> [out.json]

Environment, each required: ORACLE_MODEL, a local directory with the scenario model's configuration
files (no weights); ORACLE_MODEL_REVISION, its revision; ORACLE_LOCKFILE, the `pip freeze` of the
environment. BLIS_SGLANG_ALLOW_GOLDEN_UPDATE=1 lets it write into a testdata/golden* directory.
Every SGLANG_* variable of the caller's environment is removed from the scheduler process, which
sets only those this runner lists (KEPT_SGLANG_ENV).
"""
import json
import multiprocessing
import os
import signal
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# The SGLANG_* variables the scheduler process runs with. SGL-1: the radix cache's Python tree core
# (SGLang's default is the Rust one, python/sglang/srt/environ.py:741@297e54fd).
KEPT_SGLANG_ENV = {"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python"}
JOIN_TIMEOUT_S = 600


def server_args(scenario: dict):
    """The scenario's ServerArgs. load_format and skip_tokenizer_init are the oracle's, listed in
    oracle/mocks.json; device, attention_backend, sampling_backend and disable_cuda_graph are the
    base platform's (mocks.json, "the base platform")."""
    from sglang.srt.server_args import ServerArgs

    # HiCache, for oracle/run_hicache.py's recorded scenarios: SGLang's own fields, by name.
    hicache = {}
    if "hicache" in scenario:
        h = scenario["hicache"]
        hicache = {"enable_hierarchical_cache": True, "hicache_ratio": h["ratio"], "hicache_io_backend": h["io_backend"],
                   "hicache_mem_layout": h["mem_layout"], "hicache_write_policy": h["write_policy"]}
    return ServerArgs(
        **hicache,
        model_path=os.environ["ORACLE_MODEL"],
        load_format="dummy",
        skip_tokenizer_init=True,
        device="cpu",
        attention_backend="torch_native",
        sampling_backend="pytorch",
        disable_cuda_graph=True,
        random_seed=scenario["seed"],
        page_size=scenario["page_size"],
        max_total_tokens=scenario["max_total_tokens"],
        max_running_requests=scenario["max_running_requests"],
        chunked_prefill_size=scenario["chunked_prefill_size"],
        schedule_policy=scenario["schedule_policy"],
        disable_overlap_schedule=not scenario["overlap"],
        disable_radix_cache=scenario["disable_radix_cache"],
    )


class InitInfo:
    """The pipe run_scheduler_process reports its start-up on (transport); what it sends is
    kept for the trace."""

    def __init__(self):
        self.sent = []

    def send(self, obj):
        self.sent.append(obj)


def child(scenario: dict, sglang_root: str, out_path: str, on_step=None, setup=None) -> None:
    """Run the scenario in this process. on_step is called at the start of every loop iteration,
    and setup after the harness's replacements and before the scheduler is built (both for
    oracle/run_hicache.py)."""
    for name in [n for n in os.environ if n.startswith("SGLANG_")]:
        del os.environ[name]
    os.environ.update(KEPT_SGLANG_ENV)
    import stubs

    stubs.install()
    import harness

    run = harness.Run(scenario, sglang_root, on_step=on_step)
    harness.install(run)
    if setup is not None:
        setup()
    from sglang.srt.environ import envs
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.server_args import PortArgs

    args = server_args(scenario)
    # A step is one loop iteration only while every receive drains its arrivals and none is skipped.
    receive = {"max_recv_per_poll": envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get(),
               "scheduler_recv_interval": args.scheduler_recv_interval}
    if receive["max_recv_per_poll"] >= 0 or receive["scheduler_recv_interval"] > 1:
        raise SystemExit(f"the oracle counts a step per loop iteration and needs the default receive settings, not {receive}")
    port_args = PortArgs.init_new(args)
    pipe = InitInfo()
    run_scheduler_process(args, port_args, gpu_id=0, tp_rank=0, pp_rank=0, dp_rank=None, pipe_writer=pipe)
    init = pipe.sent[0] if pipe.sent and isinstance(pipe.sent[0], dict) else {}
    result = {
        "steps": run.steps,
        "finished": run.finished,
        "shutdown_sent": run.shutdown_sent,
        "truncated": run.truncated,
        "pool": {"max_total_num_tokens": init.get("max_total_num_tokens"), "max_req_input_len": init.get("max_req_input_len")},
        "classes": run.classes,
        "receive": receive,
        "sglang_env": dict(sorted(KEPT_SGLANG_ENV.items())),
        "resolved": {k: getattr(args, k) for k in sorted(scenario["record_server_args"])},
        "harness_problems": run.problems + harness.clock_problems(),
    }
    with open(out_path, "w") as f:
        json.dump(result, f)


# The repository files a tier-2 run reads, besides its scenario (recorded in oracle_inputs).
ORACLE_INPUTS = ("oracle/run_scheduler.py", "oracle/harness.py", "oracle/stubs.py", "oracle/traces.py",
                 "oracle/mocks.json", "oracle/harness.json", "oracle/requirements.lock")


def trace_problems(result: dict, scenario: dict) -> list:
    """Why a run's result is not a trace of SGLang's own scheduler running the scenario."""
    expected = {r["rid"] for r in scenario["requests"]}
    problems = list(result.get("harness_problems", []))
    if result["truncated"]:
        problems.append(f"the scenario did not finish within max_steps {scenario['max_steps']}")
    if set(result["finished"]) != expected:
        problems.append(f"finished {sorted(result['finished'])}, expected {sorted(expected)}")
    if not result["shutdown_sent"]:
        problems.append("the event loop returned before the harness sent its ShutdownReq")
    # The loop that launched every forward pass is SGLang's own, the one ServerArgs selects.
    want_loop = "event_loop_overlap" if scenario["overlap"] else "event_loop_normal"
    forwards = 0
    for step in result["steps"]:
        for fwd in step["forwards"]:
            forwards += 1
            loop = fwd["loop"]
            if loop.get("function") != want_loop or loop.get("upstream_code") is not True or \
                    loop.get("file") != "python/sglang/srt/managers/scheduler.py":
                problems.append(f"step {step['step']}: a forward pass ran from {loop}, not from SGLang's "
                                f"Scheduler.{want_loop} at the checkout")
    if not forwards:
        problems.append("no forward pass ran")
    if result.get("classes", {}).get("tree_core") != "UnifiedTreeCore":
        problems.append(f"the tree core is {result.get('classes', {}).get('tree_core')!r}, not the Python UnifiedTreeCore (SGL-1)")
    return problems


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__.split("Usage:", 1)[1].split("\n\n", 1)[0].strip(), file=sys.stderr)
        return 2
    sglang_root, scenario_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) == 4 else None
    sys.path.insert(0, HERE)
    import traces

    if traces.refuses_write(out):
        print(traces.REFUSAL, file=sys.stderr)
        return 2
    sys.path.insert(0, os.path.join(os.path.abspath(sglang_root), "python"))
    with open(scenario_path) as f:
        scenario = json.load(f)
    prov = traces.provenance(2, sglang_root, scenario_path, scenario, ORACLE_INPUTS)

    # run_scheduler_process sends SIGQUIT to its parent when the scheduler raises; the parent is
    # this process, which reports it.
    quit_seen = []
    signal.signal(signal.SIGQUIT, lambda *a: quit_seen.append(True))
    with tempfile.TemporaryDirectory() as tmp:
        result_path = os.path.join(tmp, "result.json")
        proc = multiprocessing.get_context("fork").Process(target=child, args=(scenario, sglang_root, result_path))
        proc.start()
        proc.join(JOIN_TIMEOUT_S)
        if proc.is_alive():
            proc.terminate()
            proc.join(10)
            print(f"the scheduler process did not finish in {JOIN_TIMEOUT_S} s; terminated", file=sys.stderr)
            return 1
        if quit_seen or proc.exitcode != 0 or not os.path.exists(result_path):
            print(f"the scheduler process failed (exit {proc.exitcode}, SIGQUIT {bool(quit_seen)})", file=sys.stderr)
            return 1
        with open(result_path) as f:
            result = json.load(f)
    problems = trace_problems(result, scenario)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1  # a failing run writes no trace
    traces.write(out, {"provenance": prov, **result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
