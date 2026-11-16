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

Environment: ORACLE_MODEL, a local directory with the scenario model's configuration files (no
weights); ORACLE_MODEL_REVISION, its revision; ORACLE_LOCKFILE, the `pip freeze` of the
environment; BLIS_SGLANG_ALLOW_GOLDEN_UPDATE=1 to write into a testdata/golden* directory.
"""
import hashlib
import json
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def server_args(scenario: dict):
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(
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


def child(scenario: dict, out_path: str) -> None:
    # SGL-1: the oracle runs the radix cache's Python tree core (SGLang's default is the Rust one,
    # python/sglang/srt/environ.py:741@297e54fd); set before SGLang reads its environment.
    os.environ["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = "python"
    import harness

    run = harness.Run(scenario)
    harness.install(run)
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.server_args import PortArgs

    args = server_args(scenario)
    port_args = PortArgs.init_new(args)
    pipe = InitInfo()
    run_scheduler_process(args, port_args, gpu_id=0, tp_rank=0, pp_rank=0, dp_rank=None, pipe_writer=pipe)
    result = {
        "steps": run.steps,
        "finished": run.finished,
        "shutdown_sent": run.shutdown_sent,
        "truncated": run.truncated,
        "init_sent": len(pipe.sent),
        "resolved": {k: getattr(args, k) for k in sorted(scenario["record_server_args"])},
    }
    with open(out_path, "w") as f:
        json.dump(result, f)


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(root: str, *args) -> str:
    try:
        return subprocess.check_output(["git", "-C", root, *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance(sglang_root: str, scenario_path: str, scenario: dict) -> dict:
    with open(os.path.join(HERE, "mocks.json")) as f:
        mocks = json.load(f)
    lockfile = os.environ.get("ORACLE_LOCKFILE")
    dirty = git(ROOT, "status", "--porcelain", "--untracked-files=no")
    head = git(ROOT, "rev-parse", "HEAD")
    inputs = {}
    for rel in ("oracle/run_scheduler.py", "oracle/harness.py", "oracle/stubs.py", "oracle/mocks.json",
                "oracle/requirements.lock"):
        if os.path.exists(os.path.join(ROOT, rel)):
            inputs[rel] = sha256(os.path.join(ROOT, rel))
    if os.path.abspath(scenario_path).startswith(ROOT + os.sep):
        inputs[os.path.relpath(os.path.abspath(scenario_path), ROOT)] = sha256(scenario_path)
    return {
        "tier": 2,
        "sglang_commit": git(sglang_root, "rev-parse", "HEAD"),
        "scenario_sha256": sha256(scenario_path),
        "mock_version": mocks["version"],
        "model": scenario["model"],
        "model_revision": os.environ.get("ORACLE_MODEL_REVISION", "unpinned"),
        "python": sys.version.split()[0],
        "platform": platform.system().lower() + "-" + platform.machine(),
        "lockfile_sha256": sha256(lockfile) if lockfile else "unrecorded",
        "requirements_lock_sha256": sha256(os.path.join(HERE, "requirements.lock"))
        if os.path.exists(os.path.join(HERE, "requirements.lock")) else "unknown",
        "blis_sglang_commit": head + ("+dirty" if dirty and head != "unknown" else ""),
        "oracle_inputs": inputs,
        "seed": scenario["seed"],
    }


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__.split("Usage:", 1)[1].split("\n\n", 1)[0].strip(), file=sys.stderr)
        return 2
    sglang_root, scenario_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) == 4 else None
    if out and "testdata/golden" in os.path.abspath(out).replace(os.sep, "/") and \
            os.environ.get("BLIS_SGLANG_ALLOW_GOLDEN_UPDATE") != "1":
        print("refusing to write a golden trace: set BLIS_SGLANG_ALLOW_GOLDEN_UPDATE=1, and add a "
              "golden-reviewed: <trace>: <reason> trailer for each trace the commit changes", file=sys.stderr)
        return 2
    sys.path.insert(0, os.path.join(os.path.abspath(sglang_root), "python"))
    sys.path.insert(0, HERE)
    with open(scenario_path) as f:
        scenario = json.load(f)

    # run_scheduler_process sends SIGQUIT to its parent when the scheduler raises; the parent is
    # this process, which reports it.
    quit_seen = []
    signal.signal(signal.SIGQUIT, lambda *a: quit_seen.append(True))
    with tempfile.TemporaryDirectory() as tmp:
        result_path = os.path.join(tmp, "result.json")
        proc = multiprocessing.get_context("fork").Process(target=child, args=(scenario, result_path))
        proc.start()
        proc.join()
        if quit_seen or proc.exitcode != 0 or not os.path.exists(result_path):
            print(f"the scheduler process failed (exit {proc.exitcode}, SIGQUIT {bool(quit_seen)})", file=sys.stderr)
            return 1
        with open(result_path) as f:
            result = json.load(f)
    expected = {r["rid"] for r in scenario["requests"]}
    problems = []
    if result["truncated"]:
        problems.append(f"the scenario did not finish within max_steps {scenario['max_steps']}")
    if set(result["finished"]) != expected:
        problems.append(f"finished {sorted(result['finished'])}, expected {sorted(expected)}")
    if not result["shutdown_sent"]:
        problems.append("the event loop returned before the harness sent its ShutdownReq")
    # The loop that launched every forward pass is SGLang's own, the one ServerArgs selects.
    want_loop = "event_loop_overlap" if scenario["overlap"] else "event_loop_normal"
    for step in result["steps"]:
        for fwd in step["forwards"]:
            loop = fwd["loop"]
            if loop.get("function") != want_loop or not loop.get("upstream_code") or \
                    loop.get("file") != "python/sglang/srt/managers/scheduler.py":
                problems.append(f"step {step['step']}: a forward pass ran from {loop}, not from SGLang's "
                                f"Scheduler.{want_loop}")
    trace = {"provenance": provenance(sglang_root, scenario_path, scenario), **result}
    text = json.dumps(trace, indent=1, sort_keys=True) + "\n"
    if out:
        with open(out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
