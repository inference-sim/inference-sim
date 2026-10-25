# Original to blis-vllm.
"""Tier-3 oracle: vLLM's EngineCore batch queue driven through a scenario, on CPU.

Each engine step is vLLM's own EngineCore.step_with_batch_queue
(vllm/v1/engine/core.py:698-810@5df63a72), called unmodified on an EngineCore that holds vLLM's
real scheduler: the AsyncScheduler when the scenario turns async scheduling on, the
Scheduler otherwise, built by vLLM's own CPU test helper (tests/v1/core/utils.py
create_scheduler) with pipeline_parallel_size from the scenario. The batch queue's size is
vLLM's own VllmConfig.max_concurrent_batches (vllm/config/vllm.py:577-588@5df63a72) for the
scenario's async, PP and model-runner settings.

What is replaced, all listed in oracle/mocks.json:

- the executor (the GPU): execute_model and sample_tokens return futures that resolve to
  the shared GPU mock's ModelRunnerOutput when the batch queue pops them, so batches complete
  in the order they were queued, as they do on one device;
- EngineCore's construction: the model, the KV-cache profiling and the worker processes
  (EngineCore.__init__, core.py:118-259) need a GPU, so the oracle builds the EngineCore with
  object.__new__ and sets the attributes step_with_batch_queue reads, to the values __init__
  would give them (batch_queue_size = max_concurrent_batches, a deque of that length, the
  scheduler, no structured output, no spec decode);
- the CPU platform's override of async scheduling: vllm/platforms/cpu.py:329-331@5df63a72
  turns async scheduling off on CPU ("not required on CPU"), so the oracle sets the
  scheduler config's async_scheduling back to the scenario's value after the platform ran,
  and builds the scheduler class get_scheduler_cls() picks for it
  (vllm/config/scheduler.py:228-236@5df63a72);
- the V2 model runner's Triton requirement: use_v2_model_runner is read from the scenario
  and set on the scheduler and the config's derived values, as vLLM's own CPU tests do
  (tests/v1/core/test_async_scheduler.py:499-517@5df63a72), because the CPU wheel has no
  Triton (inference-sim/blis-vllm#3).

No decision code is patched: schedule(), update_from_output(), _update_after_schedule and
step_with_batch_queue itself run as released.

The GPU mock decides nothing: which requests of a batch sample is read from the scheduler's
own state when the batch executes (is_prefill_chunk, which _update_after_schedule sets for
exactly that batch, core/sched/scheduler.py:1634@5df63a72), as tier 2 reads it, and what they
sample is the scenario's assigned output, the last token as EOS, consumed in completion
order. A request scheduled one step ahead of its sampled tokens (async scheduling's
placeholder step) samples once per step like any decode.

Usage: python -I oracle/run_enginecore.py <vllm checkout> <scenario.json | dir> [out dir]
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import deque
from concurrent.futures import Future


def main() -> None:
    vllm_root, path = sys.argv[1], sys.argv[2]
    if os.path.isdir(path):
        out_dir = sys.argv[3] if len(sys.argv) > 3 else path
        # Golden traces are never overwritten without review (docs/testing.md).
        if "testdata/golden" in os.path.abspath(out_dir).replace(os.sep, "/") and \
                os.environ.get("BLIS_VLLM_ALLOW_GOLDEN_UPDATE") != "1":
            sys.exit("refusing to write golden traces: set BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1, and add "
                     "a golden-reviewed: <reviewer> trailer to the commit")
        names = sorted(f for f in os.listdir(path)
                       if f.endswith(".json") and not f.endswith((".trace.json", ".metrics.json")))
        for name in names:
            trace = run_one(vllm_root, os.path.join(path, name))
            with open(os.path.join(out_dir, name[:-5] + ".trace.json"), "w") as f:
                json.dump(trace, f, indent=1, sort_keys=True)
        return
    print(json.dumps(run_one(vllm_root, path), indent=1, sort_keys=True))


def run_one(vllm_root: str, scenario_path: str) -> dict:
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    model = os.environ.get("ORACLE_MODEL", "facebook/opt-125m")

    import tests.v1.core.utils as helpers  # noqa: E402
    from tests.v1.core.utils import EOS_TOKEN_ID  # noqa: E402

    if not os.path.abspath(helpers.__file__).startswith(os.path.abspath(vllm_root) + os.sep):
        sys.exit("tests.v1.core.utils did not resolve into the vLLM checkout")
    from vllm.sampling_params import SamplingParams  # noqa: E402
    from vllm.utils.hashing import sha256  # noqa: E402
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
    from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
    from vllm.v1.engine.core import EngineCore  # noqa: E402
    from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
    from vllm.v1.request import Request  # noqa: E402

    with open(scenario_path) as f:
        sc = json.load(f)
    cfg = sc["config"]
    block_size = cfg["block_size"]
    async_sched = bool(cfg.get("async_scheduling", False))
    pp = int(cfg.get("pipeline_parallel_size", 1))
    v2 = bool(cfg.get("use_v2_model_runner", False))
    if pp > 1 and async_sched and not v2:
        # vLLM refuses to start this (vllm/config/vllm.py:2970-2974@5df63a72).
        sys.exit(f"{scenario_path}: the V1 model runner does not support PP with async scheduling")

    scheduler = helpers.create_scheduler(
        model=model,
        max_num_seqs=cfg["max_num_seqs"],
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        long_prefill_token_threshold=cfg.get("long_prefill_token_threshold", 0),
        enable_prefix_caching=cfg.get("enable_prefix_caching", False),
        num_blocks=cfg["num_blocks"],
        block_size=block_size,
        max_model_len=cfg.get("max_model_len"),
        scheduling_policy=cfg.get("scheduling_policy", "fcfs"),
        pipeline_parallel_size=pp,
        async_scheduling=async_sched,
        use_v2_model_runner=v2,
        scheduler_cls=AsyncScheduler if async_sched else Scheduler,
    )
    vllm_config = scheduler.vllm_config
    # The CPU platform turned async scheduling off (vllm/platforms/cpu.py:329-331); restore
    # the scenario's value.
    vllm_config.scheduler_config.async_scheduling = async_sched
    # vLLM's own max_concurrent_batches (vllm/config/vllm.py:577-588), evaluated with the
    # scenario's model-runner choice: the CPU wheel cannot select the V2 runner (no Triton).
    from types import SimpleNamespace  # noqa: E402
    from vllm.config import VllmConfig  # noqa: E402
    batch_queue_size = VllmConfig.max_concurrent_batches.fget(SimpleNamespace(
        parallel_config=vllm_config.parallel_config,
        scheduler_config=vllm_config.scheduler_config,
        use_v2_model_runner=v2))
    if async_sched:
        assert isinstance(scheduler, AsyncScheduler)
        assert scheduler.decode_stagger == pp

    # A bare EngineCore holding exactly what step_with_batch_queue reads.
    core = object.__new__(EngineCore)
    core.vllm_config = vllm_config
    core.log_stats = True
    core.scheduler = scheduler
    core.batch_queue_size = batch_queue_size
    core.batch_queue = deque(maxlen=batch_queue_size) if batch_queue_size > 1 else None
    core.is_mm_encoder_only = False
    core.is_pooling_model = False
    core.check_for_draft_tokens = False
    core.use_spec_decode = False
    core.async_scheduling = async_sched
    core._weight_version = "default"
    import queue as _queue
    core.aborts_queue = _queue.Queue()

    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256)
    by_id = {r["id"]: r for r in sc["requests"]}
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    sampled = {r["id"]: 0 for r in sc["requests"]}
    batches: list[dict] = []
    steps: list[dict] = []
    finished: dict[str, dict] = {}

    class Done:
        """A future that has already resolved."""

        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class Lazy:
        """A future resolved when the batch queue pops it: the GPU mock runs in completion
        order, which on one device is the order the batches were queued."""

        def __init__(self, fn):
            self._fn, self._value, self._done = fn, None, False

        def result(self):
            if not self._done:
                self._value, self._done = self._fn(), True
            return self._value

    class MockExecutor:
        """The GPU (oracle/mocks.json): which requests sample is read from the scheduler's own
        state at execute time (is_prefill_chunk, set by _update_after_schedule for exactly this
        batch); what they sample is the scenario's assigned output, at completion."""

        def execute_model(self, scheduler_output, non_block=False):
            bid = len(batches)
            samples = [rid for rid in scheduler_output.num_scheduled_tokens
                       if not scheduler.requests[rid].is_prefill_chunk]
            batches.append({
                "batch": bid,
                "scheduled": dict(sorted(scheduler_output.num_scheduled_tokens.items())),
                "preempted": sorted(scheduler_output.preempted_req_ids or []),
                "samples": sorted(samples),
            })
            if scheduler_output.total_num_scheduled_tokens == 0:
                # No sampling follows: the engine queues this future itself (core.py:733-735).
                return Done(gpu_output(scheduler_output, []))
            self._pending = (scheduler_output, samples)
            return Done(None)

        def sample_tokens(self, grammar_output, non_block=False):
            so, samples = self._pending
            if non_block:
                return Lazy(lambda: gpu_output(so, samples))
            return gpu_output(so, samples)

    core.model_executor = MockExecutor()

    def gpu_output(scheduler_output, samples):
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        tokens = []
        for rid in req_ids:
            if rid not in samples:
                tokens.append([])
                continue
            k = sampled[rid]
            outp = by_id[rid]["output"]
            # The last assigned token is sampled as EOS; a step scheduled after it (async
            # scheduling runs one step ahead) samples EOS again, which the engine ignores
            # for a request that has already finished.
            tokens.append([EOS_TOKEN_ID if k >= len(outp) - 1 else outp[k]])
            sampled[rid] = k + 1
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=tokens,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def add_arrivals(upto: int) -> None:
        while pending and pending[0]["arrival_step"] <= upto:
            r = pending.pop(0)
            params = SamplingParams(max_tokens=r.get("max_tokens", 1 << 20))
            params.update_from_generation_config({}, EOS_TOKEN_ID)
            scheduler.add_request(Request(
                request_id=r["id"], prompt_token_ids=list(r["prompt"]), sampling_params=params,
                pooling_params=None, arrival_time=float(r["arrival_step"]),
                priority=r.get("priority", 0), block_hasher=hasher))

    step = 0
    max_steps = sc.get("max_steps", 10000)
    step_fn = core.step_with_batch_queue if core.batch_queue is not None else None
    while step < max_steps:
        add_arrivals(step)
        has_work = scheduler.has_requests() or bool(core.batch_queue)
        if not has_work and not pending:
            break
        if not has_work:
            steps.append({"step": step, "idle": True})
            step += 1
            continue
        before = len(batches)
        if step_fn is not None:
            outputs, executed = step_fn()
        else:
            outputs, executed = core.step()
        done = []
        for eco in (outputs or {}).values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    done.append(o.request_id)
                    finished[o.request_id] = {"reason": str(o.finish_reason),
                                              "output_tokens": sampled[o.request_id], "step": step}
        rec = {
            "step": step,
            "scheduled_batch": batches[before]["batch"] if len(batches) > before else None,
            "queue_len": len(core.batch_queue) if core.batch_queue is not None else 0,
            "processed": outputs is not None,
            "finished": sorted(done),
        }
        rec.update(_kv(scheduler))
        steps.append(rec)
        step += 1

    here = os.path.dirname(os.path.abspath(__file__))
    return {
        "provenance": provenance(vllm_root, scenario_path, here, {
            "async_scheduling": async_sched, "pipeline_parallel_size": pp,
            "use_v2_model_runner": v2, "max_concurrent_batches": batch_queue_size,
            "scheduler_class": type(scheduler).__name__}),
        "batches": batches,
        "steps": steps,
        "finished": dict(sorted(finished.items())),
    }


def _kv(scheduler) -> dict:
    num_running, num_waiting = scheduler.get_request_counts()
    mgr = scheduler.kv_cache_manager
    blocks = {}
    for rid in sorted(r.request_id for r in scheduler.running):
        blocks[rid] = [[b.block_hash.hex() if b.block_hash is not None else None, b.ref_cnt]
                       for b in mgr.coordinator.single_type_managers[0].req_to_blocks.get(rid, [])]
    return {
        "blocks": blocks,
        "free_blocks": scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
        "num_running": num_running,
        "num_waiting": num_waiting,
        "running": sorted(r.request_id for r in scheduler.running),
        "computed": {r.request_id: r.num_computed_tokens for r in sorted(scheduler.running, key=lambda r: r.request_id)},
        "placeholders": {r.request_id: r.num_output_placeholders
                         for r in sorted(scheduler.running, key=lambda r: r.request_id)},
    }


def provenance(vllm_root: str, scenario_path: str, here: str, engine: dict) -> dict:
    if here not in sys.path:
        sys.path.insert(0, here)
    from run_scheduler import blis_vllm_commit, committed_lock_sha256, scenario_seed  # noqa: E402

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", vllm_root, *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    import vllm  # noqa: E402

    with open(scenario_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(here, "mocks.json"), "rb") as f:
        mocks = json.load(f)
    lockfile = os.environ.get("ORACLE_LOCKFILE")
    lock_digest = "unrecorded"
    if lockfile:
        with open(lockfile, "rb") as f:
            lock_digest = hashlib.sha256(f.read()).hexdigest()
    return {
        "tier": 3,
        "vllm_version": vllm.__version__,
        "vllm_distribution": importlib.metadata.version("vllm"),
        "vllm_commit": git("rev-parse", "HEAD"),
        "scenario_sha256": digest,
        "mock_version": mocks["version"],
        "model": "facebook/opt-125m",
        "model_revision": os.environ.get("ORACLE_MODEL_REVISION", "unpinned"),
        "mocks": [m["symbol"] for m in mocks["mocks"]],
        "engine": engine,
        "python": sys.version.split()[0],
        "platform": platform.system().lower() + "-" + platform.machine(),
        "lockfile_sha256": lock_digest,
        "requirements_lock_sha256": committed_lock_sha256(),
        "blis_vllm_commit": blis_vllm_commit(),
        "seed": scenario_seed(scenario_path),
    }


if __name__ == "__main__":
    main()
