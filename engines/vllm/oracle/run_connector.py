# Original to blis-vllm.
"""Tier-4 oracle: vLLM's real Scheduler with its real NIXL connector, transport mocked.

A scenario (oracle/README.md, "Connector scenarios") fixes the scheduler configuration, the
connector (NixlConnector, the pull scheduler NIXL's default selects), the engine's KV role and
the requests: each with its prompt, assigned output and arrival step, and, for a P/D request,
which side of the transfer this engine is:

  - "remote_prefill" (this engine is the decoder, kv_consumer): the prompt's KV is pulled from
    a remote prefiller. The scenario gives how many steps after the pull starts the worker's
    output reports it done ("recv_delay"; 0 is the step that started it, and a request aborted
    meanwhile is reported too; with a batch queue, the first output processed from then on), and optionally that it failed as a whole ("recv_failed") or
    which of the pulled blocks failed
    ("invalid_blocks", by index into the request's blocks the pull fills: its unhashed ones,
    what NIXL hands its worker, nixl/pull_scheduler.py:150-155@5df63a72);
  - "remote_decode" (this engine is the prefiller, kv_producer): the request's blocks are kept
    after it finishes until the decoder has read them; the scenario gives how many steps after
    the finish the worker's output reports the send done ("send_delay", at least 1).

Any request may also be aborted by its client at a step ("abort_step"): EngineCore applies
aborts after the step's model run and before update_from_output (vllm/v1/engine/core.py:
678-683@5df63a72, abort_requests -> finish_requests(FINISHED_ABORTED), :554-559).

vLLM's own CPU test helpers build the Scheduler and its connector
(tests/v1/kv_connector/unit/utils.py create_vllm_config, create_scheduler), the setup vLLM's
own cpu_test connector suites use (test_remote_prefill_lifecycle.py,
test_remote_decode_lifecycle.py, test_kv_load_failure_recovery.py). The worker side is the
mock: the transport, reporting finished_recving / finished_sending / failed_recving /
invalid_block_ids in the KVConnectorOutput at the scenario's steps, as a NIXL worker reports
a transfer it has finished. The GPU is the tier-2 mock (sampled tokens, the last one EOS).
Every replacement is listed in oracle/mocks.json; no scheduler, KV-cache or connector
decision is patched.

A scenario may also state a hybrid model's KV cache groups ("kv_groups", as tier 2 states them:
full attention and sliding windows, built from vLLM's own spec classes and handed to the
helper's create_scheduler as its kv_cache_config) and a drafter ("speculative", tier 2's: vLLM's
own SpeculativeConfig, a configuration-only draft checkpoint under oracle/drafts, with the
GPU mock's acceptance plan per request, "accepted", and the drafter's proposals after every
step that ran, as EngineCore.post_step hands them over, vllm/v1/engine/core.py:689-696@5df63a72).
NIXL is SupportsHMA (nixl/connector.py:78), so vLLM keeps the hybrid KV cache manager on with
it (vllm/config/vllm.py:2170-2192); a block-level load failure ("invalid_blocks") is refused
beside several groups, because NIXL's worker never reports one there (nixl/base_worker.py:
2910-2914) and the scheduler would refuse it (scheduler.py:3365-3374).

A scenario may also run more than one batch in flight ("async_scheduling",
"pipeline_parallel_size", "use_v2_model_runner", as tier 3 states them). The engine is then
vLLM's own EngineCore.step_with_batch_queue (vllm/v1/engine/core.py:698-810@5df63a72) over the
real (Async)Scheduler and connector, on tier 3's bare EngineCore (run_enginecore.py
engine_core). The GPU mock reads which requests sample when a batch executes and samples them
when the batch queue pops it. The transport mock's report rides on that output as its
KVConnectorOutput, evaluated at the pop. A client's abort at a step is queued on the
EngineCore's aborts queue, which vLLM drains after the next popped batch's output and before
update_from_output (core.py:780-783, 814-822). vLLM refuses a decoder or a hybrid layout with
PP > 1 in NIXL's worker (nixl/base_worker.py:757-779), so such a scenario is refused here
before it is run.

Per step it records what tier 2 records (tokens scheduled, preempted, finished, the KV state
by content) plus the connector-visible state: the requests waiting for remote KV, the ones
whose pull has landed and are not yet admitted, the finished requests whose blocks the
connector still holds (delay-freed), and the kv-fetch stage counts.

Usage: python -I oracle/run_connector.py <vllm checkout> <scenario.json | dir> [out dir]
"""

import json
import os
import sys


def main() -> None:
    vllm_root, path = sys.argv[1], sys.argv[2]
    if os.path.isdir(path):
        out_dir = sys.argv[3] if len(sys.argv) > 3 else path
        if "testdata/golden" in os.path.abspath(out_dir).replace(os.sep, "/") and \
                os.environ.get("BLIS_VLLM_ALLOW_GOLDEN_UPDATE") != "1":
            sys.exit("refusing to write golden traces: set BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1, and add "
                     "a golden-reviewed: <reviewer> trailer to the commit")
        for name in sorted(f for f in os.listdir(path)
                           if f.endswith(".json") and not f.endswith(".trace.json")):
            trace = run_one(vllm_root, os.path.join(path, name))
            with open(os.path.join(out_dir, name[:-5] + ".trace.json"), "w") as f:
                json.dump(trace, f, indent=1, sort_keys=True)
        return
    print(json.dumps(run_one(vllm_root, path), indent=1, sort_keys=True))


def _ordered(ids):
    """A set-like view of ids that iterates in sorted order, or None when empty."""
    return dict.fromkeys(sorted(ids)).keys() if ids else None


def run_one(vllm_root: str, scenario_path: str) -> dict:
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    model = os.environ.get("ORACLE_MODEL", "facebook/opt-125m")

    import tests.v1.kv_connector.unit.utils as helpers  # noqa: E402

    if not os.path.abspath(helpers.__file__).startswith(os.path.abspath(vllm_root) + os.sep):
        sys.exit("tests.v1.kv_connector.unit.utils did not resolve into the vLLM checkout")
    from vllm.sampling_params import SamplingParams  # noqa: E402
    from vllm.utils.hashing import sha256  # noqa: E402
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
    from vllm.v1.metrics.stats import KV_FETCH_STAGES  # noqa: E402
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput  # noqa: E402
    from vllm.v1.request import Request, RequestStatus  # noqa: E402

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from run_scheduler import kv_state, provenance  # noqa: E402

    with open(scenario_path) as f:
        sc = json.load(f)
    cfg = sc["config"]
    conn = cfg["connector"]
    block_size = cfg["block_size"]

    from run_scheduler import v2_runner_env  # noqa: E402

    spec = cfg.get("speculative")
    groups = cfg.get("kv_groups")
    if groups and len(groups) > 1 and any(r.get("invalid_blocks") for r in sc["requests"]):
        sys.exit(f"{scenario_path}: invalid_blocks beside several KV cache groups: NIXL's worker reports no "
                 "block-level failure under HMA (nixl/base_worker.py:2910-2914) and the scheduler refuses one "
                 "(scheduler.py:3365-3374)")
    vllm_config = helpers.create_vllm_config(
        model=model,
        max_num_seqs=cfg["max_num_seqs"],
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        block_size=block_size,
        max_model_len=cfg["max_model_len"],
        kv_connector=conn["name"],
        kv_role=conn["kv_role"],
        kv_load_failure_policy=conn.get("kv_load_failure_policy", "fail"),
        kv_connector_extra_config=conn.get("extra_config") or {},
        enable_chunked_prefill=cfg.get("enable_chunked_prefill", True),
    )
    async_sched = bool(cfg.get("async_scheduling", False))
    pp = int(cfg.get("pipeline_parallel_size", 1))
    v2 = bool(cfg.get("use_v2_model_runner", False)) or (bool(spec) and spec["method"] == "dspark")
    if pp > 1 and (conn["kv_role"] != "kv_producer" or (groups and len(groups) > 1)):
        sys.exit(f"{scenario_path}: vLLM's NIXL pull worker refuses pipeline_parallel_size > 1 for a decoder "
                 "and for a hybrid KV cache layout (nixl/base_worker.py:757-779@5df63a72)")
    if pp > 1 and async_sched and not v2:
        sys.exit(f"{scenario_path}: the V1 model runner does not support PP with async scheduling "
                 "(vllm/config/vllm.py:2970-2974@5df63a72)")
    # DSpark runs only on the V2 model runner (vllm/config/vllm.py:2976-2979@5df63a72): the
    # environment of a GPU host selects it (run_scheduler.py v2_runner_env, oracle/mocks.json).
    restore_env = v2_runner_env(v2)
    if pp > 1:
        # The pipeline-parallel size, through vLLM's own ParallelConfig and VllmConfig
        # constructors, as tier 3's create_scheduler passes it.
        import dataclasses  # noqa: E402

        from vllm.config import ParallelConfig  # noqa: E402

        vllm_config = dataclasses.replace(vllm_config, parallel_config=ParallelConfig(pipeline_parallel_size=pp))
    if spec:
        # The drafter: vLLM's own SpeculativeConfig on a configuration-only draft checkpoint, as
        # tier 2 builds it (run_scheduler.py build_scheduler), and the helper's VllmConfig built
        # again by vLLM's own constructor with it, as `vllm serve --speculative-config` builds
        # one (VllmConfig.__post_init__ derives every value the scheduler and the connector
        # read: lookahead, prefill lookahead, draft slots). Nothing is written into an object.
        import dataclasses  # noqa: E402

        from vllm.config import SpeculativeConfig  # noqa: E402

        args = {"method": spec["method"], "num_speculative_tokens": spec["num_speculative_tokens"]}
        if spec.get("draft"):
            args["model"] = os.path.join(here, "drafts", spec["draft"])
        vllm_config = dataclasses.replace(vllm_config, speculative_config=SpeculativeConfig(
            target_model_config=vllm_config.model_config, target_parallel_config=vllm_config.parallel_config,
            **args))
    # create_vllm_config always enables prefix caching and leaves the scheduling policy, the
    # long-prefill cap and the watermark at their defaults; the scenario states them
    # (environment set-up of vLLM's own config objects, read when the Scheduler is built).
    vllm_config.cache_config.enable_prefix_caching = cfg.get("enable_prefix_caching", True)
    vllm_config.scheduler_config.policy = cfg.get("scheduling_policy", "fcfs")
    vllm_config.scheduler_config.long_prefill_token_threshold = cfg.get("long_prefill_token_threshold", 0)
    vllm_config.scheduler_config.watermark = cfg.get("watermark", 0.0)
    # The helper passes max_model_len to SchedulerConfig only and builds ModelConfig without it
    # (utils.py:132-142), so the model's own 2048 would govern: the Scheduler reads
    # model_config.max_model_len (scheduler.py:145). Set it there, as `vllm serve
    # --max-model-len` does, and run the startup check the edits above bypassed.
    vllm_config.model_config.max_model_len = cfg["max_model_len"]
    vllm_config.scheduler_config.verify_max_model_len(cfg["max_model_len"])
    # The CPU platform turns async scheduling off (vllm/platforms/cpu.py:329-331); tier 3's
    # set_async sets the scenario's value back and returns vLLM's own max_concurrent_batches
    # for it, before the Scheduler reads both (defer_block_free, scheduler.py:184-192).
    from run_enginecore import set_async  # noqa: E402

    batch_queue_size = set_async(vllm_config, async_sched, v2) if (async_sched or pp > 1) else 1
    try:
        scheduler = helpers.create_scheduler(vllm_config, num_blocks=cfg["num_blocks"],
                                             kv_cache_config=kv_cache_config(cfg, vllm_config))
    finally:
        restore_env()
    init_none_hash(sha256)
    if batch_queue_size > 1:
        return run_queued(scheduler, vllm_config, batch_queue_size, sc, scenario_path, vllm_root,
                          hasher_for=lambda: get_request_block_hasher(block_size, sha256),
                          helpers=helpers, add_kv_params=add_kv_params, connector_state_of=connector_state_of)
    assert scheduler.max_model_len == cfg["max_model_len"], "the Scheduler does not read the scenario's max_model_len"
    num_spec = scheduler.num_spec_tokens
    accepted_plan = {r["id"]: list(r.get("accepted", [])) for r in sc["requests"]}
    verify_steps = {r["id"]: 0 for r in sc["requests"]}

    hasher = get_request_block_hasher(block_size, sha256)
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    by_id = {r["id"]: r for r in sc["requests"]}
    sampled_so_far = {r["id"]: 0 for r in sc["requests"]}
    finished_at = {}
    load_started = {}
    reported = set()
    finished = {}
    steps = []
    step = 0
    max_steps = sc.get("max_steps", 10000)

    def add_arrivals(upto: int) -> None:
        while pending and pending[0]["arrival_step"] <= upto:
            r = pending.pop(0)
            params = SamplingParams(max_tokens=r.get("max_tokens", 1 << 20))
            params.update_from_generation_config({}, helpers.EOS_TOKEN_ID)
            req = Request(
                request_id=r["id"], prompt_token_ids=list(r["prompt"]), sampling_params=params,
                pooling_params=None, arrival_time=float(r["arrival_step"]),
                priority=r.get("priority", 0), block_hasher=hasher)
            add_kv_params(req, r, block_size)
            scheduler.add_request(req)

    def connector_state() -> dict:
        return connector_state_of(scheduler)

    while step < max_steps:
        add_arrivals(step)
        if not scheduler.has_requests() and not pending:
            break
        out = scheduler.schedule()
        for rid, r in by_id.items():
            req = scheduler.requests.get(rid)
            if r.get("remote_prefill") and rid not in load_started and req is not None and \
                    req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                load_started[rid] = step
        record = {
            "step": step,
            "scheduled": dict(sorted(out.num_scheduled_tokens.items())),
            "preempted": sorted(out.preempted_req_ids or []),
        }
        # The GPU mock (tier 2): a request completing its known tokens samples its next
        # assigned token, the last one as EOS.
        # With a drafter, a verify step also samples the scenario's accepted count of the drafts
        # it verified, at most the real ones (a -1 placeholder is always rejected,
        # vllm/v1/sample/rejection_sampler.py:762-763), as tier 2's mock does.
        req_ids, sampled = [], []
        spec_scheduled = {}
        for rid in out.num_scheduled_tokens:
            req = scheduler.requests[rid]
            req_ids.append(rid)
            if not req.is_prefill_chunk:
                n = 1
                drafts = out.scheduled_spec_decode_tokens.get(rid, [])
                if drafts:
                    spec_scheduled[rid] = len(drafts)
                    plan, i = accepted_plan[rid], verify_steps[rid]
                    verify_steps[rid] = i + 1
                    n = 1 + min(plan[i] if i < len(plan) else 0, len([d for d in drafts if d >= 0]))
                outp = by_id[rid]["output"]
                toks = []
                for _ in range(n):
                    k = sampled_so_far[rid]
                    if k >= len(outp):
                        break
                    toks.append(helpers.EOS_TOKEN_ID if k == len(outp) - 1 else outp[k])
                    sampled_so_far[rid] = k + 1
                sampled.append(toks)
            else:
                sampled.append([])
        if spec_scheduled:
            record["spec_scheduled"] = dict(sorted(spec_scheduled.items()))
        # The transport mock: what the worker side reports this step.
        kv_out, record["transport"] = transport_report(scheduler, by_id, step, load_started, finished_at, reported)
        aborts = sorted(rid for rid, r in by_id.items() if r.get("abort_step") == step)
        if aborts:
            scheduler.finish_requests(aborts, RequestStatus.FINISHED_ABORTED)
        record["aborted"] = aborts
        mro = ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={}, pooler_output=[],
            kv_connector_output=kv_out)
        engine_outputs = scheduler.update_from_output(out, mro)
        if spec and out.total_num_scheduled_tokens > 0:
            # The drafter proposes for every request the step ran, as EngineCore.post_step hands
            # them over without async scheduling (core.py:689-696); the scheduler ignores
            # proposals for prefill chunks (scheduler.py:2560-2564). Only the count reaches a
            # decision.
            from vllm.v1.outputs import DraftTokenIds  # noqa: E402

            ids = [rid for rid in out.num_scheduled_tokens if rid in scheduler.requests]
            scheduler.update_draft_token_ids(DraftTokenIds(ids, [[0] * num_spec for _ in ids]))
        done = []
        for eco in engine_outputs.values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    done.append(o.request_id)
                    finished_at.setdefault(o.request_id, step)
                    finished[o.request_id] = {"reason": str(o.finish_reason),
                                              "output_tokens": sampled_so_far[o.request_id], "step": step}
        record["finished"] = sorted(done)
        record.update(kv_state(scheduler))
        record.update(connector_state())
        steps.append(record)
        step += 1

    prov = provenance(vllm_root, scenario_path)
    prov["tier"] = 4
    return {"provenance": prov, "steps": steps, "finished": dict(sorted(finished.items()))}


def add_kv_params(req, r: dict, block_size: int) -> None:
    """The KV-transfer parameters a P/D proxy hands each side, set on the Request (as
    create_request builds them, tests/v1/kv_connector/unit/utils.py:246-264@5df63a72)."""
    if r.get("remote_prefill"):
        req.kv_transfer_params = dict(
            do_remote_prefill=True, do_remote_decode=False, remote_engine_id="prefiller",
            remote_request_id="prefill-" + r["id"],
            remote_block_ids=list(range(-(-len(r["prompt"]) // block_size))),
            remote_host="prefiller", remote_port=1234, tp_size=1)
    elif r.get("remote_decode"):
        req.kv_transfer_params = dict(do_remote_prefill=False, do_remote_decode=True)


def connector_state_of(scheduler) -> dict:
    """The connector-visible state of the scheduler, recorded after each step."""
    from vllm.v1.metrics.stats import KV_FETCH_STAGES  # noqa: E402
    from vllm.v1.request import RequestStatus  # noqa: E402

    kv_waiting = sorted(r.request_id for r in scheduler.kv_holding_waiting
                        if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS)
    live = {r.request_id for r in scheduler.running}
    live |= {r.request_id for r in scheduler.waiting}
    live |= {r.request_id for r in scheduler.kv_holding_waiting}
    held = sorted(rid for rid, r in scheduler.requests.items()
                  if rid not in live and r.is_finished())
    return {
        "waiting_for_remote_kv": kv_waiting,
        "recv_landed": sorted(scheduler.finished_recving_kv_req_ids),
        "delay_freed": held,
        "kv_holding_waiting": sorted(r.request_id for r in scheduler.kv_holding_waiting),
        "kv_fetch_stages": [scheduler._kv_fetch_counts[s] for s in KV_FETCH_STAGES],
        "num_requests": len(scheduler.requests),
    }


def transport_report(scheduler, by_id: dict, step: int, load_started: dict, finished_at: dict, reported: set):
    """The transport mock: what the worker side reports in the output processed at a step, from
    the scenario's outcomes, and the KVConnectorOutput carrying it (None when it reports
    nothing). A transfer is reported once, in the first output processed at or after its step
    (the pull's start plus recv_delay; the finish plus send_delay). Synchronously that is the
    step itself, since every step processes an output; with a batch queue, a step that pops no
    batch hands the scheduler nothing, so the report waits for the next popped output.

    A pull lands at its step whether or not the client aborted the request since: the worker
    reports the transfer when it completes, and the scheduler then frees the blocks it held for
    it (scheduler.py:3222-3231). The blocks a pull fills are, per KV cache group, the request's
    unhashed blocks of each group (get_unhashed_block_ids_all_groups, kv_cache_manager.py:
    114-124, which NIXL hands its worker, pull_scheduler.py:149-156@5df63a72); a block-level
    failure names blocks of the one group (refused beside several)."""
    from vllm.v1.outputs import KVConnectorOutput  # noqa: E402
    from vllm.v1.request import RequestStatus  # noqa: E402

    recving, sending, failed, invalid = set(), set(), set(), set()
    for rid, r in by_id.items():
        req = scheduler.requests.get(rid)
        if req is None:
            continue
        if r.get("remote_prefill") and rid in load_started and ("recv", rid) not in reported and \
                step >= load_started[rid] + r.get("recv_delay", 0) and \
                (req.status == RequestStatus.WAITING_FOR_REMOTE_KVS or req.is_finished()):
            reported.add(("recv", rid))
            recving.add(rid)
            if r.get("recv_failed"):
                failed.add(rid)
            pulled = [[b.block_id for b in m.req_to_blocks.get(rid, []) if b.block_hash is None and not b.is_null]
                      for m in scheduler.kv_cache_manager.coordinator.single_type_managers]
            for i in r.get("invalid_blocks", []):
                if i < len(pulled[0]):
                    invalid.add(pulled[0][i])
        if r.get("remote_decode") and rid in finished_at and req.is_finished() and ("send", rid) not in reported and \
                step >= finished_at[rid] + max(r.get("send_delay", 1), 1):
            reported.add(("send", rid))
            sending.add(rid)
    kv_out = None
    if recving or sending or failed or invalid:
        # The worker reports these as sets, and the scheduler frees blocks in the order it
        # iterates them (scheduler.py:3222-3235@5df63a72), which decides which freed blocks
        # the next allocation reuses. A set of strings iterates in an order that depends on
        # PYTHONHASHSEED, so the transport mock fixes the order: by request ID, the order
        # the Go side reports them in. (Insertion-ordered dict keys behave as a set here:
        # the scheduler only iterates and tests membership.)
        kv_out = KVConnectorOutput(finished_recving=_ordered(recving), finished_sending=_ordered(sending),
                                   failed_recving=failed, invalid_block_ids=invalid)
    rec = {"recving": sorted(recving), "sending": sorted(sending), "failed": sorted(failed),
           "invalid_blocks": len(invalid)}
    return kv_out, rec


class QueuedGPU:
    """The GPU and transport mocks as tier 3's EngineCore executor (oracle/mocks.json): which
    requests of a batch sample is read from the scheduler's own is_prefill_chunk when the batch
    executes; what they sample (the scenario's output, the last token as EOS, plus the drafts
    the acceptance plan accepts, every async placeholder a real draft) is resolved when the
    batch queue pops it, with the transport mock's report for that step as the output's
    KVConnectorOutput. A pull's start is noted when the batch that parks it executes."""

    def __init__(self, scheduler, by_id, clock, load_started, finished_at, async_sched, record):
        self.scheduler, self.by_id, self.clock = scheduler, by_id, clock
        self.load_started, self.finished_at, self.async_sched = load_started, finished_at, async_sched
        self.record = record
        self.reported = set()
        self.sampled = {rid: 0 for rid in by_id}
        self.verify = {rid: 0 for rid in by_id}
        self.last = None
        self.pending = None

    def executor(self):
        return self

    def execute_model(self, so, non_block=False):
        from run_enginecore import Done, Lazy  # noqa: E402
        from vllm.v1.request import RequestStatus  # noqa: E402

        step = self.clock["step"]
        for rid, req in self.scheduler.requests.items():
            if self.by_id[rid].get("remote_prefill") and rid not in self.load_started and \
                    req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.load_started[rid] = step
        samples = [rid for rid in so.num_scheduled_tokens if not self.scheduler.requests[rid].is_prefill_chunk]
        rec = self.record()
        rec["scheduled"] = dict(sorted(so.num_scheduled_tokens.items()))
        rec["preempted"] = sorted(so.preempted_req_ids or [])
        spec = {rid: len(d) for rid, d in so.scheduled_spec_decode_tokens.items() if d}
        if spec:
            rec["spec_scheduled"] = dict(sorted(spec.items()))
        if so.total_num_scheduled_tokens == 0:
            return Lazy(lambda: self.output(so, []))
        self.last = so
        self.pending = (so, samples)
        return Done(None)

    def sample_tokens(self, grammar_output, non_block=False):
        from run_enginecore import Lazy  # noqa: E402

        so, samples = self.pending
        return Lazy(lambda: self.output(so, samples))

    def take_draft_token_ids(self):
        # The drafter after a step that ran, without async scheduling (core.py:689-696), as
        # tier 3's mock proposes: num_speculative_tokens drafts for every request the last
        # executed batch ran. Only their count reaches a decision.
        from vllm.v1.outputs import DraftTokenIds  # noqa: E402

        if self.last is None:
            return None
        ids = [rid for rid in self.last.num_scheduled_tokens if rid in self.scheduler.requests]
        k = self.scheduler.num_spec_tokens
        return DraftTokenIds(ids, [[0] * k for _ in ids])

    def output(self, so, samples):
        from tests.v1.core.utils import EOS_TOKEN_ID  # noqa: E402
        from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402

        req_ids = list(so.num_scheduled_tokens.keys())
        tokens = []
        for rid in req_ids:
            if rid not in samples:
                tokens.append([])
                continue
            outp = self.by_id[rid]["output"]
            n = 1
            drafts = so.scheduled_spec_decode_tokens.get(rid, [])
            if drafts:
                plan, i = self.by_id[rid].get("accepted", []), self.verify[rid]
                self.verify[rid] = i + 1
                real = len(drafts) if self.async_sched else len([d for d in drafts if d >= 0])
                n = 1 + min(plan[i] if i < len(plan) else 0, real)
            toks = []
            for _ in range(n):
                k = self.sampled[rid]
                toks.append(EOS_TOKEN_ID if k >= len(outp) - 1 else outp[k])
                self.sampled[rid] = k + 1
                if k >= len(outp) - 1:
                    break
            tokens.append(toks)
        kv_out, rep = transport_report(self.scheduler, self.by_id, self.clock["step"], self.load_started,
                                       self.finished_at, self.reported)
        rec = self.record()
        rec["transport"] = rep
        rec["processed"] = True
        return ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=tokens, logprobs=None, prompt_logprobs_dict={}, pooler_output=[],
            kv_connector_output=kv_out)


def run_queued(scheduler, vllm_config, batch_queue_size, sc, scenario_path, vllm_root, hasher_for, helpers,
               add_kv_params, connector_state_of) -> dict:
    """A scenario with more than one batch in flight: vLLM's own step_with_batch_queue on tier 3's
    bare EngineCore (run_enginecore.py engine_core), one call per step, then post_step."""
    from run_enginecore import engine_core  # noqa: E402
    from run_scheduler import kv_state, provenance  # noqa: E402
    from vllm.sampling_params import SamplingParams  # noqa: E402
    from vllm.v1.request import Request  # noqa: E402

    cfg = sc["config"]
    hasher = hasher_for()
    by_id = {r["id"]: r for r in sc["requests"]}
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    clock = {"step": 0}
    load_started, finished_at, finished = {}, {}, {}
    steps: list[dict] = []
    current: dict = {}
    async_sched = bool(vllm_config.scheduler_config.async_scheduling)
    gpu = QueuedGPU(scheduler, by_id, clock, load_started, finished_at, async_sched, lambda: current)
    core = engine_core(scheduler, vllm_config, batch_queue_size, async_sched, scheduler.num_spec_tokens > 0, gpu)
    max_steps = sc.get("max_steps", 10000)
    step = 0
    while step < max_steps:
        clock["step"] = step
        while pending and pending[0]["arrival_step"] <= step:
            r = pending.pop(0)
            params = SamplingParams(max_tokens=r.get("max_tokens", 1 << 20))
            params.update_from_generation_config({}, helpers.EOS_TOKEN_ID)
            req = Request(request_id=r["id"], prompt_token_ids=list(r["prompt"]), sampling_params=params,
                          pooling_params=None, arrival_time=float(r["arrival_step"]),
                          priority=r.get("priority", 0), block_hasher=hasher)
            add_kv_params(req, r, cfg["block_size"])
            scheduler.add_request(req)
        has_work = scheduler.has_requests() or bool(core.batch_queue)
        if not has_work and not pending:
            break
        current = {"step": step, "scheduled": {}, "preempted": [], "processed": False,
                   "transport": {"recving": [], "sending": [], "failed": [], "invalid_blocks": 0}}
        # A client's aborts at this step reach the EngineCore's queue (core.py:1928), which vLLM
        # drains after the next popped batch's output (core.py:780-783).
        aborts = sorted(rid for rid, r in by_id.items() if r.get("abort_step") == step)
        if aborts:
            core.aborts_queue.put_nowait(aborts)
        if has_work:
            outputs, executed = core.step_with_batch_queue()
            core.post_step(executed)
        else:
            outputs, current["idle"] = None, True
        done = []
        for eco in (outputs or {}).values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    done.append(o.request_id)
                    finished_at.setdefault(o.request_id, step)
                    finished[o.request_id] = {"reason": str(o.finish_reason),
                                              "output_tokens": gpu.sampled[o.request_id], "step": step}
        current["finished"] = sorted(done)
        current["queue_len"] = len(core.batch_queue)
        current["aborted"] = aborts
        current.update(kv_state(scheduler))
        current.update(connector_state_of(scheduler))
        steps.append(current)
        step += 1
    prov = provenance(vllm_root, scenario_path)
    prov["tier"] = 4
    prov["engine"] = {"max_concurrent_batches": batch_queue_size, "async_scheduling": async_sched,
                      "scheduler_class": type(scheduler).__name__}
    return {"provenance": prov, "steps": steps, "finished": dict(sorted(finished.items()))}


def kv_cache_config(cfg: dict, vllm_config):
    """The scenario's KV cache groups for the helper's create_scheduler (its kv_cache_config
    argument), or None for the helper's one full-attention group. Each group is one 1-head,
    1-dim float32 layer of vLLM's own FullAttentionSpec or SlidingWindowSpec, as tier 2 builds
    them (run_scheduler.py build_scheduler), each sliding window tagged with
    extra_retained_tokens = num_prefill_lookahead_tokens - 1 from the built VllmConfig, as a
    launch tags it (get_kv_cache_configs, vllm/v1/core/kv_cache_utils.py:2582-2589@5df63a72)."""
    groups = cfg.get("kv_groups")
    if not groups:
        return None
    import torch  # noqa: E402
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs  # noqa: E402
    from vllm.v1.kv_cache_interface import (  # noqa: E402
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    register_all_kvcache_specs(vllm_config)
    extra_retained = max(0, vllm_config.num_prefill_lookahead_tokens - 1)
    specs = []
    for i, g in enumerate(groups):
        common = {"block_size": cfg["block_size"], "num_kv_heads": 1, "head_size": 1, "dtype": torch.float32}
        if g["kind"] == "full":
            layer = FullAttentionSpec(**common)
        elif g["kind"] == "sliding_window":
            layer = SlidingWindowSpec(sliding_window=g["sliding_window"], extra_retained_tokens=extra_retained,
                                      **common)
        else:
            raise ValueError(f"unknown KV cache group kind {g['kind']!r}")
        specs.append(KVCacheGroupSpec([f"layer{i}"], layer))
    return KVCacheConfig(num_blocks=cfg["num_blocks"], kv_cache_tensors=[], kv_cache_groups=specs)


if __name__ == "__main__":
    main()
