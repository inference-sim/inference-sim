# Original to blis-vllm.
"""Tier-4 oracle: vLLM's real Scheduler with its real MultiConnector over a NixlConnector (pull
scheduler) and an OffloadingConnector, the workers mocked as transports.

A scenario (oracle/README.md, "MultiConnector scenarios") is an offloading scenario
(oracle/run_offloading.py: KV cache groups, CPU tier, load and store delays) whose connector is

    {"name": "MultiConnector", "kv_role": R, "kv_load_failure_policy": P,
     "extra_config": {"connectors": [
         {"kv_connector": "NixlConnector", "kv_role": ...},
         {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
          "kv_connector_extra_config": {"cpu_bytes_to_use": ..., ...}}]}}

with the two children in either order, and whose requests may carry the NIXL parameters
oracle/run_connector.py reads ("remote_prefill" with "recv_delay", "recv_failed", "invalid_blocks";
"remote_decode" with "send_delay") beside the offloading ones ("load_delay", "store_delay").
vLLM's kv_offloading_size would replace kv_connector, so a scenario names MultiConnector itself.
The scheduler-level kv_role and kv_load_failure_policy are the top-level KVTransferConfig's
(scheduler.py:164-195@5df63a72); each child sees its own KVTransferConfig.

What the real MultiConnector does (vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py
@5df63a72), all of it running here:
- children are built in config order from kv_connector_extra_config["connectors"] (:184-190, 226-249);
- get_num_new_matched_tokens asks EVERY child in order, returns (None, False) as soon as one
  returns None (a child's later siblings are then not asked), and assigns the request to the FIRST
  child that returned tokens > 0 in _requests_to_connector (:434-453);
- update_state_after_alloc gives the chosen child the real num_external_tokens and every other
  child the request's real blocks with 0 (:467-477);
- on_new_request and build_connector_meta go to all children, the metadata a tuple in child order
  (:479-494); update_connector_output hands each child its own slice of
  MultiKVConnectorWorkerMetadata (:496-515);
- request_finished(_all_groups) asks all children, the delayed free is that of any child
  returning True, the transfer parameters are merged, and _requests_to_connector is popped
  (:542-612); has_pending_push_work is any(child) (:621-622).

What is replaced (oracle/mocks.json lists the replacements run_offloading.py and run_connector.py
make; this runner adds none of its own to vLLM's objects):
- the GPU, as tier 2 (each request completing its known tokens samples its next assigned token, the
  last one as EOS);
- the OffloadingConnector's worker, as oracle/run_offloading.py (Worker): it runs on the
  offloading child's slice of the step's MultiKVConnectorMetadata, and its completions reach the
  scheduler as vLLM's own OffloadingWorkerMetadata inside a MultiKVConnectorWorkerMetadata that
  holds None for the other child (build_connector_worker_meta, :409-419);
- the NIXL worker, as oracle/run_connector.py (transport_report): a pull lands recv_delay steps
  after it started, a hold ends send_delay steps after the finish. A pull is the NIXL child's
  only when the real MultiConnector chose it for the request (_requests_to_connector): a request
  that carries remote-prefill parameters but whose load the offloading child won is not pulled
  (NIXL's worker only notifies the remote side then and reports nothing, nixl/pull_worker.py:
  444-464@5df63a72), so the mock reports nothing for it. The two workers' reports reach the
  scheduler as one KVConnectorOutput, as the engine's KVOutputAggregator hands it;
- the iteration order of finished_req_ids, as oracle/run_offloading.py (order_finished).

The engine step is EngineCore.step's (core.py:657-687@5df63a72): no schedule() while
has_requests() is false, which counts the children's pending jobs. A scenario with async
scheduling runs vLLM's own EngineCore.step_with_batch_queue on tier 3's bare EngineCore, as
run_offloading.py does. Pipeline parallelism is refused: NIXL's worker refuses it for a decoder and
under HMA (nixl/base_worker.py:757-779), and no scenario here needs it.

Per step it records tier 2's state and the connector-visible state (oracle/run_connector.py), the
NIXL transport report ("transport": recving, sending, failed, invalid_blocks), the offloading
state ("offload": loads, stores, flush, completed, jobs, pending_push, cpu, as
oracle/run_offloading.py), "pending_push" (the MultiConnector's own any(child)) and "chosen": the
request ID -> child index map read from the real connector's _requests_to_connector after the
step. The trace also holds "requests": the kv_transfer_params each request was handed, as given
(NIXL rewrites them as it runs), and "children": the connectors in config order.

Usage: python -I oracle/run_multiconnector.py <vllm checkout> <scenario.json | dir> [out dir]
"""

import copy
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
                     "a golden-reviewed: <trace>: <reason> trailer for each trace the commit changes")
        for name in sorted(f for f in os.listdir(path)
                           if f.endswith(".json") and not f.endswith(".trace.json")):
            trace = run_one(vllm_root, os.path.join(path, name))
            with open(os.path.join(out_dir, name[:-5] + ".trace.json"), "w") as f:
                json.dump(trace, f, indent=1, sort_keys=True)
        return
    print(json.dumps(run_one(vllm_root, path), indent=1, sort_keys=True))


def gpu_spec(spec) -> list:
    return [[int(b) for b in spec.block_ids], list(spec.group_sizes), list(spec.block_indices)]


def cpu_state(manager) -> dict:
    """The CPU tier's chunks in LRU order (key, chunk id, reference count), its free and
    evictable chunk counts: oracle/run_offloading.py's record."""
    from run_offloading import key_text  # noqa: E402

    policy = manager._policy
    chunks = sorted(policy.chunks.items(), key=lambda kv: policy._ranks[kv[0]])
    return {
        "chunks": [[key_text(k), int(c.chunk_id), int(c.ref_cnt)] for k, c in chunks],
        "free": manager._get_num_free_chunks(),
        "evictable": manager._num_evictable_cache_chunks,
    }


def offload_record(meta) -> dict:
    """The offloading child's jobs of one step's metadata (loads, stores, flushed jobs)."""
    return {
        "loads": [[jid, j.req_id, [int(c) for c in j.src_spec.block_ids], gpu_spec(j.dst_spec)]
                  for jid, j in sorted(meta.load_jobs.items())],
        "stores": [[jid, j.req_id, [int(c) for c in j.dst_spec.block_ids], gpu_spec(j.src_spec)]
                   for jid, j in sorted(meta.store_jobs.items())],
        "flush": sorted(meta.jobs_to_flush or []),
    }


def merge_outputs(output_cls, off_out, nixl_out):
    """The step's one KVConnectorOutput from the two workers' reports (None when both report
    nothing). Receiving requests are the union, by request ID; the sets NIXL's worker reports
    (sending, failed, invalid blocks) and the offloading worker's metadata ride along unchanged."""
    from run_connector import _ordered  # noqa: E402

    if off_out is None and nixl_out is None:
        return None
    recving = set()
    for o in (off_out, nixl_out):
        if o is not None and o.finished_recving:
            recving |= set(o.finished_recving)
    return output_cls(
        finished_recving=_ordered(recving),
        finished_sending=nixl_out.finished_sending if nixl_out is not None else None,
        failed_recving=nixl_out.failed_recving if nixl_out is not None else set(),
        invalid_block_ids=nixl_out.invalid_block_ids if nixl_out is not None else set(),
        kv_connector_worker_meta=off_out.kv_connector_worker_meta if off_out is not None else None)


class Harness:
    """The state both engine loops share: the real scheduler and its MultiConnector, the two
    transport mocks' bookkeeping, and the requests still to arrive."""

    def __init__(self, scheduler, sc, helpers, hasher, off_idx, nixl_idx, worker):
        self.scheduler, self.helpers, self.hasher = scheduler, helpers, hasher
        self.multi = scheduler.connector
        self.off_idx, self.nixl_idx, self.worker = off_idx, nixl_idx, worker
        self.cs = self.multi._connectors[off_idx].connector_scheduler
        self.by_id = {r["id"]: r for r in sc["requests"]}
        self.pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
        self.block_size = sc["config"]["block_size"]
        self.sampled = {r["id"]: 0 for r in sc["requests"]}
        self.finished: dict = {}
        self.given: dict = {}
        self.load_started: dict = {}
        self.finished_at: dict = {}
        self.reported: set = set()
        self.nixl_owned: set = set()

    def add_arrivals(self, upto: int) -> None:
        from run_connector import add_kv_params  # noqa: E402
        from vllm.sampling_params import SamplingParams  # noqa: E402
        from vllm.v1.request import Request  # noqa: E402

        while self.pending and self.pending[0]["arrival_step"] <= upto:
            r = self.pending.pop(0)
            params = SamplingParams(max_tokens=r.get("max_tokens", 1 << 20))
            params.update_from_generation_config({}, self.helpers.EOS_TOKEN_ID)
            req = Request(
                request_id=r["id"], prompt_token_ids=list(r["prompt"]), sampling_params=params,
                pooling_params=None, arrival_time=float(r["arrival_step"]),
                priority=r.get("priority", 0), block_hasher=self.hasher)
            add_kv_params(req, r, self.block_size)
            # NIXL rewrites the parameters as it runs (do_remote_prefill goes False once it has
            # planned the pull); the trace keeps what the request was handed.
            self.given[r["id"]] = copy.deepcopy(req.kv_transfer_params)
            self.scheduler.add_request(req)

    def note_pulls(self, step: int) -> None:
        """A request the scheduler has just parked for remote KV starts a pull (NIXL's) only if
        the real MultiConnector assigned it to the NIXL child: the child that returned tokens
        for it, multi_connector.py:448-452@5df63a72. One the offloading child won is parked for
        its CPU load and the NIXL worker, told 0 external tokens, only notifies the prefiller."""
        from vllm.v1.request import RequestStatus  # noqa: E402

        chosen = self.multi._requests_to_connector
        for rid, r in self.by_id.items():
            req = self.scheduler.requests.get(rid)
            if r.get("remote_prefill") and rid not in self.load_started and req is not None and \
                    req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.load_started[rid] = step
                if chosen.get(rid) == self.nixl_idx:
                    self.nixl_owned.add(rid)

    def nixl_view(self) -> dict:
        """The scenario's requests as NIXL's transport mock reads them: a remote prefill the
        offloading child won is a request with no pull."""
        return {rid: (r if not r.get("remote_prefill") or rid in self.nixl_owned
                      else dict(r, remote_prefill=False))
                for rid, r in self.by_id.items()}

    def nixl_report(self, step: int):
        from run_connector import transport_report  # noqa: E402

        return transport_report(self.scheduler, self.nixl_view(), step, self.load_started,
                                self.finished_at, self.reported)

    def note_finished(self, engine_outputs, step: int) -> list:
        done = []
        for eco in (engine_outputs or {}).values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    done.append(o.request_id)
                    self.finished_at.setdefault(o.request_id, step)
                    self.finished[o.request_id] = {"reason": str(o.finish_reason),
                                                   "output_tokens": self.sampled[o.request_id], "step": step}
        return sorted(done)

    def state(self, record: dict) -> None:
        """The state after a step: tier 2's, the connector-visible state, and the children's."""
        from run_connector import connector_state_of  # noqa: E402
        from run_scheduler import kv_state  # noqa: E402

        record.update(kv_state(self.scheduler))
        record.update(connector_state_of(self.scheduler))
        off = record["offload"]
        off["jobs"] = sorted(self.cs._jobs)
        off["pending_push"] = self.cs.has_pending_push_work()
        off["cpu"] = cpu_state(self.cs.manager)
        record["pending_push"] = self.multi.has_pending_push_work()
        record["chosen"] = dict(sorted(self.multi._requests_to_connector.items()))

    def trace(self, vllm_root: str, scenario_path: str, names: list, steps: list) -> dict:
        from run_scheduler import provenance  # noqa: E402

        prov = provenance(vllm_root, scenario_path)
        prov["tier"] = 4
        prov["cpu_chunks"] = self.cs.manager._num_chunks
        return {"provenance": prov, "steps": steps, "finished": dict(sorted(self.finished.items())),
                "requests": dict(sorted(self.given.items())), "children": names}


def run_one(vllm_root: str, scenario_path: str) -> dict:
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    model = os.environ.get("ORACLE_MODEL", "facebook/opt-125m")

    import tests.v1.kv_connector.unit.utils as helpers  # noqa: E402

    if not os.path.abspath(helpers.__file__).startswith(os.path.abspath(vllm_root) + os.sep):
        sys.exit("tests.v1.kv_connector.unit.utils did not resolve into the vLLM checkout")
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (  # noqa: E402
        MultiKVConnectorWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (  # noqa: E402
        OffloadingWorkerMetadata,
    )
    from vllm.utils.hashing import sha256  # noqa: E402
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput  # noqa: E402
    from vllm.v1.request import RequestStatus  # noqa: E402

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import run_offloading as off  # noqa: E402
    from run_scheduler import v2_runner_env  # noqa: E402

    with open(scenario_path) as f:
        sc = json.load(f)
    cfg = sc["config"]
    conn = cfg["connector"]
    if conn["name"] != "MultiConnector":
        sys.exit(f"{scenario_path}: this runner drives MultiConnector, not {conn['name']}")
    children = (conn.get("extra_config") or {}).get("connectors") or []
    names = [c.get("kv_connector") for c in children]
    if sorted(names) != ["NixlConnector", "OffloadingConnector"]:
        sys.exit(f"{scenario_path}: the oracle drives one NixlConnector and one OffloadingConnector, got {names}")
    nixl_idx, off_idx = names.index("NixlConnector"), names.index("OffloadingConnector")
    off_extra = children[off_idx].get("kv_connector_extra_config") or {}
    if off_extra.get("spec_name", "CPUOffloadingSpec") != "CPUOffloadingSpec":
        sys.exit(f"{scenario_path}: spec_name {off_extra['spec_name']!r}: the oracle runs CPUOffloadingSpec "
                 "(oracle/run_offloading.py)")
    groups = cfg.get("kv_groups")
    if groups and len(groups) > 1 and any(r.get("invalid_blocks") for r in sc["requests"]):
        sys.exit(f"{scenario_path}: invalid_blocks beside several KV cache groups: NIXL's worker reports no "
                 "block-level failure under HMA (nixl/base_worker.py:2910-2914@5df63a72)")
    spec = cfg.get("speculative")
    async_sched = bool(cfg.get("async_scheduling", False))
    if int(cfg.get("pipeline_parallel_size", 1)) > 1:
        sys.exit(f"{scenario_path}: pipeline_parallel_size > 1 is not run beside the NIXL child "
                 "(nixl/base_worker.py:757-779@5df63a72)")
    v2 = bool(cfg.get("use_v2_model_runner", False)) or (bool(spec) and spec["method"] == "dspark")
    from run_enginecore import set_async  # noqa: E402

    restore_env = v2_runner_env(v2)
    try:
        vllm_config = off.build_config(cfg, model)
        # The CPU platform turns async scheduling off (vllm/platforms/cpu.py:329-331); tier 3's
        # set_async sets the scenario's value back and returns vLLM's own max_concurrent_batches,
        # before the Scheduler reads both (defer_block_free, scheduler.py:184-192).
        batch_queue_size = set_async(vllm_config, async_sched, v2) if async_sched else 1
        scheduler = helpers.create_scheduler(vllm_config, num_blocks=cfg["num_blocks"],
                                             kv_cache_config=off.kv_cache_config(cfg, vllm_config))
    finally:
        restore_env()
    assert scheduler.max_model_len == cfg["max_model_len"]
    assert type(scheduler.connector).__name__ == "MultiConnector"

    def worker_meta(completed_jobs):
        # build_connector_worker_meta (multi_connector.py:409-419): the tuple in child order, None
        # for a child with nothing to report.
        return MultiKVConnectorWorkerMetadata(metadata=tuple(
            OffloadingWorkerMetadata(completed_jobs=completed_jobs) if i == off_idx else None
            for i in range(len(names))))

    init_none_hash(sha256)
    hasher = get_request_block_hasher(scheduler.hash_block_size, sha256)
    workers = scheduler.connector._connectors[off_idx].connector_scheduler.config.num_workers
    worker = off.Worker({r["id"]: r for r in sc["requests"]}, cfg.get("transfer") or {}, workers,
                        KVConnectorOutput, worker_meta)
    h = Harness(scheduler, sc, helpers, hasher, off_idx, nixl_idx, worker)
    if batch_queue_size > 1:
        return run_queued(h, vllm_config, batch_queue_size, sc, scenario_path, vllm_root, names)

    num_spec = scheduler.num_spec_tokens
    accepted_plan = {r["id"]: list(r.get("accepted", [])) for r in sc["requests"]}
    verify_steps = {r["id"]: 0 for r in sc["requests"]}
    steps = []
    sampled_so_far = h.sampled
    max_steps = sc.get("max_steps", 10000)
    step = 0
    while step < max_steps:
        h.add_arrivals(step)
        if not scheduler.has_requests():
            if not h.pending:
                break
            steps.append({"step": step, "idle": True})
            step += 1
            continue
        off.order_finished(scheduler)
        out = scheduler.schedule()
        h.note_pulls(step)
        meta = out.kv_connector_metadata
        record = {
            "step": step,
            "scheduled": dict(sorted(out.num_scheduled_tokens.items())),
            "preempted": sorted(out.preempted_req_ids or []),
        }
        record["offload"] = offload_record(meta.metadata[off_idx])
        # The two workers (the transport mocks), on this step's metadata, as the batch executes.
        off_out, done = worker.execute(meta.metadata[off_idx], step)
        nixl_out, record["transport"] = h.nixl_report(step)
        kv_out = merge_outputs(KVConnectorOutput, off_out, nixl_out)

        # The GPU mock (tier 2), as oracle/run_offloading.py.
        req_ids, sampled = [], []
        spec_scheduled = {}
        for rid in out.num_scheduled_tokens:
            req = scheduler.requests[rid]
            req_ids.append(rid)
            if req.is_prefill_chunk:
                sampled.append([])
                continue
            n = 1
            drafts = out.scheduled_spec_decode_tokens.get(rid, [])
            if drafts:
                spec_scheduled[rid] = len(drafts)
                plan, i = accepted_plan[rid], verify_steps[rid]
                verify_steps[rid] = i + 1
                n = 1 + min(plan[i] if i < len(plan) else 0, len([d for d in drafts if d >= 0]))
            outp = h.by_id[rid]["output"]
            toks = []
            for _ in range(n):
                k = sampled_so_far[rid]
                if k >= len(outp):
                    break
                toks.append(helpers.EOS_TOKEN_ID if k == len(outp) - 1 else outp[k])
                sampled_so_far[rid] = k + 1
            sampled.append(toks)
        if spec_scheduled:
            record["spec_scheduled"] = dict(sorted(spec_scheduled.items()))

        record["offload"]["completed"] = done
        aborts = sorted(rid for rid, r in h.by_id.items() if r.get("abort_step") == step)
        if aborts:
            scheduler.finish_requests(aborts, RequestStatus.FINISHED_ABORTED)
        record["aborted"] = aborts
        mro = ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={}, pooler_output=[],
            kv_connector_output=kv_out)
        engine_outputs = scheduler.update_from_output(out, mro)
        if spec and out.total_num_scheduled_tokens > 0:
            from vllm.v1.outputs import DraftTokenIds  # noqa: E402

            ids = [rid for rid in out.num_scheduled_tokens if rid in scheduler.requests]
            scheduler.update_draft_token_ids(DraftTokenIds(ids, [[0] * num_spec for _ in ids]))
        record["finished"] = h.note_finished(engine_outputs, step)
        h.state(record)
        steps.append(record)
        step += 1
    return h.trace(vllm_root, scenario_path, names, steps)


def queued_gpu_class():
    """The GPU and both workers as tier 3's EngineCore executor: oracle/run_offloading.py's
    QueuedGPU, with the offloading worker run on its child's slice of the metadata and the NIXL
    transport report merged into the output the batch queue pops."""
    import run_offloading as off  # noqa: E402

    class MultiQueuedGPU(off.QueuedGPU):
        def __init__(self, h, clock, async_sched, record):
            super().__init__(h.scheduler, h.by_id, clock, h.worker, async_sched, h.sampled, record, gpu_spec)
            self.h = h

        def execute_model(self, so, non_block=False):
            from run_enginecore import Done, Lazy  # noqa: E402

            step = self.clock["step"]
            self.h.note_pulls(step)
            meta = so.kv_connector_metadata.metadata[self.h.off_idx]
            rec = self.record()
            rec["scheduled"] = dict(sorted(so.num_scheduled_tokens.items()))
            rec["preempted"] = sorted(so.preempted_req_ids or [])
            spec = {rid: len(d) for rid, d in so.scheduled_spec_decode_tokens.items() if d}
            if spec:
                rec["spec_scheduled"] = dict(sorted(spec.items()))
            rec["offload"] = offload_record(meta)
            # The offloading worker runs on its metadata as the step's batch executes
            # (kv_connector_no_forward for an empty one, kv_connector_model_runner_mixin.py:25-38,
            # 66-110@5df63a72).
            kv_out, done = self.worker.execute(meta, step)
            rec["offload"]["completed"] = done
            samples = [rid for rid in so.num_scheduled_tokens
                       if not self.scheduler.requests[rid].is_prefill_chunk]
            if so.total_num_scheduled_tokens == 0:
                return Lazy(lambda: self.output(so, [], kv_out))
            self.last = so
            self.pending = (so, samples, kv_out)
            return Done(None)

        def output(self, so, samples, kv_out):
            from vllm.v1.outputs import KVConnectorOutput  # noqa: E402

            nixl_out, rep = self.h.nixl_report(self.clock["step"])
            self.record()["transport"] = rep
            return super().output(so, samples, merge_outputs(KVConnectorOutput, kv_out, nixl_out))

    return MultiQueuedGPU


def run_queued(h, vllm_config, batch_queue_size, sc, scenario_path, vllm_root, names) -> dict:
    """A scenario with more than one batch in flight: vLLM's own step_with_batch_queue on tier 3's
    bare EngineCore (run_enginecore.py engine_core), one call per step, then post_step, as
    oracle/run_offloading.py run_queued drives the OffloadingConnector. Client aborts go through
    the EngineCore's aborts queue (core.py:780-783, 814-822)."""
    from run_enginecore import engine_core  # noqa: E402

    scheduler = h.scheduler
    clock = {"step": 0}
    steps: list[dict] = []
    current: dict = {}
    async_sched = bool(vllm_config.scheduler_config.async_scheduling)
    gpu = queued_gpu_class()(h, clock, async_sched, lambda: current)
    core = engine_core(scheduler, vllm_config, batch_queue_size, async_sched, scheduler.num_spec_tokens > 0, gpu)
    import run_offloading as off  # noqa: E402

    max_steps = sc.get("max_steps", 10000)
    step = 0
    while step < max_steps:
        clock["step"] = step
        h.add_arrivals(step)
        has_work = scheduler.has_requests() or bool(core.batch_queue)
        if not has_work and not h.pending:
            break
        current = {"step": step, "scheduled": {}, "preempted": [], "processed": False,
                   "transport": {"recving": [], "sending": [], "failed": [], "invalid_blocks": 0},
                   "offload": {"loads": [], "stores": [], "flush": [], "completed": []}}
        aborts = sorted(rid for rid, r in h.by_id.items() if r.get("abort_step") == step)
        if aborts:
            core.aborts_queue.put_nowait(aborts)
        if has_work:
            off.order_finished(scheduler)
            outputs, executed = core.step_with_batch_queue()
            core.post_step(executed)
        else:
            outputs, current["idle"] = None, True
        current["finished"] = h.note_finished(outputs, step)
        current["queue_len"] = len(core.batch_queue)
        current["aborted"] = aborts
        h.state(current)
        steps.append(current)
        step += 1
    trace = h.trace(vllm_root, scenario_path, names, steps)
    trace["provenance"]["engine"] = {"max_concurrent_batches": batch_queue_size, "async_scheduling": async_sched,
                                     "scheduler_class": type(scheduler).__name__}
    return trace


if __name__ == "__main__":
    main()
