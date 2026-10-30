# Original to blis-vllm.
"""Tier-4 oracle: vLLM's real Scheduler with its real OffloadingConnector, the worker mocked.

A scenario (oracle/README.md, "Offloading scenarios") fixes the scheduler configuration, the KV
cache groups, the OffloadingConnector's kv_connector_extra_config (cpu_bytes_to_use,
blocks_per_chunk, spec_name, offload_prompt_only, eviction_policy) and the requests, each with
its prompt, assigned output and arrival step. The connector's scheduler side runs as released:
OffloadingConnectorScheduler (vllm/distributed/kv_transfer/kv_connector/v1/offloading/
scheduler.py:563-2101@5df63a72) over vLLM's own CPUOffloadingManager and LRU policy
(vllm/v1/kv_offload/cpu/manager.py:47, cpu/policies/lru.py:16), sized from cpu_bytes_to_use by
CPUOffloadingSpec (vllm/v1/kv_offload/cpu/spec.py:94-127).

The spec is CPUOffloadingSpec, vLLM's default spec_name (vllm/v1/kv_offload/factory.py:31).
TieringOffloadingSpec (vllm/v1/kv_offload/tiering/spec.py:82) needs a /dev/shm mmap region for its
scheduler-side manager (spec.py:342-350; cpu/shared_offload_region.py:107-146), which a macOS host
has not; with no secondary tiers its TieringOffloadingManager forwards every call the connector
scheduler makes to its CPUPrimaryTierOffloadingManager, a CPUOffloadingManager with the same
policy (tiering/manager.py:396-417, 569-604, 618-661, 736-781, 843-873, 912-921@5df63a72). A
scenario stating TieringOffloadingSpec is refused here rather than run on the other spec.

What is replaced, all listed in oracle/mocks.json:
- the GPU: tier 2's mock (each request completing its known tokens samples its next assigned
  token, the last one as EOS);
- the worker side (OffloadingConnectorWorker, offloading/worker.py:40-323): the transport mock.
  It follows the worker's protocol on each step's OffloadingConnectorMetadata: jobs_to_flush are
  submitted and waited for first (handle_preemptions, :198-224), then the stores the last step
  deferred and this step's loads are submitted (start_kv_transfers, :226-238), and this step's
  stores are deferred to the next step (prepare_store_kv, :240-249). A load completes
  load_delay steps after it is submitted, a store store_delay steps after (the scenario's, per
  request or for all), a waited-for job at once. Completions reach the scheduler as vLLM's own
  OffloadingWorkerMetadata (offloading/common.py:76-104) in KVConnectorOutput's
  kv_connector_worker_meta, in job-id order, and a completed load's request in
  finished_recving (worker.py:273-309), by request ID;
- the iteration order of the scheduler's finished_req_ids: a set of strings, whose order
  depends on the interpreter's hash seed. The connector stores a finished request's last
  chunks and tells the manager it finished in that order (scheduler.py:1576-1579, 1889-1896),
  which decides CPU evictions and LRU recency; before each schedule() the runner hands the
  scheduler the same set as a set that iterates by request ID.

The engine step is EngineCore.step's (vllm/v1/engine/core.py:657-687@5df63a72): no schedule() while
has_requests() is false, which counts the connector's pending transfer jobs
(has_pending_push_work, scheduler.py:1902-1908; vllm/v1/core/sched/scheduler.py:2823-2838).
Client aborts, as tier 4's NIXL runner applies them, land between the model run and
update_from_output (core.py:678-683).

A scenario with async scheduling or pipeline parallelism ("async_scheduling",
"pipeline_parallel_size", "use_v2_model_runner", as tier 3 states them) runs vLLM's own
EngineCore.step_with_batch_queue (core.py:698-810) on tier 3's bare EngineCore
(run_enginecore.py engine_core and set_async, the environment mocks oracle/mocks.json lists for
it), as oracle/run_connector.py runs NIXL: the AsyncScheduler for async scheduling, the batch
queue at vLLM's own max_concurrent_batches. The worker mock runs on a batch's metadata when the
batch executes, and the batch's output, resolved when the queue pops it, carries the worker's
report. A drafter ("speculative": DSpark on the V2 runner, or EAGLE-3 on a configuration-only
draft) is tier 2's, with each request's acceptance plan ("accepted").

Per step it records tier 2's state, tier 4's connector-visible state, and the offloading state:
the step's load and store jobs (job id, request, CPU chunks, GPU blocks by group), the jobs
flushed, the jobs completed, the jobs still pending, and the CPU tier's chunks in LRU order
(key, chunk id, reference count), its free and evictable chunk counts.

Usage: python -I oracle/run_offloading.py <vllm checkout> <scenario.json | dir> [out dir]
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


class IdOrderedSet(set):
    """A set of request IDs that iterates in sorted order: the transport-independent order the
    harness fixes for finished_req_ids (oracle/mocks.json). Membership and size are a set's."""

    def __iter__(self):
        return iter(sorted(set.__iter__(self)))


def key_text(key: bytes) -> str:
    """An OffloadKey (block hash + 4-byte group index, vllm/v1/kv_offload/base.py:29-31) as the
    trace writes it: the hash's first 8 bytes in hex, then the group."""
    return key[:8].hex() + "/" + str(int.from_bytes(key[-4:], "big"))


class Worker:
    """The worker side (OffloadingConnectorWorker) as the transport mock (oracle/mocks.json): on a
    step's OffloadingConnectorMetadata, when the batch carrying it executes, it follows the
    worker's protocol (handle_preemptions, start_kv_transfers, prepare_store_kv,
    offloading/worker.py:198-249@5df63a72), and reports the jobs complete by then, as get_finished
    does at the end of that execute (worker.py:273-309), in that batch's output."""

    def __init__(self, by_id, transfer, workers, output_cls, meta_cls):
        self.by_id, self.transfer, self.workers = by_id, transfer, workers
        self.output_cls, self.meta_cls = output_cls, meta_cls
        self.unsubmitted: list[int] = []
        self.completes_at: dict[int, int] = {}
        self.job_req: dict[int, str] = {}
        self.load_jobs: set[int] = set()

    def delay(self, rid: str, kind: str) -> int:
        r = self.by_id.get(rid, {})
        return int(r.get(kind, self.transfer.get(kind, 1)))

    def submit(self, jid: int, step: int, kind: str) -> None:
        self.completes_at[jid] = step + self.delay(self.job_req[jid], kind)

    def execute(self, meta, step: int):
        """The worker on one batch's metadata at its execute step; returns the batch's
        KVConnectorOutput (None for none) and the jobs it reports."""
        for jid, j in meta.load_jobs.items():
            self.job_req[jid] = j.req_id
            self.load_jobs.add(jid)
        for jid, j in meta.store_jobs.items():
            self.job_req[jid] = j.req_id
        flush = set(meta.jobs_to_flush or ())
        stores_now = [jid for jid in sorted(meta.store_jobs) if jid not in flush]
        self.unsubmitted.extend(jid for jid in sorted(meta.store_jobs) if jid in flush)
        for jid in self.unsubmitted:
            self.submit(jid, step, "store_delay")
        self.unsubmitted.clear()
        for jid in flush:
            if jid in self.completes_at:
                self.completes_at[jid] = step
        for jid in sorted(meta.load_jobs):
            self.submit(jid, step, "load_delay")
        self.unsubmitted.extend(stores_now)
        # The report: jobs complete in job-id order, each by every worker; a completed load's
        # request is finished_recving, by request id.
        done = sorted(jid for jid, at in self.completes_at.items() if at <= step)
        for jid in done:
            del self.completes_at[jid]
        recving = sorted({self.job_req[j] for j in done if j in self.load_jobs})
        kv_out = None
        if done:
            kv_out = self.output_cls(
                finished_recving=dict.fromkeys(recving).keys() if recving else None,
                kv_connector_worker_meta=self.meta_cls(completed_jobs={j: self.workers for j in done}))
        return kv_out, done


def build_config(cfg: dict, model: str):
    """vLLM's own config objects for the scenario, built by their constructors (no write into a
    built object): the helper's create_vllm_config (tests/v1/kv_connector/unit/utils.py:111-180)
    fixes values the scenario states, so this builds the same objects with them."""
    from vllm.config import (  # noqa: E402
        AttentionConfig,
        CacheConfig,
        DeviceConfig,
        KVTransferConfig,
        ModelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )

    conn = cfg["connector"]
    model_config = ModelConfig(model=model, trust_remote_code=True, dtype="float16", seed=42,
                               max_model_len=cfg["max_model_len"])
    scheduler_config = SchedulerConfig(
        max_num_seqs=cfg["max_num_seqs"], max_num_batched_tokens=cfg["max_num_batched_tokens"],
        max_model_len=cfg["max_model_len"], enable_chunked_prefill=cfg.get("enable_chunked_prefill", True),
        is_encoder_decoder=False, policy=cfg.get("scheduling_policy", "fcfs"),
        long_prefill_token_threshold=cfg.get("long_prefill_token_threshold", 0),
        watermark=cfg.get("watermark", 0.0))
    # Dense retention (prefix_cache_retention_interval None), the policy core/kv models: the
    # Scheduler's KV cache config and the connector (scheduler.py:224) then agree, as at launch,
    # where both read cache_config (vllm/v1/core/kv_cache_utils.py:1684-1686@5df63a72).
    cache_config = CacheConfig(block_size=cfg["block_size"], gpu_memory_utilization=0.9, cache_dtype="auto",
                               enable_prefix_caching=cfg.get("enable_prefix_caching", True),
                               prefix_cache_retention_interval=None)
    kv_transfer_config = KVTransferConfig(kv_connector=conn["name"], kv_role=conn.get("kv_role", "kv_both"),
                                          kv_connector_extra_config=conn.get("extra_config") or {})
    from vllm.config import ParallelConfig  # noqa: E402

    # The engine's world size, the connector's num_workers (offloading/scheduler.py:335): every
    # job is pending on each worker, and the CPU tier holds a copy of a block per worker
    # (cpu/spec.py:107-121).
    kw = {"parallel_config": ParallelConfig(tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
                                            pipeline_parallel_size=cfg.get("pipeline_parallel_size", 1))}
    spec = cfg.get("speculative")
    if spec:
        args = {"method": spec["method"], "num_speculative_tokens": spec["num_speculative_tokens"]}
        if spec.get("draft"):
            args["model"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drafts", spec["draft"])
        kw["speculative_config"] = SpeculativeConfig(target_model_config=model_config,
                                                     target_parallel_config=kw["parallel_config"], **args)
    return VllmConfig(scheduler_config=scheduler_config, model_config=model_config, cache_config=cache_config,
                      kv_transfer_config=kv_transfer_config, device_config=DeviceConfig("cpu"),
                      attention_config=AttentionConfig(), **kw)


def kv_cache_config(cfg: dict, vllm_config):
    """The scenario's KV cache groups: one layer each of vLLM's own FullAttentionSpec or
    SlidingWindowSpec, of the scenario's page (kv_page: num_kv_heads, head_size, float32), each
    window tagged with extra_retained_tokens = num_prefill_lookahead_tokens - 1, as a launch tags it
    (vllm/v1/core/kv_cache_utils.py:2582-2589@5df63a72). A page that is a multiple of 16 KiB makes
    the CPU chunk size, page-aligned (cpu/spec.py:118-121), the same on a 4 KiB-page and a
    16 KiB-page host."""
    import torch  # noqa: E402
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs  # noqa: E402
    from vllm.v1.kv_cache_interface import (  # noqa: E402
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    register_all_kvcache_specs(vllm_config)
    page = cfg.get("kv_page") or {}
    common = {"block_size": cfg["block_size"], "num_kv_heads": page.get("num_kv_heads", 1),
              "head_size": page.get("head_size", 1), "dtype": torch.float32}
    extra_retained = max(0, vllm_config.num_prefill_lookahead_tokens - 1)
    specs = []
    for i, g in enumerate(cfg.get("kv_groups") or [{"kind": "full"}]):
        if g["kind"] == "full":
            layer = FullAttentionSpec(**common)
        elif g["kind"] == "sliding_window":
            layer = SlidingWindowSpec(sliding_window=g["sliding_window"], extra_retained_tokens=extra_retained, **common)
        else:
            raise ValueError(f"unknown KV cache group kind {g['kind']!r}")
        specs.append(KVCacheGroupSpec([f"layer{i}"], layer))
    return KVCacheConfig(num_blocks=cfg["num_blocks"], kv_cache_tensors=[], kv_cache_groups=specs)


def run_one(vllm_root: str, scenario_path: str) -> dict:
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    model = os.environ.get("ORACLE_MODEL", "facebook/opt-125m")

    import tests.v1.kv_connector.unit.utils as helpers  # noqa: E402

    if not os.path.abspath(helpers.__file__).startswith(os.path.abspath(vllm_root) + os.sep):
        sys.exit("tests.v1.kv_connector.unit.utils did not resolve into the vLLM checkout")
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (  # noqa: E402
        OffloadingWorkerMetadata,
    )
    from vllm.sampling_params import SamplingParams  # noqa: E402
    from vllm.utils.hashing import sha256  # noqa: E402
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
    from vllm.v1.metrics.stats import KV_FETCH_STAGES  # noqa: E402
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput  # noqa: E402
    from vllm.v1.request import Request, RequestStatus  # noqa: E402

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from run_scheduler import kv_state, provenance, v2_runner_env  # noqa: E402

    with open(scenario_path) as f:
        sc = json.load(f)
    cfg = sc["config"]
    conn = cfg["connector"]
    extra = conn.get("extra_config") or {}
    if conn["name"] != "OffloadingConnector":
        sys.exit(f"{scenario_path}: this runner drives OffloadingConnector, not {conn['name']}")
    if extra.get("spec_name", "CPUOffloadingSpec") != "CPUOffloadingSpec":
        sys.exit(f"{scenario_path}: spec_name {extra['spec_name']!r}: the oracle runs CPUOffloadingSpec (see the "
                 "module docstring)")
    spec = cfg.get("speculative")
    async_sched = bool(cfg.get("async_scheduling", False))
    pp = int(cfg.get("pipeline_parallel_size", 1))
    v2 = bool(cfg.get("use_v2_model_runner", False)) or (bool(spec) and spec["method"] == "dspark")
    if pp > 1 and async_sched and not v2:
        sys.exit(f"{scenario_path}: the V1 model runner does not support PP with async scheduling "
                 "(vllm/config/vllm.py:2970-2974@5df63a72)")
    from run_enginecore import set_async  # noqa: E402

    restore_env = v2_runner_env(v2)
    try:
        vllm_config = build_config(cfg, model)
        # The CPU platform turns async scheduling off (vllm/platforms/cpu.py:329-331); tier 3's
        # set_async sets the scenario's value back and returns vLLM's own max_concurrent_batches,
        # before the Scheduler reads both (defer_block_free, scheduler.py:184-192). The helper's
        # create_scheduler then builds the AsyncScheduler for it (utils.py:207-209).
        batch_queue_size = set_async(vllm_config, async_sched, v2) if (async_sched or pp > 1) else 1
        scheduler = helpers.create_scheduler(vllm_config, num_blocks=cfg["num_blocks"],
                                             kv_cache_config=kv_cache_config(cfg, vllm_config))
    finally:
        restore_env()
    assert scheduler.max_model_len == cfg["max_model_len"]
    cs = scheduler.connector.connector_scheduler
    manager = cs.manager
    workers = cs.config.num_workers
    num_spec = scheduler.num_spec_tokens
    accepted_plan = {r["id"]: list(r.get("accepted", [])) for r in sc["requests"]}
    verify_steps = {r["id"]: 0 for r in sc["requests"]}

    init_none_hash(sha256)
    hasher = get_request_block_hasher(scheduler.hash_block_size, sha256)
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    by_id = {r["id"]: r for r in sc["requests"]}
    sampled_so_far = {r["id"]: 0 for r in sc["requests"]}
    finished = {}
    steps = []
    max_steps = sc.get("max_steps", 10000)
    transfer = cfg.get("transfer") or {}

    worker = Worker(by_id, transfer, workers, KVConnectorOutput, OffloadingWorkerMetadata)

    def add_arrivals(upto: int) -> None:
        while pending and pending[0]["arrival_step"] <= upto:
            r = pending.pop(0)
            params = SamplingParams(max_tokens=r.get("max_tokens", 1 << 20))
            params.update_from_generation_config({}, helpers.EOS_TOKEN_ID)
            scheduler.add_request(Request(
                request_id=r["id"], prompt_token_ids=list(r["prompt"]), sampling_params=params,
                pooling_params=None, arrival_time=float(r["arrival_step"]),
                priority=r.get("priority", 0), block_hasher=hasher))

    def connector_state() -> dict:
        live = {r.request_id for r in scheduler.running}
        live |= {r.request_id for r in scheduler.waiting}
        live |= {r.request_id for r in scheduler.kv_holding_waiting}
        return {
            "waiting_for_remote_kv": sorted(r.request_id for r in scheduler.kv_holding_waiting
                                            if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS),
            "recv_landed": sorted(scheduler.finished_recving_kv_req_ids),
            "delay_freed": sorted(rid for rid, r in scheduler.requests.items() if rid not in live and r.is_finished()),
            "kv_holding_waiting": sorted(r.request_id for r in scheduler.kv_holding_waiting),
            "kv_fetch_stages": [scheduler._kv_fetch_counts[s] for s in KV_FETCH_STAGES],
            "num_requests": len(scheduler.requests),
        }

    def gpu_spec(spec) -> list:
        return [[int(b) for b in spec.block_ids], list(spec.group_sizes), list(spec.block_indices)]

    def cpu_state() -> dict:
        policy = manager._policy
        chunks = sorted(policy.chunks.items(), key=lambda kv: policy._ranks[kv[0]])
        return {
            "chunks": [[key_text(k), int(c.chunk_id), int(c.ref_cnt)] for k, c in chunks],
            "free": manager._get_num_free_chunks(),
            "evictable": manager._num_evictable_cache_chunks,
        }

    if batch_queue_size > 1:
        return run_queued(scheduler, vllm_config, batch_queue_size, sc, scenario_path, vllm_root, worker,
                          add_arrivals, connector_state, cpu_state, gpu_spec, sampled_so_far, finished)

    step = 0
    while step < max_steps:
        add_arrivals(step)
        if not scheduler.has_requests():
            if not pending:
                break
            steps.append({"step": step, "idle": True})
            step += 1
            continue
        scheduler.finished_req_ids = IdOrderedSet(scheduler.finished_req_ids)
        out = scheduler.schedule()
        meta = out.kv_connector_metadata
        record = {
            "step": step,
            "scheduled": dict(sorted(out.num_scheduled_tokens.items())),
            "preempted": sorted(out.preempted_req_ids or []),
        }
        off = {
            "loads": [[jid, j.req_id, [int(c) for c in j.src_spec.block_ids], gpu_spec(j.dst_spec)]
                      for jid, j in sorted(meta.load_jobs.items())],
            "stores": [[jid, j.req_id, [int(c) for c in j.dst_spec.block_ids], gpu_spec(j.src_spec)]
                       for jid, j in sorted(meta.store_jobs.items())],
            "flush": sorted(meta.jobs_to_flush or []),
        }
        # The worker (the transport mock), on this step's metadata, as the batch executes.
        kv_out, done = worker.execute(meta, step)

        # The GPU mock.
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
            outp = by_id[rid]["output"]
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

        off["completed"] = done
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
            from vllm.v1.outputs import DraftTokenIds  # noqa: E402

            ids = [rid for rid in out.num_scheduled_tokens if rid in scheduler.requests]
            scheduler.update_draft_token_ids(DraftTokenIds(ids, [[0] * num_spec for _ in ids]))
        fin = []
        for eco in engine_outputs.values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    fin.append(o.request_id)
                    finished[o.request_id] = {"reason": str(o.finish_reason),
                                              "output_tokens": sampled_so_far[o.request_id], "step": step}
        record["finished"] = sorted(fin)
        record.update(kv_state(scheduler))
        record.update(connector_state())
        off["jobs"] = sorted(cs._jobs)
        off["pending_push"] = cs.has_pending_push_work()
        off["cpu"] = cpu_state()
        record["offload"] = off
        steps.append(record)
        step += 1

    prov = provenance(vllm_root, scenario_path)
    prov["tier"] = 4
    prov["cpu_chunks"] = manager._num_chunks
    return {"provenance": prov, "steps": steps, "finished": dict(sorted(finished.items()))}


class QueuedGPU:
    """The GPU and worker mocks as tier 3's EngineCore executor (oracle/mocks.json): when a batch
    executes, which of its requests sample is read from the scheduler's own is_prefill_chunk, and
    the worker mock runs on the batch's OffloadingConnectorMetadata (Worker.execute); the batch's
    output (the scenario's tokens, the last one as EOS, plus the drafts the acceptance plan
    accepts, every async placeholder a real draft; and the worker's KVConnectorOutput) is resolved
    when the batch queue pops it."""

    def __init__(self, scheduler, by_id, clock, worker, async_sched, sampled, record, gpu_spec):
        self.scheduler, self.by_id, self.clock, self.worker = scheduler, by_id, clock, worker
        self.async_sched, self.sampled, self.record, self.gpu_spec = async_sched, sampled, record, gpu_spec
        self.verify = {rid: 0 for rid in by_id}
        self.last = None
        self.pending = None

    def executor(self):
        return self

    def execute_model(self, so, non_block=False):
        from run_enginecore import Done, Lazy  # noqa: E402

        meta = so.kv_connector_metadata
        rec = self.record()
        rec["scheduled"] = dict(sorted(so.num_scheduled_tokens.items()))
        rec["preempted"] = sorted(so.preempted_req_ids or [])
        spec = {rid: len(d) for rid, d in so.scheduled_spec_decode_tokens.items() if d}
        if spec:
            rec["spec_scheduled"] = dict(sorted(spec.items()))
        rec["offload"] = {
            "loads": [[jid, j.req_id, [int(c) for c in j.src_spec.block_ids], self.gpu_spec(j.dst_spec)]
                      for jid, j in sorted(meta.load_jobs.items())],
            "stores": [[jid, j.req_id, [int(c) for c in j.dst_spec.block_ids], self.gpu_spec(j.src_spec)]
                       for jid, j in sorted(meta.store_jobs.items())],
            "flush": sorted(meta.jobs_to_flush or []),
        }
        # The worker runs on the metadata as the step's batch executes (kv_connector_no_forward
        # for an empty one, kv_connector_model_runner_mixin.py:25-38, 66-110@5df63a72).
        kv_out, done = self.worker.execute(meta, self.clock["step"])
        rec["offload"]["completed"] = done
        samples = [rid for rid in so.num_scheduled_tokens if not self.scheduler.requests[rid].is_prefill_chunk]
        if so.total_num_scheduled_tokens == 0:
            return Lazy(lambda: self.output(so, [], kv_out))
        self.last = so
        self.pending = (so, samples, kv_out)
        return Done(None)

    def sample_tokens(self, grammar_output, non_block=False):
        from run_enginecore import Lazy  # noqa: E402

        so, samples, kv_out = self.pending
        return Lazy(lambda: self.output(so, samples, kv_out))

    def take_draft_token_ids(self):
        # The drafter after a step that ran, without async scheduling (core.py:689-696), as tier
        # 3's mock proposes: num_speculative_tokens drafts for every request the last executed
        # batch ran. Only their count reaches a decision.
        from vllm.v1.outputs import DraftTokenIds  # noqa: E402

        if self.last is None:
            return None
        ids = [rid for rid in self.last.num_scheduled_tokens if rid in self.scheduler.requests]
        k = self.scheduler.num_spec_tokens
        return DraftTokenIds(ids, [[0] * k for _ in ids])

    def output(self, so, samples, kv_out):
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
        self.record()["processed"] = True
        return ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=tokens, logprobs=None, prompt_logprobs_dict={}, pooler_output=[],
            kv_connector_output=kv_out)


def run_queued(scheduler, vllm_config, batch_queue_size, sc, scenario_path, vllm_root, worker, add_arrivals,
               connector_state, cpu_state, gpu_spec, sampled, finished) -> dict:
    """A scenario with more than one batch in flight: vLLM's own step_with_batch_queue on tier 3's
    bare EngineCore (run_enginecore.py engine_core), one call per step, then post_step, as
    oracle/run_connector.py run_queued drives NIXL. Client aborts go through the EngineCore's aborts
    queue (core.py:780-783, 814-822)."""
    from run_enginecore import engine_core  # noqa: E402
    from run_scheduler import kv_state, provenance  # noqa: E402

    cs = scheduler.connector.connector_scheduler
    by_id = {r["id"]: r for r in sc["requests"]}
    clock = {"step": 0}
    steps: list[dict] = []
    current: dict = {}
    async_sched = bool(vllm_config.scheduler_config.async_scheduling)
    gpu = QueuedGPU(scheduler, by_id, clock, worker, async_sched, sampled, lambda: current, gpu_spec)
    core = engine_core(scheduler, vllm_config, batch_queue_size, async_sched, scheduler.num_spec_tokens > 0, gpu)
    pending = [r for r in sc["requests"]]
    max_steps = sc.get("max_steps", 10000)
    step = 0
    while step < max_steps:
        clock["step"] = step
        add_arrivals(step)
        pending = [r for r in pending if r["arrival_step"] > step]
        has_work = scheduler.has_requests() or bool(core.batch_queue)
        if not has_work and not pending:
            break
        current = {"step": step, "scheduled": {}, "preempted": [], "processed": False,
                   "offload": {"loads": [], "stores": [], "flush": [], "completed": []}}
        aborts = sorted(rid for rid, r in by_id.items() if r.get("abort_step") == step)
        if aborts:
            core.aborts_queue.put_nowait(aborts)
        if has_work:
            scheduler.finished_req_ids = IdOrderedSet(scheduler.finished_req_ids)
            outputs, executed = core.step_with_batch_queue()
            core.post_step(executed)
        else:
            outputs, current["idle"] = None, True
        done = []
        for eco in (outputs or {}).values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    done.append(o.request_id)
                    finished[o.request_id] = {"reason": str(o.finish_reason),
                                              "output_tokens": sampled[o.request_id], "step": step}
        current["finished"] = sorted(done)
        current["queue_len"] = len(core.batch_queue)
        current["aborted"] = aborts
        current.update(kv_state(scheduler))
        current.update(connector_state())
        current["offload"]["jobs"] = sorted(cs._jobs)
        current["offload"]["pending_push"] = cs.has_pending_push_work()
        current["offload"]["cpu"] = cpu_state()
        steps.append(current)
        step += 1
    prov = provenance(vllm_root, scenario_path)
    prov["tier"] = 4
    prov["cpu_chunks"] = cs.manager._num_chunks
    prov["engine"] = {"max_concurrent_batches": batch_queue_size, "async_scheduling": async_sched,
                      "scheduler_class": type(scheduler).__name__}
    return {"provenance": prov, "steps": steps, "finished": dict(sorted(finished.items()))}


if __name__ == "__main__":
    main()
