# Original to blis-vllm.
"""Tier-4 oracle for the OffloadingConnector beside Mamba KV cache groups.

vLLM's real Scheduler with its real OffloadingConnector (offloading/scheduler.py, v0.34.0), the
worker mocked, over KV cache groups of full attention, sliding windows and Mamba "align" state
(MambaSpec). It is oracle/run_offloading.py's synchronous loop, with two differences that the
release it runs against needs:

- a scenario's "kv_groups" may hold {"kind": "mamba"} entries, built as vLLM's own MambaSpec with
  mamba_cache_mode "align" (the scenario's "mamba": num_speculative_blocks,
  num_prefill_checkpoint_blocks, prefill_checkpoint_alignment), and the CacheConfig the helper
  would build is built here with mamba_cache_mode and mamba_block_size, so VllmConfig's own
  validators run on them;
- ModelRunnerOutput is built without req_id_to_index, which v0.34.0 no longer takes.

It writes nothing into a vLLM object: every replacement it makes is one oracle/mocks.json lists
for run_offloading.py (the GPU, the worker's transport, the finished_req_ids order, which it takes
from run_offloading.order_finished). A scenario with async scheduling, pipeline parallelism or a
drafter is refused: oracle/run_offloading.py runs those on the release it was written for.

Usage: python -I oracle/run_offloading_mamba.py <vllm checkout> <scenario.json | dir> [out dir]
Run it from a directory other than oracle/ (torch names a module by a relative path).
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
                     "a golden-reviewed: <trace>: <reason> trailer for each trace the commit changes")
        for name in sorted(f for f in os.listdir(path)
                           if f.endswith(".json") and not f.endswith(".trace.json")):
            trace = run_one(vllm_root, os.path.join(path, name))
            with open(os.path.join(out_dir, name[:-5] + ".trace.json"), "w") as f:
                json.dump(trace, f, indent=1, sort_keys=True)
        return
    print(json.dumps(run_one(vllm_root, path), indent=1, sort_keys=True))


def build_config(cfg: dict, model: str, mamba: dict | None):
    """vLLM's own config objects for the scenario, built by their constructors."""
    from vllm.config import (  # noqa: E402
        AttentionConfig,
        CacheConfig,
        DeviceConfig,
        KVTransferConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
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
    # A model with Mamba groups keeps vLLM's default retention (0: the replay boundaries' states only);
    # a scenario may state another multiple of the block size.
    retention = None
    cache_kw = {}
    if mamba is not None:
        cache_kw = {"mamba_cache_mode": "align", "mamba_block_size": cfg["block_size"]}
        retention = cfg.get("prefix_cache_retention_interval", 0)
    cache_config = CacheConfig(block_size=cfg["block_size"], gpu_memory_utilization=0.9, cache_dtype="auto",
                               enable_prefix_caching=cfg.get("enable_prefix_caching", True),
                               prefix_cache_retention_interval=retention, **cache_kw)
    kv_transfer_config = KVTransferConfig(kv_connector=conn["name"], kv_role=conn.get("kv_role", "kv_both"),
                                          kv_connector_extra_config=conn.get("extra_config") or {},
                                          kv_load_failure_policy=conn.get("kv_load_failure_policy", "fail"))
    parallel = ParallelConfig(tensor_parallel_size=cfg.get("tensor_parallel_size", 1))
    return VllmConfig(scheduler_config=scheduler_config, model_config=model_config, cache_config=cache_config,
                      kv_transfer_config=kv_transfer_config, device_config=DeviceConfig("cpu"),
                      attention_config=AttentionConfig(), parallel_config=parallel)


def kv_cache_config(cfg: dict, num_blocks: int, retention):
    """The scenario's KV cache groups, one layer each: vLLM's own FullAttentionSpec,
    SlidingWindowSpec or MambaSpec of the scenario's page (kv_page: num_kv_heads, head_size,
    float32). A Mamba group's page_size_padded is that same page, so every group's CPU chunk is
    one page. A page that is a multiple of 16 KiB makes the CPU chunk size the same on a 4 KiB-page
    and a 16 KiB-page host (cpu/spec.py)."""
    import torch  # noqa: E402
    from vllm.v1.kv_cache_interface import (  # noqa: E402
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    page = cfg.get("kv_page") or {}
    heads, head = page.get("num_kv_heads", 1), page.get("head_size", 1)
    common = {"block_size": cfg["block_size"], "num_kv_heads": heads, "head_size": head, "dtype": torch.float32}
    mamba = cfg.get("mamba") or {}
    specs = []
    for i, g in enumerate(cfg.get("kv_groups") or [{"kind": "full"}]):
        if g["kind"] == "full":
            layer = FullAttentionSpec(**common)
        elif g["kind"] == "sliding_window":
            layer = SlidingWindowSpec(sliding_window=g["sliding_window"], **common)
        elif g["kind"] == "mamba":
            layer = MambaSpec(
                block_size=cfg["block_size"], shapes=((mamba.get("state_bytes", 1024),),), dtypes=(torch.uint8,),
                page_size_padded=2 * cfg["block_size"] * heads * head * 4, mamba_cache_mode="align",
                num_speculative_blocks=mamba.get("num_speculative_blocks", 0),
                num_prefill_checkpoint_blocks=mamba.get("num_prefill_checkpoint_blocks", 0),
                prefill_checkpoint_alignment=mamba.get("prefill_checkpoint_alignment"))
        else:
            raise ValueError(f"unknown KV cache group kind {g['kind']!r}")
        specs.append(KVCacheGroupSpec([f"layer{i}"], layer))
    # The planner copies the CacheConfig's retention into the KVCacheConfig the Scheduler reads
    # (kv_cache_planning.py:777-779), as the connector reads the CacheConfig's own.
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=specs,
                         prefix_cache_retention_interval=retention)


def provenance(vllm_root: str, scenario_path: str) -> dict:
    import hashlib  # noqa: E402
    import platform  # noqa: E402
    import subprocess  # noqa: E402

    import run_scheduler as t2  # noqa: E402

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", vllm_root, *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    here = os.path.dirname(os.path.abspath(__file__))
    with open(scenario_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(here, "mocks.json"), "rb") as f:
        mocks = json.load(f)
    return {
        "vllm_commit": git("rev-parse", "HEAD"),
        "vllm_version": "0.34.0",
        "scenario_sha256": digest,
        "mock_version": mocks["version"],
        "model": "facebook/opt-125m",
        "model_revision": os.environ.get("ORACLE_MODEL_REVISION", "unpinned"),
        "mocks": [m["symbol"] for m in mocks["mocks"]],
        "python": sys.version.split()[0],
        "platform": platform.system().lower() + "-" + platform.machine(),
        "requirements_lock_sha256": t2.committed_lock_sha256(),
        "blis_vllm_commit": t2.blis_vllm_commit(),
        "oracle_inputs": t2.oracle_inputs(scenario_path),
        "seed": t2.scenario_seed(scenario_path),
        "tier": 4,
    }


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
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, resolve_kv_cache_block_sizes  # noqa: E402
    from vllm.v1.metrics.stats import KV_FETCH_STAGES  # noqa: E402
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput  # noqa: E402
    from vllm.v1.request import Request  # noqa: E402

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from run_offloading import Worker, key_text, order_finished  # noqa: E402
    from run_scheduler import kv_state  # noqa: E402

    with open(scenario_path) as f:
        sc = json.load(f)
    cfg = sc["config"]
    conn = cfg["connector"]
    extra = conn.get("extra_config") or {}
    if conn["name"] != "OffloadingConnector" or extra.get("spec_name", "CPUOffloadingSpec") != "CPUOffloadingSpec":
        sys.exit(f"{scenario_path}: this runner drives OffloadingConnector over CPUOffloadingSpec")
    if cfg.get("async_scheduling") or cfg.get("pipeline_parallel_size", 1) > 1 or cfg.get("speculative"):
        sys.exit(f"{scenario_path}: async scheduling, pipeline parallelism and drafters are run_offloading.py's")
    has_mamba = any(g["kind"] == "mamba" for g in cfg.get("kv_groups") or [])
    vllm_config = build_config(cfg, model, cfg.get("mamba") if has_mamba else None)
    kv_config = kv_cache_config(cfg, cfg["num_blocks"], vllm_config.cache_config.prefix_cache_retention_interval)
    # Groups of one block size: the helper's block_size is the scheduler's and the hash block size.
    sched_bs, hash_bs = resolve_kv_cache_block_sizes(kv_config, vllm_config)
    scheduler = helpers.create_scheduler(vllm_config, num_blocks=cfg["num_blocks"], kv_cache_config=kv_config,
                                         hash_block_size=hash_bs)
    assert scheduler.max_model_len == cfg["max_model_len"] and scheduler.block_size == sched_bs
    cs = scheduler.connector.connector_scheduler
    manager = cs.manager
    workers = cs.config.num_workers
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    by_id = {r["id"]: r for r in sc["requests"]}
    sampled_so_far = {r["id"]: 0 for r in sc["requests"]}
    finished = {}
    steps = []
    max_steps = sc.get("max_steps", 10000)
    transfer = cfg.get("transfer") or {}

    init_none_hash(sha256)
    hasher = get_request_block_hasher(scheduler.hash_block_size, sha256,
                                      scheduler.vllm_config.num_hash_lookahead_tokens)
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

    step = 0
    while step < max_steps:
        add_arrivals(step)
        if not scheduler.has_requests():
            if not pending:
                break
            steps.append({"step": step, "idle": True})
            step += 1
            continue
        order_finished(scheduler)
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
        kv_out, done = worker.execute(meta, step)
        req_ids, sampled = [], []
        for rid in out.num_scheduled_tokens:
            req = scheduler.requests[rid]
            req_ids.append(rid)
            if req.is_prefill_chunk:
                sampled.append([])
                continue
            outp = by_id[rid]["output"]
            k = sampled_so_far[rid]
            toks = []
            if k < len(outp):
                toks.append(helpers.EOS_TOKEN_ID if k == len(outp) - 1 else outp[k])
                sampled_so_far[rid] = k + 1
            sampled.append(toks)
        off["completed"] = done
        aborts = sorted(rid for rid, r in by_id.items() if r.get("abort_step") == step)
        if aborts:
            from vllm.v1.request import RequestStatus  # noqa: E402

            scheduler.finish_requests(aborts, RequestStatus.FINISHED_ABORTED)
        record["aborted"] = aborts
        mro = ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=sampled, logprobs=None,
                                prompt_logprobs_dict={}, pooler_output=[], kv_connector_output=kv_out)
        engine_outputs = scheduler.update_from_output(out, mro)
        fin = []
        for eco in engine_outputs.values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    fin.append(o.request_id)
                    finished[o.request_id] = {"reason": str(o.finish_reason),
                                              "output_tokens": sampled_so_far[o.request_id], "step": step}
        record["finished"] = sorted(fin)
        record.update(kv_state(scheduler))
        live = {r.request_id for r in scheduler.running} | {r.request_id for r in scheduler.waiting}
        live |= {r.request_id for r in scheduler.kv_holding_waiting}
        record.update({
            "waiting_for_remote_kv": sorted(r.request_id for r in scheduler.kv_holding_waiting
                                            if r.status.name == "WAITING_FOR_REMOTE_KVS"),
            "recv_landed": sorted(scheduler.finished_recving_kv_req_ids),
            "delay_freed": sorted(rid for rid, r in scheduler.requests.items() if rid not in live and r.is_finished()),
            "kv_holding_waiting": sorted(r.request_id for r in scheduler.kv_holding_waiting),
            "kv_fetch_stages": [scheduler._kv_fetch_counts[s] for s in KV_FETCH_STAGES],
            "num_requests": len(scheduler.requests),
        })
        off["jobs"] = sorted(cs._jobs)
        off["pending_push"] = cs.has_pending_push_work()
        off["cpu"] = cpu_state()
        record["offload"] = off
        steps.append(record)
        step += 1

    prov = provenance(vllm_root, scenario_path)
    prov["cpu_chunks"] = manager._num_chunks
    return {"provenance": prov, "steps": steps, "finished": dict(sorted(finished.items()))}


if __name__ == "__main__":
    main()
