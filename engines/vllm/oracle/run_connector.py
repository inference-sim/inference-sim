# Original to blis-vllm.
"""Tier-4 oracle: vLLM's real Scheduler with its real NIXL connector, transport mocked.

A scenario (oracle/README.md, "Connector scenarios") fixes the scheduler configuration, the
connector (NixlConnector, the pull scheduler NIXL's default selects), the engine's KV role and
the requests: each with its prompt, assigned output and arrival step, and, for a P/D request,
which side of the transfer this engine is:

  - "remote_prefill" (this engine is the decoder, kv_consumer): the prompt's KV is pulled from
    a remote prefiller. The scenario gives how many steps after the pull starts the worker's
    output reports it done ("recv_delay"; 0 is the step that started it, and a request aborted
    meanwhile is reported too), and optionally that it failed as a whole ("recv_failed") or
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
    # DSpark runs only on the V2 model runner (vllm/config/vllm.py:2976-2979@5df63a72): the
    # environment of a GPU host selects it (run_scheduler.py v2_runner_env, oracle/mocks.json).
    restore_env = v2_runner_env(bool(spec) and spec["method"] == "dspark")
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
    try:
        scheduler = helpers.create_scheduler(vllm_config, num_blocks=cfg["num_blocks"],
                                             kv_cache_config=kv_cache_config(cfg, vllm_config))
    finally:
        restore_env()
    assert scheduler.max_model_len == cfg["max_model_len"], "the Scheduler does not read the scenario's max_model_len"
    num_spec = scheduler.num_spec_tokens
    accepted_plan = {r["id"]: list(r.get("accepted", [])) for r in sc["requests"]}
    verify_steps = {r["id"]: 0 for r in sc["requests"]}

    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256)
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    by_id = {r["id"]: r for r in sc["requests"]}
    sampled_so_far = {r["id"]: 0 for r in sc["requests"]}
    finished_at = {}
    load_started = {}
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
            # The KV-transfer parameters a P/D proxy hands each side (as create_request builds
            # them, tests/v1/kv_connector/unit/utils.py:246-264@5df63a72).
            if r.get("remote_prefill"):
                req.kv_transfer_params = dict(
                    do_remote_prefill=True, do_remote_decode=False, remote_engine_id="prefiller",
                    remote_request_id="prefill-" + r["id"],
                    remote_block_ids=list(range(-(-len(r["prompt"]) // block_size))),
                    remote_host="prefiller", remote_port=1234, tp_size=1)
            elif r.get("remote_decode"):
                req.kv_transfer_params = dict(do_remote_prefill=False, do_remote_decode=True)
            scheduler.add_request(req)

    def connector_state() -> dict:
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
        recving, sending, failed, invalid = set(), set(), set(), set()
        for rid, r in by_id.items():
            req = scheduler.requests.get(rid)
            if req is None:
                continue
            # A pull lands at its step whether or not the client aborted the request since:
            # the worker reports the transfer when it completes, and the scheduler then frees
            # the blocks it held for it (scheduler.py:3222-3231).
            if r.get("remote_prefill") and rid in load_started and \
                    step == load_started[rid] + r.get("recv_delay", 0) and \
                    (req.status == RequestStatus.WAITING_FOR_REMOTE_KVS or req.is_finished()):
                recving.add(rid)
                if r.get("recv_failed"):
                    failed.add(rid)
                # The blocks the pull fills, per KV cache group: the request's unhashed blocks of
                # each group (get_unhashed_block_ids_all_groups, kv_cache_manager.py:114-124, which
                # NIXL hands its worker, pull_scheduler.py:149-156@5df63a72). A block-level failure
                # names blocks of the one group (refused above beside several).
                pulled = [[b.block_id for b in m.req_to_blocks.get(rid, []) if b.block_hash is None and not b.is_null]
                          for m in scheduler.kv_cache_manager.coordinator.single_type_managers]
                for i in r.get("invalid_blocks", []):
                    if i < len(pulled[0]):
                        invalid.add(pulled[0][i])
            if r.get("remote_decode") and rid in finished_at and req.is_finished() and \
                    step == finished_at[rid] + max(r.get("send_delay", 1), 1):
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
        record["transport"] = {"recving": sorted(recving), "sending": sorted(sending),
                               "failed": sorted(failed), "invalid_blocks": len(invalid)}
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
