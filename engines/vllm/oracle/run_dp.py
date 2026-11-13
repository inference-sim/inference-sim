# Original to blis-vllm.
"""Tier-3 oracle, data parallel: vLLM's DP engine-core loop and internal load balancer.

A scenario runs D data-parallel ranks, each one vLLM's real Scheduler (tests/v1/core/utils.py
create_scheduler). The front end routes each request with vLLM's real
DPLBAsyncMPClient.get_core_engine_for_request (vllm/v1/engine/core_client.py:1642-1697@5df63a72)
on a client object holding exactly the state that method reads, as vLLM's own tests build it
(tests/v1/engine/test_engine_core_client.py:208-222@5df63a72). Its per-engine
[waiting, running, kv_cache_usage] snapshot is published every stats_interval steps from the
ranks' own counts (the coordinator's 100 ms publication, vllm/v1/engine/coordinator.py:154,
as a step count), and a finished request leaves the client's in-flight count
(process_engine_outputs, core_client.py:1703-1710).

Every rank steps in lockstep, each step being one iteration of DPEngineCoreProc.run_busy_loop
(vllm/v1/engine/core.py:2328-2405@5df63a72): a rank with work runs EngineCore.step() with
schedule(throttle_prefills=DPEngineCoreProc._should_throttle_prefills) (:2318-2325); a rank
with nothing that ran but the wave still running executes a dummy batch; then every rank calls
the real DPEngineCoreProc._has_global_unfinished_reqs (:2407-2428), whose all-reduce
(ParallelConfig.sync_dp_state) is answered with the OR of every rank's local unfinished flag,
the cross-rank sum a process group would compute. When the wave ends the step counter resets
and the wave number advances; a request arriving at an idle wave starts the next one, as the
FIRST_REQ / START_DP_WAVE path does (core_client.py:1590-1593; core.py:2281-2298).

Replaced, all listed in oracle/mocks.json: the GPU (the shared mock, as in run_scheduler.py),
the DP process group's all-reduce (sync_dp_state), and the coordinator's stats transport. The
throttle cadence, the wave sync and the load balancer's choice run as released. The loop that
steps them is NOT vLLM's: run_busy_loop blocks on ZMQ queues and a process group, so this file
transcribes its per-step decisions (the dummy batch, the wave's end and counter reset, and a
request starting the next wave, a simplification of the coordinator's START_DP_WAVE round trip).
oracle/mocks.json lists that transcription, and a trace's dummy, waves, running and step_counter
fields are evidence only that blis-vllm matches it.

With async_scheduling (oracle/scenarios-dp-async), each rank serves as a dashboard DP
deployment does by default (vllm/config/vllm.py:1585-1644@5df63a72): its scheduler is the
AsyncScheduler and its step is its own EngineCore.step_with_batch_queue
(vllm/v1/engine/core.py:698-812@5df63a72), unmodified, with a queue of max_concurrent_batches,
the step_fn __init__ picks (core.py:242-244) and _process_engine_step calls with post_step
(:1603-1619). Each rank's DPEngineCoreProc then also holds what step_with_batch_queue reads, and
its GPU is run_enginecore.py's mock executor; both are oracle/run_enginecore.py's own
(engine_core, MockExecutor), listed in oracle/mocks.json. A rank steps while it has work
(has_work, core.py:1534-1540: requests, batches queued, or the wave running); its step's
model_executed is what the dummy batch reads.

With a "connector" in the scenario's config (oracle/scenarios-dp-nixl), each rank also holds vLLM's
real NixlConnector scheduler side, built as oracle/run_connector.py builds one (tests/v1/kv_connector/
unit/utils.py create_vllm_config, create_scheduler: one config, one Scheduler and one connector per
rank), with the rank's own kv_role. A request names the side of the P/D transfer this rank is
("remote_prefill" or "remote_decode") and, optionally, the rank the proxy sends it to ("rank": the
request's data_parallel_rank, which get_core_engine_for_request obeys without consulting the load
counts, core_client.py:1647-1650); with no "rank" the balancer chooses. The transfers are the
connector oracle's transport mock (run_connector.py transport_report), evaluated per rank on that
rank's own scheduler: a pull lands "recv_delay" steps after the rank's step that parked it, a held
producer request is released "send_delay" steps after it finished. A rank with a request waiting for
remote KV has unfinished requests (Scheduler.get_num_unfinished_requests counts it), so its wave
stays running; the rank steps (has_requests), schedules nothing and runs a dummy batch. Each rank's
record then adds what run_connector.py records. Without a "connector" nothing in this paragraph runs,
and the traces of the scenarios above are unchanged.

Usage: python -I oracle/run_dp.py <vllm checkout> <scenario.json | dir> [out dir]
"""

import json
import os
import sys
from collections import Counter
from types import SimpleNamespace


def main() -> None:
    vllm_root, path = sys.argv[1], sys.argv[2]
    if os.path.isdir(path):
        out_dir = sys.argv[3] if len(sys.argv) > 3 else path
        if "testdata/golden" in os.path.abspath(out_dir).replace(os.sep, "/") and \
                os.environ.get("BLIS_VLLM_ALLOW_GOLDEN_UPDATE") != "1":
            sys.exit("refusing to write golden traces: set BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1, and add "
                     "a golden-reviewed: <trace>: <reason> trailer for each trace the commit changes")
        for name in sorted(f for f in os.listdir(path)
                           if f.endswith(".json") and not f.endswith((".trace.json", ".metrics.json"))):
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
    from vllm.config import ParallelConfig  # noqa: E402
    from vllm.sampling_params import SamplingParams  # noqa: E402
    from vllm.utils.hashing import sha256  # noqa: E402
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
    from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
    from vllm.v1.engine.core import DPEngineCoreProc  # noqa: E402
    from vllm.v1.engine.core_client import DPLBAsyncMPClient  # noqa: E402
    from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
    from vllm.v1.request import Request  # noqa: E402

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from run_enginecore import MockGPU, engine_core, restore_async, set_async  # noqa: E402
    from run_scheduler import kv_state, provenance  # noqa: E402

    with open(scenario_path) as f:
        sc = json.load(f)
    cfg = sc["config"]
    dp = int(cfg["data_parallel_size"])
    interval = int(cfg.get("prefill_schedule_interval", 1))
    sync_interval = int(cfg.get("dp_sync_interval", 16))
    stats_interval = int(cfg.get("stats_interval", 1))
    # The coordinator's counts arrive late: what it publishes at step t is what the ranks
    # reported stats_lag steps earlier (a transport delay, oracle/mocks.json), so a snapshot
    # can miss requests the client has routed since, which the in-flight floor covers.
    stats_lag = int(cfg.get("stats_lag", 0))
    block_size = cfg["block_size"]
    # Async scheduling: each rank runs the AsyncScheduler and its own step_with_batch_queue.
    async_sched = bool(cfg.get("async_scheduling", False))

    connector = cfg.get("connector")
    roles: list[str] = []
    configs: list = []
    queue_sizes: list[int] = []
    ranks = []
    if connector:
        import tests.v1.kv_connector.unit.utils as chelpers  # noqa: E402
        from run_connector import add_kv_params, apply_scenario_config, connector_state_of  # noqa: E402
        from run_connector import QueuedGPU, transport_report  # noqa: E402
        from vllm.v1.request import RequestStatus  # noqa: E402

        if not os.path.abspath(chelpers.__file__).startswith(os.path.abspath(vllm_root) + os.sep):
            sys.exit("tests.v1.kv_connector.unit.utils did not resolve into the vLLM checkout")
        if connector["name"] != "NixlConnector":
            sys.exit(f"{scenario_path}: this runner drives NixlConnector")
        roles = list(connector.get("kv_roles") or [connector.get("kv_role", "kv_consumer")] * dp)
        if len(roles) != dp or any(r not in ("kv_producer", "kv_consumer", "kv_both") for r in roles):
            sys.exit(f"{scenario_path}: connector.kv_roles needs one of kv_producer, kv_consumer, kv_both per rank")
        for role in roles:
            vllm_config = chelpers.create_vllm_config(
                model=model, max_num_seqs=cfg["max_num_seqs"], max_num_batched_tokens=cfg["max_num_batched_tokens"],
                block_size=block_size, max_model_len=cfg["max_model_len"], kv_connector=connector["name"],
                kv_role=role, kv_load_failure_policy=connector.get("kv_load_failure_policy", "fail"),
                kv_connector_extra_config=connector.get("extra_config") or {},
                enable_chunked_prefill=cfg.get("enable_chunked_prefill", True))
            apply_scenario_config(vllm_config, cfg)
            # The CPU platform turns async scheduling off; set_async puts the scenario's value
            # back before the Scheduler reads it (run_connector.py).
            queue_sizes.append(set_async(vllm_config, async_sched, False) if async_sched else 1)
            ranks.append(chelpers.create_scheduler(vllm_config, num_blocks=cfg["num_blocks"]))
            configs.append(vllm_config)
        assert all(isinstance(s, AsyncScheduler) == async_sched for s in ranks)
    else:
        for _ in range(dp):
            ranks.append(helpers.create_scheduler(
                model=model,
                max_num_seqs=cfg["max_num_seqs"],
                max_num_batched_tokens=cfg["max_num_batched_tokens"],
                long_prefill_token_threshold=cfg.get("long_prefill_token_threshold", 0),
                enable_prefix_caching=cfg.get("enable_prefix_caching", False),
                num_blocks=cfg["num_blocks"],
                block_size=block_size,
                max_model_len=cfg.get("max_model_len"),
                scheduling_policy=cfg.get("scheduling_policy", "fcfs"),
                **({"async_scheduling": True, "scheduler_cls": AsyncScheduler} if async_sched else {}),
            ))

    # Each rank's DP engine core: exactly the state _has_global_unfinished_reqs and
    # _should_throttle_prefills read, as vLLM's own tests build it
    # (tests/v1/engine/test_engine_core.py:641-657@5df63a72).
    cores = []
    for i in range(dp):
        core = object.__new__(DPEngineCoreProc)
        core.dp_group = object()
        core.dp_sync_interval = sync_interval
        core.prefill_schedule_interval = interval
        core.step_counter = 0
        core.current_wave = 0
        core.pending_pause = False
        core.ignore_start_dp_wave = False
        core.engines_running = False
        cores.append(core)

    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256)
    by_id = {r["id"]: r for r in sc["requests"]}
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    sampled = {r["id"]: 0 for r in sc["requests"]}

    # With async scheduling each rank's DP core also holds what step_with_batch_queue reads, and
    # its GPU is the shared mock (oracle/run_enginecore.py), one per rank.
    batches: list[list[dict]] = [[] for _ in range(dp)]
    # The connector's transport and the client's aborts, per rank (run_connector.py's own state).
    clock = {"step": 0}
    load_started: dict[str, int] = {}
    finished_at: dict[str, int] = {}
    reported: set = set()
    current: list[dict] = [{} for _ in range(dp)]
    gpus: list = []
    if async_sched and connector:
        for i, core in enumerate(cores):
            assert queue_sizes[i] > 1
            gpus.append(QueuedGPU(ranks[i], by_id, clock, load_started, finished_at, True,
                                  lambda i=i: current[i]))
            engine_core(ranks[i], configs[i], queue_sizes[i], True, False, gpus[i], core=core)
    elif async_sched:
        for i, core in enumerate(cores):
            vllm_config, queue_size = restore_async(ranks[i], True, False)
            assert isinstance(ranks[i], AsyncScheduler) and queue_size > 1
            gpu = MockGPU(lambda s=ranks[i]: s, by_id, sampled, {}, {}, 0, True, batches[i],
                          ModelRunnerOutput, EOS_TOKEN_ID)
            engine_core(ranks[i], vllm_config, queue_size, True, False, gpu, core=core)
    elif not connector:
        assert all(type(s) is Scheduler for s in ranks)

    # The all-reduce: the sum over ranks (sync_dp_state's process-group collective), answered
    # from every rank's local flag of this step. A rank's call is answered once all have voted.
    votes: dict[int, bool] = {}

    def scripted_sync(dp_group, has_unfinished, pending_pause):
        return any(votes.values()), False

    ParallelConfig.sync_dp_state = staticmethod(scripted_sync)

    # The front end's load balancer, with the state get_core_engine_for_request reads.
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.engine_inflight = Counter()
    client.core_engines = [bytes([i, 0]) for i in range(dp)]
    client.lb_engines = [[0, 0, 0.0] for _ in range(dp)]
    client.eng_start_index = 0
    client.vllm_config = SimpleNamespace(use_v2_model_runner=False,
                                         parallel_config=SimpleNamespace(all2all_backend="allgather_reducescatter"))

    routed: dict[str, int] = {}
    steps = []
    finished = {}
    max_steps = sc.get("max_steps", 10000)

    history: list[list[list]] = []

    def record_counts() -> None:
        history.append([[*ranks[i].get_request_counts()[::-1], ranks[i].get_kv_cache_usage()]
                        for i in range(dp)])

    def publish_stats() -> None:
        # The coordinator rebinds lb_engines to the counts it last received
        # (core_client.py:1562-1575), stats_lag steps old.
        idx = len(history) - 1 - stats_lag
        if idx >= 0:
            client.lb_engines = [list(c) for c in history[idx]]
        else:
            client.lb_engines = [[0, 0, 0.0] for _ in range(dp)]

    def step_batch_queue(i, s, core, r_rec) -> bool:
        # The rank's busy-loop step with a batch queue: step_with_batch_queue, the step_fn
        # EngineCore.__init__ picks for a queue (core.py:242-244), then post_step, as
        # _process_engine_step runs them (core.py:1603-1619); a rank with no requests and no
        # batch queued has no work and does not step (has_work, core.py:1534-1540).
        if not (s.has_requests() or core.batch_queue):
            r_rec["processed"], r_rec["queue_len"] = False, 0
            return False
        if s.has_requests():
            r_rec["throttled"] = DPEngineCoreProc._should_throttle_prefills(core)
        before = len(batches[i])
        outputs, executed = core.step_with_batch_queue()
        core.post_step(executed)
        if len(batches[i]) > before:
            r_rec["scheduled"] = batches[i][before]["scheduled"]
        r_rec["processed"] = outputs is not None
        for eco in (outputs or {}).values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    r_rec["finished"].append(o.request_id)
                    finished[o.request_id] = {"reason": str(o.finish_reason), "rank": i,
                                              "output_tokens": sampled[o.request_id], "step": step}
                    if (eng := client.reqs_in_flight.pop(o.request_id, None)) is not None:
                        client.engine_inflight[eng] -= 1
        r_rec["finished"].sort()
        r_rec["queue_len"] = len(core.batch_queue)
        return executed

    def finish_client_side(o_request_id: str) -> None:
        # process_engine_outputs (core_client.py:1703-1710).
        if (eng := client.reqs_in_flight.pop(o_request_id, None)) is not None:
            client.engine_inflight[eng] -= 1

    def step_connector(i, s, core, r_rec) -> bool:
        """One rank's busy-loop step with its connector: run_connector.py's step (sync) or its
        queued step (async), on this rank's scheduler, with the aborts of requests routed here."""
        aborts = sorted(rid for rid, r in by_id.items() if r.get("abort_step") == step and routed.get(rid) == i)
        r_rec["aborted"] = aborts
        for rid in aborts:
            finish_client_side(rid)
        if async_sched:
            # Queued on the EngineCore's aborts queue, which vLLM drains after the next popped
            # batch's output (core.py:780-783).
            if aborts:
                core.aborts_queue.put_nowait(aborts)
            if not (s.has_requests() or core.batch_queue):
                r_rec["processed"], r_rec["queue_len"] = False, 0
                return False
            current[i] = r_rec
            if s.has_requests():
                r_rec["throttled"] = DPEngineCoreProc._should_throttle_prefills(core)
            outputs, executed = core.step_with_batch_queue()
            core.post_step(executed)
            for eco in (outputs or {}).values():
                for o in eco.outputs:
                    if o.finish_reason is not None:
                        r_rec["finished"].append(o.request_id)
                        finished_at.setdefault(o.request_id, step)
                        finished[o.request_id] = {"reason": str(o.finish_reason), "rank": i,
                                                  "output_tokens": gpus[i].sampled[o.request_id], "step": step}
                        finish_client_side(o.request_id)
            r_rec["finished"].sort()
            r_rec["queue_len"] = len(core.batch_queue)
            return executed
        if not s.has_requests():
            return False
        throttle = DPEngineCoreProc._should_throttle_prefills(core)
        r_rec["throttled"] = throttle
        out = s.schedule(throttle)
        for rid, r in by_id.items():
            req = s.requests.get(rid)
            if r.get("remote_prefill") and rid not in load_started and req is not None and \
                    req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                load_started[rid] = step
        r_rec["scheduled"] = dict(sorted(out.num_scheduled_tokens.items()))
        r_rec["preempted"] = sorted(out.preempted_req_ids or [])
        req_ids, toks = [], []
        for rid in out.num_scheduled_tokens:
            req_ids.append(rid)
            if not s.requests[rid].is_prefill_chunk:
                k, o = sampled[rid], by_id[rid]["output"]
                if k >= len(o):
                    toks.append([])
                else:
                    toks.append([EOS_TOKEN_ID if k == len(o) - 1 else o[k]])
                    sampled[rid] = k + 1
            else:
                toks.append([])
        kv_out, r_rec["transport"] = transport_report(s, by_id, step, load_started, finished_at, reported)
        if aborts:
            s.finish_requests(aborts, RequestStatus.FINISHED_ABORTED)
        outs = s.update_from_output(out, ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index={rid: j for j, rid in enumerate(req_ids)},
            sampled_token_ids=toks, logprobs=None, prompt_logprobs_dict={}, pooler_output=[],
            kv_connector_output=kv_out))
        for eco in outs.values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    r_rec["finished"].append(o.request_id)
                    finished_at.setdefault(o.request_id, step)
                    finished[o.request_id] = {"reason": str(o.finish_reason), "rank": i,
                                              "output_tokens": sampled[o.request_id], "step": step}
                    finish_client_side(o.request_id)
        r_rec["finished"].sort()
        return out.total_num_scheduled_tokens > 0

    step = 0
    record_counts()
    while step < max_steps:
        if step % stats_interval == 0:
            publish_stats()
        arrivals = []
        while pending and pending[0]["arrival_step"] <= step:
            r = pending.pop(0)
            params = SamplingParams(max_tokens=r.get("max_tokens", 1 << 20))
            params.update_from_generation_config({}, EOS_TOKEN_ID)
            req = Request(request_id=r["id"], prompt_token_ids=list(r["prompt"]), sampling_params=params,
                          pooling_params=None, arrival_time=float(r["arrival_step"]),
                          priority=r.get("priority", 0), block_hasher=hasher)
            if connector:
                add_kv_params(req, r, block_size)
            engine = client.get_core_engine_for_request(SimpleNamespace(
                request_id=r["id"], data_parallel_rank=r.get("rank") if connector else None, pooling_params=None))
            rank = client.core_engines.index(engine)
            routed[r["id"]] = rank
            arrivals.append([r["id"], rank])
            ranks[rank].add_request(req)
            # A request for an idle wave starts the next one on every rank (FIRST_REQ ->
            # START_DP_WAVE, core_client.py:1590-1593; coordinator.py:363-374; core.py:2281-2298).
            if not cores[rank].engines_running:
                for c in cores:
                    c.engines_running = True
        if not pending and not any(s.has_requests() for s in ranks) and not any(c.engines_running for c in cores) \
                and not any(c.batch_queue for c in cores if async_sched):
            break
        rec = {"step": step, "arrivals": arrivals, "ranks": []}
        clock["step"] = step
        votes.clear()
        local = []
        for i, (s, core) in enumerate(zip(ranks, cores)):
            r_rec = {"rank": i, "dummy": False, "scheduled": {}, "finished": [], "throttled": False}
            if connector:
                r_rec.update({"preempted": [], "aborted": [], "transport": {
                    "recving": [], "sending": [], "failed": [], "invalid_blocks": 0}})
                if async_sched:
                    # QueuedGPU sets it when the rank's step pops a batch.
                    r_rec["processed"] = False
            executed = False
            if connector:
                executed = step_connector(i, s, core, r_rec)
            elif async_sched:
                executed = step_batch_queue(i, s, core, r_rec)
            elif s.has_requests():
                throttle = DPEngineCoreProc._should_throttle_prefills(core)
                r_rec["throttled"] = throttle
                out = s.schedule(throttle)
                req_ids, toks = [], []
                for rid in out.num_scheduled_tokens:
                    req_ids.append(rid)
                    if not s.requests[rid].is_prefill_chunk:
                        k = sampled[rid]
                        o = by_id[rid]["output"]
                        toks.append([EOS_TOKEN_ID if k >= len(o) - 1 else o[k]])
                        sampled[rid] = k + 1
                    else:
                        toks.append([])
                outs = s.update_from_output(out, ModelRunnerOutput(
                    req_ids=req_ids, req_id_to_index={rid: j for j, rid in enumerate(req_ids)},
                    sampled_token_ids=toks, logprobs=None, prompt_logprobs_dict={}, pooler_output=[]))
                executed = out.total_num_scheduled_tokens > 0
                r_rec["scheduled"] = dict(sorted(out.num_scheduled_tokens.items()))
                for eco in outs.values():
                    for o in eco.outputs:
                        if o.finish_reason is not None:
                            r_rec["finished"].append(o.request_id)
                            finished[o.request_id] = {"reason": str(o.finish_reason), "rank": i,
                                                      "output_tokens": sampled[o.request_id], "step": step}
                            # process_engine_outputs (core_client.py:1703-1710).
                            if (eng := client.reqs_in_flight.pop(o.request_id, None)) is not None:
                                client.engine_inflight[eng] -= 1
                r_rec["finished"].sort()
            unfinished = s.has_unfinished_requests()
            if not executed and (unfinished or core.engines_running):
                # A dummy batch keeps the rank in the collective (core.py:2357-2368).
                r_rec["dummy"] = True
            local.append(unfinished)
            if connector:
                r_rec.update(kv_state(s))
                r_rec.update(connector_state_of(s))
            rec["ranks"].append(r_rec)
        for i, u in enumerate(local):
            votes[i] = u
        for i, core in enumerate(cores):
            if not core.engines_running and not local[i]:
                continue
            core.engines_running = DPEngineCoreProc._has_global_unfinished_reqs(core, local[i])
            if not core.engines_running:
                core.current_wave += 1
                core.step_counter = 0
        rec["waves"] = [c.current_wave for c in cores]
        rec["running"] = [c.engines_running for c in cores]
        rec["step_counter"] = [c.step_counter for c in cores]
        rec["counts"] = [list(s.get_request_counts()) for s in ranks]
        record_counts()
        steps.append(rec)
        step += 1

    prov = provenance(vllm_root, scenario_path)
    prov["tier"] = 3
    prov["engine"] = {"data_parallel_size": dp, "prefill_schedule_interval": interval,
                      "dp_sync_interval": sync_interval, "stats_interval": stats_interval,
                      "stats_lag": stats_lag}
    if async_sched:
        prov["engine"].update({"async_scheduling": True, "max_concurrent_batches": cores[0].batch_queue_size,
                               "scheduler_class": type(ranks[0]).__name__})
    if connector:
        prov["engine"].update({"connector": connector["name"], "kv_roles": roles})
    return {"provenance": prov, "routed": dict(sorted(routed.items())), "steps": steps,
            "finished": dict(sorted(finished.items()))}


if __name__ == "__main__":
    main()
