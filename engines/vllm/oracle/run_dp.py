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
decisions (the wave, the throttle cadence, the dummy batch, the load balancer's choice) run as
released.

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
                     "a golden-reviewed: <reviewer> trailer to the commit")
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
    from vllm.v1.engine.core import DPEngineCoreProc  # noqa: E402
    from vllm.v1.engine.core_client import DPLBAsyncMPClient  # noqa: E402
    from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
    from vllm.v1.request import Request  # noqa: E402

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from run_scheduler import provenance  # noqa: E402

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

    ranks = []
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

    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256)
    by_id = {r["id"]: r for r in sc["requests"]}
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    sampled = {r["id"]: 0 for r in sc["requests"]}
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
            engine = client.get_core_engine_for_request(SimpleNamespace(
                request_id=r["id"], data_parallel_rank=None, pooling_params=None))
            rank = client.core_engines.index(engine)
            routed[r["id"]] = rank
            arrivals.append([r["id"], rank])
            ranks[rank].add_request(req)
            # A request for an idle wave starts the next one on every rank (FIRST_REQ ->
            # START_DP_WAVE, core_client.py:1590-1593; coordinator.py:363-374; core.py:2281-2298).
            if not cores[rank].engines_running:
                for c in cores:
                    c.engines_running = True
        if not pending and not any(s.has_requests() for s in ranks) and not any(c.engines_running for c in cores):
            break
        rec = {"step": step, "arrivals": arrivals, "ranks": []}
        votes.clear()
        local = []
        for i, (s, core) in enumerate(zip(ranks, cores)):
            r_rec = {"rank": i, "dummy": False, "scheduled": {}, "finished": [], "throttled": False}
            executed = False
            if s.has_requests():
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
    return {"provenance": prov, "routed": dict(sorted(routed.items())), "steps": steps,
            "finished": dict(sorted(finished.items()))}


if __name__ == "__main__":
    main()
