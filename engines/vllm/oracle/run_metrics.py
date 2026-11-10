# Original to blis-vllm.
"""Tier-5 oracle: vLLM's own metric series for a tier-2 scenario.

It drives the scenario exactly as oracle/run_scheduler.py does (vLLM's real Scheduler, the
same GPU mock), and feeds every step's outputs through vLLM's own frontend: the
OutputProcessor builds IterationStats from the EngineCoreOutputs, and PrometheusStatLogger
records them with the step's SchedulerStats, as AsyncLLM's output handler does
(vllm/v1/engine/async_llm.py:845-905@5df63a72). After the last step it reads every vllm:*
sample from the Prometheus registry, and after every step the samples of the families
whose values the scheduler decides step by step (gauges and counters, not histograms), so a
rule that moves a count to another step, or a gauge that is wrong mid-run, cannot hide behind
an equal final total.

The only difference from run_scheduler.py is what is recorded: the metric series instead of
the per-step decisions. The clock is virtual: the step index, through the EngineCoreOutputs
timestamp and through time.time and time.monotonic in the scheduler, the engine events and
the stats modules (the clock mocks listed in oracle/mocks.json). So a run is deterministic, latency families
included; those are "derived" in the manifest and not compared, only the exact ones are.

Usage: python -I oracle/run_metrics.py <vllm checkout> <scenario.json | dir> [out dir]
(in directory mode, <name>.metrics.json beside each scenario or in [out dir]).
"""

import json
import os
import sys


# Gauges vLLM sets to the wall-clock time of the update (Gauge.set_to_current_time), so their
# value differs between two runs of the same scenario while their labels do not. The labels are
# what a step decided (the running and waiting adapters); the value is when the sample was made.
# The series records the value as 1.0 once the sample is set, and keeps every label, so the
# family is compared by its labels and a changed label set still fails the golden comparison.
WALL_CLOCK_GAUGES = frozenset({"vllm:lora_requests_info"})


def sample_value(name: str, value: float) -> float:
    """A sample's value as the series records it: a wall-clock gauge's timestamp is 1.0 when set."""
    if name in WALL_CLOCK_GAUGES:
        return 1.0 if value > 0 else 0.0
    return value


def main() -> None:
    vllm_root, path = sys.argv[1], sys.argv[2]
    if os.path.isdir(path):
        out_dir = sys.argv[3] if len(sys.argv) > 3 else path
        # Golden traces are never overwritten without review (docs/testing.md).
        if "testdata/golden" in os.path.abspath(out_dir).replace(os.sep, "/") and \
                os.environ.get("BLIS_VLLM_ALLOW_GOLDEN_UPDATE") != "1":
            sys.exit("refusing to write golden traces: set BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1, and add "
                     "a golden-reviewed: <trace>: <reason> trailer for each trace the commit changes")
        for name in sorted(f for f in os.listdir(path)
                           if f.endswith(".json") and not f.endswith((".trace.json", ".metrics.json"))):
            series = run_one(vllm_root, os.path.join(path, name))
            with open(os.path.join(out_dir, name[:-5] + ".metrics.json"), "w") as f:
                json.dump(series, f, indent=1, sort_keys=True)
        return
    print(json.dumps(run_one(vllm_root, path), indent=1, sort_keys=True))


def run_one(vllm_root: str, scenario_path: str) -> dict:
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    model = os.environ.get("ORACLE_MODEL", "facebook/opt-125m")

    from prometheus_client import REGISTRY  # noqa: E402

    from tests.v1.core.utils import EOS_TOKEN_ID, create_scheduler  # noqa: E402
    from vllm.sampling_params import SamplingParams  # noqa: E402
    from vllm.utils.hashing import sha256  # noqa: E402
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
    from vllm.v1.engine import EngineCoreRequest  # noqa: E402
    from vllm.v1.engine.output_processor import OutputProcessor  # noqa: E402
    from vllm.v1.metrics.loggers import PrometheusStatLogger  # noqa: E402
    from vllm.v1.metrics.stats import IterationStats  # noqa: E402
    from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
    from vllm.v1.request import Request, RequestStatus  # noqa: E402

    import vllm.v1.core.sched.scheduler as sched_mod  # noqa: E402
    import vllm.v1.engine as engine_mod  # noqa: E402
    import vllm.v1.metrics.stats as stats_mod  # noqa: E402

    # The virtual clock (a clock mock, oracle/mocks.json): the step index.
    clock = {"now": 0.0}

    class _VirtualTime:
        @staticmethod
        def time() -> float:
            return clock["now"]

        @staticmethod
        def monotonic() -> float:
            return clock["now"]

        def __getattr__(self, name):
            import time as real
            return getattr(real, name)

    sched_mod.time = _VirtualTime()
    stats_mod.time = _VirtualTime()
    engine_mod.time = _VirtualTime()

    with open(scenario_path) as f:
        sc = json.load(f)
    cfg = sc["config"]
    block_size = cfg["block_size"]
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    spec = cfg.get("speculative")
    if spec or cfg.get("max_loras"):
        # A speculative or LoRA scenario is built as tier 2 builds it (run_scheduler.build_scheduler):
        # vLLM's own SpeculativeConfig and LoRAConfig handed to the helper's constructors.
        import tests.v1.core.utils as helpers  # noqa: E402
        from run_scheduler import build_scheduler  # noqa: E402

        # Only the speculative and LoRA configurations are added to the tier-5 scheduler. The scheduler
        # knobs tier 2 also varies (watermark, active-sequence and scheduled-token caps, the
        # adaptive prefill threshold, LoRA, KV events and hybrid groups) are not modelled by this
        # tier's Go driver, and tier 5 has never applied them, so they are left out of the
        # configuration here as they are for a scenario without speculation.
        spec_only = {k: v for k, v in cfg.items() if k in (
            "max_num_seqs", "max_num_batched_tokens", "block_size", "num_blocks", "max_model_len",
            "long_prefill_token_threshold", "enable_prefix_caching", "scheduling_policy", "speculative",
            "max_loras", "max_cpu_loras")}
        scheduler = build_scheduler(helpers, create_scheduler, model, spec_only)
    else:
        scheduler = create_scheduler(
            model=model,
            max_num_seqs=cfg["max_num_seqs"],
            max_num_batched_tokens=cfg["max_num_batched_tokens"],
            long_prefill_token_threshold=cfg.get("long_prefill_token_threshold", 0),
            enable_prefix_caching=cfg.get("enable_prefix_caching", False),
            num_blocks=cfg["num_blocks"],
            block_size=block_size,
            max_model_len=cfg.get("max_model_len"),
            scheduling_policy=cfg.get("scheduling_policy", "fcfs"),
        )
    # A fresh registry view per scenario: the logger unregisters earlier vllm:* collectors
    # itself (unregister_vllm_metrics, vllm/v1/metrics/prometheus.py:56-69@5df63a72).
    logger = PrometheusStatLogger(scheduler.vllm_config, [0])
    frontend = OutputProcessor(None, log_stats=True)

    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256)
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    by_id = {r["id"]: r for r in sc["requests"]}
    sampled_so_far = {r["id"]: 0 for r in sc["requests"]}
    # LoRA adapter ids, numbered from 1 in name order, as run_scheduler.py numbers them.
    lora_ids = {n: i + 1 for i, n in enumerate(sorted({r["lora"] for r in sc["requests"] if r.get("lora")}))}
    accepted_plan = {r["id"]: list(r.get("accepted", [])) for r in sc["requests"]}
    verify_steps = {r["id"]: 0 for r in sc["requests"]}
    num_spec = getattr(scheduler, "num_spec_tokens", 0)
    step = 0
    max_steps = sc.get("max_steps", 10000)
    per_step = []

    def collect(include_histograms: bool) -> dict:
        out = {}
        for metric in REGISTRY.collect():
            if not metric.name.startswith("vllm:"):
                continue
            if metric.type == "histogram" and not include_histograms:
                continue
            for smp in metric.samples:
                if smp.name.endswith("_created"):
                    continue
                labels = ",".join(f"{k}={v}" for k, v in sorted(smp.labels.items())
                                  if k not in ("model_name", "engine"))
                out[smp.name + ("{" + labels + "}" if labels else "")] = sample_value(smp.name, smp.value)
        return dict(sorted(out.items()))

    while step < max_steps:
        clock["now"] = float(step)
        while pending and pending[0]["arrival_step"] <= step:
            r = pending.pop(0)
            params = SamplingParams(max_tokens=r.get("max_tokens", 1 << 20))
            params.update_from_generation_config({}, EOS_TOKEN_ID)
            lora = None
            if r.get("lora"):
                from vllm.lora.request import LoRARequest  # noqa: E402

                lora = LoRARequest(lora_name=r["lora"], lora_int_id=lora_ids[r["lora"]],
                                   lora_path="/adapters/" + r["lora"])
            core_req = EngineCoreRequest(
                request_id=r["id"], external_req_id=r["id"], prompt_token_ids=list(r["prompt"]),
                mm_features=None, sampling_params=params, pooling_params=None,
                arrival_time=float(r["arrival_step"]), lora_request=lora, cache_salt=None,
                data_parallel_rank=None, priority=r.get("priority", 0))
            frontend.add_request(core_req, None)
            scheduler.add_request(Request(
                request_id=r["id"], prompt_token_ids=list(r["prompt"]), sampling_params=params,
                pooling_params=None, arrival_time=float(r["arrival_step"]),
                priority=r.get("priority", 0), block_hasher=hasher, lora_request=lora))
        # An abort at this step, before the step runs. Two kinds, because only one reaches the
        # finish-reason counter:
        #  - "client" (the default): AsyncLLM.abort. The frontend's OutputProcessor forgets the
        #    request (async_llm.py:918), then EngineCore finishes it in the scheduler
        #    (core.py:554-559@5df63a72). No output carries the abort back, so the request is in
        #    no vllm:request_success sample.
        #  - "engine": the engine ends the request itself (pause in abort mode, shutdown with a
        #    zero timeout; core.py:1648-1651, 2089-2092@5df63a72). EngineCoreProc._send_abort_outputs,
        #    vLLM's own method, queues an output with finish_reason ABORT, which the frontend
        #    processes while it still holds the request, so the abort is counted.
        client = sorted(rid for rid, r in by_id.items() if r.get("abort_step") == step
                        and r.get("abort_by", "client") == "client")
        engine = sorted(rid for rid, r in by_id.items() if r.get("abort_step") == step
                        and r.get("abort_by") == "engine")
        if client:
            frontend.abort_requests(client, internal=True)
            scheduler.finish_requests(client, RequestStatus.FINISHED_ABORTED)
        if engine:
            import queue as _queue
            from vllm.v1.engine.core import EngineCoreProc  # noqa: E402

            class _EngineSide:
                """Only the output queue and the three methods that fill it are used."""
                def __init__(self):
                    self.output_queue = _queue.Queue()
                _send_abort_outputs = EngineCoreProc._send_abort_outputs
                _send_abort_outputs_to_client = EngineCoreProc._send_abort_outputs_to_client
                _send_finish_outputs_to_client = EngineCoreProc._send_finish_outputs_to_client

            side = _EngineSide()
            aborted = scheduler.finish_requests(engine, RequestStatus.FINISHED_ABORTED)
            side._send_abort_outputs(aborted)
            while not side.output_queue.empty():
                _, eco_abort = side.output_queue.get_nowait()
                it = IterationStats()
                frontend.process_outputs(eco_abort.outputs, float(step), it)
                logger.record(scheduler_stats=None, iteration_stats=it, engine_idx=0)
        if not scheduler.has_unfinished_requests() and not pending:
            break
        out = scheduler.schedule()
        # The shared GPU mock, as in run_scheduler.py.
        req_ids, sampled = [], []
        for rid in out.num_scheduled_tokens:
            req = scheduler.requests[rid]
            req_ids.append(rid)
            if not req.is_prefill_chunk:
                # The shared GPU mock, as in run_scheduler.py: one token, plus the drafts the
                # scenario accepts at this verify step (at most the real drafts; a -1
                # placeholder is always rejected, rejection_sampler.py:762-763).
                drafts = out.scheduled_spec_decode_tokens.get(rid, [])
                n = 1
                if drafts:
                    plan = accepted_plan[rid]
                    i = verify_steps[rid]
                    verify_steps[rid] = i + 1
                    real = len([d for d in drafts if d >= 0])
                    n = 1 + min(plan[i] if i < len(plan) else 0, real)
                toks = []
                outp = by_id[rid]["output"]
                for _ in range(n):
                    k = sampled_so_far[rid]
                    if k >= len(outp):
                        break
                    toks.append(EOS_TOKEN_ID if k == len(outp) - 1 else outp[k])
                    sampled_so_far[rid] = k + 1
                sampled.append(toks)
            else:
                sampled.append([])
        outputs = scheduler.update_from_output(out, ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={}, pooler_output=[]))
        if spec and out.total_num_scheduled_tokens > 0:
            # The drafter proposes for every request the step ran, as in run_scheduler.py.
            from vllm.v1.outputs import DraftTokenIds  # noqa: E402

            ids = [rid for rid in out.num_scheduled_tokens if rid in scheduler.requests]
            scheduler.update_draft_token_ids(DraftTokenIds(ids, [[0] * num_spec for _ in ids]))
        eco = outputs.get(0)
        if eco is not None:
            iteration = IterationStats() if eco.outputs else None
            frontend.process_outputs(eco.outputs, float(step), iteration)
            logger.record(scheduler_stats=eco.scheduler_stats, iteration_stats=iteration, engine_idx=0)
        per_step.append(collect(include_histograms=False))
        step += 1

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from run_scheduler import provenance  # noqa: E402

    prov = provenance(vllm_root, scenario_path)
    prov["tier"] = 5
    return {"provenance": prov, "steps": step, "series": collect(include_histograms=True),
            "per_step": per_step}


if __name__ == "__main__":
    main()
