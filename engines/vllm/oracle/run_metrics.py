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


def main() -> None:
    vllm_root, path = sys.argv[1], sys.argv[2]
    if os.path.isdir(path):
        out_dir = sys.argv[3] if len(sys.argv) > 3 else path
        # Golden traces are never overwritten without review (docs/testing.md).
        if "testdata/golden" in os.path.abspath(out_dir).replace(os.sep, "/") and \
                os.environ.get("BLIS_VLLM_ALLOW_GOLDEN_UPDATE") != "1":
            sys.exit("refusing to write golden traces: set BLIS_VLLM_ALLOW_GOLDEN_UPDATE=1, and add "
                     "a golden-reviewed: <reviewer> trailer to the commit")
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
    from vllm.v1.request import Request  # noqa: E402

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
                out[smp.name + ("{" + labels + "}" if labels else "")] = smp.value
        return dict(sorted(out.items()))

    while step < max_steps:
        clock["now"] = float(step)
        while pending and pending[0]["arrival_step"] <= step:
            r = pending.pop(0)
            params = SamplingParams(max_tokens=r.get("max_tokens", 1 << 20))
            params.update_from_generation_config({}, EOS_TOKEN_ID)
            core_req = EngineCoreRequest(
                request_id=r["id"], external_req_id=r["id"], prompt_token_ids=list(r["prompt"]),
                mm_features=None, sampling_params=params, pooling_params=None,
                arrival_time=float(r["arrival_step"]), lora_request=None, cache_salt=None,
                data_parallel_rank=None, priority=r.get("priority", 0))
            frontend.add_request(core_req, None)
            scheduler.add_request(Request(
                request_id=r["id"], prompt_token_ids=list(r["prompt"]), sampling_params=params,
                pooling_params=None, arrival_time=float(r["arrival_step"]),
                priority=r.get("priority", 0), block_hasher=hasher))
        if not scheduler.has_unfinished_requests() and not pending:
            break
        out = scheduler.schedule()
        # The shared GPU mock, as in run_scheduler.py.
        req_ids, sampled = [], []
        for rid in out.num_scheduled_tokens:
            req = scheduler.requests[rid]
            req_ids.append(rid)
            if not req.is_prefill_chunk:
                k = sampled_so_far[rid]
                outp = by_id[rid]["output"]
                sampled.append([EOS_TOKEN_ID if k == len(outp) - 1 else outp[k]])
                sampled_so_far[rid] = k + 1
            else:
                sampled.append([])
        outputs = scheduler.update_from_output(out, ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={}, pooler_output=[]))
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
