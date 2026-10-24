# Original to blis-vllm.
"""Tier-2 oracle: drive vLLM's real Scheduler through a scenario and record every step.

The scenario (JSON, schema in oracle/README.md) fixes the scheduler configuration, a fixed KV
capacity and the requests with their prompts, arrival steps and assigned outputs. This runs
vLLM's own Scheduler (built by vLLM's own test helper, tests/v1/core/utils.py create_scheduler)
and replaces only the GPU: each step's ModelRunnerOutput is built from the scenario's assigned
output tokens, the "shared GPU mock". It patches no scheduler, KV-cache or connector code;
oracle/mocks.json lists exactly what is replaced, and check_mocks.py fails CI if that list
names decision code.

Output (JSON on stdout): provenance, then one record per step with what the Go side must
reproduce: tokens scheduled per request, preempted and finished ids, the free block count,
every running request's blocks by content (block hash and ref count, never block ids), prefix
hits, and the waiting/running counts.

Usage: python -I oracle/run_scheduler.py <vllm checkout> <scenario.json>
"""

import hashlib
import json
import os
import subprocess
import sys


def main() -> None:
    vllm_root, scenario_path = sys.argv[1], sys.argv[2]
    # vLLM's test helpers are imported from the pinned checkout itself, never copied.
    sys.path.insert(0, vllm_root)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    # ModelConfig reads a model's config.json even though no weights are loaded. ORACLE_MODEL
    # names a local snapshot of the helper's default model at a pinned revision (see
    # oracle/README.md); its configuration does not reach any scheduling decision, which the
    # scenario's fixed num_blocks and block_size govern.
    model = os.environ.get("ORACLE_MODEL", "facebook/opt-125m")

    from tests.v1.core.utils import EOS_TOKEN_ID, create_scheduler  # noqa: E402
    from vllm.sampling_params import SamplingParams  # noqa: E402
    from vllm.utils.hashing import sha256  # noqa: E402
    from vllm.v1.core.kv_cache_utils import (  # noqa: E402
        get_request_block_hasher,
        init_none_hash,
    )
    from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
    from vllm.v1.request import Request  # noqa: E402

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

    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256)
    pending = sorted(sc["requests"], key=lambda r: (r["arrival_step"], r["id"]))
    by_id = {r["id"]: r for r in sc["requests"]}
    sampled_so_far = {r["id"]: 0 for r in sc["requests"]}
    finished = {}
    steps = []
    step = 0
    max_steps = sc.get("max_steps", 10000)

    def add_arrivals(upto: int) -> None:
        while pending and pending[0]["arrival_step"] <= upto:
            r = pending.pop(0)
            # The assigned output ends where the model samples EOS (the mock below); a
            # client budget, when the scenario states one, is max_tokens.
            params = SamplingParams(max_tokens=r.get("max_tokens", 1 << 20))
            params.update_from_generation_config({}, EOS_TOKEN_ID)
            req = Request(
                request_id=r["id"],
                prompt_token_ids=list(r["prompt"]),
                sampling_params=params,
                pooling_params=None,
                arrival_time=float(r["arrival_step"]),
                priority=r.get("priority", 0),
                block_hasher=hasher,
            )
            scheduler.add_request(req)

    while step < max_steps:
        add_arrivals(step)
        if not scheduler.has_unfinished_requests() and not pending:
            break
        out = scheduler.schedule()
        record = {
            "step": step,
            "scheduled": dict(sorted(out.num_scheduled_tokens.items())),
            "preempted": sorted(out.preempted_req_ids or []),
        }
        # The shared GPU mock: a request whose step completes its known tokens samples its
        # next assigned output token. Whether it does is read from the scheduler's own state
        # (is_prefill_chunk, set in _update_after_schedule after schedule() advanced
        # num_computed_tokens), never decided here.
        req_ids, sampled = [], []
        for rid in out.num_scheduled_tokens:
            req = scheduler.requests[rid]
            req_ids.append(rid)
            if not req.is_prefill_chunk:
                k = sampled_so_far[rid]
                outp = by_id[rid]["output"]
                # The last assigned token is sampled as EOS: that is how the assigned output
                # length reaches the engine, which stops on it as it would on a real EOS.
                sampled.append([EOS_TOKEN_ID if k == len(outp) - 1 else outp[k]])
                sampled_so_far[rid] = k + 1
            else:
                sampled.append([])
        mro = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        engine_outputs = scheduler.update_from_output(out, mro)
        done = []
        for eco in engine_outputs.values():
            for o in eco.outputs:
                if o.finish_reason is not None:
                    done.append(o.request_id)
                    finished[o.request_id] = {
                        "reason": str(o.finish_reason),
                        "output_tokens": sampled_so_far[o.request_id],
                        "step": step,
                    }
        record["finished"] = sorted(done)
        record.update(kv_state(scheduler))
        steps.append(record)
        step += 1

    print(json.dumps({
        "provenance": provenance(vllm_root, scenario_path),
        "steps": steps,
        "finished": dict(sorted(finished.items())),
    }, indent=1, sort_keys=True))


def kv_state(scheduler) -> dict:
    """The KV state compared by content: free blocks, and per running request the hash and
    reference count of each of its blocks, never the block id (block ids are an allocator
    detail two correct implementations may number differently)."""
    mgr = scheduler.kv_cache_manager
    pool = mgr.block_pool
    per_request = {}
    for rid in sorted(r.request_id for r in scheduler.running):
        blocks = mgr.coordinator.single_type_managers[0].req_to_blocks.get(rid, [])
        per_request[rid] = [
            [b.block_hash.hex() if b.block_hash is not None else None, b.ref_cnt]
            for b in blocks
        ]
    num_running, num_waiting = scheduler.get_request_counts()
    return {
        "free_blocks": pool.get_num_free_blocks(),
        "running": sorted(r.request_id for r in scheduler.running),
        "num_running": num_running,
        "num_waiting": num_waiting,
        "kv_usage": scheduler.get_kv_cache_usage(),
        "blocks": per_request,
    }


def provenance(vllm_root: str, scenario_path: str) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", vllm_root, *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    import vllm  # noqa: E402

    with open(scenario_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "mocks.json"), "rb") as f:
        mocks = json.load(f)
    return {
        "tier": 2,
        "vllm_version": vllm.__version__,
        "vllm_commit": git("rev-parse", "HEAD"),
        "scenario_sha256": digest,
        "mock_version": mocks["version"],
        "model": os.environ.get("ORACLE_MODEL", "facebook/opt-125m"),
        "model_revision": os.environ.get("ORACLE_MODEL_REVISION", "unpinned"),
        "mocks": [m["symbol"] for m in mocks["mocks"]],
        "python": sys.version.split()[0],
    }


if __name__ == "__main__":
    main()
