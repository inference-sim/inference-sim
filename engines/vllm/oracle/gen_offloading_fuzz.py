# Original to blis-vllm.
"""Seeded random OffloadingConnector scenarios for the tier-4 fuzz test (blis-vllm#14).

Each seed gives one scenario for oracle/run_offloading.py: a scheduler configuration, a GPU KV
pool small enough that its prefix cache forgets (so a later request finds its prefix only on the
CPU tier), a CPU tier of a few chunks (so stores evict and sometimes fail), blocks_per_chunk 1-3,
offload_prompt_only on or off, the worker's load and store delays, and a request stream that
reuses a few shared prefixes, with some client aborts. The same seed always gives the same
scenario, from Python's random.Random alone.

The KV page is 16 KiB per block per group (head_size = 2048 / block_size, one float32 head, K
and V), so a CPU chunk, rounded up to the host's page size (cpu/spec.py:118-121), is the same on
a 4 KiB-page and a 16 KiB-page host.

Seeds below 20000 have one full-attention KV cache group; seeds 20000 and up add one or two
sliding windows beside it (the hybrid groups the OffloadingConnector supports as SupportsHMA).
Seeds 40000 and up also run 2, 3, 4 or 6 workers (divisors of the helper model's 12 heads) (tensor_parallel_size, the connector's num_workers:
each job is pending on every worker, and the CPU tier holds a copy of a block per worker).
Seeds 60000 and up draw, from a third stream, the batch queue (async scheduling on the V1 or V2
runner, pipeline parallelism 2 with async on V2) and a drafter (DSpark K 3-7 or EAGLE-3 K 1-4,
with each request's acceptance plan), the shapes of Muse Spark's arm D. Seeds below 60000 are
unchanged.

Usage: python -I oracle/gen_offloading_fuzz.py <out dir> <first seed> <count>
"""

import json
import os
import random
import sys

HYBRID_SEEDS = 20000
TP_SEEDS = 40000
QUEUED_SEEDS = 60000


def scenario(seed: int) -> dict:
    if seed >= QUEUED_SEEDS:
        return queued_scenario(seed)
    return scenario_base(seed)


def scenario_base(seed: int) -> dict:
    rng = random.Random(seed)
    hybrid = seed >= HYBRID_SEEDS and (seed < QUEUED_SEEDS or random.Random(seed * 7919).random() < 0.6)
    block = rng.choice([2, 4, 8])
    budget = rng.choice([16, 32, 64])
    seqs = rng.randint(1, 4)
    max_len = rng.choice([64, 128])
    bpc = rng.choice([1, 1, 2, 3])
    groups = [{"kind": "full"}]
    if hybrid:
        for _ in range(rng.randint(1, 2)):
            groups.append({"kind": "sliding_window", "sliding_window": rng.choice([block, 2 * block, 3 * block + 1, 16])})
        # Groups with one spec merge into one (get_kv_cache_groups); keep the windows distinct.
        seen, uniq = set(), []
        for g in groups:
            k = (g["kind"], g.get("sliding_window"))
            if k not in seen:
                seen.add(k)
                uniq.append(g)
        groups = uniq
    shared = [[rng.randint(1, 5000) for _ in range(rng.randint(block, 4 * block * bpc))] for _ in range(rng.randint(1, 3))]
    reqs = []
    for i in range(rng.randint(2, 10)):
        plen = rng.randint(2, max_len // 2)
        prompt = []
        if rng.random() < 0.7:
            base = rng.choice(shared)
            prompt = base[: min(len(base), plen)]
        prompt += [rng.randint(1, 50000) for _ in range(plen - len(prompt))]
        room = max(1, min(12, max_len - plen - 1))
        r = {"id": f"q{i:02d}", "arrival_step": rng.randint(0, 25), "prompt": prompt,
             "output": [rng.randint(1, 50000) for _ in range(rng.randint(1, room))]}
        if rng.random() < 0.08:
            r["abort_step"] = r["arrival_step"] + rng.randint(0, 6)
        if rng.random() < 0.2:
            r["load_delay"] = rng.randint(0, 4)
        if rng.random() < 0.2:
            r["store_delay"] = rng.randint(0, 4)
        reqs.append(r)
    biggest = max(len(r["prompt"]) + len(r["output"]) for r in reqs)
    need_one = (-(-min(biggest, max_len) // block) + 1) * len(groups)
    blocks = rng.randint(need_one, need_one + 3 * len(groups) * 4)
    tp = rng.choice([2, 3, 4, 6]) if seed >= TP_SEEDS else 1
    if seed >= QUEUED_SEEDS and random.Random(seed * 7919 + 1).random() < 0.5:
        tp = 1
    chunk_bytes = 16384 * len(groups) * bpc * tp
    extra = {"cpu_bytes_to_use": chunk_bytes * rng.randint(1, 12), "offload_prompt_only": rng.random() < 0.5}
    if bpc > 1:
        extra["blocks_per_chunk"] = bpc
    cfg = {"max_num_seqs": seqs, "max_num_batched_tokens": budget, "block_size": block, "num_blocks": blocks,
           "max_model_len": max_len, "long_prefill_token_threshold": rng.choice([0, 0, rng.randint(1, budget)]),
           "enable_prefix_caching": rng.random() < 0.85, "scheduling_policy": rng.choice(["fcfs", "priority"]),
           "kv_page": {"num_kv_heads": 1, "head_size": 2048 // block},
           "transfer": {"load_delay": rng.randint(0, 3), "store_delay": rng.randint(0, 3)},
           "connector": {"name": "OffloadingConnector", "kv_role": "kv_both", "extra_config": extra}}
    if cfg["scheduling_policy"] == "priority":
        for r in reqs:
            r["priority"] = rng.randint(0, 3)
    if hybrid:
        cfg["kv_groups"] = groups
    if tp > 1:
        cfg["tensor_parallel_size"] = tp
    return {"name": f"ofuzz-{seed}", "seed": seed, "config": cfg, "requests": reqs, "max_steps": 400}


def queued_scenario(seed: int) -> dict:
    """A seed from QUEUED_SEEDS: a TP seed's draw, then, from a third stream, the batch queue and
    a drafter."""
    q = random.Random(seed * 104729 + 7)
    sc = scenario_base(seed)
    cfg = sc["config"]
    roll = q.random()
    if roll < 0.2:
        cfg["pipeline_parallel_size"] = 2
        cfg["async_scheduling"], cfg["use_v2_model_runner"] = True, True
        # The world size grows by PP, and the CPU tier holds a copy per worker.
        ex = cfg["connector"]["extra_config"]
        ex["cpu_bytes_to_use"] *= 2
    elif roll < 0.85:
        cfg["async_scheduling"] = True
        if q.random() < 0.4:
            cfg["use_v2_model_runner"] = True
    if q.random() < 0.6:
        if q.random() < 0.6:
            k = q.choice([3, 5, 7])
            cfg["speculative"] = {"method": "dspark", "num_speculative_tokens": k}
            cfg["use_v2_model_runner"] = True
        else:
            k = q.randint(1, 4)
            cfg["speculative"] = {"method": "eagle3", "num_speculative_tokens": k, "draft": "llama-eagle"}
            # The draft checkpoint has 2 attention heads, which vLLM's config check divides by TP.
            if cfg.get("tensor_parallel_size", 1) > 2:
                cfg["tensor_parallel_size"] = 2
        # vLLM refuses a budget no larger than the draft slots (vllm/config/vllm.py:2285-2290), and
        # a long-prefill cap below 1+K trips its padding assert (scheduler.py:1134-1156).
        cfg["max_num_batched_tokens"] = max(cfg["max_num_batched_tokens"], 4 * (k + 1) * cfg["max_num_seqs"])
        if 0 < cfg["long_prefill_token_threshold"] <= k + 1:
            cfg["long_prefill_token_threshold"] = 0
        for r in sc["requests"]:
            r["accepted"] = [q.randint(0, k) for _ in range(len(r["output"]))]
    sc["max_steps"] = 600
    return sc


def main() -> None:
    out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for seed in range(first, first + count):
        with open(os.path.join(out, f"ofuzz-{seed}.json"), "w") as f:
            json.dump(scenario(seed), f, indent=1)


if __name__ == "__main__":
    main()
