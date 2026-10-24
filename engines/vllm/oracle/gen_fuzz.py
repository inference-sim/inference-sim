# Original to blis-vllm.
"""Seeded random scenarios for the tier-2 fuzz test (#2021 B8).

Each seed gives one scenario: a random scheduler configuration (budget, max_num_seqs, block
size, a KV pool sized to cause preemption sometimes, prefix caching on or off, the
long-prefill cap, FCFS or priority) and a random request stream (arrivals, prompt and output
lengths, shared prefixes, priorities, client budgets). The same seed always gives the same
scenario, from Python's random.Random alone, so a failing seed reproduces anywhere. A seed
that finds a mismatch is shrunk and kept under oracle/scenarios as a permanent test.

Seeds from 5000 on (`SPEC_FROM`) add what the base generator leaves at defaults: a
speculative decoder in about half of them (EAGLE3, P-EAGLE, draft_model, PARD, MTP single
and multi-module, DSpark; per-request acceptance drawn per verify step), LoRA adapters with
max_loras, and the scheduler knobs vLLM's test helper does not expose (watermark,
max_num_active_seqs, max_num_scheduled_tokens, the adaptive long-prefill cap). Seeds below
5000 give exactly the scenarios they always gave, so the existing corpus does not move.

Usage: python -I oracle/gen_fuzz.py <out dir> <first seed> <count>
"""

import json
import os
import random
import sys


SPEC_FROM = 5000

# Speculative methods, each with the draft configuration it is built from (oracle/drafts).
SPEC_METHODS = [
    {"method": "eagle3", "draft": "llama-eagle"},
    {"method": "eagle3", "draft": "llama-eagle", "parallel_drafting": True},
    {"method": "draft_model", "draft": "llama-eagle"},
    {"method": "draft_model", "draft": "llama-eagle", "parallel_drafting": True},
    {"method": "mtp", "draft": "mtp-1", "k": 1},
    {"method": "mtp", "draft": "mtp-4"},
    {"method": "dspark"},
]


def scenario(seed: int) -> dict:
    sc = base_scenario(seed)
    if seed >= SPEC_FROM:
        extend(sc, random.Random(seed * 7919 + 1))
    return sc


def extend(sc: dict, rng: random.Random) -> None:
    cfg, reqs = sc["config"], sc["requests"]
    if rng.random() < 0.5:
        m = dict(rng.choice(SPEC_METHODS))
        k = m.pop("k", rng.randint(1, 4))
        if m["method"] == "mtp" and m["draft"] == "mtp-4":
            k = rng.randint(2, 4)
        m["num_speculative_tokens"] = k
        # vLLM refuses a budget no larger than the draft slots (vllm/config/vllm.py:2285-2290)
        # and, for multi-module MTP with prefix caching, a block smaller than K
        # (vllm/v1/core/kv_cache_coordinator.py:147-155).
        cfg["max_num_batched_tokens"] = max(cfg["max_num_batched_tokens"], 2 * k + 2)
        if m["method"] == "mtp" and m["draft"] == "mtp-4" and cfg["enable_prefix_caching"]:
            cfg["block_size"] = max(cfg["block_size"], 4)
        cfg["speculative"] = m
        for r in reqs:
            r["accepted"] = [rng.randint(0, k) if rng.random() < 0.8 else k for _ in range(len(r["output"]) + 2)]
    if rng.random() < 0.3:
        names = ["a1", "a2", "a3"][: rng.randint(1, 3)]
        cfg["max_loras"] = rng.randint(1, len(names))
        for r in reqs:
            if rng.random() < 0.7:
                r["lora"] = rng.choice(names)
    if rng.random() < 0.25:
        cfg["watermark"] = rng.choice([0.01, 0.05, 0.1])
    if rng.random() < 0.25:
        cfg["max_num_active_seqs"] = rng.randint(1, cfg["max_num_seqs"])
    if rng.random() < 0.25:
        cfg["max_num_scheduled_tokens"] = rng.randint(max(1, cfg["max_num_seqs"]), cfg["max_num_batched_tokens"])
    if cfg["long_prefill_token_threshold"] and rng.random() < 0.5:
        cfg["long_prefill_token_threshold_adaptive"] = True
    # The pool must still hold the largest request with the drafter's lookahead and the
    # watermark, or vLLM would never admit it.
    look = cfg.get("speculative", {}).get("num_speculative_tokens", 0) + 1
    biggest = max(len(r["prompt"]) + len(r["output"]) for r in reqs)
    need = -(-(min(biggest, cfg["max_model_len"]) + look) // cfg["block_size"]) + 2
    if cfg.get("watermark"):
        need = int((need + 1) / (1 - cfg["watermark"])) + 2
    cfg["num_blocks"] = max(cfg["num_blocks"], need)


def base_scenario(seed: int) -> dict:
    rng = random.Random(seed)
    block = rng.choice([1, 2, 4, 8, 16])
    budget = rng.choice([8, 16, 32, 64, 128])
    seqs = rng.randint(1, 8)
    max_len = rng.choice([64, 128, 256])
    policy = rng.choice(["fcfs", "priority"])
    caching = rng.random() < 0.6
    # vLLM refuses a threshold above max_model_len and a budget below max_num_seqs
    # (SchedulerConfig.verify_max_model_len, vllm/config/scheduler.py:314-357@5df63a72).
    threshold = rng.choice([0, 0, 0, rng.randint(1, min(budget, max_len))])
    seqs = min(seqs, budget)
    n = rng.randint(1, 12)
    shared = [rng.randint(1, 5000) for _ in range(rng.randint(0, 3 * block))]
    reqs = []
    for i in range(n):
        plen = rng.randint(1, max_len // 2)
        prompt = []
        if shared and rng.random() < 0.5:
            prompt = shared[: min(len(shared), plen)]
        prompt += [rng.randint(1, 50000) for _ in range(plen - len(prompt))]
        olen = rng.randint(1, max(1, min(24, max_len - plen - 1)))
        r = {
            "id": f"q{i:02d}",
            "arrival_step": rng.randint(0, 12),
            "prompt": prompt,
            "output": [rng.randint(1, 50000) for _ in range(olen)],
            "priority": rng.randint(0, 3) if policy == "priority" else 0,
        }
        if rng.random() < 0.2:
            r["max_tokens"] = rng.randint(1, olen + 2)
        reqs.append(r)
    # Size the pool between "just fits the largest request" and "fits everything", so some
    # seeds preempt and some do not. The full-sequence admission gate needs the largest
    # request to fit alone, or vLLM would never admit it.
    biggest = max(len(r["prompt"]) + len(r["output"]) for r in reqs)
    need_one = -(-min(biggest, max_len) // block) + 1
    need_all = sum(-(-(len(r["prompt"]) + len(r["output"])) // block) for r in reqs) + 1
    blocks = rng.randint(need_one, max(need_one, need_all))
    return {
        "name": f"fuzz-{seed}",
        "seed": seed,
        "config": {
            "max_num_seqs": seqs,
            "max_num_batched_tokens": budget,
            "block_size": block,
            "num_blocks": blocks,
            "max_model_len": max_len,
            "long_prefill_token_threshold": threshold,
            "enable_prefix_caching": caching,
            "scheduling_policy": policy,
        },
        "requests": reqs,
        "max_steps": 4000,
    }


def main() -> None:
    out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for seed in range(first, first + count):
        with open(os.path.join(out, f"fuzz-{seed}.json"), "w") as f:
            json.dump(scenario(seed), f, indent=1)


if __name__ == "__main__":
    main()
