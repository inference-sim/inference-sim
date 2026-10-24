# Original to blis-vllm.
"""Seeded random NIXL connector scenarios for the tier-4 fuzz test (#2021 B8).

Each seed gives one scenario: the engine's KV role (decoder or prefiller), the load-failure
policy, a scheduler configuration with a KV pool sized so admissions sometimes wait, and a
request stream mixing P/D requests with local ones. Decoder-side P/D requests get a pull delay,
and some get a whole-load failure or failed blocks; prefiller-side ones get a send delay. The
same seed always gives the same scenario, from Python's random.Random alone. About one request
in ten is aborted by its client a few steps after it arrives.

Usage: python -I oracle/gen_connector_fuzz.py <out dir> <first seed> <count>
"""

import json
import os
import random
import sys


def scenario(seed: int) -> dict:
    rng = random.Random(seed)
    role = rng.choice(["kv_consumer", "kv_consumer", "kv_producer"])
    policy = rng.choice(["fail", "recompute"])
    block = rng.choice([2, 4, 8])
    budget = rng.choice([8, 16, 32, 64])
    seqs = min(rng.randint(1, 6), budget)
    max_len = rng.choice([64, 128])
    caching = rng.random() < 0.7
    policy_sched = rng.choice(["fcfs", "priority"])
    threshold = rng.choice([0, 0, 0, rng.randint(1, min(budget, max_len))])
    shared = [rng.randint(1, 5000) for _ in range(rng.randint(0, 3 * block))]
    reqs = []
    for i in range(rng.randint(1, 10)):
        plen = rng.randint(2, max_len // 2)
        prompt = []
        if shared and rng.random() < 0.5:
            prompt = shared[: min(len(shared), plen)]
        prompt += [rng.randint(1, 50000) for _ in range(plen - len(prompt))]
        pd = rng.random() < 0.7
        r = {"id": f"q{i:02d}", "arrival_step": rng.randint(0, 8), "prompt": prompt,
             "priority": rng.randint(0, 3) if policy_sched == "priority" else 0}
        if role == "kv_consumer" and pd:
            r["remote_prefill"] = True
            r["recv_delay"] = rng.randint(0, 4)
            roll = rng.random()
            if roll < 0.15:
                r["recv_failed"] = True
            elif roll < 0.3:
                nblocks = -(-plen // block)
                r["invalid_blocks"] = sorted(rng.sample(range(nblocks), rng.randint(1, min(2, nblocks))))
            r["output"] = [rng.randint(1, 50000) for _ in range(rng.randint(1, max(1, min(12, max_len - plen - 1))))]
        elif role == "kv_producer" and pd:
            r["remote_decode"] = True
            r["max_tokens"] = 1
            r["send_delay"] = rng.randint(1, 4)
            r["output"] = [rng.randint(1, 50000)]
        else:
            r["output"] = [rng.randint(1, 50000) for _ in range(rng.randint(1, max(1, min(12, max_len - plen - 1))))]
        if rng.random() < 0.1:
            # The client aborts it, perhaps while its pull is in flight.
            r["abort_step"] = r["arrival_step"] + rng.randint(0, 6)
        reqs.append(r)
    biggest = max(len(r["prompt"]) + len(r["output"]) for r in reqs)
    need_one = -(-min(biggest, max_len) // block) + 1
    need_all = sum(-(-(len(r["prompt"]) + len(r["output"])) // block) for r in reqs) + 1
    blocks = rng.randint(need_one, max(need_one, need_all))
    cfg_extra = {}
    if rng.random() < 0.2:
        # A watermark: a waiting request, a landed pull among them, is admitted only if it
        # leaves this share of the pool free while others run.
        cfg_extra["watermark"] = rng.choice([0.05, 0.1, 0.2])
    return {
        "name": f"cfuzz-{seed}", "seed": seed,
        "config": {"max_num_seqs": seqs, "max_num_batched_tokens": budget, "block_size": block,
                   "num_blocks": blocks, "max_model_len": max_len, "long_prefill_token_threshold": threshold,
                   "enable_prefix_caching": caching, "scheduling_policy": policy_sched, **cfg_extra,
                   "connector": {"name": "NixlConnector", "kv_role": role, "kv_load_failure_policy": policy,
                                 "extra_config": rng.choice([{}, {"kv_recompute_threshold": 0}])}},
        "requests": reqs, "max_steps": 400,
    }


def main() -> None:
    out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for seed in range(first, first + count):
        with open(os.path.join(out, f"cfuzz-{seed}.json"), "w") as f:
            json.dump(scenario(seed), f, indent=1)


if __name__ == "__main__":
    main()
