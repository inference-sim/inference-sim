# Original to blis-vllm.
"""Seeded random NIXL connector scenarios for the tier-4 fuzz test (#2021 B8).

Each seed gives one scenario: the engine's KV role (decoder or prefiller), the load-failure
policy, a scheduler configuration with a KV pool sized so admissions sometimes wait, and a
request stream mixing P/D requests with local ones. Decoder-side P/D requests get a pull delay,
and some get a whole-load failure or failed blocks; prefiller-side ones get a send delay. The
same seed always gives the same scenario, from Python's random.Random alone. About one request
in ten is aborted by its client a few steps after it arrives.

Seeds 10000 and up also run requests into max_model_len (LENGTH_CAP_SEEDS). Seeds 20000 and up
(HYBRID_SEEDS) also draw a hybrid model's KV cache groups (full attention beside one or two
sliding windows), a drafter (DSpark with up to 7 drafts and sample_from_anchor, the default; an
EAGLE-family draft; a four-module MTP drafter, whose lookahead sets NIXL's prefill backoff) with
a per-request acceptance plan, or both. A load's block-level failures ("invalid_blocks") are
drawn only beside one group: NIXL's worker reports none under HMA (nixl/base_worker.py:2910-2914
@5df63a72).

Usage: python -I oracle/gen_connector_fuzz.py <out dir> <first seed> <count>
"""

import json
import os
import random
import sys


LENGTH_CAP_SEEDS = 10000
HYBRID_SEEDS = 20000
QUEUED_SEEDS = 30000


def out_len(rng: random.Random, max_len: int, plen: int, capped: bool) -> int:
    """An assigned output length: inside the room max_model_len leaves, or, for a capped seed,
    sometimes past it."""
    room = max(1, min(12, max_len - plen - 1))
    if capped and rng.random() < 0.5:
        return rng.randint(max_len - plen, max_len - plen + 8)
    return rng.randint(1, room)


def hybrid_dims(rng: random.Random, block: int) -> dict:
    """A hybrid seed's KV cache groups and drafter, from their own random stream."""
    out = {}
    roll = rng.random()
    if roll < 0.7:
        groups = [{"kind": "full"}, {"kind": "sliding_window", "sliding_window": rng.choice([block, 2 * block, 8, 10, 16])}]
        if rng.random() < 0.25:
            groups.append({"kind": "sliding_window", "sliding_window": rng.choice([4, 12, 24])})
        out["kv_groups"] = groups
    if roll >= 0.35 or "kv_groups" not in out:
        out["speculative"] = rng.choice([
            {"method": "dspark", "num_speculative_tokens": rng.choice([3, 5, 7, 7])},
            {"method": "mtp", "num_speculative_tokens": rng.randint(2, 4), "draft": "ernie-mtp-4"},
            {"method": "mtp", "num_speculative_tokens": 1, "draft": "ernie-mtp-4"},
        ])
    return out


def scenario(seed: int) -> dict:
    if seed >= QUEUED_SEEDS:
        return queued_scenario(seed)
    if seed >= HYBRID_SEEDS:
        return hybrid_scenario(seed)
    # Seeds from LENGTH_CAP_SEEDS on let a request run past max_model_len: its assigned output
    # is longer than the room left, so vLLM's length cap (check_stop, scheduler.py:2506) ends it.
    # Earlier seeds keep every output inside the room, so their scenarios are unchanged.
    return base_scenario(random.Random(seed), seed, seed >= LENGTH_CAP_SEEDS)


def base_scenario(rng: random.Random, seed: int, capped: bool) -> dict:
    """The scenario a seed's stream draws, one group and no drafter."""
    role = rng.choice(["kv_consumer", "kv_consumer", "kv_producer"])
    policy = rng.choice(["fail", "recompute"])
    block = rng.choice([2, 4, 8])
    budget = rng.choice([8, 16, 32, 64])
    seqs = min(rng.randint(1, 6), budget)
    max_len = rng.choice([24, 32, 48]) if capped else rng.choice([64, 128])
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
            r["output"] = [rng.randint(1, 50000) for _ in range(out_len(rng, max_len, plen, capped))]
        elif role == "kv_producer" and pd:
            r["remote_decode"] = True
            r["max_tokens"] = 1
            r["send_delay"] = rng.randint(1, 4)
            r["output"] = [rng.randint(1, 50000)]
        else:
            r["output"] = [rng.randint(1, 50000) for _ in range(out_len(rng, max_len, plen, capped))]
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


def hybrid_scenario(seed: int) -> dict:
    """A seed from HYBRID_SEEDS: the base scenario's draw (a quarter of them into
    max_model_len), then, from a second stream, a hybrid model's groups and a drafter with each
    request's acceptance plan. Beside several groups the block-level failures are dropped."""
    dims_rng = random.Random(seed * 7919 + 1)
    sc = base_scenario(random.Random(seed), seed, dims_rng.random() < 0.25)
    dims = hybrid_dims(dims_rng, sc["config"]["block_size"])
    sc["config"].update(dims)
    spec = dims.get("speculative")
    if spec and spec["method"] == "mtp" and spec["num_speculative_tokens"] > 1 and sc["config"]["enable_prefix_caching"]:
        # vLLM refuses multi-module MTP with prefix caching below a block of num_speculative_tokens
        # (kv_cache_coordinator.py:146-155@5df63a72), so such a seed draws a block that holds them.
        sc["config"]["block_size"] = max(sc["config"]["block_size"], 4)
    if spec:
        k = spec["num_speculative_tokens"]
        # The budget must exceed the draft slots (vllm/config/vllm.py:2285-2290@5df63a72).
        sc["config"]["max_num_batched_tokens"] = max(sc["config"]["max_num_batched_tokens"], 2 * k + 2)
        sc["config"]["max_num_seqs"] = min(sc["config"]["max_num_seqs"], sc["config"]["max_num_batched_tokens"])
        if 0 < sc["config"]["long_prefill_token_threshold"] < 1 + k:
            # vLLM pads a first decode to 1 + K and then caps it at the long-prefill threshold,
            # which its own assertion refuses (scheduler.py:1134-1156, 1371-1372@5df63a72): a
            # cap below 1 + K crashes vLLM, so it is raised to 1 + K.
            sc["config"]["long_prefill_token_threshold"] = 1 + k
        for r in sc["requests"]:
            r["accepted"] = [dims_rng.randint(0, k) for _ in range(len(r["output"]) + 2)]
    ngroups = len(dims.get("kv_groups", [])) or 1
    if ngroups > 1:
        for r in sc["requests"]:
            r.pop("invalid_blocks", None)
    # Every group takes its own blocks from the one pool, and a drafter's lookahead slots take
    # more: the base draw's pool is scaled so admissions still sometimes wait.
    lookahead = -(-(spec["num_speculative_tokens"] + 1) // sc["config"]["block_size"]) if spec else 0
    sc["config"]["num_blocks"] = ngroups * (sc["config"]["num_blocks"] + lookahead) + 1
    return sc


def queued_scenario(seed: int) -> dict:
    """A seed from QUEUED_SEEDS: a hybrid seed's draw (a quarter with neither groups nor a
    drafter), then, from a third stream, the batch queue."""
    q_rng = random.Random(seed * 104729 + 3)
    sc = hybrid_scenario(seed)
    cfg = sc["config"]
    if q_rng.random() < 0.25:
        cfg.pop("kv_groups", None)
        if cfg.pop("speculative", None):
            for r in sc["requests"]:
                r.pop("accepted", None)
    producer_one_group = cfg["connector"]["kv_role"] == "kv_producer" and len(cfg.get("kv_groups", [])) <= 1
    roll = q_rng.random()
    if producer_one_group and roll < 0.3:
        cfg["pipeline_parallel_size"] = q_rng.choice([2, 4])
        if q_rng.random() < 0.5:
            cfg["async_scheduling"], cfg["use_v2_model_runner"] = True, True
    else:
        cfg["async_scheduling"] = True
        if q_rng.random() < 0.4:
            cfg["use_v2_model_runner"] = True
    sc["max_steps"] = 600
    return sc


def main() -> None:
    out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for seed in range(first, first + count):
        with open(os.path.join(out, f"cfuzz-{seed}.json"), "w") as f:
            json.dump(scenario(seed), f, indent=1)


if __name__ == "__main__":
    main()
