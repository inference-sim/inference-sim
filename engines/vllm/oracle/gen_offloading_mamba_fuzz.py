# Original to blis-vllm.
"""Seeded random OffloadingConnector scenarios beside Mamba KV cache groups, for
oracle/run_offloading_mamba.py.

Each seed gives one scenario: one to three Mamba "align" groups (speculative blocks 0-2,
prefill-checkpoint blocks 0-1), usually one full-attention group beside them, a block of 16 or 32
tokens, blocks_per_chunk 1 or 2, a GPU pool small enough that its prefix cache forgets (so a later
request finds its prefix only on the CPU tier), a CPU tier of a few chunks (so stores evict), a
request stream that repeats and extends a few prompts, with some aborts, and retention 0 (vLLM's
default) or a multiple of the block. The same seed always gives the same scenario.

The KV page is 16 KiB per block per group (head_size = 2048 / block), as in
oracle/gen_offloading_fuzz.py.

Usage: python -I oracle/gen_offloading_mamba_fuzz.py <out dir> <first seed> <count>
"""

import json
import os
import random
import sys


def scenario(seed: int) -> dict:
    rng = random.Random(seed)
    block = rng.choice([16, 16, 32])
    bpc = rng.choice([1, 1, 2])
    n_mamba = rng.randint(1, 3)
    groups = [{"kind": "mamba"} for _ in range(n_mamba)]
    if rng.random() < 0.85:
        groups.append({"kind": "full"})
    mamba = {"state_bytes": 1024, "num_speculative_blocks": rng.choice([0, 0, 1, 2]),
             "num_prefill_checkpoint_blocks": rng.choice([0, 1, 1]), "prefill_checkpoint_alignment": 16}
    if mamba["num_prefill_checkpoint_blocks"] == 0:
        mamba["prefill_checkpoint_alignment"] = None
    max_len = 1024
    shared = [[rng.randint(1, 5000) for _ in range(rng.randint(2 * block, 6 * block))] for _ in range(rng.randint(1, 3))]
    reqs = []
    for i in range(rng.randint(3, 9)):
        plen = rng.randint(block // 2, 7 * block)
        prompt = []
        if rng.random() < 0.7:
            base = rng.choice(shared)
            prompt = base[: min(len(base), plen)]
            if rng.random() < 0.4:
                prompt = list(base)
        prompt += [rng.randint(1, 50000) for _ in range(max(0, plen - len(prompt)))]
        r = {"id": f"q{i:02d}", "arrival_step": rng.randint(0, 40), "prompt": prompt,
             "output": [rng.randint(1, 50000) for _ in range(rng.randint(1, 9))]}
        if rng.random() < 0.08:
            r["abort_step"] = r["arrival_step"] + rng.randint(0, 6)
        if rng.random() < 0.2:
            r["load_delay"] = rng.randint(0, 4)
        if rng.random() < 0.2:
            r["store_delay"] = rng.randint(0, 4)
        reqs.append(r)
    # Later requests repeat or extend an earlier prompt, so their state is found on the CPU tier.
    for j in range(rng.randint(1, 4)):
        src = rng.choice(reqs)
        prompt = list(src["prompt"]) + [rng.randint(1, 50000) for _ in range(rng.choice([0, 0, 3, block, 2 * block]))]
        reqs.append({"id": f"q{len(reqs):02d}", "arrival_step": rng.randint(30, 90), "prompt": prompt,
                     "output": [rng.randint(1, 50000) for _ in range(rng.randint(1, 5))]})
    biggest = max(len(r["prompt"]) + len(r["output"]) for r in reqs)
    per_req = (-(-biggest // block) + 1 + mamba["num_speculative_blocks"] + 2) * len(groups)
    blocks = rng.randint(per_req, per_req * 2)
    chunk_bytes = 16384 * len(groups) * bpc
    extra = {"cpu_bytes_to_use": chunk_bytes * rng.randint(20, 120), "offload_prompt_only": rng.random() < 0.5}
    if bpc > 1:
        extra["blocks_per_chunk"] = bpc
    cfg = {"max_num_seqs": rng.randint(1, 4), "max_num_batched_tokens": rng.choice([64, 128, 256]), "block_size": block,
           "num_blocks": blocks, "max_model_len": max_len, "long_prefill_token_threshold": 0,
           "enable_prefix_caching": True, "scheduling_policy": "fcfs",
           "kv_page": {"num_kv_heads": 1, "head_size": 2048 // block},
           "transfer": {"load_delay": rng.randint(0, 3), "store_delay": rng.randint(0, 3)},
           "connector": {"name": "OffloadingConnector", "kv_role": "kv_both", "extra_config": extra},
           "mamba": mamba, "kv_groups": groups}
    if rng.random() < 0.3:
        cfg["prefix_cache_retention_interval"] = block * bpc * rng.choice([1, 2, 4])
    return {"name": f"omamba-{seed}", "seed": seed, "config": cfg, "requests": reqs, "max_steps": 500}


def main() -> None:
    out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for seed in range(first, first + count):
        with open(os.path.join(out, f"omamba-{seed}.json"), "w") as f:
            json.dump(scenario(seed), f, indent=1)


if __name__ == "__main__":
    main()
