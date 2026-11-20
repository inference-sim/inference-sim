# Original to blis-vllm.
"""Seeded random hybrid Mamba scenarios for the tier-2 fuzz test.

Each seed gives one scenario: a model with one attention layer type (grouped full attention or
MLA) and one to three Mamba layers, a random scheduler configuration (token budget, sequences,
long-prefill cap, FCFS or priority) and a random request stream (arrivals, shared prefixes,
repeated prompts, client budgets). The same seed always gives the same scenario, from Python's
random.Random alone, so a failing seed reproduces anywhere. The scenario runs on vLLM v0.32.0's
real Scheduler through oracle/run_scheduler.py, as every tier-2 scenario does, and the Go side
reads the pair (scenario, trace) through core/sched's fuzz test (BLIS_VLLM_FUZZ_DIR).

What a seed draws, each as the engine would run it (vllm/platforms/interface.py:759-1126@5df63a72):
  - the attention block size 16, 32 or 64, which is the engine's block size;
  - "align" mode (prefix caching on, mamba block size = block size; seven seeds in ten) or "none"
    mode (prefix caching off, mamba block size = max_model_len);
  - a Mamba state of 50% to 100% of the attention page, padded to that page;
  - in align mode a prefill checkpoint in half the seeds (1 block, alignment 1 as Mamba-2 or 16 as
    KDA with flashkda), and prefix_cache_retention_interval left at vLLM's default 0 in four
    seeds in five, else dense (null) or a periodic multiple of the block size;
  - n-gram speculation of k tokens with num_speculative_blocks k in one seed in five;
  - a token budget from 8 to 128, below and above the block size;
  - a KV pool between "one request fits" and "everything fits", so some seeds preempt.

Usage: python3 -I oracle/gen_fuzz_mamba.py <out dir> <first seed> <count>
"""

import json
import os
import random
import sys

BLOCKS = [16, 32, 64]
BUDGETS = [8, 16, 24, 32, 64, 128]
MAX_LENS = [256, 512]


def scenario(seed: int) -> dict:
    rng = random.Random(seed * 6700417 + 11)
    block = rng.choice(BLOCKS)
    align = rng.random() < 0.7
    mla = rng.random() < 0.3
    mamba_layers = rng.choice([1, 1, 1, 3])
    max_len = rng.choice(MAX_LENS)
    budget = rng.choice(BUDGETS)
    seqs = min(rng.randint(1, 6), budget)
    policy = rng.choice(["fcfs", "fcfs", "priority"])
    threshold = rng.choice([0, 0, 0, rng.randint(1, min(budget, max_len))])

    if mla:
        # bfloat16, 576 elements a token (Kimi-like): page = block * 576 * 2.
        attn = {"count": 1, "spec": "mla", "head_size": 576, "elem_bytes": 2, "block_size": block}
        page = block * 576 * 2
    else:
        heads, head = rng.choice([1, 2, 4]), rng.choice([4, 8])
        attn = {"count": 1, "spec": "full", "num_kv_heads": heads, "head_size": head}
        page = 2 * block * heads * head * 4
    state = page - rng.choice([0, page // 8, page // 4, page // 2])

    ckpt, align_tokens, spec_blocks, ngram = 0, None, 0, None
    cache_extra = {}
    if align:
        if rng.random() < 0.5:
            ckpt, align_tokens = 1, rng.choice([1, 16])
        r = rng.random()
        if r < 0.1:
            cache_extra["prefix_cache_retention_interval"] = None
        elif r < 0.2:
            cache_extra["prefix_cache_retention_interval"] = block * rng.randint(1, 4)
    if rng.random() < 0.2:
        spec_blocks = rng.randint(1, 3)
        ngram = {"method": "ngram", "num_speculative_tokens": spec_blocks}
        budget = max(budget, 2 * spec_blocks + 2)
        seqs = min(seqs, budget)
        threshold = min(threshold, max_len)
    mamba = {"count": mamba_layers, "spec": "mamba", "state_bytes": state, "page_size_padded": page,
             "block_size": block if align else max_len,
             "mamba_cache_mode": "align" if align else "none",
             "num_speculative_blocks": spec_blocks, "num_prefill_checkpoint_blocks": ckpt}
    if ckpt:
        mamba["prefill_checkpoint_alignment"] = align_tokens

    # Requests: shared prefixes (some a block multiple, some not), whole-prompt repeats, outputs.
    n = rng.randint(1, 8)
    shared = [rng.randint(1, 50000) for _ in range(rng.randint(0, 5 * block))]
    reqs = []
    prompts = []
    for i in range(n):
        plen = rng.randint(1, min(max_len // 2, 6 * block))
        r = rng.random()
        if prompts and r < 0.2:
            prompt = list(rng.choice(prompts))  # a whole earlier prompt again
        else:
            prompt = shared[: min(len(shared), plen)] if shared and r < 0.6 else []
            prompt += [rng.randint(1, 50000) for _ in range(plen - len(prompt))]
        prompts.append(prompt)
        olen = rng.randint(1, max(1, min(24, max_len - len(prompt) - 1)))
        req = {"id": f"q{i:02d}", "arrival_step": rng.randint(0, 12), "prompt": prompt,
               "output": [rng.randint(1, 50000) for _ in range(olen)],
               "priority": rng.randint(0, 3) if policy == "priority" else 0}
        if rng.random() < 0.2:
            req["max_tokens"] = rng.randint(1, olen + 2)
        if ngram:
            req["accepted"] = [rng.randint(0, spec_blocks) for _ in range(olen + 2)]
        reqs.append(req)

    # The pool: every request must fit alone, with the Mamba groups' state blocks (two states, the
    # checkpoint, the speculative blocks and a copy-on-write block each), so no request waits
    # forever; a tight pool is anything above that, up to everything at once.
    mamba_blocks = mamba_layers * (4 + spec_blocks + ckpt)
    look = spec_blocks + 1

    def full_blocks(r):
        return -(-(min(len(r["prompt"]) + len(r["output"]), max_len) + look) // block)

    one = max(full_blocks(r) for r in reqs) + mamba_blocks + 2
    everything = sum(full_blocks(r) + mamba_blocks for r in reqs) + 2
    blocks = rng.randint(one, max(one, everything))

    cfg = {"max_num_seqs": seqs, "max_num_batched_tokens": budget, "block_size": block,
           "num_blocks": blocks, "max_model_len": max_len, "long_prefill_token_threshold": threshold,
           "enable_prefix_caching": align, "scheduling_policy": policy,
           "kv_layers": [attn, mamba]}
    if ngram:
        cfg["speculative"] = ngram
    if cache_extra:
        cfg["cache_config"] = cache_extra
    return {"name": f"fuzz-mamba-{seed}", "seed": seed, "config": cfg, "requests": reqs, "max_steps": 2000}


def main() -> None:
    out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for seed in range(first, first + count):
        with open(os.path.join(out, f"fuzz-mamba-{seed}.json"), "w") as f:
            json.dump(scenario(seed), f, indent=1)


if __name__ == "__main__":
    main()
