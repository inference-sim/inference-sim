# Original to blis-vllm.
"""Random scenarios for oracle/run_lora_cache.py (the worker's LoRA adapter cache).

Usage: python3 -I oracle/gen_fuzz_lora.py <out dir> <first seed> <count>

Each seed (python's random.Random) picks max_loras 1-4, a max_cpu_loras from max_loras to max_loras+4
(or none, which vLLM sets to max_loras), a pool of 2-9 adapters, and 6-30 batches of 1 to
max_loras+1 adapters drawn from the pool, with repeats. A batch of more adapters than slots is the
error vLLM raises; it is kept, one batch in about twenty.
"""
import json
import os
import random
import sys


def scenario(seed: int) -> dict:
    r = random.Random(seed)
    slots = r.randint(1, 4)
    cfg = {"max_loras": slots}
    if r.random() < 0.85:
        cfg["max_cpu_loras"] = r.randint(slots, slots + 4)
    pool = [chr(ord("a") + i) for i in range(r.randint(2, 9))]
    steps = []
    for _ in range(r.randint(6, 30)):
        n = r.randint(1, slots) if r.random() < 0.95 else slots + 1
        steps.append({"adapters": [r.choice(pool) for _ in range(n)]})
    return {"name": f"lora-fuzz-{seed}", "config": cfg, "steps": steps}


def main() -> None:
    out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for seed in range(first, first + count):
        with open(os.path.join(out, f"fuzz-lora-{seed}.json"), "w") as f:
            json.dump(scenario(seed), f)


if __name__ == "__main__":
    main()
