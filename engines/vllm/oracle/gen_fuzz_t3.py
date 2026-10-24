# Original to blis-vllm.
"""Seeded random scenarios for the tier-3 fuzz test (#2021 B8).

Each seed gives one scenario for oracle/run_enginecore.py: a random scheduler configuration
as oracle/gen_fuzz.py draws it, plus an EngineCore configuration (async scheduling on or
off, pipeline_parallel_size 1, 2 or 4, the V1 or V2 model runner), so the batch queue runs
every size vLLM's max_concurrent_batches gives (1 to 5), with the async scheduler's
placeholders, PP decode spacing and preemptions with output in flight. The same seed always
gives the same scenario, from Python's random.Random alone.

Usage: python -I oracle/gen_fuzz_t3.py <out dir> <first seed> <count>
"""

import json
import os
import sys


def scenario(seed: int) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from gen_fuzz import scenario as t2  # noqa: E402

    import random

    sc = t2(seed)
    rng = random.Random(seed * 7919 + 3)
    sc["name"] = f"fuzz-t3-{seed}"
    sc["config"]["async_scheduling"] = rng.random() < 0.6
    sc["config"]["pipeline_parallel_size"] = rng.choice([1, 1, 2, 2, 4])
    sc["config"]["use_v2_model_runner"] = rng.random() < 0.5
    if sc["config"]["async_scheduling"] and sc["config"]["pipeline_parallel_size"] > 1:
        # The V1 model runner refuses async scheduling with PP
        # (vllm/config/vllm.py:2970-2974@5df63a72), so that pairing takes the V2 runner.
        sc["config"]["use_v2_model_runner"] = True
    sc["max_steps"] = 6000
    return sc


def main() -> None:
    out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for seed in range(first, first + count):
        with open(os.path.join(out, f"fuzz-t3-{seed}.json"), "w") as f:
            json.dump(scenario(seed), f, indent=1)


if __name__ == "__main__":
    main()
