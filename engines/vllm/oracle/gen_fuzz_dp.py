# Original to blis-vllm.
"""Seeded random scenarios for the DP part of the tier-3 fuzz test (#2021 B8).

Each seed gives a scenario for oracle/run_dp.py: a scheduler configuration as
oracle/gen_fuzz.py draws it (seeds below its SPEC_FROM, so no speculative decoding), plus a
data-parallel group: 2 to 4 ranks, prefill_schedule_interval 1 to 6, dp_sync_interval 1 to
16, a stats publication every 1 to 4 steps, from seed 2000 a delay of 0 to 6 steps on the
counts the coordinator publishes (so a snapshot can miss requests the client has routed),
and arrivals spread so some waves end and new ones start. The same seed always gives the
same scenario.

Usage: python -I oracle/gen_fuzz_dp.py <out dir> <first seed> <count>
"""

import json
import os
import random
import sys


def scenario(seed: int) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from gen_fuzz import scenario as t2  # noqa: E402

    sc = t2(seed)
    rng = random.Random(seed * 104729 + 7)
    sc["name"] = f"fuzz-dp-{seed}"
    cfg = sc["config"]
    cfg["data_parallel_size"] = rng.choice([2, 2, 3, 4])
    cfg["prefill_schedule_interval"] = rng.choice([1, 1, 2, 3, 4, 6])
    cfg["dp_sync_interval"] = rng.choice([1, 2, 4, 8, 16])
    cfg["stats_interval"] = rng.randint(1, 4)
    # Seeds from 2000 on also delay the coordinator's counts (the snapshot misses routing).
    if seed >= 2000:
        cfg["stats_lag"] = rng.randint(0, 6)
    # Spread arrivals over a longer span so waves end and restart.
    gap = rng.choice([0, 0, 10, 30])
    for r in sc["requests"]:
        if gap and rng.random() < 0.4:
            r["arrival_step"] += gap
    sc["max_steps"] = 6000
    return sc


def main() -> None:
    out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for seed in range(first, first + count):
        with open(os.path.join(out, f"fuzz-dp-{seed}.json"), "w") as f:
            json.dump(scenario(seed), f, indent=1)


if __name__ == "__main__":
    main()
