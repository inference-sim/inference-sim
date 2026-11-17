# Original to blis-sglang.
"""Scenario generator for the scheduler replay (S13): seeded random scenarios of oracle/run_sched.py's
format, each a different mix of arrivals, prompt lengths, shared prefixes, generation lengths, page
size, pool size, chunk size, policy and mixed chunk. The same seed makes the same file.

Usage: python3 -I oracle/gen_fuzz.py <seed> <out.json> [--policy P] [--page N] [--mixed] [--tight]
"""
import argparse
import json
import random


def scenario(seed: int, policy: str, page: int, mixed: bool, tight: bool) -> dict:
    rng = random.Random(seed)
    n = rng.randint(4, 9)
    chunk = rng.choice([16, 32, 48, 64])
    chunk = max(chunk, 2 * page) // page * page
    shared = [1000 + i for i in range(rng.randint(8, 40))]
    reqs = []
    for i in range(n):
        plen = rng.randint(6, 70)
        if rng.random() < 0.5:
            prompt = shared[:rng.randint(1, len(shared))] + [5000 + 100 * i + j for j in range(plen)]
        else:
            prompt = [2000 + 100 * i + j for j in range(plen)]
        reqs.append({"rid": f"q{i}", "arrival_step": rng.randint(0, 6), "prompt": prompt,
                     "max_new_tokens": rng.randint(1, 24), "ignore_eos": rng.random() < 0.3})
    reqs.sort(key=lambda r: (r["arrival_step"], r["rid"]))
    pool = rng.choice([160, 224, 320]) if tight else 4096
    if page > 1:
        pool = pool // page * page
    return {
        "description": f"fuzz seed {seed}: {policy}, page {page}, mixed {mixed}, pool {pool}",
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "seed": 0, "vocab_size": 151936, "sampled_token": 13,
        "device_memory_gib": 80, "step_seconds": 0.01, "page_size": page, "max_total_tokens": pool,
        "max_running_requests": rng.choice([3, 4, 8]), "chunked_prefill_size": chunk,
        "schedule_policy": policy, "overlap": False, "disable_radix_cache": False, "max_steps": 400,
        "server_args": {"enable_mixed_chunk": True} if mixed else {},
        "record_server_args": ["chunked_prefill_size", "max_running_requests", "max_total_tokens", "page_size", "schedule_policy"],
        "requests": reqs,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--policy", default="fcfs")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--tight", action="store_true")
    a = ap.parse_args()
    with open(a.out, "w") as f:
        json.dump(scenario(a.seed, a.policy, a.page, a.mixed, a.tight), f, indent=1)
