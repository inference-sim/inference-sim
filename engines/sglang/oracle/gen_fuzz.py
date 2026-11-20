# Original to blis-sglang.
"""Scenario generator for the scheduler replay (S13): seeded random scenarios of oracle/run_sched.py's
format, each a different mix of arrivals, prompt lengths, shared prefixes, generation lengths, page
size, pool size, chunk size, policy and mixed chunk. The same seed makes the same file.

Usage: python3 -I oracle/gen_fuzz.py <seed> <out.json> [--policy P] [--page N] [--mixed] [--tight]
                                     [--priority [--low-first] [--threshold N] [--no-preempt]] [--routing]
                                     [--queue-limit N] [--priority-disabled] [--heavy] [--prefill-max N] [--oversize] [--no-cache] [--seed-arg N]
                                     [--truncate] [--overlap [--eos] [--no-consecutive-prefill-overlap]]

--priority gives each request a priority in -3..3 (some none) and turns priority scheduling on; --routing gives
each a routing key from three; --queue-limit sets max_queued_requests; --oversize adds a request SGLang
aborts at ingest for its length; --no-cache disables the radix cache (fixed-length requests then take
add_one_req_ignore_eos); --truncate sets --allow-auto-truncate (use it with --oversize: the long input is cut to the limit and runs); --overlap runs SGLang's overlap event loop (S14); --eos makes the sampled token the model's
end-of-sequence token, so each request that does not ignore it finishes at its first token, while a decode batch that
holds it is already launched (the extra discarded step); --no-consecutive-prefill-overlap sets
SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP; --heavy makes more requests that generate longer; --seed-arg sets the server's random_seed (the random policy's stream).
"""
import argparse
import json
import random


def scenario(seed: int, policy: str, page: int, mixed: bool, tight: bool, opt) -> dict:
    rng = random.Random(seed)
    n = rng.randint(10, 14) if opt.heavy else rng.randint(4, 9)
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
                     "max_new_tokens": rng.randint(20, 60) if opt.heavy else rng.randint(1, 24), "ignore_eos": rng.random() < 0.3})
    if opt.priority or opt.priority_disabled:
        for r in reqs:
            if rng.random() < 0.85:
                r["priority"] = rng.randint(-3, 3)
    if opt.routing:
        for r in reqs:
            if rng.random() < 0.8:
                r["routing_key"] = rng.choice(["ka", "kb", "kc"])
    if opt.no_cache:
        for r in reqs:
            r["ignore_eos"] = rng.random() < 0.7
    reqs.sort(key=lambda r: (r["arrival_step"], r["rid"]))
    pool = rng.choice([160, 224, 320]) if tight else 4096
    if page > 1:
        pool = pool // page * page
    server_args = {"enable_mixed_chunk": True} if mixed else {}
    if opt.priority:
        server_args["enable_priority_scheduling"] = True
        server_args["schedule_low_priority_values_first"] = opt.low_first
        server_args["priority_scheduling_preemption_threshold"] = opt.threshold
        server_args["disable_priority_preemption"] = opt.no_preempt
    if opt.priority_disabled:
        server_args["abort_on_priority_when_disabled"] = True
    if opt.prefill_max:
        server_args["prefill_max_requests"] = opt.prefill_max
    if opt.queue_limit:
        server_args["max_queued_requests"] = opt.queue_limit
    if opt.truncate:
        server_args["allow_auto_truncate"] = True
    if opt.oversize:
        reqs.append({"rid": "big", "arrival_step": 1, "prompt": [7000 + j for j in range(pool + 8)],
                     "max_new_tokens": 4, "ignore_eos": True})
        reqs.sort(key=lambda r: (r["arrival_step"], r["rid"]))
    if opt.eos:
        for r in reqs:
            r["ignore_eos"] = rng.random() < 0.4
    out = {
        "description": f"fuzz seed {seed}: {policy}, page {page}, mixed {mixed}, pool {pool}",
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "seed": opt.seed_arg, "vocab_size": 151936, "sampled_token": 151645 if opt.eos else 13,
        "device_memory_gib": 80, "step_seconds": 0.01, "page_size": page, "max_total_tokens": pool,
        "max_running_requests": rng.choice([3, 4, 8]), "chunked_prefill_size": chunk,
        "schedule_policy": policy, "overlap": opt.overlap, "disable_radix_cache": opt.no_cache, "max_steps": 1500 if opt.heavy else 400,
        "server_args": server_args,
        "record_server_args": ["chunked_prefill_size", "max_running_requests", "max_total_tokens", "page_size", "schedule_policy"],
        "requests": reqs,
    }
    if opt.no_consecutive_prefill_overlap:
        out["sglang_env"] = {"SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP": "1"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--policy", default="fcfs")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--tight", action="store_true")
    ap.add_argument("--priority", action="store_true")
    ap.add_argument("--low-first", action="store_true")
    ap.add_argument("--threshold", type=int, default=0)
    ap.add_argument("--no-preempt", action="store_true")
    ap.add_argument("--priority-disabled", action="store_true")
    ap.add_argument("--prefill-max", type=int, default=0)
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--routing", action="store_true")
    ap.add_argument("--queue-limit", type=int, default=0)
    ap.add_argument("--oversize", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--eos", action="store_true")
    ap.add_argument("--no-consecutive-prefill-overlap", action="store_true")
    ap.add_argument("--seed-arg", type=int, default=0)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        json.dump(scenario(a.seed, a.policy, a.page, a.mixed, a.tight, a), f, indent=1)
