# Original to blis-vllm.
"""Write the data-parallel scenarios that run vLLM's scheduler with enable_chunked_prefill false.

Each is an existing data-parallel scenario with chunked prefill off. vLLM then requires
max_num_batched_tokens >= max_model_len (vllm/config/scheduler.py), so the budget is raised to the model
length, and the scheduler admits a prompt only whole (scheduler.py waiting loop: a prompt that does not fit
the step's budget is not admitted in parts). The NIXL scenarios cover the roles a disaggregated deployment
runs under data parallelism: a kv_producer pool without prefix caching, a kv_consumer pool, the same with
async scheduling, and a consumer whose failed pull recomputes.

    python3 -I oracle/gen_dp_chunked_off.py
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (directory, source scenario, new name, config edits, description suffix)
CASES = [
    ("scenarios-dp-nixl", "dp2-nixl-prefill", "dp2-nixl-prefill-chunked-off", {"enable_prefix_caching": False},
     "Chunked prefill off and prefix caching off, as a disaggregated prefill pool runs: every prompt is admitted whole."),
    ("scenarios-dp-nixl", "dp2-nixl-decode", "dp2-nixl-decode-chunked-off", {},
     "Chunked prefill off."),
    ("scenarios-dp-nixl", "dp2-nixl-decode-async", "dp2-nixl-decode-async-chunked-off", {},
     "Chunked prefill off, with async scheduling, as a disaggregated decode pool runs."),
    ("scenarios-dp-nixl", "dp2-nixl-recv-failed", "dp2-nixl-recv-failed-chunked-off", {},
     "Chunked prefill off: the failed pull recomputes the prompt whole."),
    ("scenarios-dp", "dp4-lb", "dp4-lb-chunked-off", {}, "Chunked prefill off."),
    ("scenarios-dp", "dp2-kv-pressure", "dp2-kv-pressure-chunked-off", {},
     "Chunked prefill off under KV pressure: a preempted request is readmitted whole."),
]


def main() -> None:
    for directory, src, name, edits, note in CASES:
        path = os.path.join(HERE, directory, src + ".json")
        with open(path) as f:
            sc = json.load(f)
        out = copy.deepcopy(sc)
        out["name"] = name
        out["description"] = (sc.get("description", "").rstrip() + " " + note).strip()
        cfg = out["config"]
        cfg["enable_chunked_prefill"] = False
        cfg["max_num_batched_tokens"] = max(cfg["max_num_batched_tokens"], cfg["max_model_len"])
        cfg.update(edits)
        with open(os.path.join(HERE, directory, name + ".json"), "w") as f:
            json.dump(out, f)
            f.write("\n")


if __name__ == "__main__":
    main()
