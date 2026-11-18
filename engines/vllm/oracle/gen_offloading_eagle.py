# Original to blis-vllm.
"""Write the offloading scenarios whose KV config marks a group as the drafter's (is_eagle_group).

Each is an existing offloading scenario with an EAGLE-3 or an MTP drafter added (use_eagle_block_drop()
is true for eagle, eagle3 and mtp alike, vllm/config/speculative.py:2122-2149@4ba787ec) and one KV cache group
marked "eagle": true, as kv_cache_planning marks a group that holds a draft layer
(vllm/v1/core/kv_cache_planning.py:1652-1655@4ba787ec). run_offloading.py tags the group so.

    python3 -I oracle/gen_offloading_eagle.py
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "scenarios", "offloading")
K = 2

# (source scenario, new name, index of the group marked as the drafter's, config edits, drafter)
EAGLE3 = {"method": "eagle3", "num_speculative_tokens": K, "draft": "llama-eagle"}
# MTP drafters are configuration-only too (oracle/drafts/mtp-1, mtp-4): the target's own MTP layers, as
# vLLM builds them from num_nextn_predict_layers.
MTP1 = {"method": "mtp", "num_speculative_tokens": 1, "draft": "mtp-1"}
MTP4 = {"method": "mtp", "num_speculative_tokens": 3, "draft": "mtp-4"}
CASES = [
    ("offloading-lru-evict", "offloading-eagle3-lru-evict", 0, {}, EAGLE3),
    ("offloading-hybrid-swa", "offloading-eagle3-hybrid-full", 0, {}, EAGLE3),
    ("offloading-hybrid-swa", "offloading-eagle3-hybrid-window", 1, {}, EAGLE3),
    ("offloading-chunk2-prompt-only", "offloading-eagle3-chunk2", 0, {}, EAGLE3),
    ("offloading-hybrid-swa", "offloading-eagle3-hybrid-full-async", 0, {"async_scheduling": True}, EAGLE3),
    ("offloading-lru-evict", "offloading-mtp1-lru-evict", 0, {}, MTP1),
    ("offloading-hybrid-swa", "offloading-mtp1-hybrid-full", 0, {}, MTP1),
    ("offloading-hybrid-swa", "offloading-mtp1-hybrid-window", 1, {}, MTP1),
    ("offloading-hybrid-swa", "offloading-mtp4-hybrid-full", 0, {}, MTP4),
    ("offloading-hybrid-swa", "offloading-mtp1-hybrid-full-async", 0, {"async_scheduling": True}, MTP1),
]

for src, name, eagle, edits, drafter in CASES:
    sc = json.load(open(os.path.join(SRC, src + ".json")))
    c = sc["config"]
    c.update(edits)
    c["speculative"] = dict(drafter)
    k = drafter["num_speculative_tokens"]
    # vLLM refuses a token budget no larger than the draft slots (vllm/config/vllm.py:2285-2290).
    c["max_num_batched_tokens"] = max(c["max_num_batched_tokens"], 4 * (k + 1) * c["max_num_seqs"])
    groups = copy.deepcopy(c.get("kv_groups") or [{"kind": "full"}])
    groups[eagle]["eagle"] = True
    c["kv_groups"] = groups
    for i, r in enumerate(sc["requests"]):
        r["accepted"] = [(i + j) % (k + 1) for j in range(len(r["output"]))]
    sc["name"] = name
    json.dump(sc, open(os.path.join(SRC, name + ".json"), "w"), indent=1)
    print(name)
