#!/usr/bin/env python3
"""Behavioural tests for the scenario generator's engine settings.

The defect these cover (#73): every scenario was written with gpu_memory_utilization 0.9,
so deployments that differed only in it were given byte-identical predictions.

Run: python3 scripts/test_gen_aisimulate_scenarios.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_aisimulate_scenarios import emit  # noqa: E402

FAILURES = []


def ck(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def sweep(model="minimax-m2.5", stated=None, gpu="h200", gpus_per_node=8):
    s = {
        "scenario": f"{model}-{gpu}-fp8-vllm-tp4.yaml", "model": model, "gpu": gpu,
        "precision": "fp8", "framework": "vllm",
        "parallelism": {"tp_size": 4, "pp_size": 1, "attention_dp_size": 1,
                        "moe_ep_size": 1, "moe_tp_size": 4},
    }
    if gpus_per_node is not None:
        s["gpus_per_node"] = gpus_per_node
    if stated is not None:
        s["stated_config"] = stated
    return s


def engine(dep, field):
    text = emit(dep, {"1024:1024"}, {"1k1k"})
    m = re.search(rf"^ +{field}: (\S+)$", text, re.M)
    return m.group(1) if m else None


def main() -> int:
    ck("a stated gpu_memory_utilization is written as stated",
       engine(sweep(stated={"gpu_memory_utilization": 0.85}),
              "gpu_memory_utilization") == "0.85")
    ck("an unstated gpu_memory_utilization is vLLM v0.31.0's default, 0.92",
       engine(sweep(stated={}), "gpu_memory_utilization") == "0.92")
    ck("a corpus with no stated_config also takes 0.92",
       engine(sweep(), "gpu_memory_utilization") == "0.92")
    ck("a stated block_size is written as stated",
       engine(sweep(stated={"block_size": 128}), "block_size") == "128")
    ck("an unstated block_size follows the rule (16 for a plain model)",
       engine(sweep(stated={}), "block_size") == "16")
    ck("an unstated block_size follows the rule (256 for DeepSeek-V4)",
       engine(sweep("deepseek-v4-pro", stated={}), "block_size") == "256")
    ck("a stated enable_prefix_caching is written as stated",
       engine(sweep(stated={"enable_prefix_caching": False}),
              "enable_prefix_caching") == "false")
    ck("an unstated enable_prefix_caching is left to the engine default",
       engine(sweep(stated={}), "enable_prefix_caching") is None)
    mtp = sweep("deepseek-v4-pro", stated={"speculative_method": "mtp",
                                           "num_speculative_tokens": 2})
    ck("a stated speculative config is written as stated",
       (engine(mtp, "method"), engine(mtp, "num_spec_tokens")) == ("mtp", "2"))
    ck("an unstated speculative config is left out",
       engine(sweep("deepseek-v4-pro", stated={}), "method") is None)

    # Packaging is the catalog's where the chip's entry fixes it (SUB-3), and the row's otherwise.
    has_packaging = re.compile(r"^ +gpus_per_(node|rack):", re.M)
    for chip in ("gb200-nvl72", "gb300"):
        for stated in (None, 8):
            text = emit(sweep(gpu=chip, gpus_per_node=stated), {"1024:1024"}, {"1k1k"})
            ck(f"a {chip} row (row states {stated}) writes neither gpus_per_node nor gpus_per_rack",
               not has_packaging.search(text))
    wide = sweep(gpu="gb200-nvl72", gpus_per_node=None)
    wide["parallelism"] = {"tp_size": 8, "pp_size": 1, "attention_dp_size": 1,
                           "moe_ep_size": 1, "moe_tp_size": 8}
    ck("a gb200 tp8 row takes the catalog's 4 per tray: two nodes",
       engine(wide, "nodes") == "2")
    ck("an h200 row writes the node size its row states",
       engine(sweep(gpus_per_node=4), "gpus_per_node") == "4")
    try:
        emit(sweep(gpus_per_node=None), {"1024:1024"}, {"1k1k"})
        ck("an h200 row with no node size is refused", False, "emit returned")
    except SystemExit as e:
        ck("an h200 row with no node size is refused, not given 8", "gpus_per_node" in str(e))

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("PASS: 0 failure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
