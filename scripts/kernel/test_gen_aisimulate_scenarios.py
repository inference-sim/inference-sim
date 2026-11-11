#!/usr/bin/env python3
"""Behavioural tests for the scenario generator (design X19: only stated values are written).

The generator used to write vLLM v0.31.0's defaults for settings no run stated
(gpu_memory_utilization 0.92, block_size 16, max_num_seqs 256, cudagraph_mode PIECEWISE, a
cache dtype). These tests hold it to the rule that each engine value it writes is one the
corpus states, or max_model_len, which the engine requires and the generator derives from
the workload.

Run: python3 -I scripts/kernel/test_gen_aisimulate_scenarios.py
"""

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_aisimulate_scenarios as gen  # noqa: E402
from gen_aisimulate_scenarios import emit  # noqa: E402

FAILURES = []

# Engine-block keys the generator may write: the run's stated fields, the quantization the
# corpus's precision column names, and max_model_len, derived from the workload.
ALLOWED = {"quantization", "block_size", "gpu_memory_utilization", "enable_prefix_caching",
           "speculative_config", "method", "num_speculative_tokens", "max_model_len"}


def ck(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def sweep(model="minimax-m2.5", stated=None, precision="fp8", gpu="h200", gpus_per_node=None):
    s = {
        "scenario": f"{model}-{gpu}-{precision}-vllm-tp4.yaml", "model": model, "gpu": gpu,
        "precision": precision, "framework": "vllm",
        "parallelism": {"tp_size": 4, "pp_size": 1, "attention_dp_size": 1,
                        "moe_ep_size": 1, "moe_tp_size": 4},
    }
    if gpus_per_node is not None:
        s["gpus_per_node"] = gpus_per_node
    if stated is not None:
        s["stated_config"] = stated
    return s


def text(dep):
    return emit(dep, {"1024:1024"}, {"1k1k"})


def engine(dep, field):
    m = re.search(rf"^ +{field}: (\S+)$", text(dep), re.M)
    return m.group(1) if m else None


def engine_keys(doc):
    """The keys under each pool's engine: block."""
    keys, inside = set(), False
    for line in doc.splitlines():
        if re.match(r"^    engine:$", line):
            inside = True
            continue
        if inside:
            m = re.match(r"^( {6,})([a-z_]+):", line)
            if not m:
                inside = False
                continue
            keys.add(m.group(2))
    return keys


def only_stated(doc):
    return engine_keys(doc) <= ALLOWED


def fixture_catalog() -> Path:
    """A catalog holding only the packaging lines this test reads.

    This repository carries no catalog, so the test states what the real entries say:
    gb200-nvl72 and gb300 fix 4 per tray and 72 per rack (blis-substrate catalog/hardware/),
    and h200 fixes neither.
    """
    root = Path(tempfile.mkdtemp(prefix="gen-catalog-")) / "hardware"
    root.mkdir()
    for chip in ("h100", "h200", "b200", "b300", "a100-80", "a100-sxm", "l40s"):
        (root / f"{chip}.yaml").write_text("SMCount: 132\n")
    for chip in ("gb200-nvl72", "gb300"):
        (root / f"{chip}.yaml").write_text("GPUsPerNode: 4\nGPUsPerRack: 72\n")
    return root.parent


def main() -> int:
    gen.DEFAULT_CATALOG = fixture_catalog()
    emit.__defaults__ = (gen.DEFAULT_CATALOG,)
    ck("a stated gpu_memory_utilization is written as stated",
       engine(sweep(stated={"gpu_memory_utilization": 0.85}), "gpu_memory_utilization") == "0.85")
    ck("an unstated gpu_memory_utilization is left out",
       engine(sweep(stated={}), "gpu_memory_utilization") is None)
    ck("a corpus with no stated_config writes no gpu_memory_utilization",
       engine(sweep(), "gpu_memory_utilization") is None)
    ck("a stated block_size is written as stated",
       engine(sweep(stated={"block_size": 128}), "block_size") == "128")
    ck("an unstated block_size is left out, DSA and DeepSeek-V4 models included",
       all(engine(sweep(m, stated={}), "block_size") is None
           for m in ("minimax-m2.5", "glm-5", "deepseek-v4-pro")))
    ck("a stated enable_prefix_caching is written as stated",
       engine(sweep(stated={"enable_prefix_caching": False}), "enable_prefix_caching") == "false")
    ck("an unstated enable_prefix_caching is left out",
       engine(sweep(stated={}), "enable_prefix_caching") is None)
    mtp = sweep("deepseek-v4-pro", stated={"speculative_method": "mtp", "num_speculative_tokens": 2})
    ck("a stated speculative config is written as stated",
       (engine(mtp, "method"), engine(mtp, "num_speculative_tokens")) == ("mtp", "2"))
    ck("an unstated speculative config is left out",
       engine(sweep("deepseek-v4-pro", stated={}), "method") is None)
    ck("no batch cap, CUDA-graph mode or cache dtype is written",
       all(engine(sweep(stated={}), f) is None
           for f in ("max_num_seqs", "max_num_batched_tokens", "cudagraph_mode", "compilation_config", "kv_cache_dtype", "cache_dtype")))
    ck("max_model_len admits the workload",
       engine(sweep(), "max_model_len") == "32768")
    ck("precision fp4 is written as quantization nvfp4, bf16 as none",
       engine(sweep(precision="fp4"), "quantization") == "nvfp4"
       and engine(sweep(precision="bf16"), "quantization") is None)
    ck("the scenario names engine {kind: vllm, version: 0.32.0}, and no engine_version",
       re.search(r'^engine:\n  kind: vllm\n  version: "0.32.0"$', text(sweep()), re.M) is not None
       and "engine_version" not in text(sweep()))
    full = sweep("deepseek-v4-pro", stated={"block_size": 256, "gpu_memory_utilization": 0.9,
                                            "enable_prefix_caching": True,
                                            "speculative_method": "mtp",
                                            "num_speculative_tokens": 2}, precision="fp4")
    ck("every engine key written is stated or derived (X19)", only_stated(text(full)),
       str(sorted(engine_keys(text(full)) - ALLOWED)))
    # Known-bad input for the X19 check: a default written into the engine block is seen.
    bad = text(sweep()).replace("      max_model_len:", "      max_num_seqs: 256\n      max_model_len:")
    ck("the X19 check sees an unstated value (known-bad)", not only_stated(bad))

    # Packaging is the catalog's where the chip's entry fixes it (SUB-3), and the row's otherwise.
    has_packaging = re.compile(r"^ +gpus_per_(node|rack):", re.M)
    for chip in ("gb200-nvl72", "gb300"):
        for stated in (None, 8):
            out = emit(sweep(gpu=chip, gpus_per_node=stated), {"1024:1024"}, {"1k1k"})
            ck(f"a {chip} row (row states {stated}) writes neither gpus_per_node nor gpus_per_rack",
               has_packaging.search(out) is None)
    ck("an h200 row writes the node size it states",
       re.search(r"^  gpus_per_node: 4$", text(sweep(gpus_per_node=4)), re.M) is not None)
    ck("an h200 row that states none gets the corpora's 8-GPU board, and says it was assumed",
       re.search(r"^  gpus_per_node: 8$", text(sweep()), re.M) is not None
       and "gpus_per_node: 8 assumed: the corpus states none; every chip in it is an 8-GPU HGX board"
       in text(sweep()))
    ck("a node size the row states is not called assumed",
       "assumed" not in text(sweep(gpus_per_node=4)))
    for chip in ("h100", "h200", "b200", "b300", "a100-80", "a100-sxm"):
        ck(f"{chip} is on the 8-GPU HGX allowlist and gets 8, stated as assumed",
           gen.node_size({"gpu": chip, "scenario": "x.yaml"}, gen.DEFAULT_CATALOG) == (8, True))
    try:
        gen.node_size({"gpu": "l40s", "scenario": "x.yaml"}, gen.DEFAULT_CATALOG)
        ck("a chip outside the allowlist with no stated node size is refused", False)
    except SystemExit as e:
        ck("a chip outside the allowlist with no stated node size is refused (known-bad)",
           "not one of the 8-GPU HGX chips" in str(e), str(e))

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("PASS: 0 failure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
