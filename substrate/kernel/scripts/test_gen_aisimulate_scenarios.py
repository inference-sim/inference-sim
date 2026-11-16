#!/usr/bin/env python3
"""Behavioural tests for the scenario generator's engine settings.

The defect these cover (#73): every scenario was written with gpu_memory_utilization 0.9,
so deployments that differed only in it were given byte-identical predictions.

Run: python3 scripts/test_gen_aisimulate_scenarios.py
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_aisimulate_scenarios import DEFAULT_CATALOG, emit  # noqa: E402

FAILURES = []


def ck(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def sweep(model="minimax-m2.5", stated=None, gpu="h200", gpus_per_node=8, framework="vllm"):
    s = {
        "scenario": f"{model}-{gpu}-fp8-{framework}-tp4.yaml", "model": model, "gpu": gpu,
        "precision": "fp8", "framework": framework,
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
    # X19: a default is never written for an engine it does not belong to (a non-vLLM
    # deployment used to be given vLLM's 0.92).
    for fw in ("sglang", "trt", "trtllm"):
        ck(f"an unstated gpu_memory_utilization is left out under {fw}, not vLLM's 0.92",
           engine(sweep(stated={}, framework=fw), "gpu_memory_utilization") is None)
        text = emit(sweep(stated={}, framework=fw), {"1024:1024"}, {"1k1k"})
        ck(f"the {fw} scenario says the value is unstated and not vLLM's",
           "0.92" not in text and "knows no default for " + fw in text)
        ck(f"a stated gpu_memory_utilization is still written under {fw}",
           engine(sweep(stated={"gpu_memory_utilization": 0.8}, framework=fw),
                  "gpu_memory_utilization") == "0.8")
    ck("a vLLM scenario records the framework contract that supplies 0.92",
       "vllm/config/cache.py:103" in emit(sweep(stated={}), {"1024:1024"}, {"1k1k"}))
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

    # #74: a speculator the kernel does not price is never written as a target-only file.
    eagle = sweep(model="minimax-m3", stated={"speculative_method": "eagle3",
                                              "num_speculative_tokens": 3})
    ck("an unpriced speculator (minimax-m3 eagle3) emits no scenario",
       emit(eagle, {"1024:1024"}, {"1k1k"}) is None)
    # inference-sim/blis-substrate#124: a run that names its EAGLE3 checkpoint drafts with the
    # catalog's drafter entry for it, which the kernel prices; one the catalog lacks is skipped.
    for repo, entry in (("Inferact/MiniMax-M3-EAGLE3", "minimax-m3-eagle3"),
                        ("Inferact/MiniMax-M3-EAGLE3-GQA", "minimax-m3-eagle3-gqa")):
        named = sweep(model="minimax-m3", stated={"speculative_method": "eagle3",
                                                  "num_speculative_tokens": 3, "draft_model": repo})
        ck(f"an EAGLE3 run that names {repo} drafts with {entry}",
           (engine(named, "method"), engine(named, "draft_model")) == ("eagle3", entry))
    for draft in ("/data/models/MiniMax-M3-EAGLE3", "Org/Not-A-Drafter"):
        unknown = sweep(model="minimax-m3", stated={"speculative_method": "eagle3",
                                                    "num_speculative_tokens": 3, "draft_model": draft})
        ck(f"an EAGLE3 run whose draft {draft} is no catalog drafter emits no scenario",
           emit(unknown, {"1024:1024"}, {"1k1k"}) is None)
    own = sweep(model="deepseek-v4-pro", stated={"speculative_method": "mtp", "num_speculative_tokens": 2,
                                                 "draft_model": "deepseek-ai/DeepSeek-V4-Pro"})
    ck("an MTP run that names the served model as its draft drafts with its own speculator",
       engine(own, "method") == "mtp" and engine(own, "draft_model") is None)
    lower = emit(sweep(model="deepseek-v4-pro", stated={
        "speculative_method": "mtp", "num_speculative_tokens": 2,
        "draft_model": "deepseek-ai/deepseek-v4-pro"}), {"1024:1024"}, {"1k1k"}) or ""
    ck("an MTP run that names its own repository in another case drafts with its own speculator",
       "method: mtp" in lower and "draft_model:" not in lower)
    named = emit(sweep(model="minimax-m3", stated={
        "speculative_method": "eagle3", "num_speculative_tokens": 3,
        "draft_model": "Inferact/MiniMax-M3-EAGLE3"}), {"1024:1024"}, {"1k1k"}) or ""
    ck("a drafting scenario's header says the run records no draft revision",
       "# speculative: stated by the run, drafting with the catalog's minimax-m3-eagle3; "
       "the run records no draft revision." in named.splitlines())

    # The catalog is read as text: a repo or method key outside the block that owns it is a decoy.
    with tempfile.TemporaryDirectory() as d:
        cat = Path(d)
        shutil.copytree(DEFAULT_CATALOG / "hardware", cat / "hardware")
        m = cat / "models" / "decoy"
        m.mkdir(parents=True)
        (m / "model.yaml").write_text(
            "name: decoy\nbase_model:\n  repo: Other/Base\n"
            "source:\n  provider: huggingface\n  repo: Org/Decoy\n  revision: abc\n"
            "drafters: [speculator]\n")
        (m / "graph.yaml").write_text(
            "kind: ModelGraph\nname: decoy\nspeculator:\n  stack:\n    method: eagle3\n"
            "  method: decoy_mtp\n  num_spec_tokens: 1\n")
        run = sweep(model="decoy", stated={"speculative_method": "mtp", "num_speculative_tokens": 1,
                                           "draft_model": "Org/Decoy"})
        text = emit(run, {"1024:1024"}, {"1k1k"}, cat) or ""
        ck("a run naming its source repo drafts with its own speculator despite a base_model repo",
           "method: mtp" in text and "draft_model:" not in text)
        bare = sweep(model="decoy", stated={"speculative_method": "mtp", "num_speculative_tokens": 1})
        ck("the speculator's own method is read, not one nested under another of its keys",
           "method: mtp" in (emit(bare, {"1024:1024"}, {"1k1k"}, cat) or ""))
        nested = sweep(model="decoy", stated={"speculative_method": "eagle3", "num_speculative_tokens": 1})
        ck("a method only nested inside the speculator block does not make the run priced",
           emit(nested, {"1024:1024"}, {"1k1k"}, cat) is None)

    # A DSpark head shipped in the target's own repository resolves to its catalog entry, whether
    # the run names the target's repository or nothing (vLLM loads it from there).
    for draft in ("MiniMaxAI/MiniMax-M3.1-Flash", None):
        stated = {"speculative_method": "dspark", "num_speculative_tokens": 3}
        if draft:
            stated["draft_model"] = draft
        shipped = sweep(model="minimax-m3.1-flash", stated=stated)
        ck(f"a DSpark head in the target's repository resolves (draft {draft})",
           engine(shipped, "draft_model") == "minimax-m3.1-flash-dspark")
    priced = sweep(model="minimax-m3", stated={"speculative_method": "mtp",
                                               "num_speculative_tokens": 3})
    ck("a speculator the graph describes still emits, with its block",
       "method: mtp" in (emit(priced, {"1024:1024"}, {"1k1k"}) or ""))
    eagle["scenario"] = "minimax-m3-h200-fp8-vllm-tp4-eagle3.yaml"
    priced["scenario"] = "minimax-m3-h200-fp8-vllm-tp4-mtp.yaml"
    script = Path(__file__).resolve().parent / "gen_aisimulate_scenarios.py"
    with tempfile.TemporaryDirectory() as d:
        corpus = Path(d) / "corpus.json"
        for s_ in (eagle, priced):
            s_.update(workload="1024:1024", label="1k1k")
        corpus.write_text(json.dumps({"sweeps": [eagle, priced]}))
        out = Path(d) / "out"
        r = subprocess.run([sys.executable, "-I", str(script), str(corpus), "-o", str(out)],
                           capture_output=True, text=True)
        files = sorted(p.name for p in out.glob("*.yaml"))
        ck("the generator writes only the priced scenario", files == [priced["scenario"]],
           str(files))
        ck("the generator reports the skipped scenario, by name and reason",
           r.returncode == 0 and eagle["scenario"] in r.stderr and "eagle3" in r.stderr
           and "1 scenario(s) not emitted" in r.stderr, r.stderr.strip())
        (out / eagle["scenario"]).write_text("stale\n")
        r = subprocess.run([sys.executable, "-I", str(script), str(corpus), "-o", str(out),
                            "--check"], capture_output=True, text=True)
        ck("--check fails on a leftover file for an unemitted scenario",
           r.returncode == 1 and eagle["scenario"] in r.stderr, r.stderr.strip())

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("PASS: 0 failure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
