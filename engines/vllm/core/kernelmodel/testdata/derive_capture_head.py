"""Derives capture_head_derived.json: the CUDA graph capture vLLM main head resolves for the
oracle's capture cases that state neither max_cudagraph_capture_size nor cudagraph_capture_sizes.

usage: derive_capture_head.py VLLM_SOURCE > capture_head_derived.json

The oracle (core/lower/testdata/vllm_lower.json.gz) was recorded at v0.33.0, where an unstated
capture is one shared ladder; at the head it is a prefill token grid and a decode request grid per
platform. This is derived from the head's source by running its own resolver
(core/rulepack/pack/testdata/cudagraph/harness.py): "derived from source, oracle not re-run".
A maintainer with a GPU-free oracle environment re-records the oracle at the head and deletes this
file.
"""
import gzip
import json
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
root = sys.argv[1]
oracle = json.load(gzip.open(here.parent.parent / "lower" / "testdata" / "vllm_lower.json.gz"))
sel = []
for c in oracle["capture"]:
    k = c["case"]
    if c["resolved"].get("error") or k.get("enforce_eager") or k["max_num_batched_tokens"] <= 0:
        continue
    if k.get("performance_mode") == "interactivity":
        continue
    if k.get("max_capture") is not None or k.get("sizes") is not None:
        continue
    sel.append(c)
cases = [{
    "seqs": c["case"]["max_num_seqs"], "mnbt": c["case"]["max_num_batched_tokens"],
    "spec": c["case"].get("spec_tokens", 0), "family100": bool(c["case"].get("blackwell", False)),
    "mode": c["case"]["mode"], "method": "mtp" if c["case"].get("spec_tokens", 0) else "", "npredict": 1,
} for c in sel]
res = json.loads(subprocess.run(
    [sys.executable, str(here.parent.parent / "rulepack/pack/testdata/cudagraph/harness.py"), root],
    input=json.dumps(cases), capture_output=True, text=True, check=True).stdout)
out = {
    "derivation": "derived from source, oracle not re-run",
    "vllm_commit": "0fda86a35419c9c70dbfada224c838c10fbb62ad",
    "script": "core/kernelmodel/testdata/derive_capture_head.py",
    "cases": {c["case"]["name"]: {"max": r["max"]} for c, r in zip(sel, res)},
}
json.dump(out, sys.stdout, indent=1, sort_keys=True)
print()
