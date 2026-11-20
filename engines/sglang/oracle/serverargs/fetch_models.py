# Original to blis-sglang.
"""Fetch each model of cases.json as configuration only (config.json and the tokenizer and
generation configs, no weights) at the revision cases.json pins, into <dir>/<name>/.

A model whose entry states "checkpoint_files": true also gets what SGLang reads of a checkpoint beside
its configuration: hf_quant_config.json (configs/model_config.py _parse_quant_hf_config) and what
has_fp8_weights_in_checkpoint (utils/common.py@297e54fd) reads: model.safetensors.index.json and,
of the one shard that function opens, the 8-byte length and the JSON header, not the tensors. The
shard is chosen as that function chooses it, and the file is written as the first 8 + header-length
bytes of the shard; the function reads no further.

Usage: python -I oracle/serverargs/fetch_models.py <cases.json> <dir>
"""
import json
import os
import struct
import sys
import urllib.request


def _range(repo: str, revision: str, name: str, start: int, end: int) -> bytes:
    url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, name)
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end - 1)})
    with urllib.request.urlopen(req) as r:
        return r.read()


def fetch_header(repo: str, revision: str, dest: str) -> None:
    with open(os.path.join(dest, "model.safetensors.index.json")) as f:
        weight_map = json.load(f).get("weight_map", {})
    expert_files = sorted({v for k, v in weight_map.items() if "experts" in k and "weight" in k})
    shard = expert_files[0] if expert_files else next(iter(sorted(set(weight_map.values()))))
    (n,) = struct.unpack("<Q", _range(repo, revision, shard, 0, 8))
    header = _range(repo, revision, shard, 8, 8 + n)
    if len(header) != n:
        raise RuntimeError("short header read for %s/%s" % (repo, shard))
    with open(os.path.join(dest, shard), "wb") as f:
        f.write(struct.pack("<Q", n) + header)


def main(cases_path: str, out: str) -> int:
    from huggingface_hub import snapshot_download

    with open(cases_path) as f:
        models = json.load(f)["models"]
    for name, m in sorted(models.items()):
        patterns = ["config.json", "generation_config.json", "tokenizer_config.json"]
        if m.get("checkpoint_files"):
            patterns += ["model.safetensors.index.json", "hf_quant_config.json"]
        dest = "%s/%s" % (out, name)
        snapshot_download(m["repo"], revision=m["revision"], local_dir=dest, allow_patterns=patterns)
        if m.get("checkpoint_files"):
            fetch_header(m["repo"], m["revision"], dest)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
