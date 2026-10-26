# Original to blis-vllm.
"""Re-derive core/kv/testdata/block_hashes.json from vLLM's own block hasher.

Each case is a token sequence, a block size and the request's extra keys, read from the
committed file, so a case's inputs never move; its hashes are what
vLLM v0.32.0's hash_block_tokens returns (vllm/v1/core/kv_cache_utils.py:695-725@5df63a72) with
sha256 (vllm/utils/hashing.py) after init_none_hash, chained block by block with each block's
extra keys as generate_block_hash_extra_keys gives them: the LoRA key on every block, the cache
salt on the first (kv_cache_utils.py:592-595, 646-658). No hashing is reimplemented here.

The large-block cases pickle to more than 64 KiB, which CPython's pickler splits into several
frames (FRAME_SIZE_TARGET); they pin the Go pickler's framing to vLLM's bytes.

The committed file is the input: its cases are kept and their hashes recomputed, and the
large-block cases below are added once. A run whose recomputed hashes differ from the
committed ones means vLLM's hashing moved, and the file must not be overwritten unreviewed.

The file is written compact (no indentation): the large-block cases hold some 600 thousand
token ids, one per line otherwise.

The provenance block records what every oracle trace does (docs/testing.md, "Provenance"): the
vLLM version, commit and installed distribution, the Python version and the platform, the two
lock hashes, the blis-vllm commit and the mocks. The generator loads no model, runs no scenario
and replaces nothing, so it records no model and no scenario digest, its mock list is empty and
its mock version null; its inputs are the committed cases themselves.

Usage: python -I oracle/gen_block_hashes.py <vllm checkout> <committed json> > <new json>
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys


def cases(committed):
    rng = random.Random(2021)
    out = [{k: v for k, v in c.items() if k != "hashes"} for c in committed]
    have = {c["name"] for c in out}
    # Blocks whose pickle crosses 64 KiB: three-byte token ids (BININT2) put the boundary near
    # 21.8k tokens; five-byte ones (BININT) near 13.1k. Sizes straddle the boundary, including a
    # frame that fills exactly at the tuple's last element, with and without extra keys.
    for name, size, lo, hi, extra, salt in [
        ("frame-boundary-binint2-below", 21833, 300, 300, None, None),
        ("frame-boundary-binint2-at", 21834, 300, 300, None, None),
        ("frame-boundary-binint2-above", 21835, 300, 300, None, None),
        ("frame-boundary-binint2-at-lora", 21834, 300, 300, [["lora", "adapter-c", "/adapters/c"]], None),
        ("frame-boundary-binint2-at-salt", 21834, 300, 300, None, "salt"),
        # The frame closes right before a key's str (the adapter name), where only the frame
        # check before that str decides it. No case puts the boundary before a MARK tuple: a
        # block's extra keys are at most a LoRA key and a cache salt (kv_cache_utils.py:644-657),
        # so the only tuple of four or more items is the token tuple, which starts 40 bytes in.
        ("frame-boundary-before-a-str", 21831, 300, 300, [["lora", "adapter-d", "/adapters/d"]], None),
        ("large-block-binint", 16384, 70000, 151935, None, None),
        ("large-block-two-frames-each", 32768, 256, 151935, [["lora", "adapter-b", "/adapters/b"]], "s2"),
    ]:
        tokens = [rng.randint(lo, hi) for _ in range(2 * size + 3)]
        if name not in have:
            out.append({"name": name, "block_size": size, "tokens": tokens,
                        **({"extra": extra} if extra else {}), **({"salt": salt} if salt else {})})
    return out


def main() -> None:
    vllm_root = sys.argv[1]
    with open(sys.argv[2]) as f:
        committed = json.load(f)["cases"]
    from vllm.utils.hashing import sha256  # noqa: E402
    import vllm.v1.core.kv_cache_utils as kcu  # noqa: E402

    kcu.init_none_hash(sha256)
    records = []
    for c in cases(committed):
        bs, toks = c["block_size"], c["tokens"]
        lora = tuple(tuple(k) for k in c.get("extra") or [])
        parent, hashes = None, []
        for i in range(len(toks) // bs):
            keys = lora + ((("cache_salt", c["salt"]),) if c.get("salt") and i == 0 else ())
            parent = kcu.hash_block_tokens(sha256, parent, toks[i * bs:(i + 1) * bs], keys or None)
            hashes.append(bytes(parent).hex())
        records.append({**c, "hashes": hashes})
    for old, new in zip(committed, records):
        if old.get("hashes") != new["hashes"]:
            sys.exit(f"{old['name']}: vLLM's hashes differ from the committed ones; review before overwriting")

    def git(*args):
        return subprocess.run(["git", "-C", vllm_root, *args], capture_output=True, text=True).stdout.strip()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from run_scheduler import blis_vllm_commit, committed_lock_sha256  # noqa: E402

    # ORACLE_LOCKFILE names the frozen package list of the environment that ran vLLM, as for
    # every oracle trace (oracle/run_scheduler.py provenance).
    lockfile = os.environ.get("ORACLE_LOCKFILE")
    lock_digest = "unrecorded"
    if lockfile:
        with open(lockfile, "rb") as f:
            lock_digest = hashlib.sha256(f.read()).hexdigest()
    json.dump({
        "provenance": {
            "generator": "oracle/gen_block_hashes.py",
            "vllm_version": importlib.metadata.version("vllm").split("+")[0],
            "vllm_distribution": importlib.metadata.version("vllm"),
            "vllm_commit": git("rev-parse", "HEAD"),
            "functions": ["vllm.v1.core.kv_cache_utils.hash_block_tokens", "vllm.v1.core.kv_cache_utils.init_none_hash",
                          "vllm.utils.hashing.sha256"],
            "python": ".".join(map(str, sys.version_info[:3])),
            "platform": platform.system().lower() + "-" + platform.machine(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
            "lockfile_sha256": lock_digest,
            "requirements_lock_sha256": committed_lock_sha256(),
            "blis_vllm_commit": blis_vllm_commit(),
            "mock_version": None,
            "mocks": [],
        },
        "none_hash_seed": kcu._NONE_HASH_SEED,
        "none_hash": bytes(kcu.NONE_HASH).hex(),
        "cases": records,
    }, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
