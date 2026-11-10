# Original to blis-vllm.
"""Re-derive core/kv/testdata/seeded_block_hashes.json: block hashes under a stated PYTHONHASHSEED.

vLLM derives NONE_HASH, the parent of every prefix's first block, from PYTHONHASHSEED when it is
set (resolve_none_hash_seed, vllm/v1/core/kv_cache_utils.py:131-146@5df63a72); a hash then
differs from the default-seed hash of the same tokens. Each case sets the variable, runs vLLM's
own init_none_hash(sha256) and hash_block_tokens, and records the seed, NONE_HASH and the hashes.
The variable is set in this process before the call, one case at a time, and restored.

Usage: python -I oracle/gen_seeded_block_hashes.py <vllm checkout> > core/kv/testdata/seeded_block_hashes.json
"""
import json
import os
import random
import sys

SEEDS = [None, "0", "1234", "a-string-seed", ""]


def main() -> None:
    vllm_root = sys.argv[1]
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    from vllm.utils.hashing import sha256  # noqa: E402
    import vllm.v1.core.kv_cache_utils as kcu  # noqa: E402

    rng = random.Random(2022)
    tokens = [rng.randint(1, 50000) for _ in range(64)]
    lora = [["lora", "adapter-a", "/adapters/a"]]
    out = []
    for seed in SEEDS:
        if seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = seed
        kcu.init_none_hash(sha256)
        for name, extra in (("plain", None), ("lora", lora)):
            parent, hashes = None, []
            for i in range(len(tokens) // 16):
                keys = tuple(tuple(k) for k in extra) if extra else None
                parent = kcu.hash_block_tokens(sha256, parent, tokens[i * 16:(i + 1) * 16], keys)
                hashes.append(bytes(parent).hex())
            out.append({"pythonhashseed": seed, "name": name, "block_size": 16, "tokens": tokens,
                        **({"extra": extra} if extra else {}),
                        "none_hash_seed": kcu._NONE_HASH_SEED, "none_hash": bytes(kcu.NONE_HASH).hex(),
                        "hashes": hashes})
    json.dump({"vllm_commit": "5df63a7203cdb9328d824a27b682e77c7b9928be", "cases": out}, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
