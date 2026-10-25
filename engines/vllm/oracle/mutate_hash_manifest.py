# Original to blis-vllm.
"""Mutation sweep for two families of supported rows that had no mutant of their own
(verification pass 4, gap 6 of inference-sim#2021):

- the block hash (core/kv/hash.go): the "block hash with LoRA and cache-salt keys" and
  "prefix_caching_hash_algo: sha256" rows. Each mutant breaks one part of vLLM's
  hash_block_tokens / init_none_hash / pickle framing, and must be caught by core/kv's hash
  tests against the hashes vLLM v0.32.0 computed (core/kv/testdata/block_hashes.json);
- the metrics manifest generator (cmd/metricsmanifest): the "every computable vllm:* family"
  and "kv_cache_metrics, cudagraph_metrics" rows. Each mutant must be caught by the generator's
  own -check against the pinned v0.32.0 source, the check CI's metrics-manifest job runs.

Usage:
  python3 oracle/mutate_hash_manifest.py <blis-vllm checkout> <scratch dir> hash
  python3 oracle/mutate_hash_manifest.py <blis-vllm checkout> <scratch dir> manifest <vllm checkout>
Exits non-zero if a mutant survives or does not build.
"""
import os
import shutil
import subprocess
import sys

HASH = "core/kv/hash.go"
GEN = "cmd/metricsmanifest/main.go"

# (id, the rule it breaks, file, exact old text, new text)
HASH_MUTANTS = [
    ("H1", "the cache salt keys only the first block", HASH,
     '\tif k.CacheSalt == "" || i != 0 {', '\tif k.CacheSalt == "" {'),
    ("H2", "the cache salt is a key", HASH,
     '\treturn append(append([]ExtraKey(nil), k.Extra...), ExtraKey{"cache_salt", k.CacheSalt})',
     "\treturn k.Extra"),
    ("H3", "LoRA (and other) extra keys are part of the hash", HASH,
     "\t\t\tif len(extra) == 0 {\n\t\t\t\tp.none()", "\t\t\tif true {\n\t\t\t\tp.none()"),
    ("H4", "NONE_HASH is sha256 of the default seed", HASH,
     'const noneHashSeed = "vllm-none-hash"', 'const noneHashSeed = "vllm-none-hash-x"'),
    ("H5", "pickles past 64 KiB are framed as CPython frames them", HASH,
     "const frameSizeTarget = 64 * 1024", "const frameSizeTarget = 64 * 1024 * 2"),
    ("H6", "a first block chains from NONE_HASH", HASH,
     "\tif len(parent) == 0 {\n\t\tparent = noneHash\n\t}",
     "\tif len(parent) == 0 {\n\t\tparent = noneHash[:len(noneHash)-1]\n\t}"),
]
HASH_TESTS = "TestNoneHashIsVLLMs|TestBlockHashesAreVLLMs|TestAnAdapterChangesEveryBlockHash"

MANIFEST_MUTANTS = [
    ("K1", "every histogram family (_histogram_cls) is found in the source", GEN,
     "ctorRe      = regexp.MustCompile(`(_gauge_cls|_counter_cls|_histogram_cls|\\bGauge|\\bCounter|\\bHistogram)\\(`)",
     "ctorRe      = regexp.MustCompile(`(_gauge_cls|_counter_cls|\\bGauge|\\bCounter|\\bHistogram)\\(`)"),
    ("K2", "a family's enabling configuration (kv_cache_metrics, ...) is recorded", GEN,
     "\t\tf.Class, f.Reason, f.EnabledBy = c.Class, c.Reason, c.EnabledBy",
     "\t\tf.Class, f.Reason = c.Class, c.Reason"),
]


def run(scratch: str, kind: str, vllm: str | None) -> tuple[int, str]:
    if kind == "hash":
        r = subprocess.run(["go", "test", "-count=1", "-run", HASH_TESTS, "./core/kv/"],
                           cwd=scratch, capture_output=True, text=True, timeout=900)
    else:
        r = subprocess.run(["go", "run", "./cmd/metricsmanifest", "-vllm", vllm, "-check"],
                           cwd=scratch, capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def main() -> int:
    src, scratch, kind = sys.argv[1], sys.argv[2], sys.argv[3]
    vllm = os.path.abspath(sys.argv[4]) if kind == "manifest" else None
    mutants = HASH_MUTANTS if kind == "hash" else MANIFEST_MUTANTS
    bad = 0
    for mid, rule, path, old, new in mutants:
        if os.path.exists(scratch):
            shutil.rmtree(scratch)
        shutil.copytree(src, scratch, ignore=shutil.ignore_patterns(".git", "vllm-source", "oracle-model", "out"))
        target = os.path.join(scratch, path)
        text = open(target).read()
        if text.count(old) != 1:
            print(f"BROKEN   {mid} {rule}: the mutation site is not unique in {path}")
            bad += 1
            continue
        open(target, "w").write(text.replace(old, new, 1))
        code, out = run(scratch, kind, vllm)
        if code == 0:
            print(f"SURVIVED {mid} {rule}")
            bad += 1
        elif "build failed" in out or "[setup failed]" in out or "cannot use" in out or "undefined:" in out:
            print(f"BROKEN   {mid} {rule}: the mutant does not compile")
            bad += 1
        else:
            print(f"CAUGHT   {mid} {rule}")
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
