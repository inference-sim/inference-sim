# Original to blis-vllm.
"""Mutation check for the decode-step allocation test (core/sched
TestDecodeStepAllocationDoesNotGrowWithContext): apply one mutant at a time to a scratch copy of
the module, run that test, and report whether it fails.

The two mutants are the two forms of the regression the test guards: a request's whole token list
copied when a block is hashed (the review of the first oracle pull request found updateHashes
doing the equivalent), and copied on every Update. The first allocates 1.1 MB a step at 32,768
tokens over the test's 32-step window, the second 17 MB; the unmutated code allocates 26 KB.

Usage: python3 oracle/mutate_sched_alloc.py <blis-vllm checkout> <scratch dir> [S1,...]
Exits non-zero if a mutant survives or its pattern is not found.
"""
import os
import shutil
import subprocess
import sys

# (id and the regression, file, exact old text, new text)
MUTANTS = [
    ("S1 whole-sequence copy when a block is hashed", "core/sched/scheduler.go",
     "\t\tr.hashes = append(r.hashes, kv.HashBlock(parent, r.tokenRange(i*blockSize, (i+1)*blockSize), r.Keys.ExtraFor(i)))",
     "\t\tall := r.allTokens()\n"
     "\t\tr.hashes = append(r.hashes, kv.HashBlock(parent, all[i*blockSize:(i+1)*blockSize], r.Keys.ExtraFor(i)))"),
    ("S2 whole-sequence copy on every Update", "core/sched/scheduler.go",
     "func (r *Request) updateHashes(blockSize int) {\n",
     "func (r *Request) updateHashes(blockSize int) {\n\t_ = r.allTokens()\n"),
]


def main() -> int:
    src, scratch = sys.argv[1], sys.argv[2]
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    bad = 0
    for mid, path, old, new in MUTANTS:
        if only and mid.split(" ", 1)[0] not in only:
            continue
        if os.path.exists(scratch):
            shutil.rmtree(scratch)
        shutil.copytree(src, scratch, ignore=shutil.ignore_patterns(".git", "vllm-source", "oracle-model", "out"))
        target = os.path.join(scratch, path)
        text = open(target).read()
        if text.count(old) != 1:
            print(f"BROKEN   {mid}: pattern found {text.count(old)} times in {path}")
            bad += 1
            continue
        open(target, "w").write(text.replace(old, new))
        r = subprocess.run(["go", "test", "-count=1", "-run", "TestDecodeStepAllocationDoesNotGrowWithContext",
                            "./core/sched/"], cwd=scratch, capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
        if r.returncode == 0:
            print(f"SURVIVED {mid}")
            bad += 1
        elif "--- FAIL: TestDecodeStepAllocationDoesNotGrowWithContext" in out:
            print(f"CAUGHT   {mid}")
        else:
            tail = out.strip().splitlines()[-3:]
            print(f"BROKEN   {mid}: failed without the test failing: {' | '.join(tail)[:200]}")
            bad += 1
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
