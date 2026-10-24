# Original to blis-vllm.
"""Mutation sweep for blis-vllm's tier 3 (core/engine and the async half of core/sched):
apply one mutant at a time to a scratch copy of the module, run the tier-3 diff and fuzz
tests, and report whether each is caught.

Usage: python3 oracle/mutate_t3.py <blis-vllm checkout> <tier-3 fuzz dir> <scratch dir> [M1,...]
Exits non-zero if a mutant survives or breaks the build without a divergence.
"""
import os
import shutil
import subprocess
import sys

# (id, feature-matrix row it guards, file, exact old text, new text)
MUTANTS = [
    ("T1", "batch queue: max_concurrent_batches", "core/engine/engine.go",
     "\t\tif pp <= 1 {\n\t\t\treturn 2\n\t\t}", "\t\tif pp <= 1 {\n\t\t\treturn 1\n\t\t}"),
    ("T2", "batch queue: V2 async adds one", "core/engine/engine.go",
     "\t\tif v2Runner {\n\t\t\treturn pp + 1\n\t\t}", "\t\tif v2Runner {\n\t\t\treturn pp\n\t\t}"),
    ("T3", "batch queue: no blocking while the queue has room", "core/engine/engine.go",
     "if len(c.queue) < c.QueueSize && (res.Executed || c.Sched.HasRequests()) {",
     "if len(c.queue) < c.QueueSize && res.Executed {"),
    ("T4", "batch queue: completion order", "core/engine/engine.go",
     "\tb := c.queue[0]\n\tc.queue = c.queue[1:]", "\tb := c.queue[len(c.queue)-1]\n\tc.queue = c.queue[:len(c.queue)-1]"),
    ("T5", "async: output placeholders", "core/sched/scheduler.go",
     "\t\tif s.cfg.Async && !r.IsPrefillChunk {\n\t\t\t// AsyncScheduler._update_after_schedule",
     "\t\tif false && !r.IsPrefillChunk {\n\t\t\t// AsyncScheduler._update_after_schedule"),
    ("T6", "async: no extra step past max_tokens", "core/sched/scheduler.go",
     "if r.Placeholders > 0 && r.NumComputed+2-r.Placeholders >= len(r.Prompt)+r.MaxTokens {",
     "if r.Placeholders > 0 && r.NumComputed+1-r.Placeholders >= len(r.Prompt)+r.MaxTokens {"),
    ("T7", "async: stale output after preemption", "core/sched/scheduler.go",
     "\tr.numStale = r.numInFlight\n", "\tr.numStale = 0\n"),
    ("T8", "async: cache blocks on output", "core/sched/scheduler.go",
     "\t\t\t\ts.KV.CacheUpTo(r.ID, r.hashes, r.NumComputed-r.Placeholders, s.run(r))", "\t\t\t\t_ = r.hashes"),
    ("T9", "PP: V2 decode spacing", "core/sched/scheduler.go",
     "\t\t\t\tr.nextDecodeStep = s.currentStep + max(s.cfg.DecodeStagger, 1)",
     "\t\t\t\tr.nextDecodeStep = s.currentStep + 1"),
    ("T10", "async: stale resume deferred", "core/sched/scheduler.go",
     "\t\t\tif r.numStale > 0 {\n\t\t\t\t// Deliverable stale output",
     "\t\t\tif false {\n\t\t\t\t// Deliverable stale output"),
]


def main() -> int:
    src, scratch = sys.argv[1], sys.argv[3]
    fuzz = os.path.abspath(sys.argv[2])
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    env = dict(os.environ, BLIS_VLLM_T3_FUZZ_DIR=fuzz)
    bad = 0
    for mid, row, path, old, new in MUTANTS:
        if only and mid not in only:
            continue
        if os.path.exists(scratch):
            shutil.rmtree(scratch)
        shutil.copytree(src, scratch, ignore=shutil.ignore_patterns(".git", "vllm-source", "oracle-model", "out"))
        target = os.path.join(scratch, path)
        text = open(target).read()
        if text.count(old) != 1:
            print(f"BROKEN   {mid:4} {row}: pattern found {text.count(old)} times in {path}")
            bad += 1
            continue
        open(target, "w").write(text.replace(old, new))
        r = subprocess.run(["go", "test", "-count=1", "-run",
                            "TestEngineCoreAgainstVLLM|TestEngineCoreFuzzCorpusAgainstVLLM|TestMaxConcurrentBatches",
                            "./core/engine/"], cwd=scratch, env=env, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            print(f"SURVIVED {mid:4} {row}")
            bad += 1
        elif "divergence" in r.stdout or "max_concurrent_batches" in r.stdout or "want" in r.stdout or "panic" in r.stdout:
            print(f"CAUGHT   {mid:4} {row}")
        else:
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            print(f"BROKEN   {mid:4} {row}: failed without a divergence: {' | '.join(tail)[:200]}")
            bad += 1
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
