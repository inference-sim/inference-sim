# Original to blis-vllm.
"""Mutation sweep for the engine-block factory's comparison with vLLM's EngineArgs (tier 1, the
dynamic half): apply one mutant at a time to a scratch copy of the module, run
TestEngineArgsAgainstVLLM (core/vllm/engineargs_test.go, over oracle/engineargs-cases.json and
core/vllm/testdata/engineargs.json), and report whether it fails.

Each mutant moves one refusal or one default of core/vllm's resolve by one step: the boundary
mutants (E1-E3, E4, E5) are the ones a case far from the boundary lets through, which is why the
cases sit at budget 3 against 4 sequences, 5 active sequences against 4, and a long-prefill
threshold of 513 against a 512 window.

Usage: python3 oracle/mutate_engineargs.py <blis-vllm checkout> <scratch dir> [E1,...]
Exits non-zero if a mutant survives or its pattern is not found.
"""
import os
import shutil
import subprocess
import sys

TEST = "TestEngineArgsAgainstVLLM"

# (id and the rule or default it moves, file, exact old text, new text)
MUTANTS = [
    ("E1 max_num_batched_tokens below max_num_seqs is refused", "core/vllm/config.go",
     "\tif budget < seqs {", "\tif budget < seqs-1 {"),
    ("E2 max_num_active_seqs above max_num_seqs is refused", "core/vllm/config.go",
     "\tif e.MaxNumActiveSeqs > seqs {", "\tif e.MaxNumActiveSeqs > seqs+1 {"),
    ("E3 long_prefill_token_threshold above max_model_len is refused", "core/vllm/config.go",
     "\tif e.LongPrefillTokenThreshold > maxLen {", "\tif e.LongPrefillTokenThreshold > maxLen+1 {"),
    ("E4 a watermark of 1 is refused", "core/vllm/config.go",
     "if !(watermark >= 0 && watermark < 1) {", "if !(watermark >= 0 && watermark <= 1) {"),
    ("E5 a negative watermark is refused", "core/vllm/config.go",
     "if !(watermark >= 0 && watermark < 1) {", "if !(watermark > -1 && watermark < 1) {"),
    ("E6 a negative scheduler field is refused", "core/vllm/config.go",
     "\t\tif f.value < 0 {", "\t\tif f.value < -1 {"),
    ("E7 a stated 0 is refused by vLLM's rule", "core/vllm/config.go",
     "\tif len(c.statedZero) > 0 {", "\tif false {"),
    ("E8 an unstated batch limit is refused", "core/vllm/config.go",
     "\tif seqs == 0 || budget == 0 {", "\tif false {"),
    ("E9 an unknown scheduling policy is refused", "core/vllm/config.go",
     "\tcase \"\", \"fcfs\":", "\tcase \"\", \"fcfs\", \"sjf\":"),
    ("E10 the default policy is fcfs", "core/vllm/config.go",
     "\tpolicy := sched.FCFS\n", "\tpolicy := sched.Priority\n"),
    ("E11 the default watermark is 0", "core/vllm/config.go",
     "\twatermark := 0.0\n", "\twatermark := 0.01\n"),
    ("E12 a stated async_scheduling with a non-async drafter is refused", "core/vllm/config.go",
     "\tif async && e.AsyncScheduling != nil && !asyncSpecMethod(method) {", "\tif false {"),
    ("E13 a speculative method the factory does not build is refused", "core/vllm/config.go",
     "\t\tif !specMethodModelled(method) {", "\t\tif false {"),
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
        r = subprocess.run(["go", "test", "-count=1", "-run", TEST, "./core/vllm/"], cwd=scratch,
                           capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        if r.returncode == 0:
            print(f"SURVIVED {mid}")
            bad += 1
        elif f"--- FAIL: {TEST}" in out:
            print(f"CAUGHT   {mid}")
        else:
            tail = out.strip().splitlines()[-3:]
            print(f"BROKEN   {mid}: failed without the test failing: {' | '.join(tail)[:200]}")
            bad += 1
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
