# Original to blis-vllm.
"""Mutation sweep for the engine.Engine adapter and its factory (core/vllm): apply one mutant at
a time to a scratch copy of the module, run core/vllm's tests (the enginetest conformance, the
adapter's diff test against vLLM v0.32.0's tier-2 goldens, the factory's refusals), and report
whether each is caught.

Usage: python3 oracle/mutate_e24.py <blis-vllm checkout> <scratch dir> [A1,...]
Exits non-zero if a mutant survives or its pattern is not found.
"""
import os
import shutil
import subprocess
import sys

# (id, behaviour it guards, file, exact old text, new text)
MUTANTS = [    ("A1 wake after an empty schedule that left work", "core/vllm/engine.go",
    "\t\tif e.s.HasRequests() && len(e.pending) == 0 {\n\t\t\te.idleWake = now + idlePollUs", "\t\tif false {\n\t\t\te.idleWake = now + idlePollUs"),
    ("A2 drafts accepted from the outcome", "core/vllm/engine.go",
    "n += min(max(outcome.Accepted[rid], 0), real)", "n += 0"),
    ("A3 last assigned token is EOS", "core/vllm/engine.go",
    "\t\t\tif st.sampled == len(out)-1 {", "\t\t\tif false {"),
    ("A4 complete in order", "core/vllm/engine.go",
    "\tcase idx > 0:\n\t\treturn engine.Result{}, engine.ErrOutOfOrder", "\tcase idx > 99:\n\t\treturn engine.Result{}, engine.ErrOutOfOrder"),
    ("A5 in-flight limit", "core/vllm/engine.go",
    "\tif e.core.InFlight() >= e.InFlightLimit() {", "\tif e.core.InFlight() >= e.InFlightLimit()+1 {"),
    ("A6 abort frees", "core/vllm/engine.go",
    "\te.s.Abort([]string{id})", "\t_ = id"),
    ("A7 drafter proposes after a step", "core/vllm/engine.go",
    "\t\te.s.UpdateDrafts(drafts)", "\t\t_ = drafts"),
    ("A8 prefix hit reported", "core/vllm/engine.go",
    "\t\t\tsc.CachedTokens = hit", "\t\t\t_ = hit"),
    ("A9 empty batch not in flight", "core/vllm/engine.go",
    "\t\te.core.Drop(cb)\n", "\t\t_ = cb\n"),
    ("A10 strict decode", "core/vllm/config.go",
    "\tdec.KnownFields(true)\n", "\tdec.KnownFields(false)\n"),
    ("A11 version refused", "core/vllm/factory.go",
    "\tcase s.UpstreamVersion == Version:", "\tcase true:"),
    ("A12 async on by default", "core/vllm/config.go",
    "\tif pp > 1 && !v2 {\n\t\treturn false\n\t}\n\treturn true", "\tif pp > 1 && !v2 {\n\t\treturn false\n\t}\n\treturn false"),
    ("A13 max_tokens default", "core/vllm/engine.go",
    "\t\tmaxTokens = maxLen - len(r.Prompt)", "\t\tmaxTokens = 1"),
    ("A14 metrics first tokens", "core/vllm/metrics.go",
    "\t\t\tst.FirstTokens = append(st.FirstTokens, id)", "\t\t\t_ = id"),
    ("A15 counter family names", "core/vllm/metrics.go",
    "\tif base, ok := strings.CutSuffix(name, \"_total\"); ok && !isHistogram(name) {", "\tif base, ok := strings.CutSuffix(name, \"_totalX\"); ok && !isHistogram(name) {"),
    ("A16 LoRA off refuses adapters", "core/vllm/engine.go",
    "\t\tif e.maxLoRAs == 0 {", "\t\tif false {"),
    ("A17 KV holds one max_model_len sequence", "core/vllm/config.go",
    "need > c.NumGPUBlocks-1 {", "need > c.NumGPUBlocks+100000 {"),
    ("A18 the smallest pool vLLM starts is built", "core/vllm/config.go",
    "need > c.NumGPUBlocks-1 {", "need > c.NumGPUBlocks-2 {"),
    ("A19 budget at least max_num_seqs", "core/vllm/config.go",
    "\tif budget < seqs {", "\tif false {"),
    ("A20 negative scheduler fields refused", "core/vllm/config.go",
    "\t\tif f.value < 0 {", "\t\tif f.value < -1 {"),
    ("A21 NaN watermark refused", "core/vllm/config.go",
    "if !(watermark >= 0 && watermark < 1) {", "if watermark < 0 || watermark >= 1 {"),
    ("A22 unstated batch limits refused", "core/vllm/config.go",
    "\tif seqs == 0 || budget == 0 {", "\tif false {"),
    ("A23 batch queue refused until reproduced", "core/vllm/config.go",
    "\tif queue > 1 {", "\tif queue > 99 {"),
]


def main() -> int:
    src, scratch = sys.argv[1], sys.argv[2]
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    bad = 0
    for m in MUTANTS:
        mid, path, old, new = m
        tag = mid.split(" ", 1)[0]
        if only and tag not in only:
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
        r = subprocess.run(["go", "test", "-count=1", "./core/vllm/"], cwd=scratch,
                           capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            print(f"SURVIVED {mid}")
            bad += 1
        elif "--- FAIL" in r.stdout:
            print(f"CAUGHT   {mid}")
        else:
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            print(f"BROKEN   {mid}: failed without a test failure: {' | '.join(tail)[:200]}")
            bad += 1
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
