# Original to blis-vllm.
"""Mutation sweep for blis-vllm core/: apply one mutant at a time to a scratch copy of the
module, run the tier-2 diff + fuzz tests, and report whether each mutant is caught.

Usage: python3 oracle/mutate.py <blis-vllm checkout> <fuzz dir> <scratch dir> [M1,M2,...]
Prints one line per mutant: CAUGHT/SURVIVED, its id and the row it guards. Exits non-zero
if a mutant outside KNOWN_GAPS survives, or if a known gap is now caught (so the list stays
true: remove it, and the row it guards can then be marked supported).
"""
import os, shutil, subprocess, sys

# (id, feature-matrix row it guards, file, exact old text, new text)
MUTANTS = [
    ("M1", "max_num_seqs", "core/sched/scheduler.go",
     "if len(s.running) >= s.maxActive() {", "if len(s.running) > s.maxActive() {"),
    ("M2", "max_num_batched_tokens", "core/sched/scheduler.go",
     "numNew = min(numNew, tokenBudget, inputBudget)", "numNew = min(numNew, tokenBudget+1, inputBudget+1)"),
    ("M3", "chunked prefill", "core/sched/scheduler.go",
     "numNew = min(numNew, room)", "if numNew > room { break }"),
    ("M4", "long_prefill_token_threshold", "core/sched/scheduler.go",
     "if 0 < threshold && threshold < numNew {\n\t\t\tnumNew = threshold\n\t\t}\n\t\tnumNew = min(numNew, tokenBudget, inputBudget)",
     "numNew = min(numNew, tokenBudget, inputBudget)"),
    ("M5", "lone-request exemption", "core/sched/scheduler.go",
     "if eligible > 1 {", "if eligible > 0 {"),
    ("M6", "FCFS preemption victim", "core/sched/scheduler.go",
     "victim = s.running[len(s.running)-1]\n\t\t\t\ts.running = s.running[:len(s.running)-1]",
     "victim = s.running[0]\n\t\t\t\ts.running = s.running[1:]"),
    ("M7", "priority preemption victim", "core/sched/scheduler.go",
     "r.Priority > best.Priority || (r.Priority == best.Priority && r.ArrivalTime > best.ArrivalTime)",
     "r.Priority > best.Priority"),
    ("M8", "priority queue order", "core/sched/scheduler.go",
     "return a.Priority < b.Priority", "return a.Priority > b.Priority"),
    ("M9", "max_model_len input cap", "core/sched/scheduler.go",
     "numNew = min(numNew, s.cfg.MaxModelLen-r.NumComputed-1)", "numNew = min(numNew, s.cfg.MaxModelLen-r.NumComputed)"),
    ("M10", "check_stop max_model_len", "core/sched/scheduler.go",
     "case r.NumTokens() >= s.cfg.MaxModelLen || len(r.Output) >= r.MaxTokens:",
     "case len(r.Output) >= r.MaxTokens:"),
    ("M11", "check_stop max_tokens", "core/sched/scheduler.go",
     "case r.NumTokens() >= s.cfg.MaxModelLen || len(r.Output) >= r.MaxTokens:",
     "case r.NumTokens() >= s.cfg.MaxModelLen:"),
    ("M12", "no admission after preemption", "core/sched/scheduler.go",
     "if len(preempted) == 0 {\n\t\tfor tokenBudget", "if true {\n\t\tfor tokenBudget"),
    ("M13", "update order (scheduling order)", "core/sched/scheduler.go",
     "for _, id := range out.Order {", "for _, id := range sortedStrings(out.Order) {"),
    ("M14", "full-sequence admission gate", "core/sched/scheduler.go",
     "len(s.running) > 0, s.cfg.ReserveFullSequence)", "len(s.running) > 0, false)"),
    ("M15", "prefix hit capped at num_tokens-1", "core/kv/manager.go",
     "maxBlocks := (numTokens - 1) / m.blockSize", "maxBlocks := numTokens / m.blockSize"),
    ("M16", "free queue: hashed blocks to the tail", "core/kv/pool.go",
     "p.free.prependN(first)\n\tp.free.appendN(last)", "p.free.appendN(first)\n\tp.free.appendN(last)"),
    ("M17", "free order tail first", "core/kv/manager.go",
     "rev[len(bs)-1-i] = b", "rev[i] = b"),
    ("M18", "prefix caching (cache full blocks)", "core/kv/manager.go",
     "m.cache(id, hashes, min(totalComputed+numNew, numTokens))", "_ = hashes"),
    ("M19", "kv usage (null block excluded)", "core/kv/pool.go",
     "total := len(p.blocks) - 1", "total := len(p.blocks)"),
    ("M20", "evictable prefix-hit blocks counted", "core/kv/manager.go",
     "return newBlocks + evictable", "return newBlocks"),
    ("M21", "running-request skip on zero tokens", "core/sched/scheduler.go",
     "if numNew <= 0 {\n\t\t\treqIndex++\n\t\t\tcontinue\n\t\t}", "if numNew <= 0 {\n\t\t\tbreak\n\t\t}"),
    ("M22", "priority victim budget restore", "core/sched/scheduler.go",
     "tokenBudget += restored\n\t\t\t\t\tinputBudget += restored", "_ = restored"),
]


# Mutants the tier-2 scenarios cannot catch yet, each a coverage gap recorded in
# docs/testing.md. Both need a request scheduled past its known tokens (speculative decoding).
KNOWN_GAPS = {"M9", "M21"}


def main() -> int:
    src, fuzz, scratch = sys.argv[1], sys.argv[2], sys.argv[3]
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    env = dict(os.environ, BLIS_VLLM_FUZZ_DIR=fuzz)
    unexpected = 0
    for mid, row, path, old, new in MUTANTS:
        if only and mid not in only:
            continue
        if os.path.exists(scratch):
            shutil.rmtree(scratch)
        shutil.copytree(src, scratch, ignore=shutil.ignore_patterns(".git"))
        target = os.path.join(scratch, path)
        text = open(target).read()
        if text.count(old) != 1:
            print(f"BROKEN   {mid:4} {row}: pattern found {text.count(old)} times in {path}")
            unexpected += 1
            continue
        open(target, "w").write(text.replace(old, new))
        r = subprocess.run(["go", "test", "-count=1", "-run",
                            "TestDifferentialAgainstVLLM|TestFuzzCorpusAgainstVLLM", "./core/sched/"],
                           cwd=scratch, env=env, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            known = mid in KNOWN_GAPS
            print(f"SURVIVED {mid:4} {row}" + (" (known gap)" if known else ""))
            unexpected += 0 if known else 1
            continue
        if mid in KNOWN_GAPS:
            print(f"NOW CAUGHT {mid} {row}: remove it from KNOWN_GAPS")
            unexpected += 1
        elif "earliest divergence" in r.stdout or "never finished" in r.stdout or "panic" in r.stdout:
            print(f"CAUGHT   {mid:4} {row}")
        else:
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            print(f"CAUGHT?  {mid:4} {row}: {' | '.join(tail)[:200]}")
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
