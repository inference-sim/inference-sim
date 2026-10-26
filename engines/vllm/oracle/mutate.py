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
     "numNew = min(numNew, tokenBudget, inputBudget-draftSlots)", "numNew = min(numNew, tokenBudget+1, inputBudget+1-draftSlots)"),
    ("M3", "chunked prefill", "core/sched/scheduler.go",
     "numNew = min(numNew, room)", "if numNew > room { break }"),
    ("M4", "long_prefill_token_threshold", "core/sched/scheduler.go",
     "if 0 < threshold && threshold < numNew {\n\t\t\tnumNew = threshold\n\t\t}\n\t\tnumNew = min(numNew, tokenBudget, inputBudget-draftSlots)",
     "numNew = min(numNew, tokenBudget, inputBudget-draftSlots)"),
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
     "if len(preempted) == 0 {\n\t\tvar skipped", "if true {\n\t\tvar skipped"),
    ("M13", "update order (scheduling order)", "core/sched/scheduler.go",
     "for _, id := range out.Order {", "for _, id := range sortedStrings(out.Order) {"),
    ("M14", "full-sequence admission gate", "core/sched/scheduler.go",
     "FullSequenceMustFit: s.cfg.ReserveFullSequence,", "FullSequenceMustFit: false,"),
    ("M15", "prefix hit capped at num_tokens-1", "core/kv/manager.go",
     "\tmaxLen := numTokens - 1\n", "\tmaxLen := numTokens\n"),
    ("M16", "free queue: hashed blocks to the tail", "core/kv/pool.go",
     "p.free.prependN(first)\n\tp.free.appendN(last)", "p.free.appendN(first)\n\tp.free.appendN(last)"),
    ("M17", "free order tail first", "core/kv/manager.go",
     "rev[len(bs)-1-i] = b", "rev[i] = b"),
    ("M18", "prefix caching (cache full blocks)", "core/kv/manager.go",
     "\t\t\tm.cache(g, id, a.Hashes, m.cacheable(min(mainModel, a.NumTokens)), a.Run)\n", "\t\t\t_ = g\n"),
    ("M19", "kv usage (null block excluded)", "core/kv/pool.go",
     "total := len(p.blocks) - 1", "total := len(p.blocks)"),
    ("M20", "evictable prefix-hit blocks counted", "core/kv/manager.go",
     "total += newBlocks + evictable", "total += newBlocks + 0*evictable"),
    ("M21", "running-request skip on zero tokens", "core/sched/scheduler.go",
     "\t\t\treqIndex++\n\t\t\tcontinue\n\t\t}\n\t\tok := false", "\t\t\tbreak\n\t\t}\n\t\tok := false"),
    ("M22", "priority victim budget restore", "core/sched/scheduler.go",
     "tokenBudget += restored\n\t\t\t\t\tinputBudget += restored + draftSlots", "_ = restored"),
    # Speculative decoding (#2021 B3; acceptance supplied).
    ("M23", "draft slots in the input budget", "core/sched/scheduler.go",
     "\t\tinputBudget -= numNew + draftSlots\n\t\treqIndex++", "\t\tinputBudget -= numNew\n\t\treqIndex++"),
    ("M24", "rejection rollback", "core/sched/scheduler.go",
     "\t\t\t\tr.NumComputed -= rejected\n", "\t\t\t\tr.NumComputed -= 0 * rejected\n"),
    ("M25", "lookahead slots in allocate_slots", "core/kv/manager.go",
     "needSlots := min(mainModel+a.Lookahead, m.maxModelLen)", "needSlots := min(mainModel, m.maxModelLen)"),
    ("M26", "drafts are never cached", "core/kv/manager.go",
     "m.cache(g, id, a.Hashes, m.cacheable(min(mainModel, a.NumTokens)), a.Run)", "m.cache(g, id, a.Hashes, m.cacheable(mainModel), a.Run)"),
    ("M27", "EAGLE block drop", "core/kv/manager.go",
     "hit := m.fullHit(hashes, maxLen, []int{0}, m.eagle)", "hit := m.fullHit(hashes, maxLen, []int{0}, false)"),
    ("M28", "multi-module MTP reprefillable tokens", "core/kv/manager.go",
     "return max(0, numTokens-m.reprefillable)", "return max(0, numTokens)"),
    ("M29", "prefill lookahead reservation", "core/sched/scheduler.go",
     "numNew -= la - remaining", "numNew -= 0"),
    ("M30", "first-decode padding", "core/sched/scheduler.go",
     "if k > 0 && numNew == 1 && !prefillScheduled", "if false && numNew == 1 && !prefillScheduled"),
    ("M31", "verify only drafts past known tokens", "core/sched/scheduler.go",
     "if n := numNew + r.NumComputed - r.NumTokens() - r.Placeholders; n > 0 {", "if n := len(r.SpecTokens); n > 0 {"),
    ("M32", "drafts ignored for prefill chunks", "core/sched/scheduler.go",
     "\t\tif r.IsPrefillChunk {\n\t\t\tr.SpecTokens = nil\n\t\t\tcontinue\n\t\t}", ""),
    ("M33", "num_tokens_with_spec", "core/sched/scheduler.go",
     "numNew := r.numTokensWithSpec() + r.Placeholders - r.NumComputed", "numNew := r.NumTokens() + r.Placeholders - r.NumComputed"),
    ("M34", "draft slots: P-EAGLE K-1", "core/sched/spec.go",
     "\tcase parallel:\n\t\ts.DraftSlots = k - 1", "\tcase parallel:\n\t\ts.DraftSlots = k"),
    ("M35", "lookahead: EAGLE and draft_model K", "core/sched/spec.go",
     "\tcase useEagle || draftModel:\n\t\ts.Lookahead = k", "\tcase useEagle || draftModel:\n\t\ts.Lookahead = k - 1"),
    ("M36", "stop checked after each sampled token", "core/sched/scheduler.go",
     "\t\t\tif stop {\n\t\t\t\tbreak\n\t\t\t}", ""),
    ("M37", "draft_model one draft slot", "core/sched/spec.go",
     "\tcase draftModel:\n\t\ts.DraftSlots = 1", "\tcase draftModel:\n\t\ts.DraftSlots = 0"),
    # KV events (kv_events_config).
    ("M38", "BlockStored parent hash", "core/kv/pool.go",
     "\t\tif numCached > 0 {\n\t\t\tparent = hashes[numCached-1]\n\t\t}", ""),
    ("M39", "BlockRemoved on eviction", "core/kv/pool.go",
     "\tif p.events != nil {\n\t\t*p.events = append(*p.events, Event{Removed: true,", "\tif false {\n\t\t*p.events = append(*p.events, Event{Removed: true,"),
    ("M40", "BlockStored adapter name", "core/sched/scheduler.go",
     "return kv.StoredRun{Tokens: r.allTokens(), LoRA: r.LoRAName}", "return kv.StoredRun{Tokens: r.allTokens()}"),
    # Hybrid KV cache (full attention with sliding windows).
    ("M41", "sliding window skipped tokens", "core/kv/manager.go",
     "\treturn max(0, computed-g.spec.Window+1-g.extraRetained)\n", "\treturn max(0, computed-g.spec.Window-g.extraRetained)\n"),
    # Not M42: freeing out-of-window blocks on the processed-token basis
    # (kv_cache_manager.py:545-557) is equivalent here, because num_in_flight_tokens is zero
    # whenever a synchronous scheduler schedules; it binds under async scheduling (tier 3).
    ("M43", "window hit needs the window's contiguous blocks", "core/kv/manager.go",
     "need := cdiv(sg.spec.Window-1, m.blockSize) // _contiguous_blocks_for_hit, :943-955", "need := 1"),
    ("M44", "window blocks freed into null slots", "core/kv/manager.go",
     "\t\tif len(freed) > 0 {\n\t\t\tm.Pool.Free(freed)\n\t\t}\n\t}\n}", "\t\t_ = freed\n\t}\n}"),
    ("M45", "sliding window admission cap", "core/kv/manager.go",
     "required = min(required, cdiv(capTokens, m.blockSize)+1)", "required = min(required, cdiv(capTokens, m.blockSize))"),
    # Not M46: rounding the hybrid cacheable count down to the block
    # (kv_cache_coordinator.py:767-777) is equivalent here, because cache_blocks floors to
    # full blocks anyway when every group has one block size and no partial hash hits.
    ("M48", "disable_hybrid_kv_cache_manager promotes windows", "core/kv/manager.go",
     "\tif disableHybrid && hasFull {", "\tif false && disableHybrid && hasFull {"),
    ("M47", "hybrid lookup repeats until no group shortens the hit", "core/kv/manager.go",
     "\t\tif simple {\n\t\t\tbreak\n\t\t}\n", "\t\tif simple || true {\n\t\t\tbreak\n\t\t}\n"),
    # blis-vllm#20 review I10, I11.
    ("M49", "one sliding-window group looks up through the window", "core/kv/manager.go",
     "\t\tif g := m.groups[0]; g.spec.Kind == SlidingWindow {", "\t\tif g := m.groups[0]; g.spec.Kind == SlidingWindow && false {"),
    ("M50", "multi-module MTP keeps extra tokens below the window", "core/kv/manager.go",
     "\t\t\tgr.extraRetained = m.reprefillable\n", "\t\t\tgr.extraRetained = 0\n"),
    ("M51", "extra retained tokens widen the window admission cap", "core/kv/manager.go",
     "capTokens := min(g.spec.Window-1+g.extraRetained+m.maxInFlight, m.maxModelLen)",
     "capTokens := min(g.spec.Window-1+m.maxInFlight, m.maxModelLen)"),
    # One mutant per scheduler knob row, each touching only that knob, so a row's proof is a
    # mutant of its own feature (docs/feature-matrix.md, "How to read a row").
    ("M52", "watermark", "core/sched/scheduler.go",
     "wm := int(cfg.Watermark * float64(cfg.NumBlocks))", "wm := 0"),
    ("M53", "long_prefill_token_threshold_adaptive", "core/sched/scheduler.go",
     "if threshold > 0 && s.cfg.LongPrefillAdaptive {", "if false && s.cfg.LongPrefillAdaptive {"),
    ("M54", "max_num_scheduled_tokens", "core/sched/scheduler.go",
     "\tif s.cfg.MaxNumScheduledTokens > 0 {\n\t\treturn s.cfg.MaxNumScheduledTokens\n\t}",
     "\tif false && s.cfg.MaxNumScheduledTokens > 0 {\n\t\treturn s.cfg.MaxNumScheduledTokens\n\t}"),
    # A mutant of their own for three rows that had only shared ones (verification pass 1,
    # gap 11): each touches only its row's knob.
    ("M56", "max_num_active_seqs", "core/sched/scheduler.go",
     "\tif s.cfg.MaxNumActiveSeqs > 0 {\n\t\treturn s.cfg.MaxNumActiveSeqs\n\t}",
     "\tif false && s.cfg.MaxNumActiveSeqs > 0 {\n\t\treturn s.cfg.MaxNumActiveSeqs\n\t}"),
    ("M57", "block_size", "core/kv/manager.go",
     "\t\trequired := cdiv(numTokens, m.blockSize)\n", "\t\trequired := cdiv(numTokens, m.blockSize+1)\n"),
    ("M58", "max_loras per-batch skip", "core/sched/scheduler.go",
     "if s.cfg.MaxLoRAs > 0 && r.LoRA > 0 && len(scheduledLoRAs) == s.cfg.MaxLoRAs && !scheduledLoRAs[r.LoRA] {",
     "if s.cfg.MaxLoRAs > 0 && r.LoRA > 0 && len(scheduledLoRAs) > s.cfg.MaxLoRAs && !scheduledLoRAs[r.LoRA] {"),
    # blis-vllm#20 review S17.
    ("M55", "layers sharing one spec form one KV cache group", "core/kv/manager.go",
     "\tif uniform {\n\t\treturn []Group{layers[0]}\n\t}\n", "\tif uniform && false {\n\t\treturn []Group{layers[0]}\n\t}\n"),
    # swa_bounded_replay (dashboard-deepseek-v4.1-flash-replay).
    ("M59", "swa_bounded_replay", "core/sched/scheduler.go",
     "\tn := min(window, hitTokens)\n", "\tn := hitTokens\n"),
    ("M60", "swa_bounded_replay", "core/sched/scheduler.go",
     "0 < numComputed && numComputed <= replay {", "0 < numComputed && numComputed < replay {"),
    ("M61", "swa_bounded_replay", "core/sched/scheduler.go",
     "New: max(numNew-replayed, 0),", "New: numNew,"),
]


# Mutants the tier-2 scenarios cannot catch yet, each a coverage gap recorded in
# docs/testing.md.
KNOWN_GAPS: set[str] = set()


def main() -> int:
    src, scratch = sys.argv[1], sys.argv[3]
    # go test runs each package in its own directory, so the corpus path must be absolute.
    fuzz = os.path.abspath(sys.argv[2])
    if not any(f.endswith(".trace.json") for f in os.listdir(fuzz)):
        print(f"no fuzz traces in {fuzz}")
        return 1
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    env = dict(os.environ, BLIS_VLLM_FUZZ_DIR=fuzz)
    unexpected = 0
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
        elif "panic: test timed out" in r.stdout:
            print(f"BROKEN   {mid:4} {row}: the tests timed out")
            unexpected += 1
        elif "earliest divergence" in r.stdout or "never finished" in r.stdout or "panic" in r.stdout:
            print(f"CAUGHT   {mid:4} {row}")
        else:
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            print(f"BROKEN   {mid:4} {row}: failed without a divergence: {' | '.join(tail)[:200]}")
            unexpected += 1
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
