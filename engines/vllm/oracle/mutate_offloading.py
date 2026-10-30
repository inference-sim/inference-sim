# Original to blis-vllm.
"""Mutation sweep for the OffloadingConnector's port (core/connector/offloading and the
connector hooks of core/sched it reaches, core/sched/connector_offload.go): apply one mutant at
a time to a scratch copy of the module, run the offloading scenarios and fuzz corpus, and report
whether each is caught.

Usage: python3 oracle/mutate_offloading.py <blis-vllm checkout> <offloading fuzz dir> <scratch dir> [O1,...]
Exits non-zero if a mutant outside EQUIVALENT survives, if an EQUIVALENT one is caught, or if a
mutant breaks the build without a divergence.
"""
import os
import shutil
import subprocess
import sys

OFF = "core/connector/offloading/offloading.go"
CPU = "core/connector/offloading/cpu.go"
HOOK = "core/sched/connector_offload.go"
SCHED = "core/sched/scheduler.go"

# (id, behaviour it guards, file, exact old text, new text)
MUTANTS = [
    ("O1", "LRU evicts the least recent chunk", CPU,
     "return p.ranks[cands[i]] < p.ranks[cands[j]]", "return p.ranks[cands[i]] > p.ranks[cands[j]]"),
    ("O2", "a finished request's keys become most recent, head last", CPU,
     "\t\trev[len(keys)-1-i] = k", "\t\trev[i] = k"),
    ("O3", "a store's own keys are never evicted for it", CPU,
     "\t\t\tprotected[k] = true\n", "\t\t\t_ = k\n"),
    ("O4", "a store fails when too few chunks are evictable", CPU,
     "\t\tif need > m.numEvictable {\n\t\t\treturn nil, nil, false\n\t\t}\n", ""),
    ("O5", "a load pins its chunks against eviction", CPU,
     "\t\tif c.refCnt == 0 {\n\t\t\tm.lru.markNonEvictable(k)\n\t\t\tm.numEvictable--\n\t\t}\n\t\tc.refCnt++",
     "\t\tc.refCnt++"),
    ("O6", "a chunk becomes evictable only when its store completes", CPU,
     "\tif c.refCnt == 0 {\n\t\tp.evictable[k] = true\n\t}", "\tp.evictable[k] = true"),
    ("O7", "reused chunks come off the free list's tail", CPU,
     "\t\tid := m.freeList[len(m.freeList)-1]\n\t\tm.freeList = m.freeList[:len(m.freeList)-1]",
     "\t\tid := m.freeList[0]\n\t\tm.freeList = m.freeList[1:]"),
    ("O8", "a hit stops at the first missing chunk", OFF,
     "\t\tcase miss:\n\t\t\treturn n, pending\n\t\t}\n\t}\n\treturn n, pending",
     "\t\tcase miss:\n\t\t}\n\t}\n\treturn n, pending"),
    ("O9", "a chunk still being stored defers the request", OFF,
     "\tif deferred {\n\t\treturn 0, true\n\t}\n\tif len(s.loading) > 0 {",
     "\tif len(s.loading) > 0 {"),
    ("O10", "a chunk being loaded for another request defers it", OFF,
     "\t\t\t\tif s.loading[k] {\n\t\t\t\t\treturn 0, true\n\t\t\t\t}", "\t\t\t\t_ = s.loading[k]"),
    ("O11", "a request with jobs in flight is deferred", OFF,
     "\tif len(st.jobs) > 0 {\n\t\treturn 0, false, true\n\t}", ""),
    ("O12", "a sliding window's hit needs its window of chunks", OFF,
     "\t\tif run == need {\n\t\t\treturn i + need, pending\n\t\t}", "\t\tif run == max(need-1, 1) {\n\t\t\treturn i + max(need-1, 1), pending\n\t\t}"),
    ("O13", "the last prompt token is recomputed beside a window", OFF,
     "\tif len(s.swa) > 0 {\n\t\tmaxHit--\n\t}", ""),
    ("O14", "the hit's chunks are not stored again", OFF,
     "\t\tst.groups[gi].nextStored = numChunks\n", ""),
    ("O15", "offload_prompt_only stops at the prompt", OFF,
     "\t\tn = min(n, r.NumPromptTokens())", "\t\t_ = r"),
    ("O16", "a finished request stores up to its last committed token", OFF,
     "\t\t\tafter = max(r.NumPromptTokens(), r.NumTokens()-1)", "\t\t\tafter = r.NumTokens()"),
    ("O17", "a store takes each chunk's last block", OFF,
     "\t\t\t\tif gs.blockIDs[k*bpc+bpc-1] == 0 {", "\t\t\t\tif gs.blockIDs[k*bpc] == 0 {"),
    ("O18", "a failed store advances nothing", OFF,
     "\t\t\t// \"cannot store chunks\" (:1735-1740): nothing advances; the chunks are offered again.\n\t\t\tcontinue",
     "\t\t\ts.advanceStored(st, tokens)\n\t\t\tcontinue"),
    ("O19", "a preempted request's jobs are flushed", OFF,
     "\t\t\tfor j := range st.jobs {\n\t\t\t\ts.flush[j] = true\n\t\t\t}", "\t\t\t_ = st"),
    ("O20", "a job reading a reallocated block is flushed", OFF,
     "\t\tfor j := range s.pendingOf[b] {\n\t\t\ts.flush[j] = true\n\t\t}", "\t\t_ = b"),
    ("O21", "a sliding window's stale block ids are zeroed", OFF,
     "\t\t\t\tif s.allocated[gs.blockIDs[j]] {\n\t\t\t\t\tgs.blockIDs[j] = 0\n\t\t\t\t}", ""),
    ("O22", "a window stores only the blocks a hit can reach", OFF,
     "\t\t\t\tif mask != nil && !anyIn(mask, idx*bpc, bpc) {\n\t\t\t\t\tcontinue\n\t\t\t\t}", "\t\t\t\t_ = mask"),
    ("O23", "a store completion makes its chunks loadable", OFF,
     "\t\tif js.store {\n\t\t\ts.manager.completeStore(js.keys)\n\t\t} else {",
     "\t\tif js.store {\n\t\t} else {"),
    ("O24", "the engine keeps stepping while jobs are in flight", SCHED,
     " || s.hasPushWork())", ")"),
    ("O25", "finished requests are handed over in request-id order", HOOK,
     "\tsort.Strings(step.Finished)\n", "\tsort.Sort(sort.Reverse(sort.StringSlice(step.Finished)))\n"),
    ("O26", "a resumed request's block ids are its whole table", HOOK,
     "\t\tif r.NumPreemptions > 0 && !s.cfg.UseV2ModelRunner {\n\t\t\tresumed = append(resumed, r)\n\t\t\tcontinue\n\t\t}",
     ""),
    ("O27", "a running request's new block ids are the step's", HOOK,
     "\t\t\tfor _, b := range bs[min(from, len(bs)):] {", "\t\t\tfor _, b := range bs[min(from-from, len(bs)):] {"),
    ("O28", "a deferred hit skips the request this step", SCHED,
     "\t\t\t\t\tif deferred {\n\t\t\t\t\t\t// The connector cannot size the hit yet (None, :983-988).\n\t\t\t\t\t\tskip(q)\n\t\t\t\t\t\tcontinue\n\t\t\t\t\t}",
     "\t\t\t\t\t_ = deferred"),
    ("O29", "a finished request's store is flushed if its block was reallocated", OFF,
     "\t\t\t\tif s.allocated[b] {\n\t\t\t\t\ts.flush[job.ID] = true\n\t\t\t\t}", ""),
    ("O30", "a load's block range starts past the local hit", OFF,
     "\t\tfirst := cdiv(local, g.BlockSize)", "\t\tfirst := 0"),
]

# Mutants no diff test can catch, each with its argument.
EQUIVALENT = {
    "O4": "the evictable-count check is prepare_store's fast path (cpu/manager.py:315-317): without it the LRU's "
          "evict returns None whenever fewer unprotected evictable chunks than needed exist (lru.py:624-629), and "
          "the evictable count is never below that, so the store fails in the same cases",
    "O11": "a request reaches the lookup with jobs in flight only after a preemption, which flushes its jobs "
           "(scheduler.py:1856-1863); the worker waits for flushed jobs before the step runs (worker.py:198-224) "
           "and reports them that step, and a step that preempts runs no waiting pass (scheduler.py:899-917), so "
           "by the next lookup the jobs are complete",
    "O22": "with every group at one block size (New refuses others) the sliding window's alignment is the "
           "full-attention chunk, so each chunk is one alignment segment, whose last cdiv(window-1, block) >= 1 "
           "blocks are reachable (single_type_kv_cache_manager.py:1101-1110): every chunk keeps a reachable "
           "block and the mask filters none (scheduler.py:1719-1725). It filters only beside groups of several "
           "block sizes, which are refused",
    "O26": "with the V1 runner's resumed-request path a preempted request's blocks were all freed, so "
           "the blocks appended at its resumption are its whole table: both lists are equal",
}


def main() -> int:
    src, scratch = sys.argv[1], sys.argv[3]
    fuzz = os.path.abspath(sys.argv[2])
    if not any(f.endswith(".trace.json") for f in os.listdir(fuzz)):
        print(f"no offloading fuzz traces in {fuzz}")
        return 1
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    env = dict(os.environ, BLIS_VLLM_OFFLOAD_FUZZ_DIR=fuzz)
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
        r = subprocess.run(["go", "test", "-count=1", "-run", "TestOffloadingAgainstVLLM|TestOffloadingFuzzCorpusAgainstVLLM",
                            "./core/connector/offloading/"], cwd=scratch, env=env, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            if mid in EQUIVALENT:
                print(f"EQUIV    {mid:4} {row}: {EQUIVALENT[mid]}")
            else:
                print(f"SURVIVED {mid:4} {row}")
                bad += 1
        elif mid in EQUIVALENT:
            print(f"NOW CAUGHT {mid} {row}: remove it from EQUIVALENT")
            bad += 1
        elif "earliest divergence" in r.stdout or "panic:" in r.stdout:
            kind = "panic" if "earliest divergence" not in r.stdout else "divergence"
            print(f"CAUGHT   {mid:4} {row} ({kind})")
        else:
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            print(f"BROKEN   {mid:4} {row}: failed without a divergence: {' | '.join(tail)[:200]}")
            bad += 1
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
