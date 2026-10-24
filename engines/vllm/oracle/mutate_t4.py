# Original to blis-vllm.
"""Mutation sweep for blis-vllm's tier 4 (the connector paths of core/sched and core/kv, and
core/connector/nixl): apply one mutant at a time to a scratch copy of the module, run the
NIXL scenarios and the connector fuzz corpus, and report whether each is caught.

Usage: python3 oracle/mutate_t4.py <blis-vllm checkout> <connector fuzz dir> <scratch dir> [N1,...]
Exits non-zero if a mutant outside EQUIVALENT survives, if an EQUIVALENT one is caught (so the
list stays true), or if a mutant breaks the build without a divergence.
"""
import os
import shutil
import subprocess
import sys

# (id, behaviour it guards, file, exact old text, new text)
MUTANTS = [
    ("N1", "a pull covers the prompt past the local hit", "core/connector/nixl/nixl.go",
     "if n := s.remoteTokenCount(len(r.Prompt)) - localComputed; n > 0 {",
     "if n := s.remoteTokenCount(len(r.Prompt)); n > 0 {"),
    ("N2", "the prefiller holds a finished request's blocks", "core/connector/nixl/nixl.go",
     "\tdelay := len(computedBlockIDs) > 0\n", "\tdelay := false\n"),
    ("N3", "an aborted prefiller request frees at once", "core/connector/nixl/nixl.go",
     "\tif r.Status != sched.FinishedLength && r.Status != sched.FinishedStopped {\n\t\ts.notProcessed[r.ID] = true\n\t\treturn false\n\t}",
     "\tif r.Status != sched.FinishedLength && r.Status != sched.FinishedStopped {\n\t\ts.notProcessed[r.ID] = true\n\t}"),
    ("N4", "a full-prompt landing recomputes its last token", "core/sched/connector.go",
     "\tif r.NumComputed == r.NumTokens() {\n\t\tr.NumComputed = r.NumTokens() - 1\n\t}", ""),
    ("N5", "a failed load with nothing valid frees its blocks", "core/sched/connector.go",
     "\t\t} else {\n\t\t\ts.KV.Free(r.ID)\n\t\t}", "\t\t}"),
    ("N6", "a landed load's blocks are cached at promotion", "core/sched/connector.go",
     "\t} else {\n\t\ts.KV.CacheBlocks(r.ID, r.hashes, r.NumComputed, s.run(r))\n\t}", "\t}"),
    ("N7", "a promoted preempted request resumes as PREEMPTED", "core/sched/connector.go",
     "\tif r.NumPreemptions > 0 {\n\t\tr.Status = Preempted\n\t} else {\n\t\tr.Status = Waiting\n\t}",
     "\tr.Status = Waiting"),
    ("N8", "recompute skips the sync-load failures only", "core/sched/connector.go",
     "\tfor _, id := range asyncFailed {\n\t\ts.recvFailed[id] = true\n\t}\n\treturn syncFailed",
     "\tfor _, id := range asyncFailed {\n\t\ts.recvFailed[id] = true\n\t}\n\treturn append(asyncFailed, syncFailed...)"),
    ("N9", "computed tokens cut at the first invalid block", "core/sched/connector.go",
     "\t\t\tr.NumComputed = idx * s.cfg.BlockSize", "\t\t\tr.NumComputed = (idx + 1) * s.cfg.BlockSize"),
    ("N10", "kv_holding_waiting drains first", "core/sched/scheduler.go",
     "\t\t\tq := &s.holding\n\t\t\tif q.len() == 0 {\n\t\t\t\tq = &s.waiting\n\t\t\t}",
     "\t\t\tq := &s.waiting\n\t\t\tif q.len() == 0 {\n\t\t\t\tq = &s.holding\n\t\t\t}"),
    ("N11", "an async load leaves the in-flight reservation", "core/sched/scheduler.go",
     "\t\t\t\tlookahead, reserved = 0, s.inflightReserved()", "\t\t\t\tlookahead, reserved = 0, 0"),
    ("N12", "an async load's blocks are not cached yet", "core/sched/scheduler.go",
     "Reserved: reserved, DelayCache: async,", "Reserved: reserved, DelayCache: false,"),
    ("N13", "a P/D request counts as waiting to start on arrival", "core/sched/scheduler.go",
     "\t\tif r.Transfer != nil && r.Transfer.DoRemotePrefill {\n\t\t\ts.setFetchStage(r, FetchWaitingToStart)\n\t\t}",
     "\t\t_ = r.Transfer"),
    ("N14", "finished_sending frees the held blocks", "core/sched/connector.go",
     "\tfor _, id := range o.FinishedSending {\n\t\tif r := s.requests[id]; r != nil {\n\t\t\ts.freeBlocks(r)\n\t\t}\n\t}",
     "\tfor range o.FinishedSending {\n\t}"),
    ("N15", "an abort mid-pull holds the blocks until it lands", "core/sched/connector.go",
     "\t\t\tdelayFree = !s.recvDone[r.ID]", "\t\t\tdelayFree = false"),
    ("N16", "external blocks are allocated after the local hit", "core/kv/manager.go",
     "\t\t\t\tif need := cdiv(total, m.blockSize) - len(g.reqBlocks[id]); need > 0 {\n\t\t\t\t\tg.reqBlocks[id] = append(g.reqBlocks[id], m.Pool.New(need)...)\n\t\t\t\t}\n",
     ""),
    ("N17", "the connector sees the block-aligned local hit", "core/sched/scheduler.go",
     "s.connector.MatchedTokens(r, newLocal-newLocal%s.cfg.BlockSize)", "s.connector.MatchedTokens(r, newLocal)"),
    ("N18", "skipped holding requests return in order", "core/sched/scheduler.go",
     "\t\tfor i := len(skippedHolding) - 1; i >= 0; i-- {\n\t\t\ts.holding.prepend(skippedHolding[i])\n\t\t}",
     "\t\tfor _, r := range skippedHolding {\n\t\t\ts.holding.prepend(r)\n\t\t}"),
    ("N19", "the fail policy errors the failed loads", "core/sched/connector.go",
     "\tif !s.conf.RecomputeLoadFailures {\n\t\tfor id := range skip {", "\tif false {\n\t\tfor id := range skip {"),
    ("N20", "a landed pull for a finished request frees it", "core/sched/connector.go",
     "\t\t} else {\n\t\t\ts.freeBlocks(r)\n\t\t}\n\t}\n\tfor _, id := range o.FinishedSending {",
     "\t\t}\n\t}\n\tfor _, id := range o.FinishedSending {"),
    ("N21", "a decoder request frees at finish", "core/connector/nixl/nixl.go",
     "\tif !isPrefiller && !s.cfg.Bidirectional {\n\t\treturn false\n\t}",
     "\tif !isPrefiller && !s.cfg.Bidirectional && false {\n\t\treturn false\n\t}"),
    ("N22", "a load in flight is not admitted before it lands", "core/sched/connector.go",
     "\tif !s.recvDone[r.ID] {\n\t\treturn false\n\t}", ""),
    ("N23", "the waiting count includes kv_holding_waiting", "core/sched/scheduler.go",
     "func (s *Scheduler) Counts() (int, int) { return len(s.running), s.waiting.len() + s.holding.len() }",
     "func (s *Scheduler) Counts() (int, int) { return len(s.running), s.waiting.len() }"),
    ("N24", "the engine runs while a connector holds blocks", "core/sched/scheduler.go",
     "\treturn s.connector != nil && len(s.requests) > len(s.running)+s.waiting.len()+s.holding.len()",
     "\treturn false"),
]

# Mutants no input can tell apart from the code, for vLLM's NIXL pull connector on one
# full-attention KV cache group (docs/testing.md, "Tier 4"). Each reason is why, from vLLM's
# own code; a connector or layout that makes one observable must add a scenario and remove it.
EQUIVALENT = {
    # A NIXL pull is planned once (update_state_after_alloc clears do_remote_prefill,
    # pull_scheduler.py:188-189), so a preempted request is never matched for a pull again
    # and never re-enters WAITING_FOR_REMOTE_KVS with num_preemptions > 0. An offloading
    # connector, which loads on every admission, can.
    "N7": "a preempted request never waits for a NIXL pull",
    # Under "recompute" the async failures are retried when they land and are never in
    # num_scheduled_tokens (they are not running), so skipping them in update_from_output's
    # loop changes nothing; only a sync load (load_kv_async False), which NIXL never makes,
    # puts a running request among the failures.
    "N8": "NIXL makes no sync loads, so the async failures are never scheduled",
    # With one KV cache group, allocate_new_blocks (kv_cache_coordinator.py:312-345) tops the
    # table up to num_tokens_need_slot, which covers the external tokens' blocks (a prompt is
    # at most max_model_len), from the same free-queue head, so allocating the external
    # blocks first gives the same blocks in the same order. With several groups the order
    # across groups differs (issue #33775), but a NIXL load reporting invalid blocks refuses
    # several groups (scheduler.py:3365-3374).
    "N16": "the slot allocation covers the external blocks with the same pool pops",
    # A full-attention hit is whole blocks (find_longest_cache_hit,
    # single_type_kv_cache_manager.py:733-827) when the hash block size is the block size,
    # as it is for one group, so the block-aligned hit is the hit.
    "N17": "a full-attention local hit is always block-aligned",
}


def main() -> int:
    src, scratch = sys.argv[1], sys.argv[3]
    fuzz = os.path.abspath(sys.argv[2])
    if not any(f.endswith(".trace.json") for f in os.listdir(fuzz)):
        print(f"no connector fuzz traces in {fuzz}")
        return 1
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    env = dict(os.environ, BLIS_VLLM_CONNECTOR_FUZZ_DIR=fuzz)
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
        r = subprocess.run(["go", "test", "-count=1", "-run", "TestNIXLAgainstVLLM|TestNIXLFuzzCorpusAgainstVLLM",
                            "./core/connector/nixl/"], cwd=scratch, env=env, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            if mid in EQUIVALENT:
                print(f"EQUIV    {mid:4} {row}: {EQUIVALENT[mid]}")
            else:
                print(f"SURVIVED {mid:4} {row}")
                bad += 1
        elif mid in EQUIVALENT:
            print(f"NOW CAUGHT {mid} {row}: remove it from EQUIVALENT")
            bad += 1
        elif "divergence" in r.stdout or "panic" in r.stdout:
            print(f"CAUGHT   {mid:4} {row}")
        else:
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            print(f"BROKEN   {mid:4} {row}: failed without a divergence: {' | '.join(tail)[:200]}")
            bad += 1
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
