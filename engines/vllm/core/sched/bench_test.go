// Original to blis-vllm.

package sched

import (
	"fmt"
	"runtime"
	"testing"
)

// decodeAtContext builds a scheduler holding n requests of ctx prompt tokens each, runs their
// prefills to completion, and returns it ready to decode: every Schedule after that is one
// decode token for each request. The cache holds, beside each request's prompt, the blocks the
// next steps decode tokens will fill, so that none of those steps preempts: a step that
// preempts recomputes a request's whole context, which is not the cost being measured.
func decodeAtContext(tb testing.TB, n, ctx, steps int) *Scheduler {
	tb.Helper()
	const block = 16
	// The prefill phase takes a few decode steps of its own before every request decodes, so
	// the margin is steps/block + 4 blocks a request, not steps/block + 1.
	spare := steps/block + 4
	s, err := New(Config{MaxNumSeqs: n, MaxNumBatchedTokens: 1 << 19, EnableChunkedPrefill: true,
		MaxModelLen: ctx + 1<<20, BlockSize: block, NumBlocks: n*(ctx/block+spare) + 8, EnablePrefixCaching: true})
	if err != nil {
		tb.Fatal(err)
	}
	for i := 0; i < n; i++ {
		prompt := make([]int64, ctx)
		for j := range prompt {
			prompt[j] = int64(i*7919 + j%251 + 1)
		}
		if err := s.Add(&Request{ID: fmt.Sprintf("r%d", i), Prompt: prompt, MaxTokens: 1 << 20, EOS: -1}); err != nil {
			tb.Fatal(err)
		}
	}
	for steps := 0; ; steps++ {
		out := s.Schedule()
		if len(out.Scheduled) == 0 || steps > 1000 {
			tb.Fatal("the prefills did not finish")
		}
		sampled := map[string][]int64{}
		allDecoding := true
		for id, k := range out.Scheduled {
			r := s.Request(id)
			if r.NumComputed+k >= len(r.Prompt) {
				sampled[id] = []int64{int64(5 + steps%90)}
			} else {
				allDecoding = false
			}
		}
		s.Update(out, sampled)
		if allDecoding && len(out.Scheduled) == n {
			return s
		}
	}
}

// BenchmarkDecodeStepAtLongContext is one decode step of 64 requests holding 32,768 tokens each:
// schedule, then apply one sampled token each. A step that copies a request's whole token list
// when it hashes a new token costs O(context) here (updateHashes, review of the first oracle
// pull request, perspective 9: 0.79 ms a step at 32k against 0.07 ms at 512); a step that reads
// only the new block does not. TestDecodeStepAllocationDoesNotGrowWithContext is the check that
// runs in every test run; this reports the numbers (go test -bench . -benchmem ./core/sched).
//
// The scheduler is rebuilt (with the timer stopped) every benchChunk steps, with room in the
// cache for exactly that many decode steps, so a long run (-benchtime 2000x or the default)
// measures the same decode step as a short one and never one that preempts.
func BenchmarkDecodeStepAtLongContext(b *testing.B) {
	const benchChunk = 256
	for _, ctx := range []int{512, 32768} {
		b.Run(fmt.Sprintf("ctx=%d", ctx), func(b *testing.B) {
			s := decodeAtContext(b, 64, ctx, benchChunk)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if i > 0 && i%benchChunk == 0 {
					b.StopTimer()
					s = decodeAtContext(b, 64, ctx, benchChunk)
					b.StartTimer()
				}
				out := s.Schedule()
				sampled := make(map[string][]int64, len(out.Scheduled))
				for id := range out.Scheduled {
					sampled[id] = []int64{int64(5 + i%90)}
				}
				s.Update(out, sampled)
			}
		})
	}
}

// A decode step allocates the same order of memory at 32,768 tokens of context as at 512: the
// bytes allocated are deterministic where wall time is not, so this is the check that runs in
// every test run.
//
// The window is 32 steps because a request hashes one block per 16 tokens and all 64 requests
// start in phase: 32 steps hold two hash events per request, so a copy of the whole sequence made
// only when a block is hashed (inside tokenRange) is in the window, not just a copy made on every
// Update.
//
// Measured per step over this window (darwin/arm64, 64 requests): the unmutated code 26 KB at
// 512 and at 32,768 tokens; a whole-sequence copy when a block is hashed 1.1 MB at 32,768 (64 x
// 32,768 x 8 B = 16 MiB at each hash event, once per 16 steps, so 1 MiB a step); a copy on every
// Update 17 MB. The bound is 256 KiB: ten times the unmutated 26 KB and four times under the
// hash-time mutant, so allocation noise cannot reach it and the smaller mutant cannot hide under
// it. oracle/mutate_sched_alloc.py applies both mutants and shows this test fails on each.
func TestDecodeStepAllocationDoesNotGrowWithContext(t *testing.T) {
	const reqs, ctx, steps, bound = 64, 32768, 32, 256 << 10
	s := decodeAtContext(t, reqs, ctx, steps)
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	for i := 0; i < steps; i++ {
		out := s.Schedule()
		sampled := make(map[string][]int64, len(out.Scheduled))
		for id := range out.Scheduled {
			sampled[id] = []int64{int64(5 + i)}
		}
		s.Update(out, sampled)
	}
	runtime.ReadMemStats(&after)
	if perStep := (after.TotalAlloc - before.TotalAlloc) / steps; perStep > bound {
		t.Errorf("a decode step of %d requests at %d tokens of context allocates %d bytes; the bound is %d "+
			"(a whole-sequence copy per hashed block costs about %d a step, one per sampled token about %d)",
			reqs, ctx, perStep, bound, reqs*ctx*8/16, reqs*ctx*8)
	}
}
