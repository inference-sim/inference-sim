// Original to blis-vllm.

package sched

import (
	"fmt"
	"runtime"
	"testing"
)

// decodeAtContext builds a scheduler holding n requests of ctx prompt tokens each, runs their
// prefills to completion, and returns it ready to decode: every Schedule after that is one
// decode token for each request.
func decodeAtContext(tb testing.TB, n, ctx int) *Scheduler {
	tb.Helper()
	const block = 16
	s, err := New(Config{MaxNumSeqs: n, MaxNumBatchedTokens: 1 << 19, EnableChunkedPrefill: true,
		MaxModelLen: ctx + 1<<20, BlockSize: block, NumBlocks: n*(ctx/block+2) + 8, EnablePrefixCaching: true})
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
// only the new block does not. TestDecodeStepCostDoesNotGrowWithContext is the check that runs
// in every test run; this reports the numbers (go test -bench . -benchmem ./core/sched).
func BenchmarkDecodeStepAtLongContext(b *testing.B) {
	for _, ctx := range []int{512, 32768} {
		b.Run(fmt.Sprintf("ctx=%d", ctx), func(b *testing.B) {
			s := decodeAtContext(b, 64, ctx)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
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
// every test run. A step that copies each request's whole token list to hash its new token
// allocates 64 x 32,768 x 8 bytes, about 16 MiB; the bound is 2 MiB.
func TestDecodeStepAllocationDoesNotGrowWithContext(t *testing.T) {
	const reqs, ctx, bound = 64, 32768, 2 << 20
	s := decodeAtContext(t, reqs, ctx)
	var before, after runtime.MemStats
	const steps = 20
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
			"(a whole-sequence copy per sampled token costs about %d)", reqs, ctx, perStep, bound, reqs*ctx*8)
	}
}
