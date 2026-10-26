package sched

import "testing"

// Under async scheduling with a drafter, the AsyncScheduler gives a decoding request
// num_speculative_tokens placeholder drafts after every step it runs
// (AsyncScheduler._update_after_schedule, vllm/v1/core/sched/async_scheduler.py:25-58@5df63a72):
// the worker proposes the real ones itself, so the scheduler never waits for
// update_draft_token_ids, which EngineCore.post_step does not call under async scheduling
// (vllm/v1/engine/core.py:689-696@5df63a72). The next step then verifies them: a decode
// schedules 1 + K tokens, K of them the placeholders, as soon as the request is decoding.
func TestAsyncSchedulingGivesADecodePlaceholderDrafts(t *testing.T) {
	const k = 2
	s, err := New(Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 64, BlockSize: 4, NumBlocks: 40,
		MaxModelLen: 128, EnableChunkedPrefill: true, Policy: FCFS, ReserveFullSequence: true,
		Async: true, DecodeStagger: 1, MaxConcurrentBatches: 2,
		Spec: ResolveSpec(SpecOptions{Method: "mtp", NumTokens: k})})
	if err != nil {
		t.Fatal(err)
	}
	prompt := make([]int64, 8)
	for i := range prompt {
		prompt[i] = int64(100 + i)
	}
	if err := s.Add(&Request{ID: "a", Prompt: prompt, MaxTokens: 16, EOS: eosToken}); err != nil {
		t.Fatal(err)
	}
	// Step 1 prefills the prompt and samples the first token, a placeholder under async.
	first := s.Schedule()
	if got := first.Scheduled["a"]; got != len(prompt) {
		t.Fatalf("step 1 scheduled %d tokens, want the %d-token prompt", got, len(prompt))
	}
	// Step 2, scheduled before step 1's output returns, is the request's first decode. It
	// verifies K placeholder drafts: the AsyncScheduler set them after step 1.
	second := s.Schedule()
	if got := second.Scheduled["a"]; got != 1+k {
		t.Fatalf("step 2 scheduled %d tokens for a decoding request, want 1 + num_speculative_tokens = %d",
			got, 1+k)
	}
	if got := len(second.SpecDecode["a"]); got != k {
		t.Fatalf("step 2 verifies %d drafts, want the %d placeholders", got, k)
	}
	// Step 1's output arrives: one token sampled. Step 2's: the bonus token, no draft accepted.
	s.Update(first, map[string][]int64{"a": {prompt[0]}})
	s.Update(second, map[string][]int64{"a": {prompt[1]}})
	third := s.Schedule()
	if got := third.Scheduled["a"]; got != 1+k {
		t.Fatalf("step 3 scheduled %d tokens, want 1 + %d again", got, k)
	}
}
