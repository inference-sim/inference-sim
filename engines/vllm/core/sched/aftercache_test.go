// Original to blis-vllm.

package sched

import "testing"

// With speculative decoding on, update_from_output caches the blocks a running request's output
// fills, right after that output (vllm/v1/core/sched/scheduler.py:2401-2415@4ba787ec). The case it
// decides: accepted drafts complete a block whose tokens were unknown when the step was allocated,
// and the request stops on that output, so no later allocate caches the block before it is freed.
// A request arriving after then hits the block. v0.32.0 cached after output only for async
// placeholders, and the block was freed uncached.
func TestAcceptedDraftsCacheTheBlockTheyFillBeforeTheRequestIsFreed(t *testing.T) {
	s, err := New(Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 64, BlockSize: 4, NumBlocks: 40, MaxModelLen: 128,
		EnablePrefixCaching: true, Spec: ResolveSpec(SpecOptions{Method: "ngram", NumTokens: 3})})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Add(&Request{ID: "a", Prompt: []int64{101, 102, 103, 104, 105}, MaxTokens: 4, EOS: eosToken}); err != nil {
		t.Fatal(err)
	}
	out := s.Schedule()
	s.Update(out, map[string][]int64{"a": {106}})
	s.UpdateDrafts(map[string][]int64{"a": {107, 108, 109}})
	out = s.Schedule()
	if out.Scheduled["a"] != 4 {
		t.Fatalf("step 2 scheduled %v, want a's token and its 3 drafts", out.Scheduled)
	}
	// Every draft is accepted. The fourth output token is a's last (max_tokens 4), so a stops and
	// the trailing bonus token is trimmed; block 2 (tokens 105-108) is full and computed.
	if fin := s.Update(out, map[string][]int64{"a": {107, 108, 109, 110}}); len(fin) != 1 {
		t.Fatalf("a finished %v, want it stopped at max_tokens", fin)
	}
	if err := s.Add(&Request{ID: "b", Prompt: []int64{101, 102, 103, 104, 105, 106, 107, 108, 201, 202}, MaxTokens: 1, EOS: eosToken}); err != nil {
		t.Fatal(err)
	}
	out = s.Schedule()
	if got := out.Scheduled["b"]; got != 2 {
		t.Errorf("b scheduled %d tokens, want 2: both of a's full blocks (8 tokens) cached, the second by a's last output", got)
	}
}
