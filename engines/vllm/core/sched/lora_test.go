// Original to blis-vllm.

package sched

import "testing"

// max_loras counts the adapter of every running request, including one this step does not
// schedule (vllm/v1/core/sched/scheduler.py:1038-1047@4ba787ec). Under async scheduling, a
// request whose last output token is still in flight is running but not scheduled; with
// max_loras 1, a waiting request under another adapter must wait for it to finish. v0.32.0
// counted only the scheduled requests and admitted it beside.
func TestMaxLoRAsCountsARunningRequestThisStepDoesNotSchedule(t *testing.T) {
	s, err := New(Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 64, BlockSize: 4, NumBlocks: 40,
		MaxModelLen: 128, Async: true, MaxLoRAs: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Add(&Request{ID: "a", Prompt: []int64{11, 12, 13, 14}, MaxTokens: 1, EOS: eosToken, LoRA: 1, LoRAName: "A"}); err != nil {
		t.Fatal(err)
	}
	first := s.Schedule()
	if first.Scheduled["a"] != 4 {
		t.Fatalf("step 1 scheduled %v, want a's 4 prompt tokens", first.Scheduled)
	}
	if err := s.Add(&Request{ID: "b", Prompt: []int64{21, 22, 23, 24}, MaxTokens: 1, EOS: eosToken, LoRA: 2, LoRAName: "B"}); err != nil {
		t.Fatal(err)
	}
	// a's one output token is in flight, so this step has nothing to schedule for it, and its
	// adapter still fills the one LoRA slot.
	second := s.Schedule()
	if _, ok := second.Scheduled["a"]; ok {
		t.Fatalf("step 2 scheduled a (%v); its last token is in flight", second.Scheduled)
	}
	if _, ok := second.Scheduled["b"]; ok {
		t.Errorf("step 2 scheduled b beside running a (%v), want b held back: max_loras 1 and a's adapter is in use", second.Scheduled)
	}
}
