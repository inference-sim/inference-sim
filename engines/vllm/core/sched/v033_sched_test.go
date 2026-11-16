// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package sched

import "testing"

// A Mamba align prompt whose block hashes cover the drafter's lookahead stops its chunk where
// the checkpoint leaves those tokens uncomputed: get_mamba_prefill_checkpoint_position leaves
// max(1, num_hash_lookahead_tokens) tokens (vllm/v1/kv_cache_interface.py:1169-1186@4ba787ec),
// where v0.32.0 left one whatever the lookahead.
func TestMambaSplitLeavesTheHashLookaheadUncomputed(t *testing.T) {
	for _, c := range []struct {
		name      string
		lookahead int
		want      int
	}{
		// 98 tokens, block 16: (98 - 3) // 16 * 16 = 80. v0.32.0's (98 - 1) // 16 * 16 is 96.
		{"multi-module MTP, lookahead 3", 3, 80},
		// A lookahead of 1 leaves one token, as no lookahead does.
		{"EAGLE, lookahead 1", 1, 96},
		{"no lookahead", 0, 96},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := mambaCfg()
			cfg.MaxNumScheduledTokens = 512
			cfg.HashLookahead = c.lookahead
			m := NewMambaSplit(cfg)
			if got := m.Split(MambaSplitRequest{NumPrompt: 98, NumTokens: 98}, 98, 0, 0); got != c.want {
				t.Fatalf("Split of a 98-token prompt with hash lookahead %d = %d, want %d", c.lookahead, got, c.want)
			}
		})
	}
}

// DFlash and DSpark lower max_num_active_seqs to max_num_batched_tokens // num_lookahead_tokens,
// so every running request's draft pass fits in the batch (VllmConfig._set_max_num_scheduled_tokens,
// vllm/config/vllm.py:2377-2396@4ba787ec). DSpark K=3 has a lookahead of 3; a budget of 12 runs
// at most 4 requests, though 8 one-token prompts fit in it. v0.32.0 admitted all 8.
func TestDSparkCapsActiveRequestsAtItsDraftPass(t *testing.T) {
	for _, c := range []struct {
		method string
		want   int
	}{
		{"dspark", 4},
		{"dflash", 3}, // DFlash's lookahead is K+1 = 4: 12 // 4
		{"eagle3", 8}, // no draft query pass: no cap
	} {
		t.Run(c.method, func(t *testing.T) {
			s, err := New(Config{MaxNumSeqs: 8, MaxNumBatchedTokens: 12, BlockSize: 4, NumBlocks: 64,
				MaxModelLen: 64, Spec: ResolveSpec(SpecOptions{Method: c.method, NumTokens: 3})})
			if err != nil {
				t.Fatal(err)
			}
			for i := range 8 {
				id := string(rune('a' + i))
				if err := s.Add(&Request{ID: id, Prompt: []int64{int64(10 + i)}, MaxTokens: 4, EOS: eosToken}); err != nil {
					t.Fatal(err)
				}
			}
			if got := len(s.Schedule().Scheduled); got != c.want {
				t.Fatalf("%s K=3, max_num_batched_tokens 12: first step ran %d requests, want %d", c.method, got, c.want)
			}
		})
	}
}

// The budget must exceed num_lookahead_tokens (vllm/config/vllm.py:2370-2376@4ba787ec). EAGLE-3
// with K=3 takes no draft slots but writes 3 lookahead slots, so a budget of 3 is refused;
// v0.32.0 compared the draft slots (0) and started it.
func TestABudgetNoLargerThanTheLookaheadIsRefused(t *testing.T) {
	cfg := Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 3, BlockSize: 4, NumBlocks: 16, MaxModelLen: 32,
		Spec: ResolveSpec(SpecOptions{Method: "eagle3", NumTokens: 3})}
	if _, err := New(cfg); err == nil {
		t.Fatal("eagle3 K=3 with max_num_batched_tokens 3 started; vLLM v0.33.0 refuses it")
	}
	cfg.MaxNumBatchedTokens = 4
	if _, err := New(cfg); err != nil {
		t.Fatalf("eagle3 K=3 with max_num_batched_tokens 4: %v, want it to start", err)
	}
}
