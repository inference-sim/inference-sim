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
