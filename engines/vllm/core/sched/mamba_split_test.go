// Original to blis-vllm.

package sched

import (
	"testing"

	"github.com/inference-sim/blis-vllm/core/kv"
)

func mambaGroup(mode string, ckptBlocks, align int) kv.Group {
	return kv.Group{Kind: kv.Mamba, MambaCacheMode: mode, MambaCheckpointBlocks: ckptBlocks, MambaCheckpointAlignment: align}
}

// mambaCfg is a hybrid model in align mode: block 16, hash block 16, budget 64, no checkpoint
// blocks, no eagle.
func mambaCfg() MambaSplitConfig {
	return MambaSplitConfig{
		Groups:    []kv.Group{{Kind: kv.FullAttention}, mambaGroup("align", 0, 0)},
		BlockSize: 16, HashBlockSize: 16, MaxNumScheduledTokens: 64,
	}
}

type splitCase struct {
	name          string
	cfg           func(*MambaSplitConfig)
	r             MambaSplitRequest
	numNew        int
	newLocal, ext int
	want          int
}

func runSplitCases(t *testing.T, cases []splitCase) {
	t.Helper()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := mambaCfg()
			if c.cfg != nil {
				c.cfg(&cfg)
			}
			m := NewMambaSplit(cfg)
			if !m.Needed() {
				t.Fatalf("split not needed for %+v", cfg)
			}
			if got := m.Split(c.r, c.numNew, c.newLocal, c.ext); got != c.want {
				t.Fatalf("Split(%+v, numNew=%d, local=%d, ext=%d) = %d, want %d", c.r, c.numNew, c.newLocal, c.ext, got, c.want)
			}
		})
	}
}

func TestMambaSplitNeeded(t *testing.T) {
	for _, c := range []struct {
		name   string
		groups []kv.Group
		want   bool
	}{
		{"align", []kv.Group{{Kind: kv.FullAttention}, mambaGroup("align", 0, 0)}, true},
		{"none mode is identity", []kv.Group{{Kind: kv.FullAttention}, mambaGroup("none", 0, 0)}, false},
		{"no mamba group", []kv.Group{{Kind: kv.FullAttention}}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := mambaCfg()
			cfg.Groups = c.groups
			if got := NewMambaSplit(cfg).Needed(); got != c.want {
				t.Fatalf("Needed() = %v, want %v", got, c.want)
			}
		})
	}
}

func TestMambaSplitDefaultsDisableJunctionAndTail(t *testing.T) {
	cfg := mambaCfg()
	if m := NewMambaSplit(cfg); m.PartialCacheHit() || m.SharedPrefixCheckpoint() {
		t.Fatal("equal hash and block sizes must not enable partial hits or the junction stop")
	}
	cfg.HashBlockSize = 8
	if m := NewMambaSplit(cfg); m.PartialCacheHit() {
		t.Fatal("a finer hash block without enable_partial_hash_hits must not enable partial hits")
	}
	cfg.PartialHashHits = true
	m := NewMambaSplit(cfg)
	if !m.PartialCacheHit() || m.SharedPrefixCheckpoint() {
		t.Fatalf("partial=%v shared=%v; want true, false (the junction stop is opt-in)", m.PartialCacheHit(), m.SharedPrefixCheckpoint())
	}
	cfg.PartialHashHits = false
	cfg.SharedPrefixCheckpoint = true
	if NewMambaSplit(cfg).SharedPrefixCheckpoint() {
		t.Fatal("the junction stop needs partial cache hits")
	}
	cfg.PartialHashHits = true
	if !NewMambaSplit(cfg).SharedPrefixCheckpoint() {
		t.Fatal("partial hits and the manager's opt-in must enable the junction stop")
	}
}

func TestMambaSplitCheckpointBlocksNeedEveryMambaGroup(t *testing.T) {
	cfg := mambaCfg()
	cfg.Groups = []kv.Group{{Kind: kv.FullAttention}, mambaGroup("align", 1, 16), mambaGroup("align", 0, 16)}
	m := NewMambaSplit(cfg)
	r := MambaSplitRequest{NumPrompt: 100, NumTokens: 100}
	// One Mamba group without checkpoint blocks: the backend cannot save internally.
	if got := m.Split(r, 100, 0, 0); got != 96 {
		t.Fatalf("got %d, want the block boundary 96", got)
	}
}

func TestMambaSplit(t *testing.T) {
	ckpt := func(align int) func(*MambaSplitConfig) {
		return func(c *MambaSplitConfig) {
			c.Groups = []kv.Group{{Kind: kv.FullAttention}, mambaGroup("align", 1, align)}
		}
	}
	partial := func(shared bool) func(*MambaSplitConfig) {
		return func(c *MambaSplitConfig) {
			c.HashBlockSize, c.PartialHashHits, c.SharedPrefixCheckpoint = 8, true, shared
		}
	}
	runSplitCases(t, []splitCase{
		{name: "decode is identity", r: MambaSplitRequest{NumComputed: 100, NumPrompt: 100, NumTokens: 101}, numNew: 1, want: 1},
		{name: "decode with spec rows is identity", r: MambaSplitRequest{NumComputed: 100, NumPrompt: 100, NumTokens: 101}, numNew: 5, want: 5},
		{name: "start at prefill end is identity", r: MambaSplitRequest{NumComputed: 79, NumPrompt: 50, NumTokens: 80}, numNew: 1, want: 1},
		{name: "chunk ending mid-block is clipped", r: MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 40, want: 32},
		{name: "chunk ending on a boundary is kept", r: MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 48, want: 48},
		{name: "budget below one block schedules nothing", r: MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 10, want: 0},
		{name: "long-prefill threshold below a block advances sub-block",
			cfg: func(c *MambaSplitConfig) { c.LongPrefillTokenThreshold = 8 },
			r:   MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 10, want: 10},
		{name: "block wider than the budget: first chunk advances sub-block",
			cfg: func(c *MambaSplitConfig) { c.BlockSize, c.HashBlockSize = 128, 128 },
			r:   MambaSplitRequest{NumPrompt: 300, NumTokens: 300}, numNew: 64, want: 64},
		{name: "wide block: second chunk stops at the boundary",
			cfg: func(c *MambaSplitConfig) { c.BlockSize, c.HashBlockSize = 128, 128 },
			r:   MambaSplitRequest{NumComputed: 64, NumPrompt: 300, NumTokens: 300}, numNew: 64, want: 64},
		{name: "wide block: an aligned end past the start is kept even when the block exceeds the budget",
			cfg: func(c *MambaSplitConfig) { c.BlockSize, c.HashBlockSize = 128, 128 },
			r:   MambaSplitRequest{NumPrompt: 300, NumTokens: 300}, numNew: 200, want: 128},
		{name: "checkpoint position uses the hash block under partial hits: invalid alignment",
			cfg: func(c *MambaSplitConfig) { partial(false)(c); ckpt(16)(c); c.PartialHashHits = true },
			r:   MambaSplitRequest{NumPrompt: 110, NumTokens: 110}, numNew: 110, want: 96},
		{name: "checkpoint position uses the hash block under partial hits: valid alignment",
			cfg: func(c *MambaSplitConfig) { partial(false)(c); ckpt(8)(c); c.PartialHashHits = true },
			r:   MambaSplitRequest{NumPrompt: 110, NumTokens: 110}, numNew: 110, want: 110},
		{name: "wide block: chunk ending short of the boundary is kept",
			cfg: func(c *MambaSplitConfig) { c.BlockSize, c.HashBlockSize = 128, 128 },
			r:   MambaSplitRequest{NumComputed: 64, NumPrompt: 300, NumTokens: 300}, numNew: 30, want: 30},
		{name: "start mid-block stops at the next boundary",
			r: MambaSplitRequest{NumComputed: 5, NumPrompt: 100, NumTokens: 100}, numNew: 60, want: 11},
		{name: "admission counts the local and external hit in start",
			r: MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 60, newLocal: 16, ext: 5, want: 11},
		{name: "last chunk stops at the last cacheable boundary",
			r: MambaSplitRequest{NumComputed: 80, NumPrompt: 100, NumTokens: 100}, numNew: 20, want: 16},
		{name: "whole prompt in one chunk stops at the last cacheable boundary",
			r: MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 100, want: 96},
		{name: "internal checkpoint lifts the boundary stop", cfg: ckpt(16),
			r: MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 100, want: 100},
		{name: "internal checkpoint with a misaligned checkpoint is invalid", cfg: ckpt(64),
			r: MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 100, want: 96},
		{name: "checkpoint blocks without an alignment (None) are unusable", cfg: ckpt(0),
			r: MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 100, want: 96},
		{name: "internal checkpoint does not apply to a chunk short of the prefill end", cfg: ckpt(16),
			r: MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 40, want: 32},
		{name: "eagle drop backs off one block",
			cfg: func(c *MambaSplitConfig) { c.EagleBlockDrop = true },
			r:   MambaSplitRequest{NumComputed: 80, NumPrompt: 100, NumTokens: 100}, numNew: 20, want: 20},
		{name: "eagle drop stops one block earlier mid-prompt",
			cfg: func(c *MambaSplitConfig) { c.EagleBlockDrop = true },
			r:   MambaSplitRequest{NumPrompt: 100, NumTokens: 100}, numNew: 100, want: 80},
		{name: "resumed request replays its outputs past the prompt",
			r: MambaSplitRequest{NumComputed: 60, NumPrompt: 50, NumTokens: 80}, numNew: 20, want: 4},
		{name: "resumed request, whole replay in one chunk",
			r: MambaSplitRequest{NumPrompt: 50, NumTokens: 80}, numNew: 80, want: 64},
		{name: "resumed request, last token is decode", r: MambaSplitRequest{NumComputed: 79, NumPrompt: 50, NumTokens: 80},
			numNew: 1, want: 1},
		// Hash block 8 under block 16, partial hits on: the prompt's hash-aligned tail and the junction.
		{name: "junction is block-floored by default", cfg: partial(false),
			r: MambaSplitRequest{NumPrompt: 110, NumTokens: 110, SharedPrefixBoundary: 40}, numNew: 110, want: 32},
		{name: "junction is exact when the manager enables it", cfg: partial(true),
			r: MambaSplitRequest{NumPrompt: 110, NumTokens: 110, SharedPrefixBoundary: 40}, numNew: 110, want: 40},
		{name: "junction past the prompt falls back to the floored stop", cfg: partial(true),
			r: MambaSplitRequest{NumPrompt: 50, NumTokens: 80, SharedPrefixBoundary: 60}, numNew: 80, want: 48},
		{name: "junction behind the chunk adds no stop", cfg: partial(true),
			r: MambaSplitRequest{NumComputed: 64, NumPrompt: 110, NumTokens: 110, SharedPrefixBoundary: 40}, numNew: 46, want: 32},
		{name: "tail boundary stops a chunk that passes the last block boundary", cfg: partial(false),
			r: MambaSplitRequest{NumComputed: 96, NumPrompt: 110, NumTokens: 110}, numNew: 14, want: 8},
		{name: "no tail stop without partial hits",
			r: MambaSplitRequest{NumComputed: 96, NumPrompt: 110, NumTokens: 110}, numNew: 14, want: 14},
	})
}

func TestMambaCheckpointPositionAndValidity(t *testing.T) {
	for _, c := range []struct {
		tokens, hash int
		eagle        bool
		want         int
	}{
		{100, 16, false, 96}, {97, 16, false, 96}, {96, 16, false, 80}, {100, 16, true, 80}, {10, 16, true, 0}, {0, 16, false, 0},
	} {
		if got := mambaPrefillCheckpointPosition(c.tokens, c.hash, c.eagle); got != c.want {
			t.Errorf("position(%d,%d,%v) = %d, want %d", c.tokens, c.hash, c.eagle, got, c.want)
		}
	}
	// Valid: a query from 0 to 100 holds the checkpoint at 96, aligned to 16.
	if !mambaPrefillCheckpointValid(0, 100, 96, 16, 16, 16) {
		t.Error("query 0..100 with checkpoint 96 must be valid")
	}
	for name, ok := range map[string]bool{
		"no alignment":            mambaPrefillCheckpointValid(0, 100, 96, 16, 16, 0),
		"checkpoint not inside":   mambaPrefillCheckpointValid(0, 96, 96, 16, 16, 16),
		"start not hash-aligned":  mambaPrefillCheckpointValid(4, 100, 96, 16, 16, 16),
		"too close to the start":  mambaPrefillCheckpointValid(88, 100, 96, 16, 16, 8),
		"not on the alignment":    mambaPrefillCheckpointValid(0, 100, 96, 16, 16, 64),
		"column not past initial": mambaPrefillCheckpointValid(96, 110, 104, 8, 16, 8),
	} {
		if ok {
			t.Errorf("%s: valid, want invalid", name)
		}
	}
}
