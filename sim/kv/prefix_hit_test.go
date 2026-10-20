package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// The DeepSeek-V4.1 layout at 1/8 scale: an 8-token full-attention block (the scheduler
// block) and 4-token window pages; the EAGLE window lookup peeks one page past a boundary.
const phBlock = 8

var v41NoReplay = sim.KVPrefixHit{Window: 16, WindowBlock: 4, WindowEagle: true, EagleBoundaries: true}

func toks(lo, hi int) []sim.TokenID {
	out := make([]sim.TokenID, 0, hi-lo)
	for i := lo; i < hi; i++ {
		out = append(out, sim.TokenID(i))
	}
	return out
}

func cat(parts ...[]sim.TokenID) []sim.TokenID {
	var out []sim.TokenID
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// admit runs a new request through the lookup batch formation does and allocates its whole
// prompt, returning the hit in tokens.
func admit(t *testing.T, kvc *KVCacheState, id string, in []sim.TokenID) int64 {
	t.Helper()
	req := &sim.Request{ID: id, InputTokens: in}
	cached := kvc.GetCachedBlocks(req.FullInputTokens())
	if limit := (req.InputLen() - 1) / phBlock; int64(len(cached)) > limit {
		cached = cached[:limit]
	}
	cached = kvc.ReconcilePrefixHit(req, cached)
	hit := int64(len(cached)) * phBlock
	if !kvc.AllocateKVBlocks(req, hit, req.InputLen(), cached) {
		t.Fatalf("%s: allocation failed", id)
	}
	return hit
}

func newPrefixStore(p sim.KVPrefixHit) *KVCacheState {
	kvc := NewKVCacheState(64, phBlock)
	kvc.PrefixHit = p
	return kvc
}

// A follow-up turn that extends a cached prompt hits only up to the window tail the prompt
// retained, one block below its end (get_replay_boundaries with an EAGLE group), where the
// full-attention prefix alone reaches the prompt's last full block.
func TestPrefixHit_WindowGroupStopsAtTheRetainedReplayBoundary(t *testing.T) {
	a := toks(0, 40)
	b := cat(a, toks(100, 120))
	for _, tt := range []struct {
		name string
		p    sim.KVPrefixHit
		want int64
	}{
		{"full attention only", sim.KVPrefixHit{}, 40},
		{"window group, EAGLE", v41NoReplay, 32},
	} {
		kvc := newPrefixStore(tt.p)
		admit(t, kvc, "a", a)
		if got := admit(t, kvc, "b", b); got != tt.want {
			t.Errorf("%s: hit %d, want %d", tt.name, got, tt.want)
		}
	}
}

// With an EAGLE group the retained boundary sits one block below the prompt's last full
// block, where the peeked page exists; a 41-token prompt's prompt-end boundary (40) would need
// a page past its end, so without the shift it retains nothing and the next turn misses.
func TestPrefixHit_EagleBoundaryIsOneBlockBelowThePrompt(t *testing.T) {
	a := toks(0, 41)
	kvc := newPrefixStore(v41NoReplay)
	admit(t, kvc, "a", a)
	if got := admit(t, kvc, "b", cat(a, toks(100, 120))); got != 32 {
		t.Errorf("hit %d, want 32", got)
	}
}

// A request diverging inside a cached prompt, where no tail was retained, falls back to the
// last retained boundary below (here none: hit 0); its divergence point is pinned as a
// shared-prefix junction, so the next request diverging there hits it.
func TestPrefixHit_DivergenceIsPinnedAsAJunction(t *testing.T) {
	kvc := newPrefixStore(v41NoReplay)
	a := toks(0, 40)
	admit(t, kvc, "a", a)
	if got := admit(t, kvc, "c", cat(a[:20], toks(200, 240))); got != 0 {
		t.Errorf("first divergence at 20: hit %d, want 0 (no tail at or below 16)", got)
	}
	if got := admit(t, kvc, "d", cat(a[:20], toks(300, 340))); got != 16 {
		t.Errorf("second divergence at 20: hit %d, want 16 (the junction the first pinned)", got)
	}
}

// The EAGLE window lookup must match the page past a boundary it then drops: a request
// that matches a cached prompt to one token past the boundary falls to the boundary below.
func TestPrefixHit_EagleWindowPeeksOnePagePastTheBoundary(t *testing.T) {
	a := toks(0, 40)
	e := cat(a[:33], toks(400, 440))
	for _, tt := range []struct {
		name string
		p    sim.KVPrefixHit
		want int64
	}{
		{"EAGLE window", v41NoReplay, 24},
		{"plain window", sim.KVPrefixHit{Window: 16, WindowBlock: 4}, 32},
	} {
		kvc := newPrefixStore(tt.p)
		admit(t, kvc, "a", a)
		if got := admit(t, kvc, "e", e); got != tt.want {
			t.Errorf("%s: hit %d, want %d", tt.name, got, tt.want)
		}
	}
}

// A retained tail dies with the full-attention block its boundary ends: once that block is
// reused for other content, no tail is left.
func TestPrefixHit_EvictedBlockTakesItsTails(t *testing.T) {
	kvc := NewKVCacheState(6, phBlock)
	kvc.PrefixHit = v41NoReplay
	a := &sim.Request{ID: "a", InputTokens: toks(0, 40)}
	cached := kvc.ReconcilePrefixHit(a, nil)
	if !kvc.AllocateKVBlocks(a, 0, 40, cached) || len(kvc.tails) == 0 {
		t.Fatalf("a retained no tails")
	}
	kvc.ReleaseKVBlocks(a)
	filler := &sim.Request{ID: "f", InputTokens: toks(500, 548)}
	if !kvc.AllocateKVBlocks(filler, 0, 48, nil) {
		t.Fatal("filler allocation failed")
	}
	if len(kvc.tails) != 0 {
		t.Errorf("%d tails survive their blocks' reuse", len(kvc.tails))
	}
}

// An EAGLE full-attention group drops its last block from every hit.
func TestPrefixHit_FullAttentionEagleDropsOneBlock(t *testing.T) {
	kvc := newPrefixStore(sim.KVPrefixHit{FullEagle: true})
	a := toks(0, 40)
	admit(t, kvc, "a", a)
	if got := admit(t, kvc, "b", cat(a, toks(100, 120))); got != 32 {
		t.Errorf("hit %d, want 32", got)
	}
}

// SWA bounded replay drops a hit no longer than the replayed window and keeps a longer one.
func TestPrefixHit_BoundedReplayDropsAHitWithinTheWindow(t *testing.T) {
	kvc := newPrefixStore(sim.KVPrefixHit{ReplayTokens: 16})
	a := toks(0, 40)
	admit(t, kvc, "a", a)
	if got := admit(t, kvc, "short", cat(a[:20], toks(600, 640))); got != 0 {
		t.Errorf("16-token hit: %d, want 0", got)
	}
	if got := admit(t, kvc, "long", cat(a, toks(700, 720))); got != 40 {
		t.Errorf("40-token hit: %d, want 40", got)
	}
}
