package kvkey

import (
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/tokenid"
)

// randTokens returns n deterministic pseudo-random tokens for a given seed, so
// tests are reproducible (INV-6 discipline: no unseeded randomness).
func randTokens(seed int64, n int) []tokenid.TokenID {
	r := rand.New(rand.NewSource(seed))
	toks := make([]tokenid.TokenID, n)
	for i := range toks {
		toks[i] = tokenid.TokenID(r.Intn(128000))
	}
	return toks
}

// --- BC-K2: chunk keys chain hierarchically ---

// TestDeriveChunkKeys_HierarchicalPrefix verifies that the keys of a
// length-(m*stride) prefix are exactly the first m keys of any longer sequence
// (a match at chunk i implies a match at every chunk j < i).
func TestDeriveChunkKeys_HierarchicalPrefix(t *testing.T) {
	const stride = 4
	tokens := randTokens(1, stride*8) // 8 complete chunks
	full := DeriveChunkKeys("", tokens, stride)
	if len(full) != 8 {
		t.Fatalf("expected 8 chunk keys, got %d", len(full))
	}
	for m := 0; m <= 8; m++ {
		prefix := DeriveChunkKeys("", tokens[:m*stride], stride)
		if len(prefix) != m {
			t.Fatalf("prefix of %d chunks: expected %d keys, got %d", m, m, len(prefix))
		}
		for i := 0; i < m; i++ {
			if prefix[i] != full[i] {
				t.Errorf("prefix[%d]=%q != full[%d]=%q — chunk keys not hierarchical", i, prefix[i], i, full[i])
			}
		}
	}
}

// TestDeriveChunkKeys_SharedPrefixSharesKeys verifies that two sequences sharing
// their first k chunks share the first k keys, and diverge at chunk k when that
// chunk's tokens differ.
func TestDeriveChunkKeys_SharedPrefixSharesKeys(t *testing.T) {
	const stride = 3
	base := randTokens(2, stride*5)
	// A shares the first 2 chunks with base then diverges; B is base.
	a := make([]tokenid.TokenID, len(base))
	copy(a, base)
	a[2*stride] = base[2*stride] + 1 // change the first token of chunk 2

	ka := DeriveChunkKeys("", a, stride)
	kb := DeriveChunkKeys("", base, stride)
	if len(ka) != 5 || len(kb) != 5 {
		t.Fatalf("expected 5 keys each, got %d and %d", len(ka), len(kb))
	}
	for i := 0; i < 2; i++ {
		if ka[i] != kb[i] {
			t.Errorf("shared chunk %d: %q != %q", i, ka[i], kb[i])
		}
	}
	if ka[2] == kb[2] {
		t.Error("chunk 2 differs in tokens but produced identical keys")
	}
	// Divergence must propagate to all later chunks (chained).
	for i := 3; i < 5; i++ {
		if ka[i] == kb[i] {
			t.Errorf("chunk %d should differ after an earlier divergence", i)
		}
	}
}

// TestDeriveChunkKeys_PrevKeySeedsChain verifies that a non-empty prevKey changes
// the derived keys (the chain is genuinely seeded).
func TestDeriveChunkKeys_PrevKeySeedsChain(t *testing.T) {
	const stride = 4
	tokens := randTokens(3, stride*3)
	withEmpty := DeriveChunkKeys("", tokens, stride)
	withSeed := DeriveChunkKeys("deadbeef", tokens, stride)
	if len(withEmpty) != len(withSeed) {
		t.Fatalf("length mismatch: %d vs %d", len(withEmpty), len(withSeed))
	}
	if withEmpty[0] == withSeed[0] {
		t.Error("prevKey did not seed the chain: chunk 0 identical")
	}
}

// --- BC-K6: degenerate inputs ---

func TestDeriveChunkKeys_Degenerate(t *testing.T) {
	// Fewer tokens than one chunk -> nil.
	if got := DeriveChunkKeys("", []tokenid.TokenID{1, 2, 3}, 4); got != nil {
		t.Errorf("short input: expected nil, got %v", got)
	}
	// Empty tokens -> nil.
	if got := DeriveChunkKeys("", nil, 4); got != nil {
		t.Errorf("empty input: expected nil, got %v", got)
	}
	// Partial trailing chunk ignored: 9 tokens, stride 4 -> 2 keys.
	if got := DeriveChunkKeys("", randTokens(4, 9), 4); len(got) != 2 {
		t.Errorf("partial trailing chunk: expected 2 keys, got %d", len(got))
	}
}

func TestDeriveChunkKeys_NonPositiveStridePanics(t *testing.T) {
	for _, stride := range []int{0, -1, -4} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("stride %d: expected panic, got none", stride)
				}
			}()
			DeriveChunkKeys("", []tokenid.TokenID{1, 2, 3, 4}, stride)
		}()
	}
}

// --- BC-K3: interning is injective, idempotent, dense, deterministic ---

func TestInterner_InjectiveIdempotentDense(t *testing.T) {
	in := NewInterner()
	// A key stream with repeats.
	stream := []BlockKey{"a", "b", "a", "c", "b", "a", "d", "c"}
	distinct := map[BlockKey]KeyID{}
	for _, k := range stream {
		id := in.Intern(k)
		// Idempotent: interning again returns the same id.
		if id2 := in.Intern(k); id2 != id {
			t.Errorf("idempotence broken for %q: %d != %d", k, id, id2)
		}
		if prev, seen := distinct[k]; seen {
			if prev != id {
				t.Errorf("stability broken for %q: %d != %d", k, prev, id)
			}
		} else {
			distinct[k] = id
		}
	}
	// Injective: reverse map is a bijection over distinct keys.
	if in.Len() != len(distinct) {
		t.Fatalf("Len=%d != distinct=%d", in.Len(), len(distinct))
	}
	seenID := map[KeyID]bool{}
	for k, id := range distinct {
		if seenID[id] {
			t.Errorf("collision: id %d assigned to more than one key (at %q)", id, k)
		}
		seenID[id] = true
	}
	// Dense: ids are exactly 0..Len-1.
	for id := KeyID(0); int(id) < in.Len(); id++ {
		if !seenID[id] {
			t.Errorf("ids not dense: missing id %d", id)
		}
	}
}

func TestInterner_RoundTrip(t *testing.T) {
	in := NewInterner()
	keys := []BlockKey{"x", "y", "z"}
	for _, k := range keys {
		id := in.Intern(k)
		got, ok := in.Key(id)
		if !ok || got != k {
			t.Errorf("round-trip failed for %q: Key(%d)=(%q,%v)", k, id, got, ok)
		}
	}
}

func TestInterner_KeyOutOfRange(t *testing.T) {
	in := NewInterner()
	in.Intern("only")
	for _, id := range []KeyID{-1, -100, 1, 2, 1 << 40} {
		if got, ok := in.Key(id); ok || got != "" {
			t.Errorf("Key(%d) expected (\"\",false), got (%q,%v)", id, got, ok)
		}
	}
}

// TestInterner_Deterministic verifies two fresh Interners fed the identical call
// sequence produce identical id assignments (INV-6 for the primitive).
func TestInterner_Deterministic(t *testing.T) {
	stream := []BlockKey{"k1", "k2", "k1", "k3", "k4", "k2", "k5"}
	a, b := NewInterner(), NewInterner()
	for _, k := range stream {
		if ida, idb := a.Intern(k), b.Intern(k); ida != idb {
			t.Errorf("determinism broken for %q: %d != %d", k, ida, idb)
		}
	}
	if a.Len() != b.Len() {
		t.Errorf("Len mismatch: %d != %d", a.Len(), b.Len())
	}
}
