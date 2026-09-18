package kvkey

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/tokenid"
)

// FuzzDeriveChunkKeys checks robustness: for any token sequence and any positive
// stride, DeriveChunkKeys never panics and returns exactly len(tokens)/stride
// keys. The fuzzed stride is clamped to [1, 4096] so it never triggers the
// intended <=0 panic (a caller-bug guard, not a robustness failure).
func FuzzDeriveChunkKeys(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8}, 4)
	f.Add([]byte{}, 1)
	f.Add([]byte{9}, 3)
	f.Fuzz(func(t *testing.T, raw []byte, rawStride int) {
		stride := rawStride % 4096
		if stride < 0 {
			stride = -stride
		}
		stride++ // now in [1, 4096]

		tokens := make([]tokenid.TokenID, len(raw))
		for i, b := range raw {
			tokens[i] = tokenid.TokenID(b)
		}

		keys := DeriveChunkKeys("", tokens, stride)
		if want := len(tokens) / stride; len(keys) != want {
			t.Errorf("len(keys)=%d, want %d (tokens=%d stride=%d)", len(keys), want, len(tokens), stride)
		}
	})
}

// FuzzInterner checks that interning an arbitrary key stream preserves
// injectivity (distinct keys -> distinct ids) and idempotence (same key -> same
// id) regardless of input.
func FuzzInterner(f *testing.F) {
	f.Add([]byte("aabbccabc"))
	f.Add([]byte(""))
	f.Fuzz(func(t *testing.T, raw []byte) {
		in := NewInterner()
		// Derive a stream of single-byte keys plus a few multi-byte ones.
		keyToID := map[BlockKey]KeyID{}
		idToKey := map[KeyID]BlockKey{}
		for _, b := range raw {
			k := BlockKey([]byte{b})
			id := in.Intern(k)
			// Idempotent.
			if id2 := in.Intern(k); id2 != id {
				t.Fatalf("idempotence broken for %q: %d != %d", k, id, id2)
			}
			if prev, seen := keyToID[k]; seen && prev != id {
				t.Fatalf("stability broken for %q: %d != %d", k, prev, id)
			}
			if other, clash := idToKey[id]; clash && other != k {
				t.Fatalf("injectivity broken: id %d maps to %q and %q", id, other, k)
			}
			keyToID[k] = id
			idToKey[id] = k
		}
		if in.Len() != len(keyToID) {
			t.Fatalf("Len=%d != distinct=%d", in.Len(), len(keyToID))
		}
	})
}
