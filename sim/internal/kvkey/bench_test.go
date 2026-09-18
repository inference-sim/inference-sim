package kvkey

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/hash"
)

// realKeys returns n realistic 64-hex-character content keys (the actual key
// width the hot maps use), produced via the single hash source.
func realKeys(n int) []BlockKey {
	keys := make([]BlockKey, n)
	prev := ""
	for i := 0; i < n; i++ {
		h := hash.HashBlock(prev, randTokens(int64(i), 16))
		keys[i] = BlockKey(h)
		prev = h
	}
	return keys
}

func BenchmarkDeriveChunkKeys(b *testing.B) {
	tokens := randTokens(1, 2048) // ~a reasoning context
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DeriveChunkKeys("", tokens, 16)
	}
}

func BenchmarkIntern_Hit(b *testing.B) {
	in := NewInterner()
	keys := realKeys(1000)
	for _, k := range keys {
		in.Intern(k)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		in.Intern(keys[i%len(keys)])
	}
}

func BenchmarkIntern_Miss(b *testing.B) {
	keys := realKeys(b.N)
	in := NewInterner()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		in.Intern(keys[i])
	}
}

// BenchmarkProbe_StringKey and BenchmarkProbe_KeyID demonstrate the interning
// lever (evidence P): probing a map keyed on a 64-byte content string vs a map
// keyed on the interned integer handle. This is the synthetic proxy for the
// issue's "ns/op against baseline" obligation — nothing is wired in this PR, so
// there is no changed hot path to benchmark; these show the lever exists.
func BenchmarkProbe_StringKey(b *testing.B) {
	keys := realKeys(1000)
	m := make(map[BlockKey]int64, len(keys))
	for i, k := range keys {
		m[k] = int64(i)
	}
	b.ResetTimer()
	var sink int64
	for i := 0; i < b.N; i++ {
		sink += m[keys[i%len(keys)]]
	}
	_ = sink
}

func BenchmarkProbe_KeyID(b *testing.B) {
	keys := realKeys(1000)
	in := NewInterner()
	ids := make([]KeyID, len(keys))
	m := make(map[KeyID]int64, len(keys))
	for i, k := range keys {
		ids[i] = in.Intern(k)
		m[ids[i]] = int64(i)
	}
	b.ResetTimer()
	var sink int64
	for i := 0; i < b.N; i++ {
		sink += m[ids[i%len(ids)]]
	}
	_ = sink
}
