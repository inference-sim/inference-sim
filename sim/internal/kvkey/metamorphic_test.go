package kvkey

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/hash"
	"github.com/inference-sim/inference-sim/sim/internal/tokenid"
)

// TestMetamorphic_ChunkStrideEqualsHashPkg is the BC-K1 metamorphic proof: the
// same tokens entering via kvkey.DeriveChunkKeys at block stride and via the
// shared hash source (hash.ComputeBlockHashes / hash.HashBlock) produce
// byte-identical keys. This is what makes the chunk keyspace share the single
// hash function — a tier cannot grow its own hash and still match.
func TestMetamorphic_ChunkStrideEqualsHashPkg(t *testing.T) {
	blockSizes := []int{1, 2, 4, 16}
	for _, b := range blockSizes {
		for _, seed := range []int64{1, 7, 42} {
			tokens := randTokens(seed, b*10+3) // includes a partial trailing block
			got := DeriveChunkKeys("", tokens, b)

			// Reference from the sole hash source. []string cannot be slice-cast
			// to []BlockKey in Go — convert element-wise.
			ref := hash.ComputeBlockHashes(b, tokens)
			if len(got) != len(ref) {
				t.Fatalf("b=%d seed=%d: length %d != hash pkg %d", b, seed, len(got), len(ref))
			}
			for i := range ref {
				if string(got[i]) != ref[i] {
					t.Errorf("b=%d seed=%d chunk %d: %q != hash pkg %q", b, seed, i, got[i], ref[i])
				}
			}

			// And the first chunk equals a bare HashBlock over the first block.
			if len(tokens) >= b {
				want := BlockKey(hash.HashBlock("", tokens[:b]))
				if got[0] != want {
					t.Errorf("b=%d seed=%d chunk 0: %q != HashBlock %q", b, seed, got[0], want)
				}
			}
		}
	}
}

// TestMetamorphic_SeededChunkEqualsHashBlock verifies the seeded (prevKey) path
// also routes through the same HashBlock byte stream.
func TestMetamorphic_SeededChunkEqualsHashBlock(t *testing.T) {
	const b = 4
	tokens := randTokens(9, b*2)
	prev := BlockKey(hash.HashBlock("", []tokenid.TokenID{99, 98, 97, 96}))
	got := DeriveChunkKeys(prev, tokens, b)
	want0 := BlockKey(hash.HashBlock(string(prev), tokens[:b]))
	if got[0] != want0 {
		t.Errorf("seeded chunk 0: %q != HashBlock(prev) %q", got[0], want0)
	}
}
