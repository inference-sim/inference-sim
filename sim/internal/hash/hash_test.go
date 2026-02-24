package hash

import "testing"

func TestHashTokens_Deterministic(t *testing.T) {
	tokens := []int{1, 2, 3, 4, 5}
	h1 := HashTokens(tokens)
	h2 := HashTokens(tokens)
	if h1 != h2 {
		t.Errorf("HashTokens not deterministic: %q != %q", h1, h2)
	}
	if h1 == "" {
		t.Error("HashTokens returned empty string")
	}
}

func TestHashBlock_ChainsDeterministically(t *testing.T) {
	tokens := []int{10, 20, 30}
	h1 := HashBlock("", tokens)
	h2 := HashBlock("", tokens)
	if h1 != h2 {
		t.Errorf("HashBlock not deterministic: %q != %q", h1, h2)
	}
	// Different prev hash produces different result
	h3 := HashBlock("abc", tokens)
	if h1 == h3 {
		t.Error("HashBlock should produce different result with different prevHash")
	}
}

func TestComputeBlockHashes_MatchesManualChaining(t *testing.T) {
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}
	blockSize := 4
	hashes := ComputeBlockHashes(blockSize, tokens)
	if len(hashes) != 2 {
		t.Fatalf("expected 2 block hashes, got %d", len(hashes))
	}
	// Manual verification: first block hashed with empty prev, second with first's hash
	manual0 := HashBlock("", tokens[0:4])
	manual1 := HashBlock(manual0, tokens[4:8])
	if hashes[0] != manual0 {
		t.Errorf("block 0: got %q, want %q", hashes[0], manual0)
	}
	if hashes[1] != manual1 {
		t.Errorf("block 1: got %q, want %q", hashes[1], manual1)
	}
}

func TestComputeBlockHashes_PartialBlock(t *testing.T) {
	// 5 tokens with block size 4 = 1 full block (partial block ignored)
	tokens := []int{1, 2, 3, 4, 5}
	hashes := ComputeBlockHashes(4, tokens)
	if len(hashes) != 1 {
		t.Fatalf("expected 1 block hash, got %d", len(hashes))
	}
}

func TestComputeBlockHashes_Empty(t *testing.T) {
	hashes := ComputeBlockHashes(4, []int{1, 2})
	if hashes != nil {
		t.Errorf("expected nil for fewer tokens than block size, got %v", hashes)
	}
}
