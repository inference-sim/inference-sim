package kvkey

import (
	"fmt"
	"testing"
)

// TestDifferential_ChunkStrideVsVLLM is the BC-K4 differential. It asserts that
// offload keys are derived at CHUNK stride, not block stride, matching vLLM's
// per-chunk offload-key walk.
//
// vLLM (commit 63a9a5010a):
//   - config.py:28-31 — tokens_per_hash and blocks_per_chunk are separate from
//     tokens_per_block; a chunk spans blocks_per_chunk blocks.
//   - scheduler.py — hashes_per_chunk = (tokens_per_block * blocks_per_chunk) //
//     tokens_per_hash; one offload key per chunk (stepping by hashes_per_chunk).
//   - _lookup_complete_chunks — start_chunk_idx = num_computed_tokens //
//     tokens_per_chunk, so chunk i covers tokens [i*tokens_per_chunk,
//     (i+1)*tokens_per_chunk).
//
// FIDELITY NOTE: this test verifies the chunk STRIDE (chunk-granular derivation
// with hierarchical coverage), NOT byte-fidelity to vLLM's trailing-block-hash
// anchoring. BLIS chain-hashes each whole chunk with its own SHA256, a keyspace
// disjoint from the block hashes (see the package deviation note). The test
// discriminates the STRIDE: a block-stride implementation fails both the chunk
// count and the C×-ratio assertions. It cannot (and does not claim to)
// distinguish whole-chunk chaining from trailing-block-hash anchoring — both
// make chunk i a function of tokens[0:(i+1)*tokens_per_chunk].
func TestDifferential_ChunkStrideVsVLLM(t *testing.T) {
	cases := []struct {
		tokensPerBlock int // B (BLIS's tokens_per_hash)
		blocksPerChunk int // C
		numChunks      int // exact, so lengths divide cleanly
	}{
		{tokensPerBlock: 4, blocksPerChunk: 1, numChunks: 5}, // C=1: chunk == block
		{tokensPerBlock: 4, blocksPerChunk: 2, numChunks: 3},
		{tokensPerBlock: 16, blocksPerChunk: 4, numChunks: 6},
		{tokensPerBlock: 2, blocksPerChunk: 8, numChunks: 4},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("B%d_C%d", tc.tokensPerBlock, tc.blocksPerChunk), func(t *testing.T) {
			B, C := tc.tokensPerBlock, tc.blocksPerChunk
			tokensPerChunk := B * C
			numTokens := tokensPerChunk * tc.numChunks
			tokens := randTokens(int64(B*100+C), numTokens)

			blockKeys := DeriveChunkKeys("", tokens, B)              // block stride
			chunkKeys := DeriveChunkKeys("", tokens, tokensPerChunk) // chunk stride

			numBlocks := numTokens / B
			// Chunk count = floor(numBlocks / C) = floor(numTokens / (B*C)).
			if len(chunkKeys) != tc.numChunks {
				t.Fatalf("chunk count: got %d, want %d", len(chunkKeys), tc.numChunks)
			}
			if len(blockKeys) != numBlocks {
				t.Fatalf("block count: got %d, want %d", len(blockKeys), numBlocks)
			}
			// Discrimination: chunk stride yields C× fewer keys than block stride.
			// A block-stride offload-key bug would make len(chunkKeys)==numBlocks.
			if len(blockKeys) != C*len(chunkKeys) {
				t.Errorf("expected block/chunk ratio C=%d: %d != %d*%d", C, len(blockKeys), C, len(chunkKeys))
			}

			// Coverage: chunk i is a function of exactly the first (i+1) chunks
			// of tokens — vLLM's [i*tokens_per_chunk, (i+1)*tokens_per_chunk).
			for i := 0; i < len(chunkKeys); i++ {
				prefix := DeriveChunkKeys("", tokens[:(i+1)*tokensPerChunk], tokensPerChunk)
				if prefix[i] != chunkKeys[i] {
					t.Errorf("chunk %d not covered by its own token range: %q != %q", i, prefix[i], chunkKeys[i])
				}
			}

			// For C==1 the chunk keyspace coincides with the block keyspace.
			if C == 1 {
				for i := range chunkKeys {
					if chunkKeys[i] != blockKeys[i] {
						t.Errorf("C=1 chunk %d should equal block key: %q != %q", i, chunkKeys[i], blockKeys[i])
					}
				}
			}
		})
	}
}
