// Original to blis-vllm.

package kv

import (
	"reflect"
	"testing"
)

// With a hash lookahead of L, vLLM v0.33.0's BlockStored carries, for each stored block, the L
// tokens after it that its hash covers, fewer at the end of the request's tokens
// (lookahead_token_ids; cache_full_blocks and _lookahead_token_ids,
// vllm/v1/core/block_pool.py:322-354, 368-371@4ba787ec). A pool that hashes no lookahead reports
// none, as vLLM's None.
func TestBlockStoredCarriesEachBlocksLookaheadTokens(t *testing.T) {
	tokens := []int64{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
	hashes := []BlockHash{BlockHash("a"), BlockHash("b"), BlockHash("c")}
	run := StoredRun{Tokens: tokens}

	p := NewPool(8, true)
	p.EnableEvents()
	p.SetHashLookahead(3)
	blocks := p.New(2)
	p.CacheFull(blocks, hashes, 0, 2, run, 4)
	ev := p.TakeEvents()
	if len(ev) != 1 {
		t.Fatalf("caching two blocks queued %d events, want one BlockStored", len(ev))
	}
	// Block 0 ends at token 4 and block 1 at token 8, where only two of the three tokens remain.
	if want := [][]int64{{14, 15, 16}, {18, 19}}; !reflect.DeepEqual(ev[0].Lookahead, want) {
		t.Errorf("lookahead_token_ids = %v, want %v", ev[0].Lookahead, want)
	}

	// A block cached later in the run starts its own event; its lookahead runs past the tokens,
	// so it is empty, not absent.
	more := append(blocks, p.New(1)...)
	p.CacheFull(more, hashes, 2, 3, StoredRun{Tokens: append(tokens, 20, 21)}, 4)
	if ev := p.TakeEvents(); len(ev) != 1 || !reflect.DeepEqual(ev[0].Lookahead, [][]int64{{}}) {
		t.Errorf("the block ending at the last token: lookahead_token_ids %v, want one empty list", lookaheadOf(ev))
	}

	plain := NewPool(8, true)
	plain.EnableEvents()
	plain.CacheFull(plain.New(2), hashes, 0, 2, run, 4)
	if ev := plain.TakeEvents(); len(ev) != 1 || ev[0].Lookahead != nil {
		t.Errorf("a pool without lookahead: lookahead_token_ids %v, want none", lookaheadOf(ev))
	}
}

// cache_partial_block's BlockStored reports the lookahead after the partial entry's end
// (block_pool.py:598-600@4ba787ec).
func TestAPartialEntrysBlockStoredCarriesItsLookahead(t *testing.T) {
	p := NewPool(6, true)
	p.EnableEvents()
	p.SetHashLookahead(2)
	b := p.New(1)[0]
	hashes := []BlockHash{BlockHash("a"), BlockHash("b"), BlockHash("c")}
	run := StoredRun{Tokens: []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}
	p.CachePartialBlock(0, b, hashes, 8, 4, run, false)
	if ev := p.TakeEvents(); len(ev) != 1 || !reflect.DeepEqual(ev[0].Lookahead, [][]int64{{8, 9}}) {
		t.Errorf("a partial entry at 8 tokens: lookahead_token_ids %v, want [[8 9]]", lookaheadOf(ev))
	}
}

func lookaheadOf(ev []Event) [][][]int64 {
	var out [][][]int64
	for _, e := range ev {
		out = append(out, e.Lookahead)
	}
	return out
}
