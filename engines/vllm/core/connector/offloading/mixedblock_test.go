

package offloading

import (
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/core/kv"
	"github.com/inference-sim/inference-sim/engines/vllm/core/sched"
)

// A full-attention group of 16-token blocks beside a sliding-window group of 32-token blocks (the
// shape DeepSeek V4 and MiniMax-M3.1-Flash give) was refused by the connector. Its chunks are per
// group: a request that finished has stored its prefix, and a second request with the same prompt
// finds it in the CPU tier, aligned to the scheduler's block (the LCM of the groups').
func TestGroupsOfSeveralBlockSizesStoreAndLoadAPrefix(t *testing.T) {
	layers := []kv.Group{{Kind: kv.FullAttention}, {Kind: kv.SlidingWindow, Window: 64, BlockSize: 32}}
	s, err := sched.New(sched.Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 512, BlockSize: 16, NumBlocks: 128,
		MaxModelLen: 1024, EnableChunkedPrefill: true, EnablePrefixCaching: true, ReserveFullSequence: true,
		KVGroups: kv.ResolveGroups(layers, false), MaxConcurrentBatches: 1})
	if err != nil {
		t.Fatal(err)
	}
	conn, err := New(Config{Groups: []Group{{BlockSize: 16}, {BlockSize: 32, Window: 64}}, TokensPerHash: s.KV.HashBlockSize(),
		NumChunks: 64, OffloadPromptOnly: true, PrefixCaching: true, Workers: 1})
	if err != nil {
		t.Fatalf("the connector refused groups of two block sizes: %v", err)
	}
	if err := s.SetConnector(conn, sched.ConnectorConfig{Consumer: true, Producer: true}); err != nil {
		t.Fatal(err)
	}
	p := prompt(8*32+5, 100)
	add(t, s, "a", p)
	w := newWorker()
	for i := 0; i < 40 && s.HasRequests(); i++ {
		stepOnce(t, s, w, i)
	}
	b := add(t, s, "b", p)
	n, _, deferred := conn.MatchedTokensOrDefer(b, 0)
	if deferred || n == 0 {
		t.Fatalf("second request: matched %d tokens (deferred %v), want the stored prefix", n, deferred)
	}
	if n%32 != 0 {
		t.Errorf("matched %d tokens, not a multiple of the 32-token block of the larger group", n)
	}
}
