// Original to blis-vllm.

package offloading

import (
	"slices"
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/core/kv"
	"github.com/inference-sim/inference-sim/engines/vllm/core/sched"
)

// A group that shares the GPU block pool can trigger a pending store fence
// even when that group is not itself offloaded.
func TestFlushesPendingStoreWhenAnUnoffloadedGroupReusesItsBlock(t *testing.T) {
	for _, tc := range []struct {
		name      string
		blocks    int
		wantReuse bool
	}{
		{name: "ring reuses store source", blocks: 3, wantReuse: true},
		{name: "ring has a free block", blocks: 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			groups := []kv.Group{{Kind: kv.FullAttention, BlockSize: protoBlock},
				{Kind: kv.CircularBuffer, BlockSize: protoBlock / 2}}
			s, err := sched.New(sched.Config{
				MaxNumSeqs: 1, MaxNumBatchedTokens: protoBlock, BlockSize: protoBlock,
				NumBlocks: tc.blocks, MaxModelLen: 128, EnablePrefixCaching: true,
				EnableChunkedPrefill: true, MaxConcurrentBatches: 1, KVGroups: groups,
			})
			if err != nil {
				t.Fatal(err)
			}
			conn, err := New(Config{
				Groups: []Group{{BlockSize: protoBlock}}, KVGroups: []int{0},
				TokensPerHash: s.KV.HashBlockSize(), NumChunks: 4, OffloadPromptOnly: true,
				PrefixCaching: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := s.SetConnector(conn, sched.ConnectorConfig{Consumer: true, Producer: true}); err != nil {
				t.Fatal(err)
			}
			add(t, s, "first", prompt(protoBlock, 1))
			first := s.Schedule()
			meta := first.KVConnectorMetadata.(Meta)
			if len(meta.Stores) != 1 || len(meta.Stores[0].GPU) != 1 {
				t.Fatalf("first schedule must issue a store from one GPU block, got %+v", meta.Stores)
			}
			store := meta.Stores[0]
			finished := s.Update(first, map[string][]int64{"first": {7}})
			if len(finished) != 1 || finished[0].ID != "first" || !slices.Contains(conn.Jobs(), store.ID) {
				t.Fatalf("first request must finish with its store pending, finished %+v, jobs %v", finished, conn.Jobs())
			}

			add(t, s, "second", prompt(protoBlock, 100))
			second := s.Schedule()
			full, ring := s.KV.GroupBlocks(0, "second"), s.KV.GroupBlocks(1, "second")
			if len(full) != 1 || len(ring) != 1 || len(second.NewIDs) != 1 {
				t.Fatalf("second request must receive one block per group, new %v, full %v, ring %v", second.NewIDs, full, ring)
			}
			if full[0].ID == store.GPU[0] || (ring[0].ID == store.GPU[0]) != tc.wantReuse {
				t.Fatalf("setup did not isolate reuse: store %d, full %d, ring %d, want ring reuse %t",
					store.GPU[0], full[0].ID, ring[0].ID, tc.wantReuse)
			}
			flush := second.KVConnectorMetadata.(Meta).Flush
			if got := slices.Contains(flush, store.ID); got != tc.wantReuse {
				t.Errorf("store %d from GPU block %d, second request full block %d, ring block %d: flush %v, want store flushed %t",
					store.ID, store.GPU[0], full[0].ID, ring[0].ID, flush, tc.wantReuse)
			}
		})
	}
}

func TestFlushesPendingStoreWhenARunningRequestGrows(t *testing.T) {
	for _, tc := range []struct {
		name      string
		blocks    [][]int
		wantFlush bool
	}{
		{name: "GPU-only group reuses the source", blocks: [][]int{nil, {1}}, wantFlush: true},
		{name: "offloaded group reuses the source", blocks: [][]int{{1}, nil}, wantFlush: true},
		{name: "GPU-only group takes another block", blocks: [][]int{nil, {2}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			conn, err := New(Config{
				Groups: []Group{{BlockSize: protoBlock}}, KVGroups: []int{0},
				TokensPerHash: protoBlock, NumChunks: 4, PrefixCaching: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			req := &sched.Request{ID: "running", Prompt: prompt(2*protoBlock, 1)}
			conn.OnNewRequest(req)
			conn.BuildMeta(sched.StepOutput{Reqs: []sched.StepRequest{{
				ID: req.ID, NewBlocks: [][]int{{3}, {4}},
			}}})

			// An earlier request's pending store still reads block 1.
			const pendingStore = 7
			conn.watch(1, pendingStore)
			meta := conn.BuildMeta(sched.StepOutput{Reqs: []sched.StepRequest{{
				ID: req.ID, NewBlocks: tc.blocks,
			}}}).(Meta)
			if got := slices.Contains(meta.Flush, pendingStore); got != tc.wantFlush {
				t.Errorf("flush = %v, want pending store flushed: %v", meta.Flush, tc.wantFlush)
			}
		})
	}
}
