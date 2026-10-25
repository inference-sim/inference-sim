package sched

import (
	"testing"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// loadingConnector supplies every prompt but its last block asynchronously, so a new request
// waits in WAITING_FOR_REMOTE_KVS with its blocks allocated while the load is in flight.
type loadingConnector struct{ finished []string }

func (c *loadingConnector) OnNewRequest(*Request) {}
func (c *loadingConnector) MatchedTokens(r *Request, local int) (int, bool) {
	return len(r.Prompt) - 4 - local, true
}
func (c *loadingConnector) AfterAlloc(*Request, AllocatedBlocks, int) {}
func (c *loadingConnector) Finished(r *Request, _ []int) bool {
	c.finished = append(c.finished, r.ID)
	return false
}
func (c *loadingConnector) AfterOutput(TransferOutcome) {}

// An abort that names a request twice finishes it once, as vLLM's finish_requests takes the
// ids as a set (vllm/v1/core/sched/scheduler.py:2667-2668@5df63a72). Counted twice, the second
// copy would see the first's FINISHED_ABORTED status and free blocks the in-flight load is
// still writing, report the request finished twice and tell the connector twice.
func TestAbortNamingARequestTwiceFinishesItOnce(t *testing.T) {
	s, err := New(Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 64, BlockSize: 4, NumBlocks: 40,
		MaxModelLen: 128, EnableChunkedPrefill: true, Policy: FCFS, EnablePrefixCaching: true,
		ReserveFullSequence: true})
	if err != nil {
		t.Fatal(err)
	}
	c := &loadingConnector{}
	if err := s.SetConnector(c, ConnectorConfig{}); err != nil {
		t.Fatal(err)
	}
	prompt := make([]int64, 16)
	for i := range prompt {
		prompt[i] = int64(100 + i)
	}
	if err := s.Add(&Request{ID: "a", Prompt: prompt, MaxTokens: 4, EOS: eosToken, Transfer: &TransferParams{}}); err != nil {
		t.Fatal(err)
	}
	s.Schedule()
	if got := s.requests["a"].Status; got != WaitingForRemoteKVs {
		t.Fatalf("setup: the request is %v, want it waiting for its remote KV", got)
	}
	free := s.KV.Pool.NumFree()

	s.Abort([]string{"a", "a"})

	if got := s.KV.Pool.NumFree(); got != free {
		t.Errorf("free blocks %d -> %d: the blocks of an in-flight load were freed", free, got)
	}
	if len(c.finished) != 1 {
		t.Errorf("the connector was told the request finished %d times, want once", len(c.finished))
	}
	if s.finishedPending != 1 {
		t.Errorf("finishedPending = %d, want 1", s.finishedPending)
	}
}

// A request's block hashes are its blocks' keys exactly as kv.BlockHashes computes them over
// prompt and output together, LoRA key and first-block cache salt included, whether a block
// lies in the prompt, in the output, or across the boundary.
func TestRequestHashesAreTheBlockHashesOfItsTokens(t *testing.T) {
	keys := kv.BlockKeys{Extra: []kv.ExtraKey{{"lora", "a1", "a1"}}, CacheSalt: "salt"}
	prompt := []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	r := &Request{Prompt: prompt, Keys: keys}
	r.updateHashes(4)
	for _, tok := range []int64{11, 12, 13, 14, 15, 16, 17} {
		r.Output = append(r.Output, tok)
		r.updateHashes(4)
	}
	want := kv.BlockHashes(append(append([]int64(nil), prompt...), r.Output...), 4, keys)
	if len(r.hashes) != len(want) {
		t.Fatalf("%d hashes, want %d", len(r.hashes), len(want))
	}
	for i := range want {
		if string(r.hashes[i]) != string(want[i]) {
			t.Errorf("block %d: hash differs from kv.BlockHashes", i)
		}
	}
}
