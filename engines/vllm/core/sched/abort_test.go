package sched

import "testing"

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
