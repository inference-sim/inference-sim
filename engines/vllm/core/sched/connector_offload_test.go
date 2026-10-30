package sched

import (
	"reflect"
	"testing"
)

// pushConn is a connector that plans its own transfers: it records each step's output and keeps
// reporting pending work while pending is set.
type pushConn struct {
	pending bool
	steps   []StepOutput
}

func (c *pushConn) OnNewRequest(*Request)                                {}
func (c *pushConn) MatchedTokens(*Request, int) (int, bool)              { return 0, false }
func (c *pushConn) AfterAlloc(*Request, AllocatedBlocks, int)            {}
func (c *pushConn) Finished(*Request, []int) bool                        { return false }
func (c *pushConn) AfterOutput(TransferOutcome)                          {}
func (c *pushConn) HasPendingPushWork() bool                             { return c.pending }
func (c *pushConn) BuildMeta(s StepOutput) any                           { c.steps = append(c.steps, s); return len(c.steps) }
func (c *pushConn) MatchedTokensOrDefer(*Request, int) (int, bool, bool) { return 0, false, false }

func pushSched(t *testing.T) (*Scheduler, *pushConn) {
	t.Helper()
	s, err := New(Config{MaxNumSeqs: 4, MaxNumBatchedTokens: 64, BlockSize: 4, NumBlocks: 40, MaxModelLen: 64,
		EnablePrefixCaching: true, ReserveFullSequence: true})
	if err != nil {
		t.Fatal(err)
	}
	c := &pushConn{}
	if err := s.SetConnector(c, ConnectorConfig{Consumer: true, Producer: true}); err != nil {
		t.Fatal(err)
	}
	return s, c
}

// has_requests counts a connector's pending push work (scheduler.py:2823-2838): with every
// request gone, the engine keeps stepping while the connector reports work.
func TestHasRequestsCountsPendingPushWork(t *testing.T) {
	s, c := pushSched(t)
	if s.HasRequests() {
		t.Fatal("an idle scheduler has requests")
	}
	c.pending = true
	if !s.HasRequests() {
		t.Fatal("HasRequests ignores the connector's pending push work")
	}
}

// build_connector_meta sees the step's new requests with their whole block tables, running ones
// with the blocks the step appended, and the requests finished since the last step in sorted
// order (the order oracle/mocks.json fixes for finished_req_ids); its result is the step's
// metadata.
func TestBuildMetaSeesTheStep(t *testing.T) {
	s, c := pushSched(t)
	for _, id := range []string{"b", "a"} {
		if err := s.Add(&Request{ID: id, Prompt: []int64{1, 2, 3, 4, 5, 6}, MaxTokens: 3, EOS: -1}); err != nil {
			t.Fatal(err)
		}
	}
	out := s.Schedule()
	if out.KVConnectorMetadata != 1 {
		t.Fatalf("the step's metadata is %v, want the connector's", out.KVConnectorMetadata)
	}
	first := c.steps[0]
	if len(first.Reqs) != 2 || len(first.Reqs[0].NewBlocks[0]) != 2 || first.Reqs[0].Resumed {
		t.Fatalf("new requests: %+v, want two with their two blocks each", first.Reqs)
	}
	s.Update(out, map[string][]int64{"a": {7}, "b": {7}})
	out = s.Schedule() // decodes token 7 into block 1: no new block
	for _, r := range c.steps[1].Reqs {
		if r.NewBlocks != nil {
			t.Fatalf("a running request that allocated nothing reports blocks %v", r.NewBlocks)
		}
	}
	s.Abort([]string{"b", "a"})
	s.Update(out, nil)
	s.Schedule()
	if got := c.steps[2].Finished; !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("finished = %v, want [a b] in request-id order", got)
	}
	if got := c.steps[2].Order; len(got) != 0 {
		t.Fatalf("an empty step scheduled %v", got)
	}
}
