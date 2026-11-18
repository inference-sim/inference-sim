// Original to blis-sglang.

package sched

import "testing"

// Ending a request that already finished changes nothing: it keeps the reason it finished with,
// so the finish a pass reported is the one that stands.
func TestFinishOfAFinishedRequestKeepsItsReason(t *testing.T) {
	cfg := baseConfig(t)
	cfg.ChunkedPrefillSize = nil
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	s.handleGenerateRequest(Arrival{RID: "a", Prompt: []int64{1, 2, 3, 4}, MaxNewTokens: 1})
	l, err := s.Launch()
	if err != nil || l == nil {
		t.Fatalf("no pass: %v", err)
	}
	c, err := s.Complete(l)
	if err != nil || len(c.Finished) != 1 || c.Finished[0].Reason != "length" {
		t.Fatalf("the request did not finish by length: %v %+v", err, c)
	}
	s.Finish("a")
	for _, r := range s.last.Reqs {
		if r.RID() == "a" && r.FinishedReason != "length" {
			t.Fatalf("a finished request's reason became %q", r.FinishedReason)
		}
	}
}

// A request that decodes in the running batch while another request's prefill is in flight is not
// in that pass; ending it still stops its decoding.
func TestFinishOfARunningRequestWhileAnotherPrefillIsInFlight(t *testing.T) {
	cfg := baseConfig(t)
	cfg.ChunkedPrefillSize = nil
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	s.handleGenerateRequest(Arrival{RID: "x", Prompt: []int64{1, 2, 3, 4}, MaxNewTokens: 30})
	l, err := s.Launch()
	if err != nil || l == nil {
		t.Fatalf("no pass: %v", err)
	}
	if _, err := s.Complete(l); err != nil {
		t.Fatal(err)
	}
	s.handleGenerateRequest(Arrival{RID: "y", Prompt: []int64{7, 8, 9, 10}, MaxNewTokens: 2})
	if l, err = s.Launch(); err != nil || l == nil || l.Forward.RIDs[0] != "y" {
		t.Fatalf("y's prefill is not the pass in flight: %v %+v", err, l)
	}
	s.Finish("x")
	c, err := s.Complete(l)
	if err != nil || len(c.Finished) != 0 {
		t.Fatalf("completing y's prefill: %v %+v", err, c)
	}
	for i := 0; i < 8; i++ {
		l, err := s.Launch()
		if err != nil {
			t.Fatal(err)
		}
		if l == nil {
			break
		}
		for _, id := range l.Forward.RIDs {
			if id == "x" {
				t.Fatal("an ended request decoded again")
			}
		}
		if _, err := s.Complete(l); err != nil {
			t.Fatal(err)
		}
	}
}
