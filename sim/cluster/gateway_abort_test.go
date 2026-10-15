package cluster

import (
	"errors"
	"strings"
	"testing"
)

// TestRun_StuckGatewayIsAnError: a run whose gateway holds requests that no event can release
// fails instead of ending as if it succeeded, both when nothing is scheduled and when only the
// gateway's own background timers keep ticking.
func TestRun_StuckGatewayIsAnError(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tickUs int64
	}{{"no events left", 0}, {"only daemon timers", 60_000_000}} {
		t.Run(tc.name, func(t *testing.T) {
			cs := NewClusterSimulator(newTestDeploymentConfig(2), NewSliceRequestSource(newTestRequests(3)), nil)
			cs.SetGateway(&fakeGateway{swallow: true, tickUs: tc.tickUs})
			err := cs.Run()
			if err == nil || !strings.Contains(err.Error(), "gateway holds 3 request(s)") {
				t.Fatalf("Run() = %v, want an error naming the 3 stuck requests", err)
			}
		})
	}
}

// TestRun_GatewayAbortEndsTheRun: an outcome the gateway cannot account for ends the run with
// that error.
func TestRun_GatewayAbortEndsTheRun(t *testing.T) {
	boom := errors.New("unclassifiable router outcome")
	cs := NewClusterSimulator(newTestDeploymentConfig(2), NewSliceRequestSource(newTestRequests(3)), nil)
	cs.SetGateway(&fakeGateway{abortWith: boom})
	if err := cs.Run(); !errors.Is(err, boom) {
		t.Fatalf("Run() = %v, want %v", err, boom)
	}
}

// TestDisaggregation_SubRequestsKeepTheAdapter: a disaggregated request's prefill and decode
// children run on the adapter the gateway dispatched it with (the router may have rewritten it).
func TestDisaggregation_SubRequestsKeepTheAdapter(t *testing.T) {
	reqs := newTestRequests(3)
	for _, r := range reqs {
		r.Adapter = "adapter-a"
	}
	cs := NewClusterSimulator(newTestDisaggDeploymentConfig(4, 2, 2), NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)
	if len(cs.parentRequests) != 3 {
		t.Fatalf("%d parents, want 3", len(cs.parentRequests))
	}
	for _, p := range cs.parentRequests {
		if p.PrefillSubReq == nil || p.PrefillSubReq.Adapter != "adapter-a" {
			t.Errorf("parent %s: prefill child adapter %v", p.ID, p.PrefillSubReq)
		}
		if p.DecodeSubReq == nil || p.DecodeSubReq.Adapter != "adapter-a" {
			t.Errorf("parent %s: decode child missing or without the adapter", p.ID)
		}
	}
}
