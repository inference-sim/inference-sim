package llmd

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/embed"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// TestRejectionBucket_AllowList: each rejection kind BLIS knows maps to exactly one INV-1 bucket,
// and Unclassified, or a kind a newer contract adds, is reported as unclassified rather than
// counted as shedding.
func TestRejectionBucket_AllowList(t *testing.T) {
	want := map[embed.RejectKind]cluster.Rejection{
		embed.NoEndpoints:      cluster.RejectedRouting,
		embed.Unavailable:      cluster.RejectedRouting,
		embed.TTLExpired:       cluster.RejectedGatewayExpired,
		embed.Saturated:        cluster.RejectedGatewayCapacity,
		embed.ContextCancelled: cluster.RejectedClientTimeout,
		embed.Admission:        cluster.RejectedAdmission,
	}
	for _, k := range append(embed.RejectKinds(), "a-kind-from-a-newer-contract") {
		got, ok := rejectionBucket(k)
		w, known := want[k]
		if ok != known || (ok && got != w) {
			t.Errorf("rejectionBucket(%s) = (%v, %v), want (%v, %v)", k, got, ok, w, known)
		}
	}
}

// TestUnclassifiableOutcomeAbortsTheRunThroughTheRouter: an outcome BLIS does not know, arriving
// through the embedded router's reject path, ends the run with an error that names it.
func TestUnclassifiableOutcomeAbortsTheRunThroughTheRouter(t *testing.T) {
	cs := cluster.NewClusterSimulator(testDeployment(2), cluster.NewSliceRequestSource(testRequests(5, 1, 1000, 0)), nil)
	gw := New(Options{Model: "test-model"})
	injected := false
	gw.observe = func(kind, detail string) {
		if kind == "request" && !injected {
			injected = true
			f := gw.flights[detail]
			f.held = false
			gw.pending--
			gw.reject(f, &embed.Rejection{Kind: embed.Unclassified, Msg: "inference error: Internal - injected internal error"})
		}
	}
	cs.SetGateway(gw)
	err := cs.Run()
	if err == nil || !strings.Contains(err.Error(), "cannot classify") || !strings.Contains(err.Error(), "injected internal error") {
		t.Fatalf("Run() = %v, want the unclassifiable outcome to abort the run", err)
	}
}

// kvBlocks is the router's KV-event index for inst's endpoint on tier ("" for every tier), read
// through the contract's test capability. A router without it fails the test: the fidelity checks
// that use it are never skipped.
func kvBlocks(t *testing.T, gw *Gateway, inst cluster.InstanceID, tier string) map[uint64]bool {
	t.Helper()
	insp, ok := gw.router.(embed.KVIndexInspector)
	if !ok {
		t.Fatalf("the router (%T) does not implement embed.KVIndexInspector", gw.router)
	}
	got, err := insp.KVBlocks(gw.endpoints[inst].ep.Address(), tier)
	if err != nil {
		t.Fatal(err)
	}
	return got
}
