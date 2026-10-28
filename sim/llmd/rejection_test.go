package llmd

import (
	"fmt"
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
	cs := newCluster(testDeployment(2), cluster.NewSliceRequestSource(testRequests(5, 1, 1000, 0)), nil)
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

// TestRandomStreamScopeIsThePoolIdentity: the router's random streams are seeded with the pool's
// identity, the same string for both pool forms as before the router contract (a different scope
// would move every random draw, INV-6).
func TestRandomStreamScopeIsThePoolIdentity(t *testing.T) {
	cs := newCluster(testDeployment(2), cluster.NewSliceRequestSource(testRequests(2, 1, 1000, 0)), nil)
	gw := New(Options{Model: "test-model"})
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatal(err)
	}
	if gw.rt.scope != "Deployment.apps default/epp" {
		t.Errorf("standalone pool scope %q", gw.rt.scope)
	}
	res := runPool(t, poolManifests, map[string]string{"app": "vllm"})
	if res.gw.rt.scope != "InferencePool.inference.networking.k8s.io default/vllm" {
		t.Errorf("InferencePool scope %q", res.gw.rt.scope)
	}
}

// TestRandBeforeScopePanics: a random draw before the pool is known is a hosting fault, never a
// stream seeded without the pool's identity.
func TestRandBeforeScopePanics(t *testing.T) {
	rt := newDESRuntime(nil, 1, 0)
	defer func() {
		if r := recover(); r == nil || !strings.Contains(fmt.Sprint(r), "before its pool was known") {
			t.Fatalf("recover() = %v, want the unscoped-draw panic", r)
		}
	}()
	rt.Rand("picker")
}

// TestCheckEngineSignals_Order pins which fault is reported when two of BLIS's signal checks fail
// in different plugin types: prefix indexes, then metrics sources, then extractors.
func TestCheckEngineSignals_Order(t *testing.T) {
	prefix := []embed.PrefixIndex{{Name: "p", BlockSize: 64, EngineType: "vllm"}}
	metrics := []embed.MetricsSource{{Name: "m", Path: "/stats"}}
	extractor := []embed.MetricsExtractor{{Name: "x", DefaultEngine: "sglang"}}
	for _, tc := range []struct {
		name string
		req  embed.Requirements
		want string
	}{
		{"prefix before metrics", embed.Requirements{PrefixIndexes: prefix, MetricsSources: metrics}, "cuts block keys"},
		{"metrics before extractor", embed.Requirements{MetricsSources: metrics, Extractors: extractor}, `scrapes "/stats"`},
		{"prefix before extractor", embed.Requirements{PrefixIndexes: prefix, Extractors: extractor}, "cuts block keys"},
	} {
		if err := checkEngineSignals(tc.req, nil, 16); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", tc.name, err, tc.want)
		}
	}
}
