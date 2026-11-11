package vllmengine

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/core/vllm"
)

// recorder is the optional method a host reads by type assertion on the engine.Engine.
type recorder interface{ RecordSelected(*load.Data) }

// selectedBy builds the pool with a private Data (Input.Data nil), so the open-time
// registration lands in a Data nobody reads, and returns what registering the engine with a
// fresh host Data puts there.
func selectedBy(t *testing.T, scenario string, pool int) ([]load.Selected, engine.Engine) {
	t.Helper()
	e, err := New(Input{Spec: engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, scenario), Pool: pool}})
	if err != nil {
		t.Fatal(err)
	}
	r, ok := e.(recorder)
	if !ok {
		t.Fatalf("%s pool %d built %T, which has no RecordSelected(*load.Data)", scenario, pool, e)
	}
	host := embeddedData(t)
	if n := len(host.Report().Selected); n != 0 {
		t.Fatalf("a fresh host Data holds %d entries before registration", n)
	}
	r.RecordSelected(nil)
	if n := len(host.Report().Selected); n != 0 {
		t.Fatalf("RecordSelected(nil) changed the host Data to %d entries", n)
	}
	r.RecordSelected(host)
	return host.Report().Selected, e
}

// openedWith is what the same pool puts in a Data the host passes at build time: the entries
// RecordSelected must reproduce for an engine built with a private Data.
func openedWith(t *testing.T, scenario string, pool int) []load.Selected {
	t.Helper()
	host := embeddedData(t)
	_, err := New(Input{Spec: engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, scenario), Pool: pool}, Data: host})
	if err != nil {
		t.Fatal(err)
	}
	return host.Report().Selected
}

// The engine New returns registers its pricing kernel with the host's Data on request: a
// plain engine, a data-parallel group (one shared pricer), each pool of a prefill/decode
// deployment (separate kernels), and several instances of one pool.
func TestRecordSelectedRegistersEachPricingKernel(t *testing.T) {
	Register()
	cases := []struct {
		name, scenario string
		pool           int
		group          bool
	}{
		{"plain engine", "d058-032.yaml", 0, false},
		{"dp=2 group", "d079-032.yaml", 0, true},
		{"prefill pool", "muse-h100-roce-1p1d.yaml", 0, false},
		{"decode pool", "muse-h100-roce-1p1d.yaml", 1, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, e := selectedBy(t, c.scenario, c.pool)
			if len(got) == 0 {
				t.Fatal("RecordSelected put no coefficient in the host's Data")
			}
			if _, isGroup := e.(*vllm.DPGroup); isGroup != c.group {
				t.Fatalf("built %T, want a DPGroup: %v", e, c.group)
			}
			if want := openedWith(t, c.scenario, c.pool); !reflect.DeepEqual(got, want) {
				t.Errorf("RecordSelected recorded %d entries, a build given the Data records %d", len(got), len(want))
			}
		})
	}
	// Several instances of one pool, each registered once into one host Data.
	host := embeddedData(t)
	for range 3 {
		e, err := New(Input{Spec: engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: roots(t, "d058-032.yaml")}})
		if err != nil {
			t.Fatal(err)
		}
		e.(recorder).RecordSelected(host)
	}
	if got, want := host.Report().Selected, openedWith(t, "d058-032.yaml", 0); !reflect.DeepEqual(got, want) {
		t.Errorf("three instances recorded %d entries, one instance records %d", len(got), len(want))
	}
}

// Layer 10 of blis-substrate makes Kernel.RecordSelected register the kernel as an observer of
// the run's Data, so the report says at read time which entries a priced step read outside
// the fit range, and records accumulate across kernels. This test needs that.
func TestRecordSelectedFlagsExtrapolationFromOneOfTwoKernels(t *testing.T) {
	t.Skip("needs blis-substrate layer 10 (observer registration: the Result's extrapolated flag is read " +
		"from the registered kernels at report time). To enable after the re-pin: add a registry overlay whose " +
		"coefficient entry for the d058 pool has a narrow fit_range, build two engines for it, price a step " +
		"outside that range on only one, register both with one host Data, and assert the Result marks the " +
		"entry extrapolated although the other kernel never read it outside its range")
}
