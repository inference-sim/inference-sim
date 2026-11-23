// Moved from inference-sim/blis-schemas/spec/deployment/validate_test.go at 70246460a379f8b1898fbf6c36a3091364118950 (TestAllReduceRequestConflict).

package engineconfig

import (
	"math"
	"testing"
)

// TestAllReduceRequestConflict covers the two ways a deployment can state the
// all-reduce choice inconsistently. The engine has a boolean; a resolved name is an
// override, so the two must agree.
func TestAllReduceRequestConflict(t *testing.T) {
	yes, no := true, false
	if p := engineOnly(Engine{DisableCustomAllReduce: &yes, AllReduceBackend: "custom"}).Validate(); p.OK() {
		t.Error("requesting the custom kernel while disabling it should fail")
	}
	// The reverse is a warning: the engine would use the kernel where reachable, so
	// naming nccl is a statement the layout may override rather than a contradiction.
	p2 := engineOnly(Engine{DisableCustomAllReduce: &no, AllReduceBackend: "nccl"}).Validate()
	if !p2.OK() {
		t.Errorf("naming nccl with the kernel enabled should warn, not fail:\n%s", p2.Error())
	}
	if len(p2.All()) == 0 {
		t.Error("expected a warning about the overridden request")
	}
	// Stating only the boolean is the ordinary case.
	if p := engineOnly(Engine{DisableCustomAllReduce: &yes}).Validate(); !p.OK() {
		t.Errorf("the boolean alone should be accepted:\n%s", p.Error())
	}
}

// TestRejects is the engine half of blis-schemas' deployment TestRejects: each stated value is
// one the engine block's own types refuse.
func TestRejects(t *testing.T) {
	neg := -1
	for _, c := range []struct {
		name string
		e    Engine
	}{
		{"utilization above one", Engine{GPUMemoryUtilization: 1.5}},
		{"non-finite utilization", Engine{GPUMemoryUtilization: math.NaN()}},
		{"negative block size", Engine{BlockSize: -1}},
		// The engine types both DBO thresholds ge=0 whether or not overlap is on.
		{"negative DBO threshold with DBO off", Engine{DBO: &DBO{Enabled: false, DecodeTokenThreshold: &neg}}},
		{"negative DBO prefill threshold with DBO on", Engine{DBO: &DBO{Enabled: true, PrefillTokenThreshold: &neg}}},
		{"max_model_len below auto", Engine{MaxModelLen: -2}},
		{"negative interleave size", Engine{CPKVCacheInterleaveSize: -1}},
		{"speculative with zero drafts", Engine{Speculative: &Speculative{Method: "mtp", NumSpecTokens: 0}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if p := engineOnly(c.e).Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
	if p := engineOnly(Engine{GPUMemoryUtilization: 0.9, BlockSize: 16}).Validate(); !p.OK() {
		t.Errorf("a valid block was refused:\n%s", p.Error())
	}
}
