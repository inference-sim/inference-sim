package latencykernel

import (
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/load"
)

// inference-sim/blis-substrate#116: the tensor-parallel all-reduce is priced from the lane of the
// kernel the engine resolved. The cases below price a decode step of a TP4 dense model on an H100
// (a part with a custom-kernel lane and an NCCL lane) under each statement the engine can make, and
// compare what a step costs.

// laneDecodeStep is the time of one batch of 32 decoding requests with no overlap between resources
// (the all-reduce hides behind HBM time in the overlapped estimate), in nanoseconds.
func laneDecodeStep(t *testing.T, in Inputs) (time.Duration, *Kernel) {
	t.Helper()
	k, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	reqs := make([]kernel.ReqShape, 32)
	for i := range reqs {
		reqs[i] = kernel.ReqShape{Scheduled: 1, Computed: 2048, PromptLen: 1024}
	}
	return k.StepTime(kernel.Batch{Reqs: reqs, DecodeThreshold: 1}).NoOverlap, k
}

func stateAllReduce(w plan.AllReduceBackend, sm bool) func(*plan.Lowered) {
	return func(l *plan.Lowered) {
		l.Plan.Collectives.AllReduce = w
		l.Reported.SMAllReduce = sm
	}
}

func lanePrice(t *testing.T, d *load.Data, f func(*plan.Lowered)) (time.Duration, *Kernel) {
	t.Helper()
	return laneDecodeStep(t, planCase(t, d, "llama-3.1-70b-instruct", "h100", 1, "tp: 4, pp: 1, dp: 1", f))
}

func assumed(k *Kernel, name string) bool {
	for _, o := range k.Provenance() {
		if o.Name == name && o.Set == KernelAssumptionSet {
			return true
		}
	}
	return false
}

func TestAnNCCLAllReducePricesTheNCCLLaneAndTheCustomKernelThePlainOne(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	custom, kc := lanePrice(t, d, stateAllReduce(plan.AllReduceCustom, true))
	nccl, kn := lanePrice(t, d, stateAllReduce(plan.AllReduceNCCL, false))
	if nccl <= custom {
		t.Errorf("a step under NCCL all-reduce costs %v and under the custom kernel %v: NCCL's "+
			"small-message floor is the larger (H100, 4 ranks: 9.3 us against 3.1 us), so the step must be slower", nccl, custom)
	}
	if assumed(kc, "all_reduce_lane") || assumed(kn, "all_reduce_lane") {
		t.Error("a lane the registry carries is read as measured; no all_reduce_lane assumption is due")
	}
	if got := kn.Resolved().AllReduceBackend; got != "nccl" {
		t.Errorf("Resolved().AllReduceBackend = %q, want nccl", got)
	}
}

// An engine that predates Collectives.AllReduce states only whether the reduction runs on SMs.
func TestAnUnstatedAllReduceFollowsSMAllReduce(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, sm := range []bool{true, false} {
		legacy, _ := lanePrice(t, d, stateAllReduce(plan.AllReduceUnstated, sm))
		word := plan.AllReduceNCCL
		if sm {
			word = plan.AllReduceCustom
		}
		stated, _ := lanePrice(t, d, stateAllReduce(word, sm))
		if legacy != stated {
			t.Errorf("SMAllReduce %t: an unstated kernel prices %v, the stated %q %v", sm, legacy, word, stated)
		}
	}
}

// A kernel no sweep measures (FlashInfer's, until it is timed) is priced from the custom lane and
// the substitution is recorded, not silent.
func TestAnAllReduceKernelWithNoLaneIsPricedFromTheCustomLaneAndRecorded(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	custom, _ := lanePrice(t, d, stateAllReduce(plan.AllReduceCustom, true))
	fi, k := lanePrice(t, d, stateAllReduce(plan.AllReduceFlashInferTRTLLM, true))
	if fi != custom {
		t.Errorf("FlashInfer all-reduce has no lane, so it prices as the custom lane: %v against %v", fi, custom)
	}
	if !assumed(k, "all_reduce_lane") {
		t.Error("the substitution of the custom lane for a FlashInfer all-reduce must be an assumption in Provenance")
	}
	if got := k.Resolved().AllReduceBackend; got != "flashinfer_trtllm" {
		t.Errorf("Resolved().AllReduceBackend = %q, want the engine's word", got)
	}
}

// A width the NCCL lane lacks falls back to the custom lane with an assumption, rather than
// refusing a deployment the default lane prices. Synthetic lanes: NCCL at 2 ranks only.
func TestAMissingNCCLWidthFallsBackToTheCustomLaneWithAnAssumption(t *testing.T) {
	var entries []coefficient.Entry
	for _, w := range []int{2, 4, 8} {
		entries = append(entries, tripleFor("all_reduce", "fp16", "h200", w, 3)...)
		entries = append(entries, tripleFor("all_gather", "fp16", "h200", w, 3)...)
		entries = append(entries, tripleFor("reduce_scatter", "fp16", "h200", w, 3)...)
		entries = append(entries, tripleFor("alltoall", "fp16", "h200", w, 3)...)
	}
	entries = append(entries, tripleFor("all_reduce_nccl", "fp16", "h200", 2, 12)...)
	c := coeffs(t, entries)
	k := &Kernel{chip: hardware.Chip{Name: "h200"}, allReduce: plan.AllReduceNCCL}
	k.layout.TP = 8
	if err := k.liftCollectiveFloors(c); err != nil {
		t.Fatalf("an NCCL lane without the width must fall back, not refuse: %v", err)
	}
	key := collKey{Op: model.OpAllReduce, Group: price.GroupTP}
	if got := k.collectiveFloors[key]; got != 3*time.Microsecond {
		t.Errorf("the 8-rank all-reduce floor is %v, want the custom lane's 3us", got)
	}
	if !assumed(&Kernel{assumptions: k.assumptions, origins: k.assumptions}, "all_reduce_lane") {
		t.Error("the fallback must be recorded as an all_reduce_lane assumption")
	}

	// With the width present, the NCCL lane is read.
	entries = append(entries, tripleFor("all_reduce_nccl", "fp16", "h200", 8, 12)...)
	k = &Kernel{chip: hardware.Chip{Name: "h200"}, allReduce: plan.AllReduceNCCL}
	k.layout.TP = 8
	if err := k.liftCollectiveFloors(coeffs(t, entries)); err != nil {
		t.Fatal(err)
	}
	if got := k.collectiveFloors[key]; got != 12*time.Microsecond {
		t.Errorf("the 8-rank NCCL all-reduce floor is %v, want the NCCL lane's 12us", got)
	}
	// Other collectives are not the all-reduce: the all-gather keeps its own lane.
	if got := k.collectiveFloors[collKey{Op: model.OpAllGather, Group: price.GroupTP}]; got != 3*time.Microsecond {
		t.Errorf("an all-gather under NCCL all-reduce reads %v, want its own 3us lane", got)
	}
}

// The engine states one resolution: a kernel on SMs cannot be reported as not on SMs.
func TestAnAllReduceWordThatContradictsSMAllReduceIsRefused(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	in := planCase(t, d, "llama-3.1-70b-instruct", "h100", 1, "tp: 4, pp: 1, dp: 1", stateAllReduce(plan.AllReduceNCCL, true))
	if _, err := New(in); err == nil || !strings.Contains(err.Error(), "Collectives.AllReduce") {
		t.Fatalf("NCCL stated with SMAllReduce true must be refused naming the field; got %v", err)
	}
	in = planCase(t, d, "llama-3.1-70b-instruct", "h100", 1, "tp: 4, pp: 1, dp: 1", stateAllReduce("ring", false))
	if _, err := New(in); err == nil {
		t.Fatal("an unknown all-reduce word must be refused")
	}
}

// A custom request the engine declined (a width its kernel refuses, a group that leaves the node)
// is priced on the lane that runs, NCCL's, and the override is kept for the Result.
func TestADeclinedCustomRequestPricesNCCLAndKeepsItsOverride(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	declined, k := lanePrice(t, d, func(l *plan.Lowered) {
		stateAllReduce(plan.AllReduceNCCL, false)(l)
		l.Overrides = []plan.Override{{Field: "allreduce_backend", Requested: "custom", Resolved: "nccl",
			Reason: "the kernel supports only certain rank counts"}}
	})
	asked, _ := lanePrice(t, d, stateAllReduce(plan.AllReduceNCCL, false))
	custom, _ := lanePrice(t, d, stateAllReduce(plan.AllReduceCustom, true))
	if declined != asked || declined <= custom {
		t.Errorf("a declined custom request costs %v, an NCCL request %v, a custom kernel %v: it must be the NCCL price", declined, asked, custom)
	}
	ov := k.Resolved().Overrides
	if len(ov) != 1 || ov[0].Requested != "custom" || ov[0].Resolved != "nccl" {
		t.Errorf("the declined request must stay in Resolved().Overrides; got %+v", ov)
	}
}
