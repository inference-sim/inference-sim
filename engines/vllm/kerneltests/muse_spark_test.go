package latencykernel_test

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/kernel/kerneltest"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/vocab"
)

// Muse-Spark-Open-816B-A42B, NVFP4, on the three clusters it is served on: H100 over RoCE at
// PP2 x TP8 (the best layout in internal deployment measurements), H200 over InfiniBand at the
// same layout, and GB200 at TP8 x EP8 over two trays of one NVL72 rack.

func museBatches() []kernel.Batch {
	decode := func(n, ctx int) kernel.Batch {
		b := kernel.Batch{DecodeThreshold: 1}
		for range n {
			b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: ctx - 1, PromptLen: ctx})
		}
		return b
	}
	return []kernel.Batch{
		decode(1, 4096), decode(4, 60000), decode(64, 90000),
		{Reqs: []kernel.ReqShape{{Scheduled: 8192, PromptLen: 60000}}, DecodeThreshold: 1},
	}
}

// A pipeline-parallel Muse pool prices one stage per pipeline rank, by the StepEstimate law:
// the stages sum exactly to the step. Each rank holds half the layers' weights and KV.
func TestMuseSparkPricesTwoPipelineStages(t *testing.T) {
	pp := fixture(t, "muse-h100-pp2tp8.yaml")
	kerneltest.TestStages(t, pp, museBatches())
	for _, b := range museBatches() {
		if got := len(pp.StepTime(b).Stages); got != 2 {
			t.Fatalf("%d stages at pp 2", got)
		}
	}
	in := fixtureInputs(t, "muse-h100-pp2tp8.yaml")
	in.Deployment.Pools[0].Parallel.PP = 1
	in.Deployment.Pools[0].Parallel.TP = 8
	one, err := newKernel(in)
	if err != nil {
		t.Fatal(err)
	}
	if one.StepTime(museBatches()[1]).Stages != nil {
		t.Error("a one-stage pool reported stages")
	}
	if w, w1 := pp.FixedBytes().Weights, one.FixedBytes().Weights; !(w < w1*6/10 && w > w1*4/10) {
		t.Errorf("pp 2 holds %d weight bytes a rank against %d at pp 1: want about half", w, w1)
	}
	if kv, kv1 := pp.SequenceVariableBytes(16), one.SequenceVariableBytes(16); kv*2 != kv1 {
		t.Errorf("pp 2 holds %d KV bytes a block against %d at pp 1: want half", kv, kv1)
	}
}

// The DSpark drafter named by speculative.draft_model is priced: its passes cost time on the
// last stage only, and its weights and KV are resident there.
func TestMuseSparkPricesItsNamedDrafter(t *testing.T) {
	plain := fixture(t, "muse-h100-pp2tp8.yaml")
	spec := fixture(t, "muse-h100-pp2tp8-dspark.yaml")
	b := museBatches()[1]
	for i := range b.Reqs {
		b.Reqs[i].Scheduled = 8
	}
	p, s := plain.StepTime(b), spec.StepTime(b)
	// The first stage's host cost moves too, because under speculation a batch of 1 + k tokens
	// a request is a uniform decode and replays the decode graph; the drafter's passes are the
	// last stage's alone, so that stage gains the most.
	last := s.Stages[1].NoOverlap - p.Stages[1].NoOverlap
	first := s.Stages[0].NoOverlap - p.Stages[0].NoOverlap
	if !(last > 0 && last > first) {
		t.Errorf("the drafter added %v to the last stage and %v to the first", last, first)
	}
	if !(spec.FixedBytes().Weights > plain.FixedBytes().Weights) {
		t.Error("the drafter's weights are not resident")
	}
	if !(spec.SequenceVariableBytes(16) > plain.SequenceVariableBytes(16)) {
		t.Error("the drafter's KV is not resident")
	}
}

// The three clusters price apart, and a fabric's measured efficiency moves only what crosses
// the fabric: an H100 RoCE pipeline's stage-boundary send, never a GB200 pool that stays inside
// its rack's NVLink domain.
func TestMuseSparkPricesItsFabricPerCluster(t *testing.T) {
	b := museBatches()[3]
	roce := fixture(t, "muse-h100-pp2tp8.yaml").StepTime(b)
	ib := fixture(t, "muse-h200-pp2tp8.yaml").StepTime(b)
	nvl := fixture(t, "muse-gb200-tp8ep8.yaml").StepTime(b)
	if roce.NoOverlap == ib.NoOverlap || ib.NoOverlap == nvl.NoOverlap || roce.NoOverlap == nvl.NoOverlap {
		t.Errorf("clusters price alike: roce %v, ib %v, nvlink %v", roce.NoOverlap, ib.NoOverlap, nvl.NoOverlap)
	}
	derate := func(name, fabric string, eff float64) kernel.StepEstimate {
		in := fixtureInputs(t, name)
		in.Coefficients = append(in.Coefficients, &coefficient.Set{Kind: "CoefficientSet", Name: "fabric",
			Coefficients: []coefficient.Entry{{Name: "fabric_bandwidth_efficiency", Value: eff,
				Units: vocab.UnitDimensionless, Method: vocab.MethodMeasured,
				Scope: coefficient.Scope{Fabric: []string{fabric}}}}})
		k, err := newKernel(in)
		if err != nil {
			t.Fatal(err)
		}
		for _, a := range k.Provenance() {
			if a.Name == "fabric_bandwidth_efficiency" && strings.Contains(a.Set, "kernel") {
				t.Errorf("%s: the stated efficiency was recorded as assumed", name)
			}
		}
		return k.StepTime(b)
	}
	if got := derate("muse-h100-pp2tp8.yaml", "roce-200g", 0.5); !(got.Stages[0].NoOverlap > roce.Stages[0].NoOverlap) {
		t.Errorf("a half-efficient RoCE fabric left the boundary send's stage at %v (nominal %v)",
			got.Stages[0].NoOverlap, roce.Stages[0].NoOverlap)
	}
	if got := derate("muse-gb200-tp8ep8.yaml", "ib-400g", 0.5); got.NoOverlap != nvl.NoOverlap {
		t.Errorf("an InfiniBand efficiency moved a pool inside one NVL72 rack: %v against %v",
			got.NoOverlap, nvl.NoOverlap)
	}
}
