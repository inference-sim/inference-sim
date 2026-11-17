package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"
	"github.com/inference-sim/blis-substrate/load"
)

// An engine that states a per-rank GPU buffer in its plan has it come out of the KV budget: the
// budget is what the part allows minus FixedBytes().Total(). The case is vLLM's draft-logits
// buffer under probabilistic draft sampling, max_num_seqs x K x vocab x the head's 2 bytes
// (inference-sim/blis-substrate#164).
func TestAStatedGPUBufferShrinksTheKVBudgetByItsBytes(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	const maxNumSeqs, k, vocab, headBytes = 1024, 5, 129280, 2
	draftLogits := int64(maxNumSeqs) * k * vocab * headBytes
	build := func(bufs ...plan.Buffer) int64 {
		in := planCase(t, d, "kimi-k2.5-nvfp4", "b300", 1, "tp: 4, pp: 1, dp: 1", func(l *plan.Lowered) {
			l.Plan.Buffers = bufs
		})
		return mustKernel(t, in).FixedBytes().Total()
	}
	none := build()
	got := build(plan.Buffer{Name: "draft_logits", Bytes: units.Bytes(draftLogits), Basis: plan.BasisRank})
	if got-none != draftLogits {
		t.Errorf("the fixed memory grew by %d B, want the stated buffer's %d B", got-none, draftLogits)
	}
	// A second buffer adds; the KV budget is non-increasing in what the engine states.
	two := build(plan.Buffer{Name: "draft_logits", Bytes: units.Bytes(draftLogits), Basis: plan.BasisRank},
		plan.Buffer{Name: "other", Bytes: units.Bytes(1 << 20), Basis: plan.BasisRank})
	if two-got != 1<<20 {
		t.Errorf("a second stated buffer added %d B, want %d", two-got, 1<<20)
	}
	// A larger max_num_seqs states a larger buffer and leaves less.
	if half := build(plan.Buffer{Name: "draft_logits", Bytes: units.Bytes(draftLogits / 4), Basis: plan.BasisRank}); half >= got {
		t.Errorf("a quarter of the buffer left fixed memory %d B, not below %d B", half, got)
	}
}

// Host buffers and buffers not stated per rank are not a rank's GPU memory and are not charged;
// a plan that states no buffers is priced exactly as before.
func TestBuffersThatAreNotARanksGPUMemoryAreNotCharged(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	fixed := func(bufs ...plan.Buffer) int64 {
		in := planCase(t, d, "kimi-k2.5-nvfp4", "b300", 1, "tp: 4, pp: 1, dp: 1", func(l *plan.Lowered) {
			l.Plan.Buffers = bufs
		})
		return mustKernel(t, in).FixedBytes().Total()
	}
	base := fixed()
	if got := fixed(plan.Buffer{Name: "pinned", Bytes: units.Bytes(1 << 30), Basis: plan.BasisRank, Host: true},
		plan.Buffer{Name: "shared", Bytes: units.Bytes(1 << 30), Basis: plan.BasisNode, Host: true}); got != base {
		t.Errorf("host buffers changed the GPU fixed memory by %d B", got-base)
	}
	if f := mustKernel(t, planCase(t, d, "kimi-k2.5-nvfp4", "b300", 1, "tp: 4, pp: 1, dp: 1", nil)).FixedBytes(); f.EngineBuffers != 0 {
		t.Errorf("a plan with no buffers was charged %d B", f.EngineBuffers)
	}
}
