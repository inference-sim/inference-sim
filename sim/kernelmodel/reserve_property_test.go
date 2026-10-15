package kernelmodel

import (
	"testing"

	"pgregory.net/rapid"
)

// Setting HBM aside for LoRA adapters shrinks the KV pool and never grows it: zero reserved is
// the plain budget, more reserved never yields more blocks, and the blocks always fit in what
// is left of the allocatable bytes once each GPU's share (reserved / tp) is taken.
func TestKVBudgetReserving_ShrinksThePoolMonotonically(t *testing.T) {
	m := open(t, "llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml")
	plain, err := m.KVBudget()
	if err != nil {
		t.Fatal(err)
	}
	if zero, _ := m.KVBudgetReserving(0); zero != plain {
		t.Fatalf("reserving nothing changed the budget: %+v vs %+v", zero, plain)
	}
	tp := int64(m.Kernel().Resolved().TensorParallel())
	rapid.Check(t, func(rt *rapid.T) {
		a := rapid.Int64Range(0, 64<<30).Draw(rt, "a")
		b := rapid.Int64Range(a, 64<<30).Draw(rt, "b")
		ba, errA := m.KVBudgetReserving(a)
		bb, errB := m.KVBudgetReserving(b)
		if errA != nil {
			if errB == nil {
				rt.Fatalf("reserving %d failed but the larger %d succeeded", a, b)
			}
			return
		}
		if errB == nil && bb.TotalBlocks > ba.TotalBlocks {
			rt.Fatalf("reserving %d gave %d blocks, more than %d gave at %d", b, bb.TotalBlocks, ba.TotalBlocks, a)
		}
		if used := ba.PerRankBlocks * ba.PerBlockBytes; used > plain.AllocatableBytes-a/tp {
			rt.Fatalf("reserving %d: %d blocks use %d bytes, beyond the %d left", a, ba.PerRankBlocks, used,
				plain.AllocatableBytes-a/tp)
		}
	})
}
