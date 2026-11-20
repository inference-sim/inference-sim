// Moved from inference-sim/sim/engine/adapter_cost.go at ca532abca209af41425c007974a5f3ef336f5981.

package runconfig

import (
	"fmt"
	"math"
)

// AdapterCost is the read-only query bridge over the LoRA adapter cost model. It
// is owned by sim/ so the cold-load pre-admission gate can charge load latency
// without importing sim/lora (Principle I: no reverse import); the concrete
// model (rank-derived cost terms) is core/lora's CostModel.
//
// The interface is scoped to what its consumers read today (R13): LoadLatency for
// the cold-load gate (#1466), StepOverheadFactor for the latency-model wrapper
// (WithAdapterOverhead, #1467), and AdapterReservedBytes for the kernel's KV
// sizing (#1468).
type AdapterCost interface {
	// LoadLatency returns the one-time cold-load latency of an adapter id in µs
	// (>= 0). An empty (base-model) or unregistered id returns 0 — it never gates.
	LoadLatency(id string) float64

	// StepOverheadFactor returns the multiplicative per-step compute-overhead
	// factor for a batch (>= 1.0), applied to the latency model's step price by
	// WithAdapterOverhead. It is exactly 1.0 when the batch carries no adapter ids, so a
	// no-adapter step is byte-identical to a pre-feature build (INV-6).
	StepOverheadFactor(batch []*Request) float64

	// AdapterReservedBytes returns the fixed, capacity-based HBM reservation in
	// bytes (>= 0): adapter_capacity × per-slot footprint (sized from the max
	// declared rank). Constant for the model's lifetime (static reservation,
	// D2/INV-L4) — the kernel sets it aside once at startup when sizing the KV
	// pool. Returns 0
	// when no adapters or no capacity are configured (INV-6 no-op).
	AdapterReservedBytes() float64
}

// WithAdapterOverhead wraps a latency model so every step it prices is multiplied by the
// batch's LoRA compute-overhead factor. It is how the per-step adapter cost applies to the
// latency model, which is built outside the simulator -- blis-latency-kernel, whose pricing
// knows no adapters. A nil accessor returns the model unchanged, so a run without adapters
// is byte-identical (INV-6).
func WithAdapterOverhead(m LatencyModel, ac AdapterCost) LatencyModel {
	if ac == nil {
		return m
	}
	return adapterOverheadModel{LatencyModel: m, ac: ac}
}

type adapterOverheadModel struct {
	LatencyModel
	ac AdapterCost
}

// StepTime is the wrapped model's price times the batch's overhead factor. A factor of exactly
// 1 (a batch with no adapters) leaves the price untouched. The AdapterCost contract is a
// finite factor >= 1, and an accessor breaking it is a bug: it panics rather than being read
// as "no overhead", which would hide it (R1), or reaching the clock as a NaN (INV-3).
func (m adapterOverheadModel) StepTime(batch []*Request) int64 {
	base := m.LatencyModel.StepTime(batch)
	factor := m.ac.StepOverheadFactor(batch)
	if math.IsNaN(factor) || math.IsInf(factor, 0) || factor < 1.0 {
		panic(fmt.Sprintf("AdapterCost.StepOverheadFactor = %v; the contract is a finite factor >= 1", factor))
	}
	if factor == 1.0 {
		return base
	}
	scaled := float64(base) * factor
	if scaled >= float64(math.MaxInt64/2) {
		return math.MaxInt64 / 2
	}
	return max(1, int64(scaled))
}
