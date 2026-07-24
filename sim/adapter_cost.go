package sim

// AdapterCost is the read-only query bridge over the LoRA adapter cost model. It
// is owned by sim/ so the cold-load pre-admission gate can charge load latency
// without importing sim/lora (Principle I: no reverse import); the concrete
// model (rank-derived cost terms) lives in sim/lora and is wired in via
// NewAdapterCostFunc, mirroring NewAdapterRegistryFunc / NewResidentAdapterSetFunc.
//
// The interface is scoped to what its consumers read today (R13): LoadLatency for
// the cold-load gate (#1466), StepOverheadFactor for the latency backends
// (#1467), and AdapterReservedBytes for the KV-capacity module (#1468).
type AdapterCost interface {
	// LoadLatency returns the one-time cold-load latency of an adapter id in µs
	// (>= 0). An empty (base-model) or unregistered id returns 0 — it never gates.
	LoadLatency(id string) float64

	// StepOverheadFactor returns the multiplicative per-step compute-overhead
	// factor for a batch (>= 1.0), applied identically by both latency backends
	// (R23). It is exactly 1.0 when the batch carries no adapter ids, so a
	// no-adapter step is byte-identical to a pre-feature build (INV-6).
	StepOverheadFactor(batch []*Request) float64

	// AdapterReservedBytes returns the fixed, capacity-based HBM reservation in
	// bytes (>= 0): adapter_capacity × per-slot footprint (sized from the max
	// declared rank). Constant for the model's lifetime (static reservation,
	// D2/INV-L4) — the KV-capacity module subtracts it once at startup. Returns 0
	// when no adapters or no capacity are configured (INV-6 no-op).
	AdapterReservedBytes() float64
}

// NewAdapterCostFunc builds an AdapterCost from a LoRAConfig (rank registry +
// cost coefficients). Registered by sim/lora's init() (import sim/lora to wire
// it); nil until then, in which case the Simulator leaves cold-load gating inert
// (INV-6). Returns an error for a malformed config (missing/non-positive divisor
// coefficients), which the caller maps to fatality.
var NewAdapterCostFunc func(cfg LoRAConfig) (AdapterCost, error)

// BuildAdapterCost constructs the adapter-cost accessor for a config, or returns
// (nil, nil) when the LoRA subsystem is inert: no adapters declared, no capacity
// set, or sim/lora not linked (registration funcs nil). It centralizes the
// activation condition in one place (R4) so every consumer agrees on exactly when
// adapter costs apply — the cold-load gate + resident set (NewSimulator) and the
// per-step overhead factor threaded into the latency backends (sim/cluster).
//
// The returned accessor is a pure, stateless query object derived entirely from
// the config; constructing two from the same config yields behaviorally identical
// instances (which is why the gate and the latency model may each build their own
// without shared state). The gating condition matches NewSimulator's resident-set
// wiring exactly, so residentAdapters and adapterCost stay non-nil together.
func BuildAdapterCost(cfg SimConfig) (AdapterCost, error) {
	if !cfg.HasAdapters() || cfg.AdapterCapacity == nil ||
		NewResidentAdapterSetFunc == nil || NewAdapterCostFunc == nil {
		return nil, nil
	}
	return NewAdapterCostFunc(cfg.LoRAConfig)
}
