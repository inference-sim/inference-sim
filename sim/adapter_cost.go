package sim

// AdapterCost is the read-only query bridge over the LoRA adapter cost model. It
// is owned by sim/ so the cold-load pre-admission gate can charge load latency
// without importing sim/lora (Principle I: no reverse import); the concrete
// model (rank-derived cost terms) lives in sim/lora and is wired in via
// NewAdapterCostFunc, mirroring NewAdapterRegistryFunc / NewResidentAdapterSetFunc.
//
// The interface is intentionally scoped to what the gate consumes today (R13):
// only LoadLatency. The concrete cost model additionally derives the per-step
// compute-overhead factor and the HBM footprint; those enter this interface when
// the latency and memory PRs consume them.
type AdapterCost interface {
	// LoadLatency returns the one-time cold-load latency of an adapter id in µs
	// (>= 0). An empty (base-model) or unregistered id returns 0 — it never gates.
	LoadLatency(id string) float64
}

// NewAdapterCostFunc builds an AdapterCost from a LoRAConfig (rank registry +
// cost coefficients). Registered by sim/lora's init() (import sim/lora to wire
// it); nil until then, in which case the Simulator leaves cold-load gating inert
// (INV-6). Returns an error for a malformed config (missing/non-positive divisor
// coefficients), which the caller maps to fatality.
var NewAdapterCostFunc func(cfg LoRAConfig) (AdapterCost, error)
