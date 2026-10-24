package sim

import "fmt"

// NewEngine builds one instance's engine from its resolved configuration: its KV store
// (single-tier, tiered or the offload chain, by cfg.KVCacheConfig), its latency model wrapped
// with the LoRA per-step overhead, and the Simulator that steps them. It is the one place an
// engine is assembled, so the cluster holds the engine without naming its parts (#2021 E2.2).
//
// id names the instance; offload tiers shared across one node's engines are keyed by it
// (OffloadTierSharing). The KV store comes from sim/kv's registered constructor
// (NewKVStoreFromConfig), so a binary that builds engines links sim/kv.
//
// Errors: a nil cfg.LatencyModel (the caller builds it -- blis-latency-kernel's adapter in
// production -- and there is none to fall back to, R1), a malformed LoRA cost
// configuration, an unregistered KV constructor, or any NewSimulator refusal.
func NewEngine(id string, cfg SimConfig) (*Simulator, error) {
	// The KV slot reservation is the speculative config's (#1627), never set apart from it.
	kvCfg := cfg.KVCacheConfig
	kvCfg.Speculative = cfg.KV()
	kvCfg.Speculative.MaxModelLen = cfg.MaxModelLen
	// The engines of one node share each secondary offload tier (OffloadTierSharing).
	kvCfg.Offload = kvCfg.Offload.ForInstance(id)
	if NewKVStoreFromConfig == nil {
		return nil, fmt.Errorf("NewEngine(%s): NewKVStoreFromConfig not registered: import sim/kv", id)
	}
	kvStore := NewKVStoreFromConfig(kvCfg, cfg.Seed)
	// Build the LoRA adapter-cost accessor (nil when the subsystem is inert) and
	// wrap the latency model with it so the per-step compute overhead applies
	// (#1467, R23). BuildAdapterCost is pure and
	// stateless; NewSimulator (called below) builds its own instance for the
	// cold-load gate from the same config — two behaviorally identical accessors,
	// no shared state. A nil accessor leaves StepTime byte-identical to pre-feature (INV-6).
	adapterCost, err := BuildAdapterCost(cfg)
	if err != nil {
		return nil, fmt.Errorf("NewEngine(%s): adapter cost model: %w", id, err)
	}
	// The latency model is built by the caller (blis-latency-kernel's adapter in production);
	// the simulator only times steps with it. There is no model to fall back to, so an absent
	// one is a construction error rather than a silent default (R1).
	if cfg.LatencyModel == nil {
		return nil, fmt.Errorf("NewEngine(%s): SimConfig.LatencyModel is nil; "+
			"the caller must supply the latency model", id)
	}
	latencyModel := WithAdapterOverhead(cfg.LatencyModel, adapterCost)
	s, err := NewSimulator(cfg, kvStore, latencyModel)
	if err != nil {
		return nil, fmt.Errorf("NewEngine(%s): %w", id, err)
	}
	return s, nil
}
