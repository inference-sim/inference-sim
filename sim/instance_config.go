package sim

// InstanceConfig is the configuration of one engine instance as the host holds it: the settings
// the host itself reads (the run's horizon and seed, the instance's GPU, tensor parallelism,
// running-request bound and model, the KV block size its handoffs move in, its API-server front
// end and its host overheads), beside the engine's own configuration, which the host holds
// unopened. Only the build site that registers the engine (inference-sim cmd/engine_build.go)
// builds or reads Engine; the host passes it, resolved for the instance's pool role, back to the
// engine factory (cluster.DeploymentConfig.NewEngine).
type InstanceConfig struct {
	// Horizon ends the run: no event after it runs.
	Horizon int64
	// Seed is the run's seed (INV-6).
	Seed int64
	// GPU, TP, MaxNumSeqs and Model are the instance's hardware, tensor-parallel degree,
	// running-request bound and model name, which placement, the autoscaler and the router read.
	GPU string
	TP  int
	// GPUs is the GPUs one instance occupies: tp x pp x pcp x dp (GPUsPerInstance), because an
	// instance is the whole data-parallel group its engine runs. 0 means unstated, and the
	// instance then occupies TP of them (GPUFootprint).
	GPUs       int
	MaxNumSeqs int64
	Model      string
	// BlockSizeTokens is the KV block in tokens a P/D handoff moves and is priced in.
	BlockSizeTokens int64
	// Engine is the engine's own configuration, opaque to the host.
	Engine any
}

// GPUsPerInstance is the GPUs one engine instance occupies for a parallel layout: the product of
// its tensor, pipeline, prefill-context and data-parallel widths, each floored at 1 (an unset
// width is one). It is the one count placement, node span, cost and the Result's GPU totals read.
func GPUsPerInstance(tp, pp, pcp, dp int) int {
	return max(tp, 1) * max(pp, 1) * max(pcp, 1) * max(dp, 1)
}

// GPUFootprint is the GPUs an instance built from c reserves in a node pool: GPUs when stated,
// else TP, and at least one.
func (c InstanceConfig) GPUFootprint() int {
	if c.GPUs > 0 {
		return c.GPUs
	}
	return max(c.TP, 1)
}
