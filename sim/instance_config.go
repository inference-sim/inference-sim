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
	GPU        string
	TP         int
	MaxNumSeqs int64
	Model      string
	// BlockSizeTokens is the KV block in tokens a P/D handoff moves and is priced in.
	BlockSizeTokens int64
	// Frontend is the API-server render stage (INV-R1).
	Frontend FrontendConfig
	// Engine is the engine's own configuration, opaque to the host.
	Engine any
}
