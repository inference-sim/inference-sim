package sim

// RoutingSnapshot is a lightweight view of one instance's live state, read by the model
// autoscaler. Routing itself is the llm-d router's and reads the model servers' published
// metrics, not these snapshots.
type RoutingSnapshot struct {
	ID                    string
	QueueDepth            int
	BatchSize             int
	KVUtilization         float64
	FreeKVBlocks          int64
	CacheHitRate          float64
	InFlightRequests      int     // Requests dispatched to this instance but not yet completed
	PreemptionCount       int64   // Cumulative preemption events since instance start (monotonically increasing); read live at each autoscaler tick
	Model                 string  // Model served by this instance; DefaultCollector groups snapshots by it
	GPUType               string  // GPU hardware type (e.g. "A100-80GB"); set by buildRouterState from the instance; DefaultCollector's variant key
	TPDegree              int     // Tensor-parallel degree; set by buildRouterState from the instance; DefaultCollector's variant key
	CostPerHour           float64 // Node pool cost in $/hr; set by buildRouterState from NodePool.CostPerHour; read by DefaultCollector
	TotalKvCapacityTokens int64   // Total KV cache capacity in tokens (TotalBlocks × BlockSizeTokens); used by V2SaturationAnalyzer
	KvTokensInUse         int64   // Current KV cache occupancy in tokens (UsedBlocks × BlockSizeTokens); used by V2SaturationAnalyzer
	TTFT                  float64 // μs; 0 if not yet available
	ITL                   float64 // μs; 0 if not yet available
	DispatchRate          float64 // req/s completed by this instance; 0 if not yet available
	AvgInTokens           float64 // average input tokens per completed request; 0 if not yet available
	AvgOutTokens          float64 // average output tokens per completed request; 0 if not yet available
	MaxBatchSize          float64 // server-configured max batch size; 0 if not yet available
	// ResidentAdapters is never populated: buildRouterState leaves it nil and nothing reads
	// it. LoRA-aware routing is the llm-d router's lora-affinity-scorer, which reads the
	// adapter sets each engine publishes in vllm:lora_requests_info (sim/engine_metrics.go).
	ResidentAdapters map[string]bool
}

// EffectiveLoad returns the total effective load on this instance (no production caller
// since BLIS's own routing and admission were removed in #2000):
// QueueDepth + BatchSize + InFlightRequests.
func (s RoutingSnapshot) EffectiveLoad() int {
	return s.QueueDepth + s.BatchSize + s.InFlightRequests
}

// NewRoutingSnapshot creates a RoutingSnapshot with the given instance ID.
// All numeric fields are zero-valued.
func NewRoutingSnapshot(id string) RoutingSnapshot {
	if id == "" {
		panic("NewRoutingSnapshot: id must not be empty")
	}
	return RoutingSnapshot{ID: id}
}
