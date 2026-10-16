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
	PreemptionCount       int64   // Cumulative preemption events since instance start (monotonically increasing; Immediate by default, Periodic when --snapshot-refresh-interval > 0)
	Model                 string  // Model served by this instance; used by buildRouterState() for per-model filtering
	GPUType               string  // GPU hardware type (e.g. "A100-80GB"); populated by buildRouterState() from instance config
	TPDegree              int     // Tensor-parallel degree; populated by buildRouterState() from instance config
	CostPerHour           float64 // Node pool cost in $/hr; populated by buildRouterState() from NodePool.CostPerHour
	TotalKvCapacityTokens int64   // Total KV cache capacity in tokens (TotalBlocks × BlockSizeTokens); used by V2SaturationAnalyzer
	KvTokensInUse         int64   // Current KV cache occupancy in tokens (UsedBlocks × BlockSizeTokens); used by V2SaturationAnalyzer
	TTFT                  float64 // μs; 0 if not yet available
	ITL                   float64 // μs; 0 if not yet available
	DispatchRate          float64 // req/s completed by this instance; 0 if not yet available
	AvgInTokens           float64 // average input tokens per completed request; 0 if not yet available
	AvgOutTokens          float64 // average output tokens per completed request; 0 if not yet available
	MaxBatchSize          float64 // server-configured max batch size; 0 if not yet available
	// ResidentAdapters is the set of LoRA adapter ids currently resident on this
	// instance (membership set; #1469, PR6). Consumed by the lora-affinity scorer
	// for placement affinity. Freshness (R17, INV-7): Periodic — refreshed by
	// CachedSnapshotProvider at --snapshot-refresh-interval (default 50ms), Immediate
	// when the interval is 0. Zero value (nil) ⇒ no adapter resident ⇒ scorer neutral,
	// preserving byte-identical routing when the LoRA subsystem is inert (INV-6).
	ResidentAdapters map[string]bool
}

// EffectiveLoad returns the total effective load on this instance:
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
