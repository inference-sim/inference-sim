package cluster

// ClusterMetrics contains aggregate metrics for the entire cluster simulation
type ClusterMetrics struct {
	CompletedRequests int
	TotalRequests     int
	TotalInputTokens  int64
	TotalOutputTokens int64
	SimDuration       int64
	PerInstance       map[InstanceID]*InstanceMetrics
}

// InstanceMetrics contains metrics for a single instance
type InstanceMetrics struct {
	CompletedRequests  int
	TotalInputTokens   int64
	TotalOutputTokens  int64
	PeakWaitQueueDepth int
	PeakBatchSize      int
}
