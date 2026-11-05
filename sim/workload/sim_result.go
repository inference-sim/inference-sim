package workload

// SimResult is one row of the file `blis replay --results-path` writes: a completed request's
// server-side latencies in microseconds (simulation ticks) and its token counts. SLOClass,
// Model and ITLMeanUs are omitted from the JSON when zero or empty.
type SimResult struct {
	RequestID    int     `json:"request_id"`
	TTFT         float64 `json:"ttft_us"` // server-side TTFT in microseconds
	E2E          float64 `json:"e2e_us"`  // server-side E2E in microseconds
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	SLOClass     string  `json:"slo_class,omitempty"`   // SLO tier (e.g., "standard", "batch"); empty if not set
	Model        string  `json:"model,omitempty"`       // model tag; empty if not set
	ITLMeanUs    float64 `json:"itl_mean_us,omitempty"` // mean ITL in microseconds; 0 if not recorded
}
