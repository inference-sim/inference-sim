// Package evaluation describes a measured serving run: the record a prediction is
// compared against.
//
// It exists because a schema repository that validates only inputs can check that a
// scenario is well formed and never check whether the model built on it is right. A
// published serving report carries exactly the fields needed for that comparison —
// concurrency, the two latency metrics, throughput, KV utilization, cache hit rate —
// and without a schema for them, a comparison is a one-off script each time.
//
// The record is deliberately close to what benchmark harnesses already emit. It is
// not an attempt to design a better measurement format; it is a place to put the one
// that exists so a cost model can be scored against it.
package evaluation

// Run is one measured deployment sweep.
type Run struct {
	Kind string `yaml:"kind"` // "EvaluationRun"
	Name string `yaml:"name"`

	// Scenario names the scenario this run measured, which is what makes the
	// comparison meaningful: a prediction and a measurement of different
	// deployments are two facts, not an error bar.
	Scenario string `yaml:"scenario"`

	// Harness and HarnessVersion identify what produced the numbers. Two harnesses
	// disagree on how a latency percentile is computed, so a comparison across them
	// needs the reader to know.
	Harness        string `yaml:"harness"`
	HarnessVersion string `yaml:"harness_version,omitempty"`

	// EngineVersion is the engine the run exercised. It may differ from the
	// scenario's, and when it does the comparison is across versions; recording both
	// makes that visible rather than hidden.
	EngineVersion string `yaml:"engine_version"`

	// Date is the measurement date, ISO 8601. A figure from a year ago and one from
	// last week are not interchangeable when kernels move weekly.
	Date string `yaml:"date,omitempty"`

	Points []Point `yaml:"points"`
}

// Point is one concurrency level. Every field is as measured; nothing here is
// derived, so a reader can recompute a derived quantity and check it.
type Point struct {
	// Concurrency is the offered load: the harness's simultaneous request count.
	Concurrency int `yaml:"concurrency"`

	// CompletedRequests is how many finished at this level. It matters because a
	// level that completes fewer requests than a lower one has collapsed, and
	// throughput alone can hide that.
	CompletedRequests int `yaml:"completed_requests,omitempty"`

	// Throughput, in tokens per second. Output and input are separate because an
	// input-heavy workload judged on output throughput looks idle while doing more
	// total work than a balanced one.
	OutputTokensPerSec float64 `yaml:"output_tokens_per_sec"`
	InputTokensPerSec  float64 `yaml:"input_tokens_per_sec,omitempty"`

	// Latency, in milliseconds. These are the two quantities a cost model predicts,
	// and they are not interchangeable: time-to-first-token includes admission and
	// prefill, inter-token latency is per emitted token during decode.
	TTFTms float64 `yaml:"ttft_ms"`
	ITLms  float64 `yaml:"itl_ms"`
	// EndToEndms includes completion overhead, which lands on it and not on TTFT.
	EndToEndms float64 `yaml:"e2e_ms,omitempty"`

	// MeanISL and MeanOSL are the realized sequence lengths, which can differ from
	// the workload's nominal shape once a run truncates or an early stop fires.
	MeanISL int `yaml:"mean_isl,omitempty"`
	MeanOSL int `yaml:"mean_osl,omitempty"`

	// KVUtilization is the fraction of the KV pool in use, in [0, 1]. It is the
	// measurement that checks a memory model rather than a timing one, which is why
	// a schema covering only latency would leave the occupancy half unvalidated.
	KVUtilization float64 `yaml:"kv_utilization,omitempty"`
	// PrefixCacheHitRate is the fraction of prompt tokens served from cache, in
	// [0, 1]. It is an input to a prediction as much as an output of a run.
	PrefixCacheHitRate float64 `yaml:"prefix_cache_hit_rate,omitempty"`
	// Preemptions counts requests evicted and recomputed. A non-zero value means the
	// deployment was over capacity, and comparing a prediction against such a point
	// tests the scheduler model rather than the cost model.
	Preemptions int `yaml:"preemptions,omitempty"`
}
