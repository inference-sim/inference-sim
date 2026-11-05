package evaluation

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// Validate performs field-level validation of a measured run.
func (r *Run) Validate() *validate.Problems {
	p := &validate.Problems{}
	if r.Kind != "EvaluationRun" {
		p.Field("kind", "must be %q, got %q", "EvaluationRun", r.Kind)
	}
	for field, v := range map[string]string{
		"name": r.Name, "scenario": r.Scenario, "harness": r.Harness,
		"engine_version": r.EngineVersion,
	} {
		if v == "" {
			p.Field(field, "required")
		}
	}
	if len(r.Points) == 0 {
		p.Field("points", "a run with no points measured nothing")
	}
	seen := map[int]bool{}
	for i, pt := range r.Points {
		at := fmt.Sprintf("points[%d]", i)
		if pt.Concurrency < 1 {
			p.Field(at+".concurrency", "must be at least 1")
		} else if seen[pt.Concurrency] {
			p.Field(at+".concurrency", "duplicate concurrency level %d", pt.Concurrency)
		}
		seen[pt.Concurrency] = true

		// Each measured float is checked finite before its range check: a NaN or Inf
		// would pass every ordered comparison and reach a consumer as a garbage metric.
		if p.FiniteField(at+".output_tokens_per_sec", pt.OutputTokensPerSec) &&
			pt.OutputTokensPerSec < 0 {
			p.Field(at+".output_tokens_per_sec", "must not be negative")
		}
		if p.FiniteField(at+".ttft_ms", pt.TTFTms) && pt.TTFTms < 0 {
			p.Field(at+".ttft_ms", "must not be negative")
		}
		if p.FiniteField(at+".itl_ms", pt.ITLms) && pt.ITLms < 0 {
			p.Field(at+".itl_ms", "must not be negative")
		}
		// The optional throughput/latency metrics carry no range check of their own, but
		// a non-finite one is still garbage that must not reach a prediction.
		p.FiniteField(at+".input_tokens_per_sec", pt.InputTokensPerSec)
		p.FiniteField(at+".e2e_ms", pt.EndToEndms)
		for field, v := range map[string]float64{
			"kv_utilization": pt.KVUtilization, "prefix_cache_hit_rate": pt.PrefixCacheHitRate,
		} {
			if p.FiniteField(at+"."+field, v) && (v < 0 || v > 1) {
				p.Field(at+"."+field, "must lie in [0, 1], got %v", v)
			}
		}
		if pt.Preemptions < 0 {
			p.Field(at+".preemptions", "must not be negative")
		}
		// A point with throughput but no latency cannot be compared against a
		// prediction, which is the record's purpose.
		if pt.OutputTokensPerSec > 0 && pt.TTFTms == 0 && pt.ITLms == 0 {
			p.Warnf("%s: throughput is recorded with neither latency metric, so no prediction can be scored against this point", at)
		}
	}
	return p
}
