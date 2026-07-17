package workload

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Adapter-cost fidelity comparison vs the Agullo Digital Twin (DT) reference
// (US5, #1470). The DT's simulate() emits workload-AGGREGATE ttft (mean, ms) and
// throughput (tokens/s) — not per-adapter — so the comparison is per reference
// CONFIG (Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct), matching what the DT
// actually reports.
//
// Fidelity is bounded, not ground truth (design §15): the DT itself is ~17–21%
// SMAPE on TTFT vs a real H100 but only ~5% on throughput, and BLIS ports the DT
// adapter *deltas* onto its own separately-calibrated base — so absolute TTFT is
// expected to diverge (base + queueing-regime mismatch) while the throughput term
// (the compute-overhead physics, D4) is the leg the DT calibrates most tightly.
// The report therefore also carries a delta-normalized (aware ÷ blind) diagnostic
// that cancels the base and isolates the ported adapter physics.

// DTAggregate is one side (adapter-aware or adapter-blind) of a DT simulate()
// result. Field names match the DT driver's output dict.
type DTAggregate struct {
	TTFTMs           float64 `json:"ttft"`              // mean TTFT in milliseconds
	OutputThroughput float64 `json:"output_throughput"` // completed output tokens/s
	TotalThroughput  float64 `json:"total_throughput"`  // (input+output) tokens/s
}

// DTReference is a committed DT reference fixture for one config, holding the
// adapter-aware run and the adapter-blind baseline (both physics models off).
type DTReference struct {
	Model        string      `json:"model"`
	AdapterAware DTAggregate `json:"adapter_aware"`
	AdapterBlind DTAggregate `json:"adapter_blind"`
}

// LoadDTReference reads a DT reference JSON fixture (produced offline by
// lora-control/experiments/export_dt_reference.py).
func LoadDTReference(path string) (*DTReference, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading adapter reference %s: %w", path, err)
	}
	var ref DTReference
	if err := json.Unmarshal(data, &ref); err != nil {
		return nil, fmt.Errorf("parsing adapter reference %s: %w", path, err)
	}
	if ref.AdapterAware.TTFTMs <= 0 || ref.AdapterAware.OutputThroughput <= 0 {
		return nil, fmt.Errorf("adapter reference %s: adapter_aware ttft/output_throughput must be positive (got ttft=%v tput=%v)",
			path, ref.AdapterAware.TTFTMs, ref.AdapterAware.OutputThroughput)
	}
	return &ref, nil
}

// BLISAggregate is the BLIS side of the comparison, distilled from a MetricsOutput.
type BLISAggregate struct {
	TTFTMs           float64 // ttft_mean_ms
	OutputThroughput float64 // total_output_tokens / vllm_estimated_duration_s
}

// AdapterMetricComparison is one metric's BLIS-vs-DT error for a single config.
type AdapterMetricComparison struct {
	Metric    string   `json:"metric"` // "ttft" | "throughput"
	DTValue   float64  `json:"dt_value"`
	BLISValue float64  `json:"blis_value"`
	MAPE      float64  `json:"mape"`             // |blis-dt|/|dt| (fraction; single-config APE)
	Within    bool     `json:"within_threshold"` // MAPE <= threshold
	DeltaMAPE *float64 `json:"delta_mape,omitempty"`
}

// AdapterReferenceReport is the per-config comparison result.
type AdapterReferenceReport struct {
	Model     string                    `json:"model"`
	Threshold float64                   `json:"mape_threshold"`
	Metrics   []AdapterMetricComparison `json:"metrics"`
	AllWithin bool                      `json:"all_within_threshold"`
}

// ape returns the absolute percentage error |a-b|/|b| as a fraction, or +Inf if
// the reference b is zero (undefined — surfaced rather than silently zero).
func ape(blis, dt float64) float64 {
	if dt == 0 {
		return math.Inf(1)
	}
	return math.Abs(blis-dt) / math.Abs(dt)
}

// CompareAdapterReference compares BLIS aggregate metrics against a DT reference
// config on TTFT and (output) throughput. When blind is non-nil it also computes
// the delta-normalized (aware ÷ blind) ratio error, which cancels the base-latency
// mismatch and isolates the ported adapter physics. threshold is the MAPE bound
// (SC-007 target 0.20). A metric with a zero DT denominator yields MAPE=+Inf and
// is reported as out-of-threshold (never a silent pass).
func CompareAdapterReference(ref *DTReference, aware BLISAggregate, blind *BLISAggregate, threshold float64) *AdapterReferenceReport {
	rep := &AdapterReferenceReport{Model: ref.Model, Threshold: threshold, AllWithin: true}

	type spec struct {
		name     string
		dt, blis float64
		dtBase   float64  // blind DT value (for delta ratio)
		blisBase *float64 // blind BLIS value (for delta ratio)
	}
	var blisTTFTBase, blisTputBase *float64
	if blind != nil {
		blisTTFTBase = &blind.TTFTMs
		blisTputBase = &blind.OutputThroughput
	}
	specs := []spec{
		{"ttft", ref.AdapterAware.TTFTMs, aware.TTFTMs, ref.AdapterBlind.TTFTMs, blisTTFTBase},
		{"throughput", ref.AdapterAware.OutputThroughput, aware.OutputThroughput, ref.AdapterBlind.OutputThroughput, blisTputBase},
	}

	for _, s := range specs {
		m := AdapterMetricComparison{Metric: s.name, DTValue: s.dt, BLISValue: s.blis}
		m.MAPE = ape(s.blis, s.dt)
		m.Within = m.MAPE <= threshold
		// Delta-normalized: compare aware/blind ratios (base cancels).
		if s.blisBase != nil && s.dtBase != 0 && *s.blisBase != 0 {
			dtRatio := s.dt / s.dtBase
			blisRatio := s.blis / *s.blisBase
			d := ape(blisRatio, dtRatio)
			m.DeltaMAPE = &d
		}
		if !m.Within {
			rep.AllWithin = false
		}
		rep.Metrics = append(rep.Metrics, m)
	}
	return rep
}
