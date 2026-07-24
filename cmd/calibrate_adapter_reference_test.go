package cmd

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// T043 (US5, SC-007, #1470): fixture-based validation of the BLIS-vs-DT adapter
// fidelity comparison for the two calibrated reference configs. The fixtures under
// testdata/dt-reference/ are real: DT simulate() aggregates (from the GPULLM fork)
// and BLIS `replay --metrics-path` aggregates over the identical seeded arrival
// stream (see testdata/dt-reference/README.md).
//
// Honest-scoping outcome (design §15 falsification path — the *mechanism* is the
// deliverable, the bound is an empirical result):
//   - THROUGHPUT (the compute-overhead term, D4) lands WITHIN 20% MAPE for both
//     configs → SC-007 throughput leg validated.
//   - TTFT lands FAR outside 20% because BLIS ports adapter deltas onto its own
//     separately-calibrated base (§6), which is ~100× faster than the DT's H100
//     base fit; the DT's large absolute TTFT is queueing amplification BLIS's
//     faster base never enters. TTFT is also the DT's own weakest axis (§15). This
//     leg is therefore marked UNSUPPORTED — asserted here as > 20%, never a silent
//     pass.
//
// Caveat documented in README: the canonical DT workload does not saturate BLIS,
// so the throughput agreement partly reflects both tracking offered load; a
// saturating-workload D4 validation is future work.
func TestSC007_AdapterFidelity_Fixtures(t *testing.T) {
	const threshold = 0.20
	dir := "testdata/dt-reference"
	configs := []string{"qwen-2.5-7b-instruct", "llama-3.1-8b-instruct"}

	for _, cfg := range configs {
		t.Run(cfg, func(t *testing.T) {
			ref, err := workload.LoadDTReference(filepath.Join(dir, cfg+".dt.json"))
			if err != nil {
				t.Fatal(err)
			}
			aware := loadBLISAggregate(filepath.Join(dir, cfg+".blis-aware.json"))
			blind := loadBLISAggregate(filepath.Join(dir, cfg+".blis-blind.json"))
			rep := workload.CompareAdapterReference(ref, aware, &blind, threshold)

			byName := map[string]workload.AdapterMetricComparison{}
			for _, m := range rep.Metrics {
				byName[m.Metric] = m
			}

			// SC-007 throughput leg: WITHIN 20% for both configs (validated).
			tput := byName["throughput"]
			if !tput.Within {
				t.Errorf("%s throughput MAPE = %.1f%% exceeds %.0f%% — SC-007 throughput leg regressed",
					cfg, tput.MAPE*100, threshold*100)
			}
			// Delta-normalized throughput diagnostic present and within bound.
			if tput.DeltaMAPE == nil {
				t.Errorf("%s throughput delta-normalized MAPE missing (blind fixture should enable it)", cfg)
			} else if *tput.DeltaMAPE > threshold {
				t.Errorf("%s throughput delta-normalized MAPE = %.1f%% exceeds %.0f%%", cfg, *tput.DeltaMAPE*100, threshold*100)
			}

			// TTFT leg: UNSUPPORTED — must be reported as exceeding the bound, not
			// silently within it (base/queueing-regime mismatch, §15).
			ttft := byName["ttft"]
			if ttft.Within {
				t.Errorf("%s ttft MAPE = %.1f%% is within %.0f%% — unexpected; the TTFT leg is documented UNSUPPORTED and must be flagged, not silently passed",
					cfg, ttft.MAPE*100, threshold*100)
			}

			// Because the TTFT leg exceeds, the config is not fully within threshold.
			if rep.AllWithin {
				t.Errorf("%s AllWithin=true but the TTFT leg is unsupported — no silent pass", cfg)
			}
		})
	}
}
