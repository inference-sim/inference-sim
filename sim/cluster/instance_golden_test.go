package cluster

import (
	"math"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// One instance, its engine behind the contract as `blis run` builds it, runs every golden
// dataset workload to completion and keeps the single-instance laws: request conservation
// (INV-1, the five-term form: one instance's own metrics, with no router, gateway queue or
// encode pool above it), causality (INV-5: TTFT >= 0 and E2E >= TTFT) and a clock that
// advanced (INV-3). Ported from the compatibility host loop's TestInstanceSimulator_
// GoldenDataset_Equivalence when that loop was deleted: the laws are the instance's, whatever
// drives its engine.
func TestInstance_GoldenDataset_KeepsTheInstanceLaws(t *testing.T) {
	dataset := testutil.LoadGoldenDataset(t)
	if len(dataset.Tests) == 0 {
		t.Fatal("Golden dataset contains no test cases")
	}
	for _, tc := range dataset.Tests {
		t.Run(tc.Model, func(t *testing.T) {
			instance := newCompatInstance("test-instance", instanceConfigOf(simConfig{
				Horizon:             math.MaxInt64,
				Seed:                tc.Seed,
				KVCacheConfig:       cengine.NewKVCacheConfig(tc.TotalKVBlocks, tc.BlockSizeInTokens, 0, 0, 0, 0),
				BatchConfig:         cengine.NewBatchConfig(tc.MaxNumSeqs, tc.MaxNumBatchedTokens, tc.LongPrefillTokenThreshold),
				LatencyModel:        testFakeLatency(),
				ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), tc.Model, tc.Hardware, tc.TP, 1, false, tc.MaxModelLen),
			}))
			requests := testGenerateRequests(tc.Seed, math.MaxInt64, tc.Rate/1e6,
				tc.NumRequests, tc.PrefixTokens,
				tc.PromptTokens, tc.PromptTokensStdev, tc.PromptTokensMin, tc.PromptTokensMax,
				tc.OutputTokens, tc.OutputTokensStdev, tc.OutputTokensMin, tc.OutputTokensMax)
			for _, req := range requests {
				instance.InjectRequest(req)
			}
			instance.Run()
			m := instance.Metrics()

			assertInstanceINV1Conservation(t, m, tc.NumRequests, tc.Model+"/"+tc.Workload)
			if m.CompletedRequests == 0 {
				t.Fatalf("%s: no request completed", tc.Model)
			}
			for reqID, e2e := range m.RequestE2Es {
				ttft, hasTTFT := m.RequestTTFTs[reqID]
				if !hasTTFT {
					t.Errorf("INV-5 causality: request %q has E2E but no TTFT", reqID)
					continue
				}
				if ttft < 0 {
					t.Errorf("INV-5 causality: request %q TTFT = %f, want >= 0", reqID, ttft)
				}
				if e2e < ttft {
					t.Errorf("INV-5 causality: request %q E2E (%f) < TTFT (%f)", reqID, e2e, ttft)
				}
			}
			if tc.NumRequests > 0 && m.SimEndedTime <= 0 {
				t.Errorf("INV-3 clock monotonicity: SimEndedTime = %d, want > 0 for %d requests",
					m.SimEndedTime, tc.NumRequests)
			}
		})
	}
}
