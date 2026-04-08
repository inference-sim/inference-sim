package cluster

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/workload"
	"gopkg.in/yaml.v3"
)

// rooflineGoldenDataset is the root structure of testdata/roofline_iter29.json.
type rooflineGoldenDataset struct {
	Description string                   `json:"description"`
	Backend     string                   `json:"backend"`
	Experiments []rooflineGoldenExperiment `json:"experiments"`
}

// rooflineGoldenExperiment holds a single iter29 experiment configuration and
// its expected simulation outputs using the roofline backend.
type rooflineGoldenExperiment struct {
	ID                  string          `json:"id"`
	Model               string          `json:"model"`
	ModelConfigDir      string          `json:"model_config_dir"`
	Hardware            string          `json:"hardware"`
	TP                  int             `json:"tp"`
	MaxNumSeqs          int             `json:"max_num_seqs"`
	MaxNumBatchedTokens int             `json:"max_num_batched_tokens"`
	TotalKVBlocks       int64           `json:"total_kv_blocks"`
	CPUKVBlocks         int64           `json:"cpu_kv_blocks"`
	Workload            json.RawMessage `json:"workload"`
	Expected            goldenExpected  `json:"expected"`
}

// loadRooflineGoldenDataset reads testdata/roofline_goldendataset.json.
func loadRooflineGoldenDataset(t *testing.T) *rooflineGoldenDataset {
	t.Helper()
	path := filepath.Join(goldenRepoRoot(), "testdata", "roofline_goldendataset.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("loadRooflineGoldenDataset: %v", err)
	}
	var ds rooflineGoldenDataset
	if err := json.Unmarshal(data, &ds); err != nil {
		t.Fatalf("loadRooflineGoldenDataset: parse: %v", err)
	}
	return &ds
}

// TestRoofline_GoldenDataset verifies that the roofline latency backend
// produces byte-for-byte identical TTFT and E2E predictions to the iter29
// roofline baseline.
//
// This test serves as a regression guard for the analytical roofline model.
// Any change to roofline FLOPs/bandwidth calculations, MoE handling, or TP
// communication modeling will be caught by this test.
//
// Companion invariant assertions (conservation, causality, non-zero TTFT) catch
// bugs independently of the golden values, satisfying the "test laws not just
// values" principle from principles.md.
//
// Golden values must not be regenerated in place. If the roofline backend
// behavior needs to change intentionally, rename the backend (e.g.
// "roofline-v2") and add a new dataset. Renaming prevents silent behavioral
// regressions from accumulating.
//
// Regression guard: This test will catch any unintended changes to roofline
// model calculations, including Scout MoE interleaved architecture handling
// (issue #877), weight bandwidth calculations, or TP all-reduce modeling.
func TestRoofline_GoldenDataset(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping roofline golden dataset test in short mode (-short flag)")
	}

	root := goldenRepoRoot()
	ds := loadRooflineGoldenDataset(t)
	if len(ds.Experiments) == 0 {
		t.Fatal("golden dataset has no experiments")
	}
	if ds.Backend != "roofline" {
		t.Fatalf("unexpected backend in dataset: got %q, want \"roofline\"", ds.Backend)
	}

	hwConfigPath := filepath.Join(root, "hardware_config.json")

	for _, exp := range ds.Experiments {
		exp := exp // capture loop variable
		t.Run(fmt.Sprintf("exp_%s_%s", exp.ID, exp.Model), func(t *testing.T) {
			t.Parallel() // each sub-test owns independent state; safe to run concurrently
			// ── Load model + hardware configuration ──────────────────────────
			mcPath := filepath.Join(root, exp.ModelConfigDir, "config.json")
			mc, err := latency.GetModelConfig(mcPath)
			if err != nil {
				t.Fatalf("GetModelConfig(%s): %v", mcPath, err)
			}
			hc, err := latency.GetHWConfig(hwConfigPath, exp.Hardware)
			if err != nil {
				t.Fatalf("GetHWConfig(%s): %v", exp.Hardware, err)
			}

			// ── Build roofline latency model ─────────────────────────────────
			// Roofline uses zero coefficients (pure analytical model)
			coeffs := sim.NewLatencyCoeffs(
				[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // beta coeffs (unused in roofline)
				[]float64{0, 0, 0},                      // alpha coeffs (unused in roofline)
			)
			hwCfg := sim.NewModelHardwareConfig(*mc, hc, exp.Model, exp.Hardware, exp.TP, ds.Backend, 0)

			// Validate that the backend is accepted; fail fast with a clear error.
			if _, err := latency.NewLatencyModel(coeffs, hwCfg); err != nil {
				t.Fatalf("NewLatencyModel: %v", err)
			}

			// ── Decode workload spec and generate requests ───────────────────
			var ws goldenWorkloadSpec
			if err := yaml.Unmarshal(exp.Workload, &ws); err != nil {
				t.Fatalf("decode workload: %v", err)
			}
			if ws.InferencePerf == nil {
				t.Fatalf("experiment %s: workload JSON missing inference_perf field", exp.ID)
			}

			ipSpec := &workload.InferencePerfSpec{
				SharedPrefix: ws.InferencePerf.SharedPrefix,
			}
			for _, s := range ws.InferencePerf.Stages {
				ipSpec.Stages = append(ipSpec.Stages, workload.StageSpec{
					Rate:     s.Rate,
					Duration: s.Duration,
				})
			}
			fullSpec, err := workload.ExpandInferencePerfSpec(ipSpec, ws.Seed)
			if err != nil {
				t.Fatalf("ExpandInferencePerfSpec: %v", err)
			}
			wl, err := workload.GenerateWorkload(fullSpec, math.MaxInt64, int64(ws.NumRequests))
			if err != nil {
				t.Fatalf("GenerateWorkload: %v", err)
			}
			var onDone func(*sim.Request, int64) []*sim.Request
			if len(wl.Sessions) > 0 {
				sessionMgr := workload.NewSessionManager(wl.Sessions)
				if wl.FollowUpBudget >= 0 {
					sessionMgr.SetFollowUpBudget(wl.FollowUpBudget)
				}
				onDone = sessionMgr.OnComplete
			}

			// ── Configure and run ClusterSimulator ──────────────────────────
			cfg := DeploymentConfig{
				SimConfig: sim.SimConfig{
					Horizon:             math.MaxInt64,
					Seed:                ws.Seed,
					KVCacheConfig:       sim.NewKVCacheConfig(exp.TotalKVBlocks, 16, exp.CPUKVBlocks, 0.9, 0.2, 0),
					BatchConfig:         sim.NewBatchConfig(int64(exp.MaxNumSeqs), int64(exp.MaxNumBatchedTokens), 0),
					LatencyCoeffs:       coeffs,
					ModelHardwareConfig: hwCfg,
				},
				NumInstances: 1,
			}
			cs := NewClusterSimulator(cfg, wl.Requests, onDone)
			if err := cs.Run(); err != nil {
				t.Fatalf("ClusterSimulator.Run: %v", err)
			}

			m := cs.AggregatedMetrics()

			// ── Invariant: request conservation (INV-1) ───────────────────
			if m.StillQueued != 0 {
				t.Errorf("INV-1: still_queued=%d, want 0 (requests stuck in wait queue)", m.StillQueued)
			}
			if m.StillRunning != 0 {
				t.Errorf("INV-1: still_running=%d, want 0 (requests stuck in running batch)", m.StillRunning)
			}
			if m.DroppedUnservable != 0 {
				t.Errorf("INV-1: dropped_unservable=%d, want 0", m.DroppedUnservable)
			}
			if m.TimedOutRequests != 0 {
				t.Errorf("INV-1: timed_out_requests=%d, want 0", m.TimedOutRequests)
			}
			if cs.RejectedRequests() != 0 {
				t.Errorf("INV-1: rejected_requests=%d, want 0", cs.RejectedRequests())
			}
			if m.CompletedRequests != exp.Expected.CompletedRequests {
				t.Errorf("completed_requests: got %d, want %d", m.CompletedRequests, exp.Expected.CompletedRequests)
			}

			// ── Invariant: token conservation ────────────────────────────
			if m.TotalInputTokens != exp.Expected.TotalInputTokens {
				t.Errorf("total_input_tokens: got %d, want %d", m.TotalInputTokens, exp.Expected.TotalInputTokens)
			}
			if m.TotalOutputTokens != exp.Expected.TotalOutputTokens {
				t.Errorf("total_output_tokens: got %d, want %d", m.TotalOutputTokens, exp.Expected.TotalOutputTokens)
			}

			// ── Compute output metrics from raw Metrics struct ────────────
			sortedTTFTs := goldenSortedValues(m.RequestTTFTs)
			sortedE2Es := goldenSortedValues(m.RequestE2Es)
			allITLs := make([]float64, len(m.AllITLs))
			for i, v := range m.AllITLs {
				allITLs[i] = float64(v)
			}
			sort.Float64s(allITLs)

			ttftMean := sim.CalculateMean(sortedTTFTs)
			ttftP90 := sim.CalculatePercentile(sortedTTFTs, 90)
			ttftP99 := sim.CalculatePercentile(sortedTTFTs, 99)
			e2eMean := sim.CalculateMean(sortedE2Es)
			e2eP90 := sim.CalculatePercentile(sortedE2Es, 90)
			e2eP99 := sim.CalculatePercentile(sortedE2Es, 99)
			itlMean := sim.CalculateMean(allITLs)

			// ── Invariant: causality (INV-5) ──────────────────────────────
			if ttftMean <= 0 {
				t.Errorf("ttft_mean: got %.6f, want > 0 (every request must have a first token)", ttftMean)
			}
			if ttftMean > e2eMean {
				t.Errorf("causality: ttft_mean %.6f > e2e_mean %.6f", ttftMean, e2eMean)
			}

			// ── Golden values: byte-for-byte identical floats ─────────────
			// relTol=1e-9 catches floating-point rounding from platform
			// differences while rejecting any behavioral change to the
			// roofline backend.
			//
			// To update these values, the roofline backend MUST be renamed
			// first (e.g. "roofline-v2"). See comment on
			// TestRoofline_GoldenDataset.
			const relTol = 1e-9
			goldenAssertApprox(t, "ttft_mean_ms", exp.Expected.TTFTMeanMs, ttftMean, relTol)
			goldenAssertApprox(t, "ttft_p90_ms", exp.Expected.TTFTP90Ms, ttftP90, relTol)
			goldenAssertApprox(t, "ttft_p99_ms", exp.Expected.TTFTP99Ms, ttftP99, relTol)
			goldenAssertApprox(t, "e2e_mean_ms", exp.Expected.E2EMeanMs, e2eMean, relTol)
			goldenAssertApprox(t, "e2e_p90_ms", exp.Expected.E2EP90Ms, e2eP90, relTol)
			goldenAssertApprox(t, "e2e_p99_ms", exp.Expected.E2EP99Ms, e2eP99, relTol)
			goldenAssertApprox(t, "itl_mean_ms", exp.Expected.ITLMeanMs, itlMean, relTol)
		})
	}
}

