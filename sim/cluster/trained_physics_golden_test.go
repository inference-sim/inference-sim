package cluster

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/workload"
	"gopkg.in/yaml.v3"
)

// trainedPhysicsGoldenDataset is the root structure of testdata/trained_physics_iter29.json.
type trainedPhysicsGoldenDataset struct {
	Description string                     `json:"description"`
	Backend     string                     `json:"backend"`
	AlphaCoeffs []float64                  `json:"alpha_coeffs"`
	BetaCoeffs  []float64                  `json:"beta_coeffs"`
	Experiments []trainedPhysicsExperiment `json:"experiments"`
}

// trainedPhysicsExperiment holds a single iter29 experiment configuration and
// its expected simulation outputs.
type trainedPhysicsExperiment struct {
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
	Expected            goldenExpected `json:"expected"`
}

// loadTrainedPhysicsGoldenDataset reads testdata/trained_physics_iter29.json.
func loadTrainedPhysicsGoldenDataset(t *testing.T) *trainedPhysicsGoldenDataset {
	t.Helper()
	path := filepath.Join(goldenRepoRoot(), "testdata", "trained_physics_iter29.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("loadTrainedPhysicsGoldenDataset: %v", err)
	}
	var ds trainedPhysicsGoldenDataset
	if err := json.Unmarshal(data, &ds); err != nil {
		t.Fatalf("loadTrainedPhysicsGoldenDataset: parse: %v", err)
	}
	return &ds
}

// TestTrainedPhysics_GoldenDataset verifies that the trained-physics latency
// backend produces byte-for-byte identical TTFT and E2E predictions to the
// iter29 coefficients baseline (overall loss 34.5675%, derived via sequential
// golden section search).
//
// Companion invariant assertions (conservation, causality, non-zero TTFT) catch
// bugs independently of the golden values, satisfying the "test laws not just
// values" principle from principles.md.
//
// Golden values must not be regenerated in place. If the trained-physics backend
// behavior needs to change intentionally, rename the backend (e.g.
// "trained-physics-v2") and add a new dataset. Renaming prevents silent
// behavioral regressions from accumulating across training iterations.
//
// Regression guard for issue #965: inference_perf SLOClass "batch" caused
// deferred-queue serialization, inflating TTFT 6-100× across all experiments.
// This test will catch any recurrence of that regression pattern.
func TestTrainedPhysics_GoldenDataset(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping trained-physics golden dataset test in short mode (-short flag)")
	}

	root := goldenRepoRoot()
	ds := loadTrainedPhysicsGoldenDataset(t)
	if len(ds.Experiments) == 0 {
		t.Fatal("golden dataset has no experiments")
	}
	if ds.Backend != "trained-physics" {
		t.Fatalf("unexpected backend in dataset: got %q, want \"trained-physics\"", ds.Backend)
	}

	hwConfigPath := filepath.Join(root, "hardware_config.json")

	// In update mode: collect results from parallel subtests, then write JSON.
	// t.Cleanup runs after ALL subtests (including parallel ones) complete.
	var (
		mu      sync.Mutex
		updates = make(map[int]goldenExpected, len(ds.Experiments))
	)
	if *updateGolden {
		t.Cleanup(func() {
			for i, expected := range updates {
				ds.Experiments[i].Expected = expected
			}
			data, err := json.MarshalIndent(ds, "", "  ")
			if err != nil {
				t.Errorf("marshal updated trained-physics dataset: %v", err)
				return
			}
			path := filepath.Join(goldenRepoRoot(), "testdata", "trained_physics_iter29.json")
			if err := os.WriteFile(path, append(data, '\n'), 0644); err != nil {
				t.Errorf("write %s: %v", path, err)
				return
			}
			t.Logf("updated %s (%d experiments)", path, len(updates))
		})
	}

	for i, exp := range ds.Experiments {
		i, exp := i, exp // capture loop variables
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

			// ── Build trained-physics latency model with iter29 coefficients ─
			// Note: NewLatencyCoeffs(betaCoeffs, alphaCoeffs) — order matters.
			coeffs := sim.NewLatencyCoeffs(ds.BetaCoeffs, ds.AlphaCoeffs)
			hwCfg := sim.NewModelHardwareConfig(*mc, hc, exp.Model, exp.Hardware, exp.TP, ds.Backend, 0)

			// Validate that the backend is accepted; fail fast with a clear error.
			if _, err := latency.NewLatencyModel(coeffs, hwCfg); err != nil {
				t.Fatalf("NewLatencyModel: %v", err)
			}

			// ── Decode workload spec and generate requests ───────────────────
			// SharedPrefixSpec uses yaml struct tags; use yaml.v3 to decode
			// (JSON is valid YAML so this works for either format).
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
			// Use GenerateWorkload (not GenerateRequests) so that multi-turn chat
			// sessions are handled via SessionManager callbacks — matching the CLI
			// pipeline exactly and producing byte-identical request counts.
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
					Horizon:       math.MaxInt64,
					Seed:          ws.Seed,
					// KV offload/transfer parameters match the training runner's constants
				// (--kv-offload-threshold 0.9, --kv-transfer-bandwidth 0.2).
				KVCacheConfig: sim.NewKVCacheConfig(exp.TotalKVBlocks, 16, exp.CPUKVBlocks, 0.9, 0.2, 0),
					BatchConfig:   sim.NewBatchConfig(int64(exp.MaxNumSeqs), int64(exp.MaxNumBatchedTokens), 0),
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
			// Full conservation: every leakage path must be zero and the
			// completed count must match golden. With horizon=MaxInt64 and no
			// admission/routing rejections, all requests must complete.
			// Catches silent-drop bugs independently of golden values.
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
			// ── Compute output metrics from raw Metrics struct ────────────
			sortedTTFTs := goldenSortedValues(m.RequestTTFTs)
			sortedE2Es := goldenSortedValues(m.RequestE2Es)
			allITLs := make([]float64, len(m.AllITLs))
			for j, v := range m.AllITLs {
				allITLs[j] = float64(v)
			}
			sort.Float64s(allITLs)

			ttftMean := sim.CalculateMean(sortedTTFTs)
			ttftP90 := sim.CalculatePercentile(sortedTTFTs, 90)
			ttftP99 := sim.CalculatePercentile(sortedTTFTs, 99)
			e2eMean := sim.CalculateMean(sortedE2Es)
			e2eP90 := sim.CalculatePercentile(sortedE2Es, 90)
			e2eP99 := sim.CalculatePercentile(sortedE2Es, 99)
			itlMean := sim.CalculateMean(allITLs)

			// ── Invariant: causality (INV-5) — always checked ─────────────
			if ttftMean <= 0 {
				t.Errorf("ttft_mean: got %.6f, want > 0 (every request must have a first token)", ttftMean)
			}
			if ttftMean > e2eMean {
				t.Errorf("causality: ttft_mean %.6f > e2e_mean %.6f", ttftMean, e2eMean)
			}

			// ── Update mode: record computed values, skip assertions ───────
			if *updateGolden {
				mu.Lock()
				updates[i] = goldenExpected{
					CompletedRequests: m.CompletedRequests,
					TotalInputTokens:  m.TotalInputTokens,
					TotalOutputTokens: m.TotalOutputTokens,
					TTFTSumUs:         m.TTFTSum,
					TTFTMeanMs:        ttftMean,
					TTFTP90Ms:         ttftP90,
					TTFTP99Ms:         ttftP99,
					E2EMeanMs:         e2eMean,
					E2EP90Ms:          e2eP90,
					E2EP99Ms:          e2eP99,
					ITLMeanMs:         itlMean,
				}
				mu.Unlock()
				return
			}

			// ── Golden value assertions ────────────────────────────────────
			if m.CompletedRequests != exp.Expected.CompletedRequests {
				t.Errorf("completed_requests: got %d, want %d", m.CompletedRequests, exp.Expected.CompletedRequests)
			}
			if m.TotalInputTokens != exp.Expected.TotalInputTokens {
				t.Errorf("total_input_tokens: got %d, want %d", m.TotalInputTokens, exp.Expected.TotalInputTokens)
			}
			if m.TotalOutputTokens != exp.Expected.TotalOutputTokens {
				t.Errorf("total_output_tokens: got %d, want %d", m.TotalOutputTokens, exp.Expected.TotalOutputTokens)
			}
			if m.TTFTSum != exp.Expected.TTFTSumUs {
				t.Errorf("ttft_sum_us: got %d, want %d", m.TTFTSum, exp.Expected.TTFTSumUs)
			}
			// relTol=1e-9 catches floating-point rounding from platform
			// differences while rejecting any behavioral change to the
			// trained-physics backend. To update these values, run with -update-golden.
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
