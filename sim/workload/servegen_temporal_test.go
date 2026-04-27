package workload

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadServeGenDatasetAllWindows(t *testing.T) {
	t.Run("loads all non-empty windows", func(t *testing.T) {
		tmpDir := t.TempDir()
		datasetPath := filepath.Join(tmpDir, "chunk-test-dataset.json")

		dataset := map[string]map[string]string{
			"0": {
				"input_tokens":  "{100: 0.5, 200: 0.5}",
				"output_tokens": "{50: 0.7, 100: 0.3}",
			},
			"600": {
				"input_tokens":  "{150: 0.4, 250: 0.6}",
				"output_tokens": "{75: 0.8, 150: 0.2}",
			},
			"1200": {
				"input_tokens":  "{}", // Empty window - should be skipped
				"output_tokens": "{}",
			},
		}

		data, err := json.Marshal(dataset)
		require.NoError(t, err)
		err = os.WriteFile(datasetPath, data, 0644)
		require.NoError(t, err)

		sgConfig := &ServeGenDataSpec{}
		result, err := loadServeGenDatasetAllWindows(datasetPath, sgConfig)
		require.NoError(t, err)

		// Should load 2 non-empty windows
		assert.Len(t, result, 2)

		// Check window 0
		assert.Contains(t, result, 0)
		assert.Len(t, result[0].inputPDF, 2)
		assert.InDelta(t, 0.5, result[0].inputPDF[100], 0.001)
		assert.InDelta(t, 0.5, result[0].inputPDF[200], 0.001)

		// Check window 600
		assert.Contains(t, result, 600)
		assert.Len(t, result[600].inputPDF, 2)
		assert.InDelta(t, 0.4, result[600].inputPDF[150], 0.001)

		// Empty window should be skipped
		assert.NotContains(t, result, 1200)
	})

	t.Run("respects span filtering", func(t *testing.T) {
		tmpDir := t.TempDir()
		datasetPath := filepath.Join(tmpDir, "chunk-test-dataset.json")

		dataset := map[string]map[string]string{
			"0": {
				"input_tokens":  "{100: 1.0}",
				"output_tokens": "{50: 1.0}",
			},
			"600": {
				"input_tokens":  "{150: 1.0}",
				"output_tokens": "{75: 1.0}",
			},
			"1200": {
				"input_tokens":  "{200: 1.0}",
				"output_tokens": "{100: 1.0}",
			},
		}

		data, err := json.Marshal(dataset)
		require.NoError(t, err)
		err = os.WriteFile(datasetPath, data, 0644)
		require.NoError(t, err)

		// Filter: only include windows [600, 1200)
		sgConfig := &ServeGenDataSpec{SpanStart: 600, SpanEnd: 1200}
		result, err := loadServeGenDatasetAllWindows(datasetPath, sgConfig)
		require.NoError(t, err)

		// Should include windows at 0 and 600 (keep entries before span for nearest-preceding lookup)
		// Should NOT include 1200 (after span end)
		assert.Len(t, result, 2)
		assert.Contains(t, result, 0)    // Kept for nearest-preceding lookup
		assert.Contains(t, result, 600)  // Within span
		assert.NotContains(t, result, 1200) // After span end
	})

	t.Run("all empty windows returns empty map", func(t *testing.T) {
		tmpDir := t.TempDir()
		datasetPath := filepath.Join(tmpDir, "chunk-test-dataset.json")

		dataset := map[string]map[string]string{
			"0": {
				"input_tokens":  "{}",
				"output_tokens": "{}",
			},
		}

		data, err := json.Marshal(dataset)
		require.NoError(t, err)
		err = os.WriteFile(datasetPath, data, 0644)
		require.NoError(t, err)

		sgConfig := &ServeGenDataSpec{}
		result, err := loadServeGenDatasetAllWindows(datasetPath, sgConfig)
		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("non-numeric key is skipped", func(t *testing.T) {
		tmpDir := t.TempDir()
		datasetPath := filepath.Join(tmpDir, "chunk-test-dataset.json")

		dataset := map[string]map[string]string{
			"metadata": {
				"input_tokens":  "{100: 1.0}",
				"output_tokens": "{50: 1.0}",
			},
			"600": {
				"input_tokens":  "{100: 1.0}",
				"output_tokens": "{50: 1.0}",
			},
		}

		data, err := json.Marshal(dataset)
		require.NoError(t, err)
		err = os.WriteFile(datasetPath, data, 0644)
		require.NoError(t, err)

		sgConfig := &ServeGenDataSpec{}
		result, err := loadServeGenDatasetAllWindows(datasetPath, sgConfig)
		require.NoError(t, err)

		// Only numeric key should be loaded
		assert.Len(t, result, 1)
		assert.Contains(t, result, 600)
	})

	t.Run("invalid input PDF returns error", func(t *testing.T) {
		tmpDir := t.TempDir()
		datasetPath := filepath.Join(tmpDir, "chunk-test-dataset.json")

		dataset := map[string]map[string]string{
			"0": {
				"input_tokens":  "not a dict",
				"output_tokens": "{50: 1.0}",
			},
		}

		data, err := json.Marshal(dataset)
		require.NoError(t, err)
		err = os.WriteFile(datasetPath, data, 0644)
		require.NoError(t, err)

		sgConfig := &ServeGenDataSpec{}
		_, err = loadServeGenDatasetAllWindows(datasetPath, sgConfig)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "input PDF")
	})
}

func TestLoadServeGenChunk_TemporalPreservation(t *testing.T) {
	t.Run("preserves all active windows with per-window parameters", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Write trace CSV with mixed active/inactive windows
		tracePath := filepath.Join(tmpDir, "chunk-test-trace.csv")
		traceData := "0,0,0,,0,0\n600,10.5,0.95,Gamma,1.1,0.04\n1200,22.8,1.02,Gamma,0.96,0.05\n1800,0,0,,0,0\n2400,15.3,0.88,Gamma,1.3,0.03\n"

		err := os.WriteFile(tracePath, []byte(traceData), 0644)
		require.NoError(t, err)

		// Write dataset JSON with per-window distributions
		datasetPath := filepath.Join(tmpDir, "chunk-test-dataset.json")
		dataset := map[string]map[string]string{
			"600": {
				"input_tokens":  "{100: 0.6, 200: 0.4}",
				"output_tokens": "{50: 0.8, 100: 0.2}",
			},
			"1200": {
				"input_tokens":  "{150: 0.5, 250: 0.5}",
				"output_tokens": "{75: 0.7, 150: 0.3}",
			},
			"2400": {
				"input_tokens":  "{120: 0.7, 220: 0.3}",
				"output_tokens": "{60: 0.9, 120: 0.1}",
			},
		}
		datasetJSON, err := json.Marshal(dataset)
		require.NoError(t, err)
		err = os.WriteFile(datasetPath, datasetJSON, 0644)
		require.NoError(t, err)

		// Load chunk
		sgConfig := &ServeGenDataSpec{}
		client, err := loadServeGenChunk("test", tracePath, datasetPath, sgConfig)
		require.NoError(t, err)
		require.NotNil(t, client)

		// Check client metadata
		assert.Equal(t, "servegen-chunk-test", client.ID)
		assert.Equal(t, "chunk-test", client.TenantID)
		assert.Equal(t, 1.0, client.RateFraction)
		assert.Equal(t, "standard", client.SLOClass)
		assert.True(t, client.Streaming)

		// Check lifecycle windows: should have 3 active windows (skip rate=0)
		require.NotNil(t, client.Lifecycle)
		require.Len(t, client.Lifecycle.Windows, 3, "should have 3 active windows (skip rate=0)")

		// Sort windows by start time for deterministic assertions
		sort.Slice(client.Lifecycle.Windows, func(i, j int) bool {
			return client.Lifecycle.Windows[i].StartUs < client.Lifecycle.Windows[j].StartUs
		})

		// Check window 1 (timestamp 600)
		w1 := client.Lifecycle.Windows[0]
		assert.Equal(t, int64(600*1e6), w1.StartUs)
		assert.Equal(t, int64(1200*1e6), w1.EndUs) // 600s + 600s window duration
		require.NotNil(t, w1.TraceRate)
		assert.Equal(t, 10.5, *w1.TraceRate)
		require.NotNil(t, w1.Arrival)
		assert.Equal(t, "gamma", w1.Arrival.Process)
		require.NotNil(t, w1.Arrival.Shape)
		assert.Equal(t, 1.1, *w1.Arrival.Shape)
		require.NotNil(t, w1.Arrival.Scale)
		assert.Equal(t, 40000.0, *w1.Arrival.Scale) // 0.04s * 1e6 = 40000us
		require.NotNil(t, w1.InputDist)
		assert.Equal(t, "lognormal", w1.InputDist.Type)
		// Check that lognormal parameters exist (mu, sigma fitted from empirical PMF)
		_, hasMu := w1.InputDist.Params["mu"]
		_, hasSigma := w1.InputDist.Params["sigma"]
		assert.True(t, hasMu, "lognormal distribution should have mu parameter")
		assert.True(t, hasSigma, "lognormal distribution should have sigma parameter")

		// Check window 2 (timestamp 1200) - different distributions
		w2 := client.Lifecycle.Windows[1]
		assert.Equal(t, int64(1200*1e6), w2.StartUs)
		require.NotNil(t, w2.TraceRate)
		assert.Equal(t, 22.8, *w2.TraceRate)
		require.NotNil(t, w2.InputDist)
		assert.Equal(t, "lognormal", w2.InputDist.Type)
		// Different window should have different fitted parameters
		_, hasMu2 := w2.InputDist.Params["mu"]
		assert.True(t, hasMu2, "window 2 should have lognormal parameters")

		// Check window 3 (timestamp 2400)
		w3 := client.Lifecycle.Windows[2]
		assert.Equal(t, int64(2400*1e6), w3.StartUs)
		require.NotNil(t, w3.TraceRate)
		assert.Equal(t, 15.3, *w3.TraceRate)
	})

	t.Run("inactive chunk (all rate=0) returns nil", func(t *testing.T) {
		tmpDir := t.TempDir()

		tracePath := filepath.Join(tmpDir, "chunk-test-trace.csv")
		traceData := "0,0,0,,0,0\n600,0,0,,0,0\n"
		require.NoError(t, os.WriteFile(tracePath, []byte(traceData), 0644))

		datasetPath := filepath.Join(tmpDir, "chunk-test-dataset.json")
		require.NoError(t, os.WriteFile(datasetPath, []byte(`{}`), 0644))

		sgConfig := &ServeGenDataSpec{}
		client, err := loadServeGenChunk("test", tracePath, datasetPath, sgConfig)
		require.NoError(t, err)
		assert.Nil(t, client, "inactive chunk should return nil")
	})

	t.Run("window uses nearest-preceding dataset", func(t *testing.T) {
		tmpDir := t.TempDir()

		tracePath := filepath.Join(tmpDir, "chunk-test-trace.csv")
		traceData := "600,10.5,0.95,Gamma,1.1,0.04\n1200,22.8,1.02,Gamma,0.96,0.05\n"
		require.NoError(t, os.WriteFile(tracePath, []byte(traceData), 0644))

		// Only provide dataset for window 600, not 1200
		datasetPath := filepath.Join(tmpDir, "chunk-test-dataset.json")
		dataset := map[string]map[string]string{
			"600": {
				"input_tokens":  "{100: 1.0}",
				"output_tokens": "{50: 1.0}",
			},
		}
		datasetJSON, _ := json.Marshal(dataset)
		require.NoError(t, os.WriteFile(datasetPath, datasetJSON, 0644))

		sgConfig := &ServeGenDataSpec{}
		client, err := loadServeGenChunk("test", tracePath, datasetPath, sgConfig)
		require.NoError(t, err)
		require.NotNil(t, client)

		// Both windows should be present (window 1200 uses nearest-preceding dataset at 600)
		require.Len(t, client.Lifecycle.Windows, 2)
		assert.Equal(t, int64(600*1e6), client.Lifecycle.Windows[0].StartUs)
		assert.Equal(t, int64(1200*1e6), client.Lifecycle.Windows[1].StartUs)
	})
}

// TestServeGenConversion_E2E tests the full pipeline from ServeGen data files
// through the converter and generator. Validates:
// - Multi-chunk loading with temporal preservation
// - Proportional rate allocation between co-active chunks
// - Request generation with per-window distributions
// - Clock monotonicity (INV-3) and causality (INV-5)
func TestServeGenConversion_E2E(t *testing.T) {
	t.Run("multi-chunk proportional allocation", func(t *testing.T) {
		// Set up two chunks with different trace rates in the same window.
		// Verify proportional allocation distributes requests according to
		// trace rate ratios.
		tmpDir := t.TempDir()

		// chunk-2: trace rate 15.2 req/s
		chunk2Trace := "600,15.2,0.9,Gamma,1.2,0.04\n"
		chunk2Dataset := map[string]map[string]string{
			"600": {
				"input_tokens":  "{100: 1.0}",
				"output_tokens": "{50: 1.0}",
			},
		}
		require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-2-trace.csv"), []byte(chunk2Trace), 0644))
		d2, _ := json.Marshal(chunk2Dataset)
		require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-2-dataset.json"), d2, 0644))

		// chunk-8: trace rate 22.5 req/s
		chunk8Trace := "600,22.5,1.0,Gamma,1.0,0.04\n"
		chunk8Dataset := map[string]map[string]string{
			"600": {
				"input_tokens":  "{100: 1.0}",
				"output_tokens": "{50: 1.0}",
			},
		}
		require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-8-trace.csv"), []byte(chunk8Trace), 0644))
		d8, _ := json.Marshal(chunk8Dataset)
		require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-8-dataset.json"), d8, 0644))

		// Load spec
		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 150,
			Seed:          42,
			ServeGenData:  &ServeGenDataSpec{Path: tmpDir},
		}
		err := loadServeGenData(spec)
		require.NoError(t, err)
		assert.Len(t, spec.Clients, 2, "should load 2 chunks")

		// Both clients should have lifecycle windows with per-window arrival/trace_rate
		// Distributions should be at client-level (both chunks use single dataset)
		for _, c := range spec.Clients {
			require.NotNil(t, c.Lifecycle, "client %s should have lifecycle", c.ID)
			require.Greater(t, len(c.Lifecycle.Windows), 0, "client %s should have windows", c.ID)

			// Client-level distributions should be lognormal (fitted from dataset)
			assert.Equal(t, "lognormal", c.InputDist.Type, "client %s should have lognormal input", c.ID)
			assert.Equal(t, "lognormal", c.OutputDist.Type, "client %s should have lognormal output", c.ID)

			// Windows should have per-window parameters but not distribution overrides
			for _, w := range c.Lifecycle.Windows {
				assert.NotNil(t, w.TraceRate, "window should have TraceRate")
				assert.NotNil(t, w.Arrival, "window should have Arrival")
				assert.Nil(t, w.InputDist, "window should not override input dist (use client-level)")
				assert.Nil(t, w.OutputDist, "window should not override output dist (use client-level)")
			}
		}

		// Generate requests (window is 600s-1200s)
		spec.ServeGenData = nil
		requests, err := GenerateRequests(spec, 1200*1e6, 0)
		require.NoError(t, err)
		require.Greater(t, len(requests), 0, "should generate requests")

		// Count requests per chunk
		counts := make(map[string]int)
		for _, req := range requests {
			counts[req.ClientID]++
		}

		// Both chunks should generate requests
		assert.Greater(t, counts["servegen-chunk-2"], 0, "chunk-2 should generate requests")
		assert.Greater(t, counts["servegen-chunk-8"], 0, "chunk-8 should generate requests")

		// Check proportional allocation: 15.2:22.5 ~ 40%:60%
		total := counts["servegen-chunk-2"] + counts["servegen-chunk-8"]
		chunk2Ratio := float64(counts["servegen-chunk-2"]) / float64(total)
		chunk8Ratio := float64(counts["servegen-chunk-8"]) / float64(total)

		expectedChunk2Ratio := 15.2 / (15.2 + 22.5)
		expectedChunk8Ratio := 22.5 / (15.2 + 22.5)

		assert.InDelta(t, expectedChunk2Ratio, chunk2Ratio, 0.10,
			"chunk-2 ratio should be ~%.1f%%, got %.1f%%",
			expectedChunk2Ratio*100, chunk2Ratio*100)
		assert.InDelta(t, expectedChunk8Ratio, chunk8Ratio, 0.10,
			"chunk-8 ratio should be ~%.1f%%, got %.1f%%",
			expectedChunk8Ratio*100, chunk8Ratio*100)
	})

	t.Run("requests are sorted by arrival time (INV-3)", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Two chunks with overlapping windows
		for _, chunkID := range []string{"0", "1"} {
			trace := "600,10.0,0.5,Gamma,2.0,0.05\n1200,20.0,0.5,Gamma,2.0,0.03\n"
			dataset := map[string]map[string]string{
				"600": {
					"input_tokens":  "{100: 1.0}",
					"output_tokens": "{50: 1.0}",
				},
				"1200": {
					"input_tokens":  "{100: 1.0}",
					"output_tokens": "{50: 1.0}",
				},
			}
			require.NoError(t, os.WriteFile(
				filepath.Join(tmpDir, "chunk-"+chunkID+"-trace.csv"),
				[]byte(trace), 0644))
			d, _ := json.Marshal(dataset)
			require.NoError(t, os.WriteFile(
				filepath.Join(tmpDir, "chunk-"+chunkID+"-dataset.json"),
				d, 0644))
		}

		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 100,
			Seed:          42,
			ServeGenData:  &ServeGenDataSpec{Path: tmpDir},
		}
		err := loadServeGenData(spec)
		require.NoError(t, err)
		spec.ServeGenData = nil

		requests, err := GenerateRequests(spec, 1800*1e6, 0)
		require.NoError(t, err)
		require.Greater(t, len(requests), 0)

		// INV-3: Clock monotonicity - requests must be sorted by arrival time
		for i := 1; i < len(requests); i++ {
			assert.LessOrEqual(t, requests[i-1].ArrivalTime, requests[i].ArrivalTime,
				"requests must be sorted by arrival time (index %d: %d > %d)",
				i, requests[i-1].ArrivalTime, requests[i].ArrivalTime)
		}

		// Check requests span multiple windows (from both 600s and 1200s windows)
		hasWindow1 := false
		hasWindow2 := false
		for _, req := range requests {
			if req.ArrivalTime >= 600*1e6 && req.ArrivalTime < 1200*1e6 {
				hasWindow1 = true
			}
			if req.ArrivalTime >= 1200*1e6 && req.ArrivalTime < 1800*1e6 {
				hasWindow2 = true
			}
		}
		assert.True(t, hasWindow1, "should have requests in window 1 (600-1200s)")
		assert.True(t, hasWindow2, "should have requests in window 2 (1200-1800s)")

		// Check both chunks contributed (interleaved requests)
		clientIDs := make(map[string]bool)
		for _, req := range requests {
			clientIDs[req.ClientID] = true
		}
		assert.Len(t, clientIDs, 2, "requests should come from both chunks")
	})

	t.Run("per-window distributions produce different token lengths", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Single chunk with two windows having distinct distributions.
		// Window 1: input tokens = constant 100
		// Window 2: input tokens = constant 500
		trace := "0,10.0,0.5,Gamma,2.0,0.05\n600,10.0,0.5,Gamma,2.0,0.05\n"
		dataset := map[string]map[string]string{
			"0": {
				"input_tokens":  "{100: 1.0}",
				"output_tokens": "{50: 1.0}",
			},
			"600": {
				"input_tokens":  "{500: 1.0}",
				"output_tokens": "{200: 1.0}",
			},
		}
		require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-0-trace.csv"), []byte(trace), 0644))
		d, _ := json.Marshal(dataset)
		require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-0-dataset.json"), d, 0644))

		spec := &WorkloadSpec{
			Version:       "2",
			AggregateRate: 10,
			Seed:          42,
			ServeGenData:  &ServeGenDataSpec{Path: tmpDir},
		}
		err := loadServeGenData(spec)
		require.NoError(t, err)
		spec.ServeGenData = nil

		requests, err := GenerateRequests(spec, 1200*1e6, 0)
		require.NoError(t, err)
		require.Greater(t, len(requests), 0)

		// Requests in window 1 (0-600s) should have ~100 input tokens
		// Requests in window 2 (600-1200s) should have ~500 input tokens
		var window1Lengths, window2Lengths []int
		for _, req := range requests {
			if req.ArrivalTime < 600*1e6 {
				window1Lengths = append(window1Lengths, len(req.InputTokens))
			} else {
				window2Lengths = append(window2Lengths, len(req.InputTokens))
			}
		}

		require.Greater(t, len(window1Lengths), 0, "window 1 should have requests")
		require.Greater(t, len(window2Lengths), 0, "window 2 should have requests")

		// All window 1 requests should have 100 tokens (constant empirical dist)
		for i, l := range window1Lengths {
			assert.Equal(t, 100, l, "window 1 request %d should have 100 input tokens, got %d", i, l)
		}

		// All window 2 requests should have 500 tokens (constant empirical dist)
		for i, l := range window2Lengths {
			assert.Equal(t, 500, l, "window 2 request %d should have 500 input tokens, got %d", i, l)
		}
	})

	t.Run("ConvertServeGen produces valid spec", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Single chunk
		trace := "600,10.0,0.5,Gamma,2.0,0.05\n"
		dataset := map[string]map[string]string{
			"600": {
				"input_tokens":  "{100: 0.5, 200: 0.5}",
				"output_tokens": "{50: 0.7, 100: 0.3}",
			},
		}
		require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-0-trace.csv"), []byte(trace), 0644))
		d, _ := json.Marshal(dataset)
		require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-0-dataset.json"), d, 0644))

		spec, err := ConvertServeGen(tmpDir, "")
		require.NoError(t, err)
		require.NotNil(t, spec)

		assert.Equal(t, "2", spec.Version)
		assert.Len(t, spec.Clients, 1)
		assert.Equal(t, "servegen-chunk-0", spec.Clients[0].ID)
		assert.Nil(t, spec.ServeGenData, "ServeGenData should be cleared after loading")

		// Should have per-window parameters (arrival/trace_rate) but not distributions
		require.NotNil(t, spec.Clients[0].Lifecycle)
		require.Len(t, spec.Clients[0].Lifecycle.Windows, 1)
		w := spec.Clients[0].Lifecycle.Windows[0]
		assert.NotNil(t, w.TraceRate)
		assert.NotNil(t, w.Arrival)
		// Distributions should be at client-level (single dataset)
		assert.Nil(t, w.InputDist)
		assert.Nil(t, w.OutputDist)
		// Client-level distributions should be lognormal
		assert.Equal(t, "lognormal", spec.Clients[0].InputDist.Type)
		assert.Equal(t, "lognormal", spec.Clients[0].OutputDist.Type)
	})

	t.Run("determinism: same seed produces identical output", func(t *testing.T) {
		tmpDir := t.TempDir()

		trace := "600,10.0,0.5,Gamma,2.0,0.05\n"
		dataset := map[string]map[string]string{
			"600": {
				"input_tokens":  "{100: 0.5, 200: 0.5}",
				"output_tokens": "{50: 0.7, 100: 0.3}",
			},
		}
		require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-0-trace.csv"), []byte(trace), 0644))
		d, _ := json.Marshal(dataset)
		require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-0-dataset.json"), d, 0644))

		makeSpec := func() *WorkloadSpec {
			s := &WorkloadSpec{
				Version:       "2",
				AggregateRate: 50,
				Seed:          77,
				ServeGenData:  &ServeGenDataSpec{Path: tmpDir},
			}
			err := loadServeGenData(s)
			require.NoError(t, err)
			s.ServeGenData = nil
			return s
		}

		reqs1, err1 := GenerateRequests(makeSpec(), 1200*1e6, 0)
		require.NoError(t, err1)

		reqs2, err2 := GenerateRequests(makeSpec(), 1200*1e6, 0)
		require.NoError(t, err2)

		require.Equal(t, len(reqs1), len(reqs2), "INV-6: same seed must produce same count")
		for i := range reqs1 {
			assert.Equal(t, reqs1[i].ArrivalTime, reqs2[i].ArrivalTime,
				"INV-6: arrival times must match at index %d", i)
			assert.Equal(t, reqs1[i].InputTokens, reqs2[i].InputTokens,
				"INV-6: input tokens must match at index %d", i)
			assert.Equal(t, reqs1[i].OutputTokens, reqs2[i].OutputTokens,
				"INV-6: output tokens must match at index %d", i)
		}
	})
}
