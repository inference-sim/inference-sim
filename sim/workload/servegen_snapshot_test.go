package workload

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServeGenTimeWindowExtraction_Midnight(t *testing.T) {
	// GIVEN ServeGen data with chunks active at different times
	tmpDir := t.TempDir()

	// Chunk 2: active at Hour 0 (midnight window: 0-1800s)
	// Trace has 10-minute granularity, dataset has 6-hour granularity
	chunk2Trace := "0,5.2,0.8,Weibull,1.5,0.02\n" +
		"600,4.1,0.9,Weibull,1.5,0.025\n" +
		"1200,3.8,0.85,Weibull,1.4,0.02\n"
	chunk2Dataset := map[string]map[string]string{
		// Only Hour 0 dataset entry (all three trace windows use this)
		"0": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"},
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-2-trace.csv"), []byte(chunk2Trace), 0644))
	d, _ := json.Marshal(chunk2Dataset)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-2-dataset.json"), d, 0644))

	// Chunk 8: active at Hour 8 (morning window, should be excluded from midnight)
	chunk8Trace := "28800,9.8,1.2,Gamma,2.0,0.05\n"
	chunk8Dataset := map[string]map[string]string{
		// Hour 6 dataset entry (Hour 8 traces use nearest-preceding at Hour 6)
		"21600": {"input_tokens": "{300: 0.4, 400: 0.6}", "output_tokens": "{100: 0.5, 150: 0.5}"},
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-8-trace.csv"), []byte(chunk8Trace), 0644))
	d, _ = json.Marshal(chunk8Dataset)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-8-dataset.json"), d, 0644))

	// WHEN converting with --time midnight
	spec, err := ConvertServeGen(tmpDir, "midnight")

	// THEN only chunks with windows in midnight range are included
	require.NoError(t, err)
	require.NotNil(t, spec)

	// Should only have chunk-2 (active in midnight window)
	assert.Len(t, spec.Clients, 1, "should only include chunk-2")
	assert.Equal(t, "servegen-chunk-2", spec.Clients[0].ID)

	// Should have 3 lifecycle windows (0-600, 600-1200, 1200-1800)
	require.NotNil(t, spec.Clients[0].Lifecycle)
	assert.Len(t, spec.Clients[0].Lifecycle.Windows, 3)

	// Verify window bounds
	w1 := spec.Clients[0].Lifecycle.Windows[0]
	assert.Equal(t, int64(0*1e6), w1.StartUs)   // Hour 0:00
	assert.Equal(t, int64(600*1e6), w1.EndUs)   // Hour 0:10

	w2 := spec.Clients[0].Lifecycle.Windows[1]
	assert.Equal(t, int64(600*1e6), w2.StartUs) // Hour 0:10
	assert.Equal(t, int64(1200*1e6), w2.EndUs)  // Hour 0:20

	w3 := spec.Clients[0].Lifecycle.Windows[2]
	assert.Equal(t, int64(1200*1e6), w3.StartUs) // Hour 0:20
	assert.Equal(t, int64(1800*1e6), w3.EndUs)   // Hour 0:30

	// ServeGen temporal parity: aggregate_rate=0 signals absolute rate mode
	// The peak rate (5.2 req/s) is logged but not used for proportional allocation.
	// Each window's trace_rate is used directly as the absolute arrival rate.
	assert.Equal(t, 0.0, spec.AggregateRate)
}

func TestServeGenTimeWindowExtraction_Morning(t *testing.T) {
	// GIVEN ServeGen data spanning midnight and morning
	tmpDir := t.TempDir()

	// Chunk active at Hour 8 (morning window: 28800-30600s)
	// Trace has 10-minute granularity, uses Hour 6 dataset (nearest-preceding)
	trace := "28800,10.5,1.0,Gamma,2.0,0.05\n" +
		"29400,12.3,1.1,Gamma,2.1,0.05\n" +
		"30000,11.8,1.05,Gamma,2.05,0.05\n"
	dataset := map[string]map[string]string{
		// ServeGen datasets at 6-hour intervals (0, 21600, 43200...)
		"0":     {"input_tokens": "{150: 0.4, 250: 0.6}", "output_tokens": "{70: 0.5, 110: 0.5}"},
		"21600": {"input_tokens": "{200: 0.5, 300: 0.5}", "output_tokens": "{80: 0.6, 120: 0.4}"},
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-15-trace.csv"), []byte(trace), 0644))
	d, _ := json.Marshal(dataset)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-15-dataset.json"), d, 0644))

	// WHEN converting with --time morning
	spec, err := ConvertServeGen(tmpDir, "morning")

	// THEN chunk is included with correct windows
	require.NoError(t, err)
	assert.Len(t, spec.Clients, 1)
	assert.Equal(t, "servegen-chunk-15", spec.Clients[0].ID)

	// Should have 3 lifecycle windows (28800-29400, 29400-30000, 30000-30600)
	require.NotNil(t, spec.Clients[0].Lifecycle)
	assert.Len(t, spec.Clients[0].Lifecycle.Windows, 3)

	// ServeGen temporal parity: aggregate_rate=0 signals absolute rate mode
	assert.Equal(t, 0.0, spec.AggregateRate)
}

func TestServeGenTimeWindowExtraction_MultipleChunksParallel(t *testing.T) {
	// GIVEN multiple chunks with overlapping activity in afternoon window
	tmpDir := t.TempDir()

	// Chunk A: active 50400-52200 (full afternoon window: 14:00-14:30)
	// Uses Hour 12 dataset (nearest-preceding)
	traceA := "50400,8.0,0.9,Weibull,1.5,0.02\n" +
		"51000,9.5,1.0,Weibull,1.6,0.02\n" +
		"51600,8.2,0.95,Weibull,1.55,0.02\n"
	datasetA := map[string]map[string]string{
		// ServeGen datasets at 6-hour intervals (0, 21600, 43200...)
		"0":     {"input_tokens": "{90: 1.0}", "output_tokens": "{40: 1.0}"},
		"21600": {"input_tokens": "{95: 1.0}", "output_tokens": "{45: 1.0}"},
		"43200": {"input_tokens": "{100: 1.0}", "output_tokens": "{50: 1.0}"},
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-20-trace.csv"), []byte(traceA), 0644))
	d, _ := json.Marshal(datasetA)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-20-dataset.json"), d, 0644))

	// Chunk B: active 50400 only (first 10 min)
	traceB := "50400,6.2,0.8,Gamma,2.0,0.05\n"
	datasetB := map[string]map[string]string{
		// ServeGen datasets at 6-hour intervals
		"0":     {"input_tokens": "{180: 1.0}", "output_tokens": "{70: 1.0}"},
		"21600": {"input_tokens": "{190: 1.0}", "output_tokens": "{75: 1.0}"},
		"43200": {"input_tokens": "{200: 1.0}", "output_tokens": "{80: 1.0}"},
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-35-trace.csv"), []byte(traceB), 0644))
	d, _ = json.Marshal(datasetB)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-35-dataset.json"), d, 0644))

	// WHEN converting with --time afternoon
	spec, err := ConvertServeGen(tmpDir, "afternoon")

	// THEN both chunks are included
	require.NoError(t, err)
	assert.Len(t, spec.Clients, 2)

	// Verify both chunks present
	chunkIDs := []string{spec.Clients[0].ID, spec.Clients[1].ID}
	assert.Contains(t, chunkIDs, "servegen-chunk-20")
	assert.Contains(t, chunkIDs, "servegen-chunk-35")

	// ServeGen temporal parity: aggregate_rate=0 signals absolute rate mode
	// The peak instantaneous aggregate (14.2 = 8.0 + 6.2 at t=50400) is logged
	// but not used. Each window's trace_rate is the absolute arrival rate.
	assert.Equal(t, 0.0, spec.AggregateRate)
}

func TestServeGenTimeWindowExtraction_NoActiveChunks(t *testing.T) {
	// GIVEN chunks active outside the midnight window
	tmpDir := t.TempDir()

	// Chunk active at Hour 20 (outside all windows)
	trace := "72000,5.0,0.8,Gamma,2.0,0.05\n"
	dataset := map[string]map[string]string{
		// Hour 18 dataset entry (Hour 20 uses nearest-preceding)
		"64800": {"input_tokens": "{100: 1.0}", "output_tokens": "{50: 1.0}"},
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-99-trace.csv"), []byte(trace), 0644))
	d, _ := json.Marshal(dataset)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-99-dataset.json"), d, 0644))

	// WHEN converting with --time midnight
	_, err := ConvertServeGen(tmpDir, "midnight")

	// THEN error because no active chunks in window
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no valid chunks found")
}

func TestServeGenTimeWindowExtraction_InvalidWindow(t *testing.T) {
	// GIVEN valid ServeGen data
	tmpDir := t.TempDir()
	trace := "0,5.0,0.8,Gamma,2.0,0.05\n"
	dataset := map[string]map[string]string{
		"0": {"input_tokens": "{100: 1.0}", "output_tokens": "{50: 1.0}"},
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-0-trace.csv"), []byte(trace), 0644))
	d, _ := json.Marshal(dataset)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-0-dataset.json"), d, 0644))

	// WHEN converting with invalid time window
	_, err := ConvertServeGen(tmpDir, "invalid")

	// THEN error
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid time window")
}

func TestServeGenTimeWindowExtraction_LognormalFitting(t *testing.T) {
	// GIVEN ServeGen data with empirical PDFs
	tmpDir := t.TempDir()

	trace := "0,5.0,0.8,Weibull,1.5,0.02\n"
	dataset := map[string]map[string]string{
		"0": {
			"input_tokens":  "{100: 0.3, 200: 0.4, 300: 0.3}",
			"output_tokens": "{50: 0.5, 100: 0.3, 150: 0.2}",
		},
	}
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-0-trace.csv"), []byte(trace), 0644))
	d, _ := json.Marshal(dataset)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "chunk-0-dataset.json"), d, 0644))

	// WHEN converting with time window
	spec, err := ConvertServeGen(tmpDir, "midnight")

	// THEN distributions are fitted as lognormal at client-level
	require.NoError(t, err)
	require.Len(t, spec.Clients, 1)
	require.NotNil(t, spec.Clients[0].Lifecycle)
	require.Len(t, spec.Clients[0].Lifecycle.Windows, 1)

	client := spec.Clients[0]

	// Since all windows use the same dataset, distributions should be at client-level
	assert.Equal(t, "lognormal", client.InputDist.Type)
	assert.Equal(t, "lognormal", client.OutputDist.Type)

	// Should have mu and sigma parameters
	_, hasMu := client.InputDist.Params["mu"]
	_, hasSigma := client.InputDist.Params["sigma"]
	assert.True(t, hasMu, "input dist should have mu parameter")
	assert.True(t, hasSigma, "input dist should have sigma parameter")

	_, hasMu = client.OutputDist.Params["mu"]
	_, hasSigma = client.OutputDist.Params["sigma"]
	assert.True(t, hasMu, "output dist should have mu parameter")
	assert.True(t, hasSigma, "output dist should have sigma parameter")

	// Windows should NOT have distribution overrides (use client-level)
	w := spec.Clients[0].Lifecycle.Windows[0]
	assert.Nil(t, w.InputDist, "window should not override input dist")
	assert.Nil(t, w.OutputDist, "window should not override output dist")
}
