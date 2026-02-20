package cmd

import (
	"bytes"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/stretchr/testify/assert"
)

func TestPrintKVCacheMetrics_Nonzero_PrintsSection(t *testing.T) {
	// GIVEN nonzero KV cache metrics
	var buf bytes.Buffer

	// WHEN we print to the buffer
	printKVCacheMetrics(&buf, 0.05, 0.75, 0.02)

	// THEN the output must contain the KV cache section
	output := buf.String()
	assert.Contains(t, output, "=== KV Cache Metrics ===")
	assert.Contains(t, output, "Preemption Rate:")
	assert.Contains(t, output, "Cache Hit Rate:")
	assert.Contains(t, output, "KV Thrashing Rate:")
}

func TestPrintKVCacheMetrics_AllZero_NoOutput(t *testing.T) {
	// GIVEN all-zero KV cache metrics
	var buf bytes.Buffer

	// WHEN we print to the buffer
	printKVCacheMetrics(&buf, 0, 0, 0)

	// THEN no output
	assert.Empty(t, buf.String())
}

func TestPrintPerSLOMetrics_MultipleClasses_PrintsSorted(t *testing.T) {
	// GIVEN per-SLO distributions with multiple classes
	var buf bytes.Buffer
	sloMetrics := map[string]*cluster.SLOMetrics{
		"batch": {
			TTFT: cluster.Distribution{Mean: 100, P99: 200, Count: 10},
			E2E:  cluster.Distribution{Mean: 500, P99: 800, Count: 10},
		},
		"realtime": {
			TTFT: cluster.Distribution{Mean: 50, P99: 80, Count: 5},
			E2E:  cluster.Distribution{Mean: 200, P99: 300, Count: 5},
		},
	}

	// WHEN we print per-SLO metrics
	printPerSLOMetrics(&buf, sloMetrics)

	// THEN output must contain the section and classes in sorted order
	output := buf.String()
	assert.Contains(t, output, "=== Per-SLO Metrics ===")
	// "batch" must appear before "realtime" (alphabetical)
	batchIdx := bytes.Index([]byte(output), []byte("batch"))
	realtimeIdx := bytes.Index([]byte(output), []byte("realtime"))
	assert.True(t, batchIdx < realtimeIdx, "SLO classes must be sorted alphabetically")
}

func TestPrintPerSLOMetrics_SingleClass_NoOutput(t *testing.T) {
	// GIVEN per-SLO distributions with only one class
	var buf bytes.Buffer
	sloMetrics := map[string]*cluster.SLOMetrics{
		"default": {
			TTFT: cluster.Distribution{Mean: 100, P99: 200, Count: 10},
			E2E:  cluster.Distribution{Mean: 500, P99: 800, Count: 10},
		},
	}

	// WHEN we print per-SLO metrics
	printPerSLOMetrics(&buf, sloMetrics)

	// THEN no output (single class = no differentiation)
	assert.Empty(t, buf.String())
}
