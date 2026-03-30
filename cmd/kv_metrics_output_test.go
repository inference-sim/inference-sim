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

// BC-T6: printPerTenantMetrics is a no-op when map is nil.
//
// GIVEN printPerTenantMetrics called with nil map
// WHEN output is captured
// THEN nothing is written to the writer
func TestPrintPerTenantMetrics_Nil_NoOutput(t *testing.T) {
	var buf bytes.Buffer

	printPerTenantMetrics(&buf, nil)

	assert.Empty(t, buf.String())
}

// BC-T7: printPerTenantMetrics emits correct section, sorted tenant lines, and Jain index.
//
// GIVEN a map with two tenants "alice" and "bob" with near-equal token totals
// WHEN printPerTenantMetrics is called
// THEN output contains the section header, "alice" before "bob" (lexicographic),
//
//	each line contains request count and token total,
//	and a Jain Fairness Index line appears last with value >= 0.99
func TestPrintPerTenantMetrics_TwoTenants_CorrectOutput(t *testing.T) {
	var buf bytes.Buffer
	perTenant := map[string]*cluster.TenantMetrics{
		"alice": {TenantID: "alice", CompletedRequests: 50, TotalTokensServed: 12500},
		"bob":   {TenantID: "bob", CompletedRequests: 50, TotalTokensServed: 12480},
	}

	printPerTenantMetrics(&buf, perTenant)

	output := buf.String()
	assert.Contains(t, output, "=== Per-Tenant Metrics ===", "section header must be present")
	assert.Contains(t, output, "alice", "alice must appear in output")
	assert.Contains(t, output, "bob", "bob must appear in output")
	assert.Contains(t, output, "requests=50", "request count must appear")
	assert.Contains(t, output, "tokens=12500", "alice token total must appear in output")
	assert.Contains(t, output, "Jain Fairness Index:", "Jain index line must be present")

	// alice must appear before bob (lexicographic order, R2/INV-6)
	aliceIdx := bytes.Index([]byte(output), []byte("alice"))
	bobIdx := bytes.Index([]byte(output), []byte("bob"))
	assert.True(t, aliceIdx < bobIdx, "tenants must be listed in lexicographic order")

	// Jain line must appear after per-tenant lines
	jainIdx := bytes.Index([]byte(output), []byte("Jain Fairness Index:"))
	assert.True(t, jainIdx > bobIdx, "Jain index line must appear after per-tenant lines")
}

// BC-T8: printPerTenantMetrics with a single-tenant map prints the section (does not suppress).
//
// GIVEN printPerTenantMetrics called with a one-entry map {TenantID: "alice"}
// WHEN output is captured
// THEN the section header is present, "alice" appears, and Jain Fairness Index: 1.0000 is present.
// NOTE: This codifies the intentional divergence from printPerSLOMetrics, which suppresses
// single-class output via a len<=1 guard. printPerTenantMetrics must NOT add that guard.
func TestPrintPerTenantMetrics_SingleTenant_PrintsSection(t *testing.T) {
	var buf bytes.Buffer
	perTenant := map[string]*cluster.TenantMetrics{
		"alice": {TenantID: "alice", CompletedRequests: 5, TotalTokensServed: 1000},
	}

	printPerTenantMetrics(&buf, perTenant)

	output := buf.String()
	assert.Contains(t, output, "=== Per-Tenant Metrics ===", "section header must be present for single tenant")
	assert.Contains(t, output, "alice", "tenant name must appear")
	assert.Contains(t, output, "Jain Fairness Index: 1.0000", "Jain=1.0 must be present for single tenant")
}
