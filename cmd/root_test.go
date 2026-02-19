package cmd

import (
	"bytes"
	"io"
	"os"
	"testing"
	"time"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
)

func TestSaveResults_MetricsPrintedToStdout(t *testing.T) {
	// GIVEN a Metrics struct with completed requests
	m := sim.NewMetrics()
	m.CompletedRequests = 5
	m.TotalInputTokens = 100
	m.TotalOutputTokens = 50
	m.SimEndedTime = 1_000_000 // 1 second in ticks
	m.RequestTTFTs["r1"] = 100.0
	m.RequestE2Es["r1"] = 500.0
	m.RequestSchedulingDelays["r1"] = 50
	m.AllITLs = []int64{10, 20, 30}

	// Capture stdout
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// WHEN SaveResults is called
	m.SaveResults("test", 1_000_000, 1000, time.Now(), "")

	// Restore stdout and read captured output
	_ = w.Close()
	os.Stdout = old
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	output := buf.String()

	// THEN the metrics JSON MUST appear on stdout (BC-1)
	assert.Contains(t, output, "Simulation Metrics", "metrics header must be on stdout")
	assert.Contains(t, output, "completed_requests", "metrics JSON must be on stdout")
}
