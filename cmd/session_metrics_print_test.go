package cmd

import (
	"bytes"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

func TestPrintSessionMetrics_Nil_NoOutput(t *testing.T) {
	var buf bytes.Buffer
	printSessionMetrics(&buf, nil)
	if buf.Len() != 0 {
		t.Errorf("expected no output for nil SessionMetrics, got: %q", buf.String())
	}
}

func TestPrintSessionMetrics_OutputContainsKeys(t *testing.T) {
	sm := &cluster.SessionMetrics{
		SessionCount:    3,
		TTFTCold:        cluster.NewDistribution([]float64{50.0, 60.0}),
		TTFTWarm:        cluster.NewDistribution([]float64{20.0, 22.0}),
		SessionDuration: cluster.NewDistribution([]float64{700.0, 750.0}),
	}
	var buf bytes.Buffer
	printSessionMetrics(&buf, sm)
	out := buf.String()
	for _, want := range []string{"Session Metrics", "Sessions: 3", "TTFT cold", "TTFT warm", "Session duration"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\nfull output:\n%s", want, out)
		}
	}
}

func TestPrintSessionMetrics_WarmOnly_ColdAbsent(t *testing.T) {
	// When TTFTCold.Count == 0, the cold line must be absent
	sm := &cluster.SessionMetrics{
		SessionCount:    1,
		TTFTCold:        cluster.NewDistribution(nil),
		TTFTWarm:        cluster.NewDistribution([]float64{25.0}),
		SessionDuration: cluster.NewDistribution(nil),
	}
	var buf bytes.Buffer
	printSessionMetrics(&buf, sm)
	out := buf.String()
	if strings.Contains(out, "TTFT cold") {
		t.Errorf("output must not contain 'TTFT cold' when count is 0\nfull output:\n%s", out)
	}
	if !strings.Contains(out, "TTFT warm") {
		t.Errorf("output must contain 'TTFT warm'\nfull output:\n%s", out)
	}
}
