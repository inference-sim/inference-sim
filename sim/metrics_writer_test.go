package sim

import (
	"path/filepath"
	"strings"
	"testing"
)

// SIM-2, #661: the per-request rows file (--metrics-path) that cannot be written is the
// emitter's error, naming the failure, not a logged warning.
func TestWriteMetricsFile_AnUnwritablePathIsReturned(t *testing.T) {
	m := NewMetrics()
	path := filepath.Join(t.TempDir(), "missing", "metrics.json")
	err := m.WriteMetricsFile(m.BuildOutput("cluster"), path)
	if err == nil || !strings.Contains(err.Error(), "writing JSON file") {
		t.Fatalf("an unwritable metrics file was reported as written: err = %v", err)
	}
}
