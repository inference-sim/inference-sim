package saturation

import (
	"path/filepath"
	"strings"
	"testing"
)

// SIM-2, #661: a saturation report that cannot be written is the writer's error, naming the
// path.
func TestWriteCombinedReport_AnUnwritablePathIsReturned(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing", "sat.json")
	err := WriteCombinedReport(path, NewInMemoryCollector(), nil)
	if err == nil || !strings.Contains(err.Error(), path) {
		t.Fatalf("an unwritable saturation report was reported as written: err = %v", err)
	}
}
