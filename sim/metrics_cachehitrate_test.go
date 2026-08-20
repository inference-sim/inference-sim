package sim

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestEmitOutput_CacheHitRate_FileOnly verifies the aggregate cache_hit_rate (#1583)
// is written to the --metrics-path file but NOT to the stdout MetricsOutput shape
// (BC-5, BC-9, INV-6). BuildOutput (the value marshaled to stdout) must leave the
// field nil so omitempty drops it; the file marshal populates it.
func TestEmitOutput_CacheHitRate_FileOnly(t *testing.T) {
	m := NewMetrics()
	m.CacheHitRate = 0.625

	// The stdout shape comes from BuildOutput — the field must be absent there.
	out := m.BuildOutput("cluster")
	if out.CacheHitRate != nil {
		t.Fatalf("BuildOutput (stdout shape) must not set cache_hit_rate; got %v", *out.CacheHitRate)
	}
	stdoutJSON, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(stdoutJSON), "cache_hit_rate") {
		t.Errorf("stdout MetricsOutput must not contain cache_hit_rate:\n%s", stdoutJSON)
	}

	// The --metrics-path file must contain the value.
	dir := t.TempDir()
	fpath := filepath.Join(dir, "metrics.json")
	if err := m.EmitOutput(out, fpath); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(fpath)
	if err != nil {
		t.Fatal(err)
	}
	var fileOut MetricsOutput
	if err := json.Unmarshal(data, &fileOut); err != nil {
		t.Fatal(err)
	}
	if fileOut.CacheHitRate == nil {
		t.Fatal("metrics-path file must contain cache_hit_rate")
	}
	if *fileOut.CacheHitRate != 0.625 {
		t.Errorf("file cache_hit_rate = %v, want 0.625", *fileOut.CacheHitRate)
	}
}

// TestEmitOutput_CacheHitRate_ZeroWrittenToFile verifies a zero hit rate is still
// written to the file (a valid observation), since EmitOutput sets the pointer
// unconditionally in the file branch.
func TestEmitOutput_CacheHitRate_ZeroWrittenToFile(t *testing.T) {
	m := NewMetrics()
	m.CacheHitRate = 0.0
	out := m.BuildOutput("cluster")
	dir := t.TempDir()
	fpath := filepath.Join(dir, "metrics.json")
	if err := m.EmitOutput(out, fpath); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(fpath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "cache_hit_rate") {
		t.Errorf("file must record cache_hit_rate even when 0:\n%s", data)
	}
}
