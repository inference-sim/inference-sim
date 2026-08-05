// sim/saturation/replay_test.go
package saturation

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestReplayOneDetector_EventOrderingAndCount verifies each request contributes
// exactly two events (arrival + completion) and the collector records one verdict
// per event, with arrival preceding completion for the same request.
func TestReplayOneDetector_EventOrderingAndCount(t *testing.T) {
	reqs := []sim.RequestMetrics{
		{ID: "request_1", ArrivedAt: 2.0, E2E: 100}, // out of order to exercise sort
		{ID: "request_0", ArrivedAt: 1.0, E2E: 100},
	}
	det, err := BuildDetector("composite", SaturationConfig{})
	if err != nil {
		t.Fatalf("BuildDetector: %v", err)
	}
	collector := NewInMemoryCollector()
	ReplayOneDetector(det, reqs, collector)

	recs := collector.Records()
	if len(recs) != 4 {
		t.Fatalf("expected 4 verdict records (2 events × 2 requests), got %d", len(recs))
	}
	// Timestamps must be non-decreasing (deterministic (time, type, id) order).
	for i := 1; i < len(recs); i++ {
		if recs[i].Timestamp < recs[i-1].Timestamp {
			t.Errorf("records not time-ordered at %d: %d < %d", i, recs[i].Timestamp, recs[i-1].Timestamp)
		}
	}
	// First event is request_0's arrival at 1.0s = 1_000_000µs.
	if recs[0].Timestamp != 1_000_000 {
		t.Errorf("first event timestamp = %d, want 1000000 (earliest arrival)", recs[0].Timestamp)
	}
}

// TestReplayOneDetector_Deterministic verifies two runs over the same input
// produce identical records (INV-6).
func TestReplayOneDetector_Deterministic(t *testing.T) {
	reqs := []sim.RequestMetrics{
		{ID: "request_0", ArrivedAt: 0, E2E: 100},
		{ID: "request_1", ArrivedAt: 1, E2E: 200},
		{ID: "request_2", ArrivedAt: 2, E2E: 150},
	}
	run := func() []TraceRecord {
		det, _ := BuildDetector("composite", SaturationConfig{})
		c := NewInMemoryCollector()
		ReplayOneDetector(det, reqs, c)
		return c.Records()
	}
	a, b := run(), run()
	if len(a) != len(b) {
		t.Fatalf("record count differs: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].Timestamp != b[i].Timestamp || a[i].Detector != b[i].Detector ||
			a[i].Result.Level != b[i].Result.Level || a[i].Result.Score != b[i].Result.Score ||
			a[i].Result.Confidence != b[i].Result.Confidence {
			t.Errorf("record %d differs: %+v vs %+v", i, a[i], b[i])
		}
	}
}

// TestReplayOneDetector_ResetsDetector verifies the detector is Reset() at the
// start of each replay, so reusing one detector across two legs starts clean.
func TestReplayOneDetector_ResetsDetector(t *testing.T) {
	det, _ := BuildDetector("composite", SaturationConfig{})

	firstReqs := []sim.RequestMetrics{{ID: "request_0", ArrivedAt: 0, E2E: 100}}
	c1 := NewInMemoryCollector()
	ReplayOneDetector(det, firstReqs, c1)

	// Reuse the same detector on a fresh input; if Reset weren't called, the first
	// leg's events would still be accumulated and the counts would differ.
	c2 := NewInMemoryCollector()
	ReplayOneDetector(det, firstReqs, c2)

	if len(c1.Records()) != len(c2.Records()) {
		t.Fatalf("reused detector gave different record counts: %d vs %d", len(c1.Records()), len(c2.Records()))
	}
	// Final verdict must match — a clean reset makes the second leg identical.
	last1 := c1.Records()[len(c1.Records())-1]
	last2 := c2.Records()[len(c2.Records())-1]
	if last1.Result.Level != last2.Result.Level || last1.Result.Score != last2.Result.Score {
		t.Errorf("reused detector not reset: %+v vs %+v", last1.Result, last2.Result)
	}
}

// TestWriteCombinedReport_EmptyInput_WritesEmptyTrace verifies zero records
// writes {"trace":[]} (valid JSON), not an error.
func TestWriteCombinedReport_EmptyInput_WritesEmptyTrace(t *testing.T) {
	c := NewInMemoryCollector()
	path := filepath.Join(t.TempDir(), "empty.json")
	if err := WriteCombinedReport(path, c); err != nil {
		t.Fatalf("WriteCombinedReport: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	got := string(data)
	want := "{\n  \"trace\": []\n}\n"
	if got != want {
		t.Errorf("empty trace output = %q, want %q", got, want)
	}
}

// TestWriteCombinedReport_ByteIdentical verifies two writes of the same records
// are byte-identical (sorted signal keys, INV-6).
func TestWriteCombinedReport_ByteIdentical(t *testing.T) {
	reqs := []sim.RequestMetrics{
		{ID: "request_0", ArrivedAt: 0, E2E: 100},
		{ID: "request_1", ArrivedAt: 1, E2E: 200},
	}
	write := func() []byte {
		det, _ := BuildDetector("composite", SaturationConfig{})
		c := NewInMemoryCollector()
		ReplayOneDetector(det, reqs, c)
		path := filepath.Join(t.TempDir(), "t.json")
		if err := WriteCombinedReport(path, c); err != nil {
			t.Fatalf("write: %v", err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		return data
	}
	a, b := write(), write()
	if string(a) != string(b) {
		t.Errorf("report not byte-identical across runs")
	}
}

// TestValidateReportPath verifies up-front writability checking: an empty path is
// a no-op, a writable path succeeds, an unwritable path errors.
func TestValidateReportPath(t *testing.T) {
	if err := ValidateReportPath(""); err != nil {
		t.Errorf("empty path should be a no-op, got: %v", err)
	}
	ok := filepath.Join(t.TempDir(), "ok.json")
	if err := ValidateReportPath(ok); err != nil {
		t.Errorf("writable path should succeed, got: %v", err)
	}
	// A path inside a non-existent directory is unwritable.
	bad := filepath.Join(t.TempDir(), "nope", "x.json")
	if err := ValidateReportPath(bad); err == nil {
		t.Error("expected error for unwritable path")
	}
}

// TestSinks_NoOpAndCollector verifies the two sink implementations: NoOpSink
// discards, InMemoryCollector preserves order.
func TestSinks_NoOpAndCollector(t *testing.T) {
	noop := NewNoOpSink()
	noop.Record(1, "composite", Result{})
	noop.Close() // must not panic

	c := NewInMemoryCollector()
	c.Record(10, "composite", Result{Score: 0.1})
	c.Record(20, "composite", Result{Score: 0.2})
	c.Close()
	recs := c.Records()
	if len(recs) != 2 || recs[0].Timestamp != 10 || recs[1].Timestamp != 20 {
		t.Errorf("collector did not preserve order: %+v", recs)
	}
}
