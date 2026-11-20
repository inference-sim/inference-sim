package workload

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/trace"
)

// The loader refuses a trace whose phases and phase column disagree, and a phased trace
// round-trips through the export and the load.
func TestLoadTraceV2_PhasesAndThePhaseColumnMustAgree(t *testing.T) {
	rec := func(id int, phase string) TraceRecord {
		return TraceRecord{RequestID: id, InputTokens: 4, OutputTokens: 2, Status: trace.StatusOK, Phase: phase}
	}
	hdr := func(phases ...trace.Phase) *TraceHeader {
		return &TraceHeader{Version: TraceVersion, TimeUnit: "microseconds", Mode: trace.ModeConverted, Phases: phases}
	}
	for _, c := range []struct {
		name    string
		header  *TraceHeader
		records []TraceRecord
		want    string
	}{
		{"a phase no request names", hdr(trace.Phase{Name: "a"}, trace.Phase{Name: "b"}), []TraceRecord{rec(0, "a")}, "names no request"},
		{"a request in an unknown phase", hdr(trace.Phase{Name: "a"}), []TraceRecord{rec(0, "a"), rec(1, "z")}, "not a phase of the header"},
		{"a phase label without a phase list", hdr(), []TraceRecord{rec(0, "a")}, "lists no phases"},
		{"a request without a phase", hdr(trace.Phase{Name: "a"}), []TraceRecord{rec(0, "a"), rec(1, "")}, "every request names one"},
		{"", hdr(trace.Phase{Name: "a"}, trace.Phase{Name: "b"}), []TraceRecord{rec(0, "a"), rec(1, "b")}, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			h, d := filepath.Join(dir, "t.yaml"), filepath.Join(dir, "t.csv")
			if err := ExportTraceV2(c.header, c.records, h, d); err != nil {
				t.Fatal(err)
			}
			tr, err := LoadTraceV2(h, d)
			switch {
			case c.want == "" && err != nil:
				t.Fatalf("a consistent phased trace is refused: %v", err)
			case c.want == "" && (len(tr.Header.Phases) != 2 || tr.Records[1].Phase != "b"):
				t.Errorf("phases did not survive the export and load: %+v %+v", tr.Header.Phases, tr.Records)
			case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
				t.Errorf("got %v, want an error containing %q", err, c.want)
			}

		})
	}
}
