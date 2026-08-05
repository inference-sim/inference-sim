// sim/saturation/replay.go
package saturation

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// CombinedReport is the on-disk shape of a saturation trace: one JSON object
// with a "trace" array of per-event verdicts. It is deliberately a single-key
// object (not a bare array) so #1519's bank can add sibling keys without a
// format break.
type CombinedReport struct {
	Trace []TraceRecord `json:"trace"`
}

// ReplayOneDetector streams one detector over completed request metrics and
// records its verdict after every event to the sink. It is the single uniform
// loop the whole feature relies on: every detector is a streaming detector
// (#1515), so there is no per-detector special case.
//
// Each request contributes two events — an Arrival at its arrival time and a
// Completion at arrival+E2E — ordered deterministically by
// (timestamp, event-type, request-id). The detector is Reset() first so a
// detector reused across the run/observe legs starts from a clean state.
//
// Input adapters (run/replay from the sim, observe from the real server) all
// feed this same loop; the only difference is the []sim.RequestMetrics source.
// Zero requests produce zero events and thus an empty (but valid) trace.
func ReplayOneDetector(detector Detector, requests []sim.RequestMetrics, sink TraceSink) {
	detector.Reset()

	events := make([]Event, 0, 2*len(requests))
	for _, r := range requests {
		arrivalUs := int64(r.ArrivedAt * 1e6)   // seconds → µs
		completionUs := arrivalUs + int64(r.E2E*1e3) // + E2E (ms → µs)
		events = append(events,
			Event{
				Timestamp: arrivalUs,
				Type:      Arrival,
				RequestID: r.ID,
			},
			Event{
				Timestamp: completionUs,
				Type:      Completion,
				RequestID: r.ID,
				LatencyMs: r.E2E,
			},
		)
	}

	// Deterministic order: (timestamp, event-type, request-id). Arrival (0) sorts
	// before Completion (1) at an equal timestamp; request-id breaks any remaining
	// tie. sort.Slice is not stable, but the three-key comparator is a total order
	// (request-ids are unique per request, and a request's own arrival precedes its
	// completion by construction), so the result is fully determined.
	sort.Slice(events, func(i, j int) bool {
		if events[i].Timestamp != events[j].Timestamp {
			return events[i].Timestamp < events[j].Timestamp
		}
		if events[i].Type != events[j].Type {
			return events[i].Type < events[j].Type
		}
		return events[i].RequestID < events[j].RequestID
	})

	name := detector.Name()
	for _, e := range events {
		detector.Observe(e)
		sink.Record(e.Timestamp, name, detector.Detect())
	}
	sink.Close()
}

// WriteCombinedReport serializes the collected verdicts as a {"trace":[...]}
// JSON object to path. Map keys inside each Result's Signals are sorted by
// encoding/json, so two identical runs produce byte-identical files (INV-6).
// A collector with no records writes {"trace":[]} — valid JSON, not an error.
func WriteCombinedReport(path string, collector *InMemoryCollector) error {
	records := collector.Records()
	if records == nil {
		records = []TraceRecord{}
	}
	report := CombinedReport{Trace: records}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal saturation trace: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write saturation trace %s: %w", path, err)
	}
	return nil
}

// ValidateReportPath checks up front that path is writable, so an unwritable
// destination fails fast (before the simulation runs) rather than after. It
// opens the path for writing; if the file did not already exist, the probe file
// it creates is removed so a later Fatalf can't leave a confusing 0-byte
// artifact. An empty path is a no-op.
//
// This is a fast-fail convenience, not a guarantee — a standard TOCTOU window
// remains between validation and the final WriteCombinedReport (permissions or
// disk state can change), whose own error is still surfaced by the caller. The
// point is to catch the common misconfiguration (bad dir, no permission) early.
func ValidateReportPath(path string) error {
	if path == "" {
		return nil
	}
	_, statErr := os.Stat(path)
	preexisting := statErr == nil

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("saturation report path %s is not writable: %w", path, err)
	}
	_ = f.Close()

	// Remove the probe file only if we created it (don't touch a pre-existing one).
	if !preexisting {
		_ = os.Remove(path)
	}
	return nil
}
