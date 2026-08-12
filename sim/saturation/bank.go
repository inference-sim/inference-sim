// sim/saturation/bank.go
package saturation

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

// rosterOrder is the fixed, canonical order of every detector the bank can
// drive (#1519). The bank always fans out — and therefore records — in this
// order regardless of the order names were supplied on the CLI, so
// `--detectors threshold,composite` produces a byte-identical trace to
// `--detectors composite,threshold` and to `--detectors all` restricted to those
// two (INV-6). Adding a detector means adding one entry here.
var rosterOrder = []string{"composite", "threshold", "backlog-drift"}

// AllDetectorNames returns a fresh copy of the full roster in canonical order.
// `--detectors all` expands to exactly this list. The copy prevents callers from
// mutating the shared roster (R8: no exported mutable slice/map escaping).
func AllDetectorNames() []string {
	out := make([]string, len(rosterOrder))
	copy(out, rosterOrder)
	return out
}

// rosterRank maps a detector name to its index in rosterOrder for canonical
// sorting; a name absent from the roster sorts last (it will already have been
// rejected by NewBank before any sort).
func rosterRank(name string) int {
	for i, n := range rosterOrder {
		if n == name {
			return i
		}
	}
	return len(rosterOrder)
}

// Bank holds and drives a roster of streaming detectors over ONE deterministic
// replay of completed request metrics (sourced from run/replay's sim or observe's
// real server), so every selected detector is scored on a byte-identical event
// sequence in a single pass (#1519). It reimplements no Detector method — it only
// multiplexes the streaming replay across N detectors, recording every
// (event, detector) verdict to the shared sink. The collected records are reduced
// to a per-detector stdout label by saturation.ReduceAll in cmd (#1517); the bank
// itself only produces the trace.
type Bank struct {
	detectors []Detector
	sink      TraceSink
}

// NewBank builds a bank driving exactly the named detectors over the shared sink,
// in canonical roster order. names is the already-resolved selection: `all`
// expands to AllDetectorNames() before this call; a comma-list arrives verbatim.
//
// Behavior:
//   - Names are validated, de-duplicated, and re-ordered into rosterOrder, so the
//     output is independent of CLI order and of duplicate entries (INV-6).
//   - An unknown name is a hard error listing the valid names (R1) — never a
//     silent drop.
//   - An empty selection is an error: the bank must drive at least one detector.
//   - Config-block ownership is enforced over the selected SET
//     (checkBlockOwnershipSet): a tuning block whose owning detector is NOT among
//     the selected names is a hard error (R1), mirroring the single-detector
//     path — never a silent drop. `all` selects every owner, so a shared config
//     is fine; a subset that omits a detector whose block was supplied errors.
//   - Each selected detector is then built via buildDetector, which applies the
//     block that belongs to it. Value errors within a selected detector's own
//     block still surface (range/finiteness), never a panic (R6).
func NewBank(names []string, cfg SaturationConfig, sink TraceSink) (*Bank, error) {
	if len(names) == 0 {
		return nil, fmt.Errorf("saturation bank: no detectors selected; valid: %s", strings.Join(rosterOrder, ", "))
	}

	// De-duplicate while validating. Reject unknown names up front (R1).
	seen := make(map[string]bool, len(names))
	unique := make([]string, 0, len(names))
	for _, name := range names {
		if rosterRank(name) == len(rosterOrder) {
			return nil, fmt.Errorf("unknown saturation detector %q; valid: %s", name, strings.Join(rosterOrder, ", "))
		}
		if seen[name] {
			continue
		}
		seen[name] = true
		unique = append(unique, name)
	}

	// Canonical roster order so the trace is independent of CLI argument order.
	sort.Slice(unique, func(i, j int) bool {
		return rosterRank(unique[i]) < rosterRank(unique[j])
	})

	// Reject a tuning block whose owning detector is not in the selection (R1),
	// matching the single-detector path's ownership contract. `all` selects every
	// owner so it passes trivially; the check bites only on a subset that omits a
	// detector whose block the user supplied.
	if err := checkBlockOwnershipSet(unique, cfg); err != nil {
		return nil, err
	}

	detectors := make([]Detector, 0, len(unique))
	for _, name := range unique {
		det, err := buildDetector(name, cfg)
		if err != nil {
			return nil, err
		}
		detectors = append(detectors, det)
	}

	return &Bank{detectors: detectors, sink: sink}, nil
}

// Run replays the completed requests once and fans every event out to every
// detector, recording one verdict per (event, detector) to the sink in roster
// order. It is the bank's sole public driver (the multi-detector analogue of
// ReplayOneDetector). The collected records are read back from the sink and
// reduced to a per-detector stdout label by saturation.ReduceAll (#1517).
//
// The streaming detectors derive rate from the arrival events themselves, so
// there is no total-arrivals parameter (the retired sim.BatchClassifier seam
// carried one that was never read). Zero requests produce zero events and an
// empty (valid) trace. The nil return is retained for symmetry with a future
// fallible driver and for a uniform call shape; it never fails today.
func (b *Bank) Run(requests []sim.RequestMetrics) error {
	for _, det := range b.detectors {
		det.Reset()
	}
	b.replayEvents(buildSortedEvents(requests))
	return nil
}

// replayEvents streams a pre-sorted event sequence through every detector,
// fanning each event out in roster order and recording each detector's verdict.
func (b *Bank) replayEvents(events []Event) {
	for _, e := range events {
		b.fanout(e)
	}
}

// fanout delivers one event to every detector in roster order and records each
// resulting verdict. Detectors are independent (one detector's Observe never
// touches another's state), so the roster order fixes only the interleaving of
// records in the trace, not any detector's own verdict sequence — which is why a
// subset detector's records are byte-identical to its records under `all`.
func (b *Bank) fanout(e Event) {
	for _, det := range b.detectors {
		det.Observe(e)
		b.sink.Record(e.Timestamp, det.Name(), det.Detect())
	}
}

// Close signals end-of-stream to the sink. Separate from Run so a caller can
// replay, read the collected records, and then flush — mirroring how
// ReplayOneDetector closes its sink after the single-detector stream.
func (b *Bank) Close() { b.sink.Close() }
