package workload

import (
	"encoding/json"
	"fmt"
)

// Accessors and checks that only this package's tests call. No binary calls them, so they live
// in a test file (design LIF-7); each is moved here unchanged.

// OTelSessionID extracts the session identifier from a raw OTel trace.
// Prefers the top-level-per-span session_id, then trace_id. Errors if neither
// is present on any span.
func OTelSessionID(raw []byte) (string, error) {
	var tr otelTrace
	if err := json.Unmarshal(raw, &tr); err != nil {
		return "", fmt.Errorf("parsing OTel trace: %w", err)
	}
	return sessionIDFromSpans(tr.Spans)
}

// Terminal is the number of sessions in a terminal state.
func (s SessionStates) Terminal() int {
	return s.Completed + s.Cancelled + s.HorizonInterrupted + s.BudgetExhausted
}

// bufCap returns the capacity of the underlying array as int64 (matching the
// type signature of Len/Slice/Append). Unexported: same-package memory tests
// access it via white-box; production code has no need.
func (s *sessionTokenBuffer) bufCap() int64 {
	return int64(cap(s.b))
}

// hasUniqueSessionIDs is a test helper: true iff all pooled round-0 requests
// carry distinct SessionIDs.
func (d *SessionPoolDriver) hasUniqueSessionIDs() bool {
	seen := make(map[string]struct{}, len(d.queued))
	for _, r := range d.queued {
		if _, dup := seen[r.SessionID]; dup {
			return false
		}
		seen[r.SessionID] = struct{}{}
	}
	return true
}

// IsValidSLOClass reports whether name is a valid v2 SLO class.
// Valid classes: "", "critical", "standard", "sheddable", "batch", "background".
func IsValidSLOClass(name string) bool {
	return validSLOClasses[name]
}
