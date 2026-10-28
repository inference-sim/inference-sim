// Moved from blis-latency-kernel/rank_width_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/coefficient"

	"github.com/inference-sim/blis-latency-kernel/internal/resolve"
)

// The three coefficients liftCollectiveFloors needs for one (op, dtype, width). A
// width holding fewer than all three cannot price a collective, so measuredWidths
// does not count it.
func tripleFor(op, dtype, chip string, ranks int, floor float64) []coefficient.Entry {
	stem := op + "_" + dtype + "_" + itoa(ranks) + "rank_" + chip
	return []coefficient.Entry{
		{Name: "collective_floor_" + stem, Value: floor},
		{Name: "collective_peak_rate_" + stem, Value: 200},
		{Name: "collective_transition_rate_" + stem, Value: 150},
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// coeffs builds a resolved lookup over entries with no scope, so every entry applies.
func coeffs(t *testing.T, entries []coefficient.Entry) *resolve.Coefficients {
	t.Helper()
	set := &coefficient.Set{Name: "test-set", Coefficients: entries}
	c, err := resolve.Load([]*coefficient.Set{set}, resolve.Scope{})
	if err != nil {
		t.Fatalf("resolve.Load: %v", err)
	}
	return c
}
