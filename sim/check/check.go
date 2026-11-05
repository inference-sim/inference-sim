// Package check holds one validator per kind of value an input to the simulator carries
// (design SIM-11, X15). Every input that carries a kind calls its validator: a command-line
// flag, a workload specification and a trace file. A value refused through one input is
// therefore refused through every other, with the same rule and a message that names the
// field and the value.
//
// Each validator returns nil or an error whose text begins with the field name it was given.
// The caller names the field as its input spells it: "--gateway-latency" for a flag,
// "router.gateway_latency_us" for a trace-header key, "client[0].think_time_us" for a
// workload-specification key.
package check

import (
	"fmt"
	"math"
)

// DurationUs refuses a negative span of simulated time in microseconds: a delay, a latency, a
// think time, a deadline or a timestamp. Zero is accepted; each input says what zero means
// (no delay, no deadline, the time origin).
func DurationUs(field string, v int64) error {
	if v < 0 {
		return fmt.Errorf("%s: %d µs is negative; a span of simulated time must be >= 0", field, v)
	}
	return nil
}

// DurationMs is DurationUs for an input that states milliseconds.
func DurationMs(field string, v int64) error {
	if v < 0 {
		return fmt.Errorf("%s: %d ms is negative; a span of simulated time must be >= 0", field, v)
	}
	return nil
}

// TokenCount refuses a negative number of tokens. Zero is accepted: a request with no prompt
// is dropped as unservable when it reaches the gateway (sim/llmd), and a zero prefix or output
// length means none.
func TokenCount(field string, v int64) error {
	if v < 0 {
		return fmt.Errorf("%s: %d tokens is negative; a token count must be >= 0", field, v)
	}
	return nil
}

// Count refuses a negative count of things other than tokens (requests, chunks, priority
// levels).
func Count(field string, v int64) error {
	if v < 0 {
		return fmt.Errorf("%s: %d is negative; a count must be >= 0", field, v)
	}
	return nil
}

// Fraction refuses a value outside [0, 1], NaN and the infinities.
func Fraction(field string, v float64) error {
	if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
		return fmt.Errorf("%s: %v is not a fraction; it must be a finite number in [0, 1]", field, v)
	}
	return nil
}

// PositiveFinite refuses a value that is not a finite number greater than zero: a rate, a
// ratio of rates, a distribution parameter that must be positive.
func PositiveFinite(field string, v float64) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("%s: %v is not a finite number", field, v)
	}
	if v <= 0 {
		return fmt.Errorf("%s: %v must be > 0", field, v)
	}
	return nil
}

// LatencyTargetMs refuses a service-level latency target in milliseconds (TTFT, ITL or E2E)
// that is not a finite number greater than zero. An input whose encoding cannot tell an absent
// target from a zero one (a trace header's goodput_slo_targets, whose keys are omitted when
// zero) calls it only for a stated, non-zero value.
func LatencyTargetMs(field string, v float64) error {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return fmt.Errorf("%s: %v ms is not a finite number", field, v)
	}
	if v <= 0 {
		return fmt.Errorf("%s: %v ms must be > 0; a latency target is a positive span of time", field, v)
	}
	return nil
}
