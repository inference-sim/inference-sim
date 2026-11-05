package check

import (
	"math"
	"strings"
	"testing"
)

// Each validator has its refusal pair (SIM-9): the smallest value it refuses and the closest
// value it accepts, and every refusal names the field and the value (SIM-11).
func TestValidators_RefusalPairs(t *testing.T) {
	cases := []struct {
		name     string
		accepted func(field string) error
		refused  func(field string) error
		value    string
	}{
		{"DurationUs", func(f string) error { return DurationUs(f, 0) }, func(f string) error { return DurationUs(f, -1) }, "-1"},
		{"DurationMs", func(f string) error { return DurationMs(f, 0) }, func(f string) error { return DurationMs(f, -1) }, "-1"},
		{"TokenCount", func(f string) error { return TokenCount(f, 0) }, func(f string) error { return TokenCount(f, -1) }, "-1"},
		{"Count", func(f string) error { return Count(f, 0) }, func(f string) error { return Count(f, -1) }, "-1"},
		{"Fraction low", func(f string) error { return Fraction(f, 0) }, func(f string) error { return Fraction(f, -1e-9) }, "-1e-09"},
		{"Fraction high", func(f string) error { return Fraction(f, 1) }, func(f string) error { return Fraction(f, 1.0000001) }, "1.0000001"},
		{"Fraction NaN", func(f string) error { return Fraction(f, 0.5) }, func(f string) error { return Fraction(f, math.NaN()) }, "NaN"},
		{"PositiveFinite zero", func(f string) error { return PositiveFinite(f, 1e-300) }, func(f string) error { return PositiveFinite(f, 0) }, "0"},
		{"PositiveFinite Inf", func(f string) error { return PositiveFinite(f, math.MaxFloat64) }, func(f string) error { return PositiveFinite(f, math.Inf(1)) }, "+Inf"},
		{"LatencyTargetMs zero", func(f string) error { return LatencyTargetMs(f, 1e-9) }, func(f string) error { return LatencyTargetMs(f, 0) }, "0"},
		{"LatencyTargetMs NaN", func(f string) error { return LatencyTargetMs(f, 1) }, func(f string) error { return LatencyTargetMs(f, math.NaN()) }, "NaN"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if err := c.accepted("the_field"); err != nil {
				t.Errorf("the closest accepted value is refused: %v", err)
			}
			err := c.refused("the_field")
			if err == nil {
				t.Fatal("the smallest refused value is accepted")
			}
			if !strings.HasPrefix(err.Error(), "the_field: ") || !strings.Contains(err.Error(), c.value) {
				t.Errorf("the refusal must begin with the field and name the value %s, got %q", c.value, err)
			}
		})
	}
}
