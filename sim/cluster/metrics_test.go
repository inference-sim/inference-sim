package cluster

import (
	"math"
	"testing"
)

// TestDistribution_FromValues_ComputesCorrectStats verifies BC-2.
func TestDistribution_FromValues_ComputesCorrectStats(t *testing.T) {
	tests := []struct {
		name      string
		values    []float64
		wantCount int
		wantMin   float64
		wantMax   float64
		wantMean  float64
	}{
		{
			name:      "single value",
			values:    []float64{100.0},
			wantCount: 1,
			wantMin:   100.0,
			wantMax:   100.0,
			wantMean:  100.0,
		},
		{
			name:      "multiple values",
			values:    []float64{10.0, 20.0, 30.0, 40.0, 50.0},
			wantCount: 5,
			wantMin:   10.0,
			wantMax:   50.0,
			wantMean:  30.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewDistribution(tt.values)
			if d.Count != tt.wantCount {
				t.Errorf("Count: got %d, want %d", d.Count, tt.wantCount)
			}
			if d.Min != tt.wantMin {
				t.Errorf("Min: got %f, want %f", d.Min, tt.wantMin)
			}
			if d.Max != tt.wantMax {
				t.Errorf("Max: got %f, want %f", d.Max, tt.wantMax)
			}
			if d.Mean != tt.wantMean {
				t.Errorf("Mean: got %f, want %f", d.Mean, tt.wantMean)
			}
			// P99 of [10,20,30,40,50] should be close to 50
			if tt.name == "multiple values" && d.P99 < 40.0 {
				t.Errorf("P99: got %f, expected >= 40.0", d.P99)
			}
		})
	}
}

// TestDistribution_EmptyValues_ReturnsZero verifies edge case.
func TestDistribution_EmptyValues_ReturnsZero(t *testing.T) {
	d := NewDistribution([]float64{})
	if d.Count != 0 {
		t.Errorf("Count: got %d, want 0", d.Count)
	}
	if d.Mean != 0 {
		t.Errorf("Mean: got %f, want 0", d.Mean)
	}
}

// suppress unused import
var _ = math.Floor
