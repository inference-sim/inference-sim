package sim

import (
	"testing"
)

func TestGenerateWorkloadDistribution_ZeroRate_Panics(t *testing.T) {
	// GIVEN a simulator with RequestRate = 0
	// WHEN generateWorkloadDistribution is called
	// THEN it panics instead of entering an infinite loop (#202)
	sim := &Simulator{
		Metrics: &Metrics{RequestRate: 0},
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for zero RequestRate, but did not panic")
		}
	}()
	sim.generateWorkloadDistribution()
}

func TestGenerateWorkloadDistribution_NegativeRate_Panics(t *testing.T) {
	// GIVEN a simulator with RequestRate = -1
	// WHEN generateWorkloadDistribution is called
	// THEN it panics
	sim := &Simulator{
		Metrics: &Metrics{RequestRate: -1},
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for negative RequestRate, but did not panic")
		}
	}()
	sim.generateWorkloadDistribution()
}
