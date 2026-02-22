package sim

import (
	"testing"
)

func TestGenerateWorkloadDistribution_ZeroRate_Panics(t *testing.T) {
	// GIVEN a simulator with requestRate = 0
	// WHEN generateWorkloadDistribution is called
	// THEN it panics instead of entering an infinite loop (#202)
	sim := &Simulator{
		requestRate: 0,
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for zero requestRate, but did not panic")
		}
	}()
	sim.generateWorkloadDistribution()
}

func TestGenerateWorkloadDistribution_NegativeRate_Panics(t *testing.T) {
	// GIVEN a simulator with requestRate = -1
	// WHEN generateWorkloadDistribution is called
	// THEN it panics
	sim := &Simulator{
		requestRate: -1,
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for negative requestRate, but did not panic")
		}
	}()
	sim.generateWorkloadDistribution()
}
