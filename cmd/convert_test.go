package cmd

import (
	"testing"

	"blis/sim/workload"
)

// TestConvertServeGenCmd_NewFlagsRegistered verifies BC-12: new flags
// --window-duration-seconds and --drain-timeout-seconds are registered
// with correct defaults.
func TestConvertServeGenCmd_NewFlagsRegistered(t *testing.T) {
	windowFlag := convertServeGenCmd.Flags().Lookup("window-duration-seconds")
	if windowFlag == nil {
		t.Fatal("flag --window-duration-seconds not found")
	}
	if windowFlag.DefValue != "600" {
		t.Errorf("--window-duration-seconds default: got %q, want \"600\"", windowFlag.DefValue)
	}

	drainFlag := convertServeGenCmd.Flags().Lookup("drain-timeout-seconds")
	if drainFlag == nil {
		t.Fatal("flag --drain-timeout-seconds not found")
	}
	if drainFlag.DefValue != "180" {
		t.Errorf("--drain-timeout-seconds default: got %q, want \"180\"", drainFlag.DefValue)
	}
}

// TestConvertServeGenCmd_TimeFilterFlag verifies that --time flag is optional
// and accepts valid period values (midnight, morning, afternoon).
func TestConvertServeGenCmd_TimeFilterFlag(t *testing.T) {
	timeFlag := convertServeGenCmd.Flags().Lookup("time")
	if timeFlag == nil {
		t.Fatal("flag --time not found")
	}
	if timeFlag.DefValue != "" {
		t.Errorf("--time default: got %q, want \"\" (empty/optional)", timeFlag.DefValue)
	}

	// Verify flag usage text mentions valid values
	usage := timeFlag.Usage
	if usage == "" {
		t.Error("--time flag should have usage text")
	}
	// Check that usage mentions the valid period names
	expectedSubstrings := []string{"midnight", "morning", "afternoon"}
	for _, substr := range expectedSubstrings {
		found := false
		for i := 0; i <= len(usage)-len(substr); i++ {
			if usage[i:i+len(substr)] == substr {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("--time usage should mention %q, got: %s", substr, usage)
		}
	}
}

// TestFilterCohortsByPeriod verifies that filterCohortsByPeriod correctly
// filters cohorts by ID prefix and preserves spec metadata.
func TestFilterCohortsByPeriod(t *testing.T) {
	// Create a spec with cohorts from all three periods and all fields populated
	spec := &workload.WorkloadSpec{
		Version:       "2",
		Seed:          42,
		AggregateRate: 0,
		Category:      "test",
		Horizon:       100000000,
		NumRequests:   1000,
		Cohorts: []workload.CohortSpec{
			{ID: "midnight-background", Population: 5},
			{ID: "midnight-critical", Population: 6},
			{ID: "morning-background", Population: 5},
			{ID: "morning-critical", Population: 6},
			{ID: "afternoon-background", Population: 5},
			{ID: "afternoon-critical", Population: 6},
		},
	}

	tests := []struct {
		period       string
		wantCount    int
		wantFirstID  string
		wantSecondID string
	}{
		{"midnight", 2, "midnight-background", "midnight-critical"},
		{"morning", 2, "morning-background", "morning-critical"},
		{"afternoon", 2, "afternoon-background", "afternoon-critical"},
	}

	for _, tt := range tests {
		t.Run(tt.period, func(t *testing.T) {
			filtered := filterCohortsByPeriod(spec, tt.period)

			// Check all spec metadata preserved
			if filtered.Version != spec.Version {
				t.Errorf("Version: got %q, want %q", filtered.Version, spec.Version)
			}
			if filtered.Seed != spec.Seed {
				t.Errorf("Seed: got %d, want %d", filtered.Seed, spec.Seed)
			}
			if filtered.AggregateRate != spec.AggregateRate {
				t.Errorf("AggregateRate: got %f, want %f", filtered.AggregateRate, spec.AggregateRate)
			}
			if filtered.Category != spec.Category {
				t.Errorf("Category: got %q, want %q", filtered.Category, spec.Category)
			}
			if filtered.Horizon != spec.Horizon {
				t.Errorf("Horizon: got %d, want %d", filtered.Horizon, spec.Horizon)
			}
			if filtered.NumRequests != spec.NumRequests {
				t.Errorf("NumRequests: got %d, want %d", filtered.NumRequests, spec.NumRequests)
			}

			// Check cohort count
			if len(filtered.Cohorts) != tt.wantCount {
				t.Errorf("Cohort count: got %d, want %d", len(filtered.Cohorts), tt.wantCount)
			}

			// Check cohort IDs match expected period
			if len(filtered.Cohorts) > 0 && filtered.Cohorts[0].ID != tt.wantFirstID {
				t.Errorf("First cohort ID: got %q, want %q", filtered.Cohorts[0].ID, tt.wantFirstID)
			}
			if len(filtered.Cohorts) > 1 && filtered.Cohorts[1].ID != tt.wantSecondID {
				t.Errorf("Second cohort ID: got %q, want %q", filtered.Cohorts[1].ID, tt.wantSecondID)
			}

			// Verify all cohorts have correct prefix
			prefix := tt.period + "-"
			for _, cohort := range filtered.Cohorts {
				if len(cohort.ID) < len(prefix) || cohort.ID[:len(prefix)] != prefix {
					t.Errorf("Cohort ID %q does not start with prefix %q", cohort.ID, prefix)
				}
			}
		})
	}
}
