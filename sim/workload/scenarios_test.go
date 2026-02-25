package workload

import (
	"testing"
)

func TestScenarios_ValidateUnderV2Rules(t *testing.T) {
	// Verify all built-in scenarios pass v2 validation after auto-upgrade
	scenarios := []struct {
		name string
		spec *WorkloadSpec
	}{
		{"BurstyTraffic", ScenarioBurstyTraffic(42, 10.0)},
		{"UnfairTenants", ScenarioUnfairTenants(42, 10.0)},
		{"PrefixHeavy", ScenarioPrefixHeavy(42, 10.0)},
		{"MixedSLO", ScenarioMixedSLO(42, 10.0)},
	}
	for _, tc := range scenarios {
		t.Run(tc.name, func(t *testing.T) {
			UpgradeV1ToV2(tc.spec)
			if err := tc.spec.Validate(); err != nil {
				t.Errorf("scenario %s failed validation after upgrade: %v", tc.name, err)
			}
		})
	}
}

func TestScenarios_UseV2TierNames(t *testing.T) {
	// Verify scenarios use v2 tier names directly (no upgrade needed)
	validV2Tiers := map[string]bool{
		"": true, "critical": true, "standard": true, "sheddable": true, "batch": true, "background": true,
	}

	scenarios := []*WorkloadSpec{
		ScenarioBurstyTraffic(42, 10.0),
		ScenarioUnfairTenants(42, 10.0),
		ScenarioPrefixHeavy(42, 10.0),
		ScenarioMixedSLO(42, 10.0),
	}
	for _, spec := range scenarios {
		for _, c := range spec.Clients {
			if !validV2Tiers[c.SLOClass] {
				t.Errorf("client %q uses non-v2 SLO class %q", c.ID, c.SLOClass)
			}
		}
	}
}
