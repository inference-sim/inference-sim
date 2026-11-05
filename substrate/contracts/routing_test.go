package contracts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
)

// A routing profile set loads strictly, validates through the bundle, and a scenario that
// names one resolves it against the registry's routing namespace.
func TestRoutingProfileSetLoadsAndResolves(t *testing.T) {
	good := write(t, "r.yaml", `kind: RoutingProfileSet
name: capture
role: sensitivity
profiles:
  - model: m
    category: chat
    phase: decode
    placement: linear
    expert_parallel: 8
    method: measured
    provenance: {source: "hf://x@rev", tier: 2}
    points:
      - {batch_tokens: 64, token_imbalance: 1.61, expert_imbalance: 0.77}
`)
	s, err := LoadRoutingProfileSet(good)
	if err != nil {
		t.Fatal(err)
	}
	if rep := Validate(Bundle{RoutingProfiles: []*routing.ProfileSet{s}}); !rep.OK() {
		t.Fatalf("a valid set failed validation:\n%v", rep.Problems())
	}
	s.Profiles[0].Points[0].TokenImbalance = 0.5
	if rep := Validate(Bundle{RoutingProfiles: []*routing.ProfileSet{s}}); rep.OK() {
		t.Error("the bundle did not validate its routing sets")
	}
	typo := write(t, "t.yaml", "kind: RoutingProfileSet\nname: s\nrole: default\nprofiles:\n  - model: m\n    token_imbalence: 2\n")
	if _, err := LoadRoutingProfileSet(typo); err == nil {
		t.Error("a misspelled key loaded silently")
	}

	reg := t.TempDir()
	if err := os.MkdirAll(filepath.Join(reg, "routing"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(reg, "routing", "capture.yaml"), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	sc := scenarioRef()
	sc.Routing = &routing.Pin{Profiles: "not-a-set"}
	msg := ValidateAgainstCatalog(Bundle{Scenario: sc}, catalogFixtures, reg).Field.Error()
	if !strings.Contains(msg, "references.routing.profiles") {
		t.Errorf("a missing routing set was not reported:\n%s", msg)
	}
	sc.Routing = &routing.Pin{}
	if rep := Validate(Bundle{Scenario: sc}); rep.OK() {
		t.Error("an empty routing pin was accepted")
	}
}
