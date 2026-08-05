package sim

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestLoRABundleTriple_KnownExpands pins the shipped "lora-affinity" bundle to its
// exact {routing, eviction, creation} expansion (B-7, DD-B7-3). This is the contract
// the CLI relies on when it applies bundle knobs, so a silent change to any of the
// three seam names would be a behavior change and must break this test.
func TestLoRABundleTriple_KnownExpands(t *testing.T) {
	triple, ok := LoRABundleTriple("lora-affinity")
	assert.True(t, ok, "lora-affinity must be a registered bundle")
	assert.Equal(t, PolicyTriple{
		Routing:  "route-to-holder",
		Eviction: "rank-aware",
		Creation: "pre-placement",
	}, triple)
}

// TestLoRABundleTriple_UnknownAndEmpty verifies the fail-fast contract: an unknown
// name and the empty string both return ok=false, so a caller never silently applies
// a zero triple. The empty string is deliberately NOT a bundle — it means "no bundle
// selected", distinct from the seam validity maps where "" denotes the baseline.
func TestLoRABundleTriple_UnknownAndEmpty(t *testing.T) {
	for _, name := range []string{"", "bogus", "LoRA-Affinity"} {
		_, ok := LoRABundleTriple(name)
		assert.Falsef(t, ok, "LoRABundleTriple(%q) should report ok=false", name)
		assert.Falsef(t, IsValidLoRABundle(name), "IsValidLoRABundle(%q) should be false", name)
	}
}

// TestLoRAStrategyBundles_AllNamesRegistered is a consistency law: every seam name
// referenced by every registered bundle MUST resolve in that seam's own validity set
// (routing via IsValidRoutingPolicy, eviction/creation via the hook-backed
// ValidEvictionPolicyNames/ValidCreationPolicyNames registered by sim/lora's init,
// reached here through the lora_import_test.go blank import). This catches a bundle
// that references a policy that was renamed or never registered — a fatal at run time
// otherwise. It is intentionally implementation-agnostic: it iterates whatever bundles
// exist, so a newly added bundle is checked automatically.
func TestLoRAStrategyBundles_AllNamesRegistered(t *testing.T) {
	evictionOK := setOf(ValidEvictionPolicyNames())
	creationOK := setOf(ValidCreationPolicyNames())
	// Guard: the hooks must actually be registered, else the test is vacuous.
	assert.NotEmpty(t, evictionOK, "eviction validity hook not registered (missing lora blank import?)")
	assert.NotEmpty(t, creationOK, "creation validity hook not registered (missing lora blank import?)")

	for _, name := range ValidLoRABundleNames() {
		triple, ok := LoRABundleTriple(name)
		assert.Truef(t, ok, "ValidLoRABundleNames returned %q but LoRABundleTriple missed it", name)
		assert.Truef(t, IsValidRoutingPolicy(triple.Routing),
			"bundle %q routing %q is not a valid routing policy", name, triple.Routing)
		assert.Truef(t, evictionOK[triple.Eviction],
			"bundle %q eviction %q is not a registered eviction policy", name, triple.Eviction)
		assert.Truef(t, creationOK[triple.Creation],
			"bundle %q creation %q is not a registered creation policy", name, triple.Creation)
	}
}

// TestPolicyProvenance_JSONOmittedWhenNil verifies the omit-when-inert contract
// (DD-B7-4, INV-6): a MetricsOutput with a nil PolicyProvenance marshals WITHOUT the
// "policy_provenance" key, so an all-baseline / adapter-blind run is byte-identical to
// pre-B-7. This is the JSON-level law behind computeLoRAProvenance returning nil.
func TestPolicyProvenance_JSONOmittedWhenNil(t *testing.T) {
	data, err := json.Marshal(MetricsOutput{})
	assert.NoError(t, err)
	assert.NotContains(t, string(data), "policy_provenance",
		"nil PolicyProvenance must omit the key entirely")
}

// TestPolicyProvenance_JSONPresentWhenSet verifies the SC-006 reproducibility
// contract: when set, the provenance record round-trips all three effective seam
// names, so a result is reproducible from its record alone.
func TestPolicyProvenance_JSONPresentWhenSet(t *testing.T) {
	out := MetricsOutput{PolicyProvenance: &PolicyTriple{
		Routing:  "route-to-holder",
		Eviction: "rank-aware",
		Creation: "pre-placement",
	}}
	data, err := json.Marshal(out)
	assert.NoError(t, err)
	s := string(data)
	assert.Contains(t, s, "policy_provenance")

	var round struct {
		Provenance *PolicyTriple `json:"policy_provenance"`
	}
	assert.NoError(t, json.Unmarshal(data, &round))
	assert.Equal(t, out.PolicyProvenance, round.Provenance)
}

func setOf(names []string) map[string]bool {
	m := make(map[string]bool, len(names))
	for _, n := range names {
		m[n] = true
	}
	return m
}
