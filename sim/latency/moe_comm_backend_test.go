package latency

import "testing"

// TestMoeCommFamilyFor verifies the backend-name → volume-family mapping for all
// seven vLLM MoE all-to-all backends (mirroring VLLM_ALL2ALL_BACKEND at
// vllm@f6ec81c7 vllm/envs.py:186), plus the unknown-name error path.
//
// The two families differ in the PHYSICAL communication volume they move (#1419):
//   - all-gather family (naive, allgather_reducescatter): dispatch all-gathers /
//     combine reduce-scatters dense hidden states — volume ∝ tokens·hidden, NO top_k.
//   - all2all family (pplx, deepep_*, mori, flashinfer_all2allv): each token routes
//     to its top_k expert-owning ranks — volume ∝ tokens·top_k·hidden.
func TestMoeCommFamilyFor(t *testing.T) {
	tests := []struct {
		name       string
		wantFamily moeCommFamily
	}{
		{"naive", commFamilyAllGather},
		{"allgather_reducescatter", commFamilyAllGather},
		{"pplx", commFamilyAll2All},
		{"deepep_high_throughput", commFamilyAll2All},
		{"deepep_low_latency", commFamilyAll2All},
		{"mori", commFamilyAll2All},
		{"flashinfer_all2allv", commFamilyAll2All},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := moeCommFamilyFor(tc.name)
			if err != nil {
				t.Fatalf("moeCommFamilyFor(%q) returned error: %v", tc.name, err)
			}
			if got != tc.wantFamily {
				t.Errorf("moeCommFamilyFor(%q) = %v, want %v", tc.name, got, tc.wantFamily)
			}
		})
	}
}

// TestMoeCommFamilyFor_Unknown verifies that an unrecognized backend name is a
// hard error (R1: no silent fallback that would mask a typo'd CLI flag).
func TestMoeCommFamilyFor_Unknown(t *testing.T) {
	if _, err := moeCommFamilyFor("bogus-backend"); err == nil {
		t.Fatal("moeCommFamilyFor(\"bogus-backend\") must return an error, got nil")
	}
}

// TestDefaultMoECommBackend_IsValid locks in that the default backend name is
// itself a member of the valid set and resolves to the all-gather family (it is
// vLLM's general-purpose default, allgather_reducescatter).
func TestDefaultMoECommBackend_IsValid(t *testing.T) {
	fam, err := moeCommFamilyFor(DefaultMoECommBackend)
	if err != nil {
		t.Fatalf("DefaultMoECommBackend %q must be a valid backend: %v", DefaultMoECommBackend, err)
	}
	if fam != commFamilyAllGather {
		t.Errorf("DefaultMoECommBackend %q family = %v, want commFamilyAllGather", DefaultMoECommBackend, fam)
	}

	// Every name advertised in ValidMoECommBackends must resolve without error —
	// otherwise the CLI validator and the family mapper would disagree.
	for _, name := range ValidMoECommBackends {
		if _, err := moeCommFamilyFor(name); err != nil {
			t.Errorf("ValidMoECommBackends entry %q does not resolve: %v", name, err)
		}
	}
}
