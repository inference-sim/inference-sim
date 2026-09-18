package cmd

import (
	"math"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestComposeLoRAScorer_AppendsToDefaultProfile verifies that with no explicit
// profile the lora-affinity scorer is composed onto the default dimensions rather
// than replacing them (#1469, T039).
func TestComposeLoRAScorer_AppendsToDefaultProfile(t *testing.T) {
	got, err := composeLoRAScorer(nil, 2.0)
	require.NoError(t, err)

	want := append(sim.DefaultScorerConfigs(), sim.ScorerConfig{Name: "lora-affinity", Weight: 2.0})
	assert.Equal(t, want, got, "empty base ⇒ default profile + lora-affinity")
}

// TestComposeLoRAScorer_AppendsToExplicitProfile verifies composition onto a
// user-supplied --routing-scorers profile preserves the existing dimensions.
func TestComposeLoRAScorer_AppendsToExplicitProfile(t *testing.T) {
	base := []sim.ScorerConfig{{Name: "queue-depth", Weight: 1.0}}
	got, err := composeLoRAScorer(base, 3.0)
	require.NoError(t, err)

	assert.Equal(t, []sim.ScorerConfig{
		{Name: "queue-depth", Weight: 1.0},
		{Name: "lora-affinity", Weight: 3.0},
	}, got)
	// The base slice must not be mutated (no aliasing of its backing array).
	assert.Equal(t, []sim.ScorerConfig{{Name: "queue-depth", Weight: 1.0}}, base,
		"composeLoRAScorer must not mutate the caller's base slice")
}

// TestComposeLoRAScorer_RejectsInvalidWeight enforces the finite-positive contract.
func TestComposeLoRAScorer_RejectsInvalidWeight(t *testing.T) {
	for _, w := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		_, err := composeLoRAScorer(nil, w)
		assert.Errorf(t, err, "weight %v must be rejected", w)
	}
}

// TestComposeLoRAScorer_RejectsDoubleSpecification guards against declaring
// lora-affinity via both --routing-scorers and --lora-scorer-weight.
func TestComposeLoRAScorer_RejectsDoubleSpecification(t *testing.T) {
	base := []sim.ScorerConfig{{Name: "lora-affinity", Weight: 1.0}}
	_, err := composeLoRAScorer(base, 2.0)
	assert.Error(t, err, "lora-affinity already in profile must be rejected")
}
