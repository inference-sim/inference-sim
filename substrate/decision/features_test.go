package decision

import (
	"math"
	"testing"
)

// The 29 features of every ordered pair of ten real points equal what the training pipeline
// computes, including the NaN of a simulator that did not answer and of settings a point lacks.
func TestPairFeaturesEqualTheTrainingPipelinesOnRealPoints(t *testing.T) {
	fx, pts := loadFeatureFixture(t)
	if len(fx.Feats) != len(PairFeatureNames) {
		t.Fatalf("Python has %d features, Go %d", len(fx.Feats), len(PairFeatureNames))
	}
	for i := range fx.Feats {
		if fx.Feats[i] != PairFeatureNames[i] {
			t.Fatalf("feature %d is %q in Python and %q in Go", i, fx.Feats[i], PairFeatureNames[i])
		}
	}
	nan := 0
	for _, pr := range fx.Pairs {
		got, err := PairFeatures(PairFeatureNames, pts[pr.A], pts[pr.B])
		if err != nil {
			t.Fatal(err)
		}
		for k, w := range pr.X {
			want := float32(w)
			if math.IsNaN(float64(want)) {
				nan++
				if !math.IsNaN(float64(got[k])) {
					t.Errorf("pair (%d,%d) %s: got %v, Python has missing", pr.A, pr.B, PairFeatureNames[k], got[k])
				}
				continue
			}
			if got[k] != want {
				t.Errorf("pair (%d,%d) %s: got %v, Python %v", pr.A, pr.B, PairFeatureNames[k], got[k], want)
			}
		}
	}
	if nan == 0 {
		t.Error("the fixture has no missing feature")
	}
}

// The stage's 54 inputs are the training pipeline's stage inputs: the artifact fixture was written by it,
// so its feature list is the reference. The columns this stage does not have are missing.
func TestTheStageInputsAreTheTrainingPipelinesAndTheUnavailableOnesAreMissing(t *testing.T) {
	s := loadArtifactStage(t)
	if len(s.Features) != len(StageFeatureNames) {
		t.Fatalf("artifact has %d features, Go knows %d", len(s.Features), len(StageFeatureNames))
	}
	for i := range s.Features {
		if s.Features[i] != StageFeatureNames[i] {
			t.Errorf("feature %d: artifact %q, Go %q", i, s.Features[i], StageFeatureNames[i])
		}
	}
	_, pts := loadFeatureFixture(t)
	row, err := PairFeatures(StageFeatureNames, pts[0], pts[1])
	if err != nil {
		t.Fatal(err)
	}
	for i, n := range StageFeatureNames {
		if i >= len(PairFeatureNames) && !math.IsNaN(float64(row[i])) {
			t.Errorf("%s should be missing, got %v", n, row[i])
		}
	}
	if _, err := PairFeatures([]string{"dX", "model_id"}, pts[0], pts[1]); err == nil {
		t.Error("an unknown feature name was accepted (no model or chip identifier is an input)")
	}
}
