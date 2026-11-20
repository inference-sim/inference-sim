package validate_test

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/validate/registry"
)

// registryRoot is this repository's registry. A test runs in its package directory, so the
// registry is one level up.
const registryRoot = "../registry"

// TestEveryCoefficientSetLoadsAndValidates loads every committed coefficient set and runs its
// own checks, including the duplicate (name, scope) check, against the file as committed.
// Sets are discovered with registry.SetPaths, the predicate the validator command uses, so a
// set that the command would read cannot be skipped here.
func TestEveryCoefficientSetLoadsAndValidates(t *testing.T) {
	sets, err := registry.SetPaths(registryRoot)
	if err != nil {
		t.Fatalf("registry.SetPaths(%s): %v", registryRoot, err)
	}
	// A floor rather than "not empty": eight sets are committed, so a discovery bug that
	// found one would otherwise pass while testing almost nothing.
	const floor = 8
	if len(sets) < floor {
		t.Fatalf("found %d coefficient sets under %s, want at least %d", len(sets), registryRoot, floor)
	}
	for _, path := range sets {
		s, err := contracts.LoadCoefficientSet(path)
		if err != nil {
			t.Errorf("LoadCoefficientSet(%s): %v", filepath.Base(path), err)
			continue
		}
		if len(s.Coefficients) == 0 {
			t.Errorf("%s: parsed no entries", filepath.Base(path))
			continue
		}
		if p := s.Validate(); !p.OK() {
			t.Errorf("%s failed validation:\n%s", filepath.Base(path), p.Error())
		}
	}
}

// TestEveryAcceptanceSetLoadsAndValidates does the same for acceptance profile sets.
func TestEveryAcceptanceSetLoadsAndValidates(t *testing.T) {
	paths, err := registry.AcceptancePaths(registryRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatalf("found no acceptance sets under %s", registryRoot)
	}
	for _, path := range paths {
		s, err := contracts.LoadAcceptanceProfileSet(path)
		if err != nil {
			t.Errorf("%s: %v", filepath.Base(path), err)
			continue
		}
		if p := s.Validate(); !p.OK() {
			t.Errorf("%s failed validation:\n%s", filepath.Base(path), p.Error())
		}
	}
}

// museDSpark is an unpinned Muse DSpark run on vLLM at K 7 with the draft sampler and
// verifier its deployment states, drawn from the committed registry.
func museDSpark(t *testing.T, thinking acceptance.Thinking, k int) acceptance.Request {
	t.Helper()
	paths, err := registry.AcceptancePaths(registryRoot)
	if err != nil {
		t.Fatal(err)
	}
	var sets []*acceptance.ProfileSet
	for _, path := range paths {
		s, err := contracts.LoadAcceptanceProfileSet(path)
		if err != nil {
			t.Fatalf("%s: %v", filepath.Base(path), err)
		}
		sets = append(sets, s)
	}
	return acceptance.Request{
		Model: "muse-spark-open-816b-a42b-nvfp4", Drafter: "muse-spark-open-dspark",
		SpeculativeMethod: "dspark", NumSpecTokens: k,
		DraftSampler: acceptance.DraftProbabilistic, Verifier: acceptance.VerifierBlock,
		Engine: "vllm", DefaultDraftSampler: acceptance.DraftGreedy, DefaultVerifier: acceptance.VerifierStandard,
		Hardware: "gb200-nvl72", Architecture: "MuseSparkForConditionalGeneration",
		Registry: sets,
		Workload: acceptance.Workload{Category: "coding", Thinking: thinking},
	}
}

// TestUnpinnedMuseDSparkResolvesToItsMeasurement: a Muse DSpark run on vLLM with thinking on,
// category coding and no pin takes the measured profile by the exact step, and its value lies in
// the range the internal deployment measurements gave (2.62 to 2.78 at K 7), not the 4.05 that
// SGLang's DFlash2 curve lent it by the other_drafter step.
func TestUnpinnedMuseDSparkResolvesToItsMeasurement(t *testing.T) {
	r, err := acceptance.Resolve(museDSpark(t, acceptance.ThinkingOn, 7))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != acceptance.StepExact {
		t.Errorf("resolved by step %q (differed %v), want exact", r.Step, r.Differed)
	}
	if r.AcceptanceLength < 2.62 || r.AcceptanceLength > 2.78 {
		t.Errorf("acceptance length %v is outside the measured 2.62 to 2.78", r.AcceptanceLength)
	}
	if len(r.PerPositionRates) != 7 || r.RatesDerived {
		t.Errorf("rates %v (derived %v), want seven measured rates", r.PerPositionRates, r.RatesDerived)
	}
	if r.MeasuredOn != "" {
		t.Errorf("recorded as measured on %q for a run on the same chip", r.MeasuredOn)
	}
}

// TestMuseDSparkOutsideTheMeasurementStillBorrows states what the measurement does not cover:
// thinking off has no profile and is refused, and another K borrows from the measured K 7.
func TestMuseDSparkOutsideTheMeasurementStillBorrows(t *testing.T) {
	if r, err := acceptance.Resolve(museDSpark(t, acceptance.ThinkingOff, 7)); err == nil {
		t.Errorf("thinking off resolved (step %q, AL %v) with no measurement behind it", r.Step, r.AcceptanceLength)
	}
	// A deployment that states neither sampler nor verifier runs the engine's defaults (greedy,
	// standard). It borrows the measured profile by the same-drafter other-settings step, which
	// names the fields that differ, and stays inside the measured range.
	def := museDSpark(t, acceptance.ThinkingOn, 7)
	def.DraftSampler, def.Verifier = acceptance.DraftGreedy, acceptance.VerifierStandard
	if r, err := acceptance.Resolve(def); err != nil {
		t.Errorf("engine-default settings: %v", err)
	} else if r.Step != acceptance.StepOtherSettings || r.AcceptanceLength < 2.62 || r.AcceptanceLength > 2.78 {
		t.Errorf("engine-default settings resolved by %q at AL %v, want same_drafter_other_settings in 2.62 to 2.78",
			r.Step, r.AcceptanceLength)
	}
	r, err := acceptance.Resolve(museDSpark(t, acceptance.ThinkingOn, 5))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != acceptance.StepOtherK {
		t.Errorf("K 5 resolved by %q, want same_drafter_other_k", r.Step)
	}
}
