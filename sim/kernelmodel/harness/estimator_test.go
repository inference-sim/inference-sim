package harness

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/kernelmodel/internal/artifacts"
)

func hopperRepos() kernelmodel.Repos {
	return kernelmodel.Repos{
		Scenarios: kernelmodel.DefaultScenarios(),
		Catalog:   kernelmodel.DefaultCatalog(),
		Registry:  kernelmodel.DefaultRegistry(),
	}
}

// testSweep returns a real sweep from the corpus rather than a hand-built one, so the test
// exercises the same inputs the score does.
func testSweep(t *testing.T) Sweep {
	t.Helper()
	c, err := LoadCorpus(
		artifacts.Measurement(t, "aisimulate_e2e.json"))
	if err != nil {
		t.Fatalf("corpus: %v", err)
	}
	for _, sw := range c.Sweeps {
		if sw.Scenario == "gpt-oss-120b-h200-fp4-vllm-tp4.yaml" && sw.Framework == "vllm" {
			return sw
		}
	}
	t.Fatal("gpt-oss-120b-h200-fp4-vllm-tp4 not in the corpus")
	return Sweep{}
}

// An unset Estimator must give the same answer as one set explicitly to the kernel. The
// comparison's baseline is the kernel arm, so if the zero value drifted the published 10.41%
// would silently become a different measurement.
func TestUnsetEstimatorIsTheKernel(t *testing.T) {
	cfg := Config{Repos: hopperRepos(), Admission: AdmissionKernelKV, Seed: 42}
	unset, err := Run(testSweep(t), 8, cfg)
	if err != nil {
		t.Fatalf("unset: %v", err)
	}
	cfg.Estimator = EstimatorKernel
	explicit, err := Run(testSweep(t), 8, cfg)
	if err != nil {
		t.Fatalf("explicit: %v", err)
	}
	if unset.MeanITLUs != explicit.MeanITLUs {
		t.Errorf("zero value is not the kernel: %.6f vs %.6f",
			unset.MeanITLUs, explicit.MeanITLUs)
	}
}

// An analytic arm cannot be built any more; asking for one must fail loudly rather than
// quietly score the kernel under another name.
func TestRetiredEstimatorsAreRefused(t *testing.T) {
	for _, e := range []Estimator{EstimatorRoofline, EstimatorTrainedPhysics} {
		cfg := Config{Repos: hopperRepos(), Admission: AdmissionKernelKV, Seed: 42, Estimator: e}
		if _, err := Run(testSweep(t), 8, cfg); err == nil {
			t.Errorf("%s: Run succeeded; a retired estimator must be refused", e)
		}
	}
}
