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

func hopperBackends() BackendPaths {
	return BackendPaths{
		Catalog:  kernelmodel.DefaultCatalog(),
		HWConfig: "../../../hardware_config.json",
		Defaults: "../../../defaults.yaml",
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

// Swapping the step-time model must change the step time and NOTHING else. The KV budget and
// the number of completions are decided by the kernel for every arm, so if an analytic arm
// moved them the comparison would be mixing admission behaviour into a step-time result.
func TestAnalyticArmsChangeStepTimeNotAdmission(t *testing.T) {
	cfg := Config{Repos: hopperRepos(), Admission: AdmissionKernelKV, Seed: 42,
		Backends: hopperBackends()}
	base, err := Run(testSweep(t), 8, cfg)
	if err != nil {
		t.Fatalf("kernel: %v", err)
	}
	for _, e := range []Estimator{EstimatorRoofline, EstimatorTrainedPhysics} {
		cfg.Estimator = e
		got, err := Run(testSweep(t), 8, cfg)
		if err != nil {
			t.Fatalf("%s: %v", e, err)
		}
		if got.KVBlocks != base.KVBlocks {
			t.Errorf("%s: KV budget moved, %d vs %d -- the arms no longer share admission",
				e, got.KVBlocks, base.KVBlocks)
		}
		if got.MeanITLUs == base.MeanITLUs {
			t.Errorf("%s: step time is identical to the kernel's (%.4f); the arm is not "+
				"actually being used", e, got.MeanITLUs)
		}
		if got.MeanITLUs <= 0 {
			t.Errorf("%s: produced no measurement (%.4f)", e, got.MeanITLUs)
		}
	}
}
