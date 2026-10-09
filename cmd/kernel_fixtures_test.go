package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// kernelTestScenario is the committed blis-latency-kernel scenario in-process CLI tests run on:
// a colocated TP4 deployment, quick to price.
const kernelTestScenario = "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"

// setupKernelTestFixtures points the package-level deployment at kernelTestScenario for an
// in-process run or replay -- the pinned kernel module's scenario fixtures and the vendored
// catalog and registry -- and restores the previous deployment when the test ends. It returns
// the catalog root (for catalogPath) and, for the call sites that still assign it, an empty
// hardware-config path: the kernel reads the chip from the catalog.
func setupKernelTestFixtures(t *testing.T) (catalogDir, hwPath string) {
	t.Helper()
	scenarios, catalog, registry := kernelRepos(t)
	savedScenario, savedDir, savedRegistry, savedOpened := kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened
	savedExperts, savedTopK := kernelDeploymentExperts, kernelDeploymentTopK
	savedDP, savedEP, savedSpec, savedSpecMethod := dataParallelism, enableExpertParallel, numSpeculativeTokens, speculativeMethod
	savedPrefix := noEnablePrefixCaching
	t.Cleanup(func() {
		kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened = savedScenario, savedDir, savedRegistry, savedOpened
		kernelDeploymentExperts, kernelDeploymentTopK = savedExperts, savedTopK
		dataParallelism, enableExpertParallel, numSpeculativeTokens, speculativeMethod = savedDP, savedEP, savedSpec, savedSpecMethod
		noEnablePrefixCaching = savedPrefix
	})
	kernelScenario, kernelScenarioDir, kernelRegistry, kernelOpened = kernelTestScenario, scenarios, registry, (*kernelmodel.Model)(nil)
	return catalog, ""
}

// setupKernelTestFixturesWithDefaults is setupKernelTestFixtures plus the repository's own
// defaults.yaml, which carries the LoRA cost coefficients a run reads.
func setupKernelTestFixturesWithDefaults(t *testing.T) (catalogDir, hwPath, defaultsPath string) {
	t.Helper()
	catalogDir, _ = setupKernelTestFixtures(t)
	return catalogDir, hwPath, "../defaults.yaml"
}

// kernelScenariosDir and kernelRegistryDir are the scenario and registry roots for a test that
// passes them as flags: registerSimConfigFlags resets the package vars setupKernelTestFixtures
// set, so an in-process ParseFlags must name them again.
func kernelScenariosDir(t *testing.T) string {
	t.Helper()
	scenarios, _, _ := kernelRepos(t)
	return scenarios
}

func kernelRegistryDir(t *testing.T) string {
	t.Helper()
	_, _, registry := kernelRepos(t)
	return registry
}

// copyKernelCatalog copies the vendored catalog into a fresh directory and returns its root,
// for a test that varies one catalog file while the kernel still resolves the rest.
func copyKernelCatalog(t *testing.T) string {
	t.Helper()
	_, catalog, _ := kernelRepos(t)
	root := filepath.Join(t.TempDir(), "catalog")
	if err := os.CopyFS(root, os.DirFS(catalog)); err != nil {
		t.Fatalf("copy catalog %s: %v", catalog, err)
	}
	return root
}

// kernelDeployment is a single-instance cluster config priced by the kernel for
// kernelTestScenario and sized from the kernel's Settings -- KV pool, block size, batch caps,
// max-model-len -- the way `blis run` builds one, for library-level tests that drive
// cluster.NewClusterSimulator directly. batchOpts are applied to the batch config.
type kernelDeployment struct {
	Config     cluster.DeploymentConfig
	Model      *kernelmodel.Model
	BlockSize  int64
	BlockBytes int64 // the kernel's per-rank bytes for one KV block
}

func newKernelDeployment(t *testing.T, seed int64, kvOpts []sim.KVCacheOption, batchOpts ...sim.BatchOption) kernelDeployment {
	t.Helper()
	scenarios, catalog, registry := kernelRepos(t)
	m, err := kernelmodel.Open(kernelTestScenario, kernelmodel.Repos{Scenarios: scenarios, Catalog: catalog, Registry: registry})
	if err != nil {
		t.Fatal(err)
	}
	st, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	dep := m.Deployment()
	mc := sim.ModelConfig{NumLocalExperts: dep.Experts, NumExpertsPerTok: dep.ExpertsPerTok}
	block := int64(st.BlockSize)
	return kernelDeployment{
		Config: cluster.DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon:       10_000_000,
				Seed:          seed,
				KVCacheConfig: sim.NewKVCacheConfig(st.KVBlocks, block, 0, 0.9, 0, 0, kvOpts...),
				BatchConfig:   sim.NewBatchConfig(int64(st.MaxNumSeqs), int64(st.MaxNumBatchedTokens), 0, batchOpts...),
				ModelHardwareConfig: sim.NewModelHardwareConfig(mc, sim.HardwareCalib{}, strings.ToLower(dep.Model),
					dep.Hardware, dep.TP, 1, dep.ExpertParallel, "", sim.LatencyBackendKernel, int64(st.MaxModelLen)),
				PolicyConfig:         sim.NewPolicyConfig("fcfs", ""),
				LatencyModelOverride: m,
			},
			NumInstances:    1,
			AdmissionPolicy: "always-admit",
			RoutingPolicy:   "round-robin",
		},
		Model:      m,
		BlockSize:  block,
		BlockBytes: m.Kernel().SequenceVariableBytes(int(block)),
	}
}
