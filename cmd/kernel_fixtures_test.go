package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-vllm/compat/contract"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/llmd"
)

// kernelTestScenario is the committed blis-latency-kernel scenario in-process CLI tests run on:
// a colocated TP4 deployment, quick to price.
const kernelTestScenario = "gpt-oss-120b-h200-fp4-vllm-tp4.yaml"

// setupKernelTestFixtures points the package-level deployment at kernelTestScenario for an
// in-process run or replay -- the pinned kernel module's scenario fixtures and the vendored
// catalog and registry -- and restores the previous deployment when the test ends. It returns
// the catalog root (for catalogPath); the kernel reads the chip from the catalog.
func setupKernelTestFixtures(t *testing.T) (catalogDir string) {
	t.Helper()
	scenarios, catalog, registry := kernelRepos(t)
	savedScenario, savedDir, savedRegistry, savedOpened := kernelScenario, kernelScenarioDir, registryPath, kernelOpened
	savedExperts, savedTopK := kernelDeploymentExperts, kernelDeploymentTopK
	savedDP, savedEP, savedSpec, savedSpecMethod := dataParallelism, enableExpertParallel, numSpeculativeTokens, speculativeMethod
	savedPrefix := noEnablePrefixCaching
	t.Cleanup(func() {
		kernelScenario, kernelScenarioDir, registryPath, kernelOpened = savedScenario, savedDir, savedRegistry, savedOpened
		kernelDeploymentExperts, kernelDeploymentTopK = savedExperts, savedTopK
		dataParallelism, enableExpertParallel, numSpeculativeTokens, speculativeMethod = savedDP, savedEP, savedSpec, savedSpecMethod
		noEnablePrefixCaching = savedPrefix
	})
	kernelScenario, kernelScenarioDir, registryPath, kernelOpened = kernelTestScenario, scenarios, registry, (*kernelmodel.Model)(nil)
	return catalog
}

// setupKernelTestFixturesWithDefaults is setupKernelTestFixtures plus the repository's own
// defaults.yaml, which carries the LoRA cost coefficients a run reads.
func setupKernelTestFixturesWithDefaults(t *testing.T) (catalogDir, defaultsPath string) {
	t.Helper()
	catalogDir = setupKernelTestFixtures(t)
	return catalogDir, "../defaults.yaml"
}

// kernelScenariosDir and kernelRegistryDir are the scenario and registry roots for a test that
// passes them as flags: registerSimConfigFlags resets the package vars setupKernelTestFixtures
// set, so an in-process ParseFlags must name them again.
func kernelScenariosDir(t *testing.T) string {
	t.Helper()
	scenarios, _, _ := kernelRepos(t)
	// The scenarios are a copy in which the one the tests run lists the registry set a storage
	// tier is priced from (cost-model-offload-tiers), as an offload run needs.
	dir := t.TempDir()
	if err := os.CopyFS(dir, os.DirFS(scenarios)); err != nil {
		t.Fatal(err)
	}
	with, err := scenarioDirWithOffloadSet(scenarios, kernelTestScenario)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(with) }()
	raw, err := os.ReadFile(filepath.Join(with, kernelTestScenario))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, kernelTestScenario), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
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
	// Config is the instance's engine configuration; deploymentOf wraps it into a cluster's.
	Config     engineConfig
	Model      *kernelmodel.Model
	BlockSize  int64
	BlockBytes int64 // the kernel's per-rank bytes for one KV block
}

func newKernelDeployment(t *testing.T, seed int64, kvOpts []cengine.KVCacheOption, batchOpts ...cengine.BatchOption) kernelDeployment {
	t.Helper()
	scenarios, catalog, registry := kernelRepos(t)
	m, err := kernelmodel.Open(kernelTestScenario, reposOf(scenarios, catalog, registry))
	if err != nil {
		t.Fatal(err)
	}
	st, err := m.Settings()
	if err != nil {
		t.Fatal(err)
	}
	dep := m.Deployment()
	mc := modelConfig{NumLocalExperts: dep.Experts, NumExpertsPerTok: dep.ExpertsPerTok}
	block := int64(st.BlockSize)
	return kernelDeployment{
		Config: engineConfig{
			Horizon:       10_000_000,
			Seed:          seed,
			KVCacheConfig: cengine.NewKVCacheConfig(st.KVBlocks, block, 0, 0.9, 0, 0, kvOpts...),
			BatchConfig:   cengine.NewBatchConfig(int64(st.MaxNumSeqs), int64(st.MaxNumBatchedTokens), 0, batchOpts...),
			ModelHardwareConfig: cengine.NewModelHardwareConfig(mc, strings.ToLower(dep.Model),
				dep.Hardware, dep.TP, 1, dep.ExpertParallel, int64(st.MaxModelLen)),
			PolicyConfig: cengine.NewPolicyConfig("fcfs", ""),
			LatencyModel: m,
		},
		Model:      m,
		BlockSize:  block,
		BlockBytes: m.Kernel().SequenceVariableBytes(int(block)),
	}
}

// deploymentOf is a one-instance cluster configuration whose engine is the compatibility engine
// behind the contract, built from ec (compatEngineFactory).
func deploymentOf(ec engineConfig) cluster.DeploymentConfig {
	return cluster.DeploymentConfig{InstanceConfig: instanceConfigOf(ec), NumInstances: 1,
		NewEngine: compatEngineFactory(0)}
}

// newRoutedCluster builds a cluster over reqs, with the instance engine configuration ec, routed
// the way `blis run` routes one with no router flags: the embedded llm-d router with its default
// configuration. Every cluster run needs a gateway (ClusterSimulator.Run refuses to start without
// one).
func newRoutedCluster(ec engineConfig, reqs []*sim.Request) *cluster.ClusterSimulator {
	cfg := deploymentOf(ec)
	cs := cluster.NewClusterSimulator(cfg, cluster.NewSliceRequestSource(reqs), nil)
	cs.SetGateway(newGateway(routerInputs{EPPFlags: append([]string(nil), llmd.DefaultEPPFlags()...)}, cfg.Model))
	return cs
}

// compatEngineFactory builds each instance's compatibility engine behind engine.Engine (blis-vllm
// compat/contract) from its configuration, driven by a sim.EngineDriver. No run builds this
// engine since engine selection refuses vLLM 0.29.0 and 0.31.0 (LIF-6); the library-level tests
// in this package still drive the host through it, until they move onto the v0.32.0 engine
// (blis-vllm deletes compat/ in M4, release plan §3.3).
func compatEngineFactory(acceptance float64) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	// Each instance's engine is built from its own configuration, resolved for its pool role (a
	// P/D prefill or decode pool's overrides), as the compatibility engine's own host loop builds
	// it (sim.NewEngine).
	driver := func(eng engine.Engine, cfg engineConfig) (*sim.EngineDriver, error) {
		dcfg := sim.EngineDriverConfig{Horizon: cfg.Horizon, Acceptance: acceptance, ResumeGapFromLastToken: true,
			ScrapeEngineMetrics: true, ApplyAtStepStart: true}
		hostOverheads(&dcfg, cfg.LatencyModel)
		// This engine has no kernel to price its admission: the fake latency model's queueing time
		// is a delay.
		if m := cfg.LatencyModel; m != nil {
			dcfg.QueueingTime = func(req *sim.Request) int64 {
				return m.QueueingTime(&cengine.Request{InputTokens: make([]cengine.TokenID, req.InputLen())})
			}
		}
		return sim.NewEngineDriver(eng, dcfg), nil
	}
	return func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		cfg := engineConfigIn(ic)
		eng, err := contract.New(fmt.Sprintf("instance_%d", index), cfg)
		if err != nil {
			return nil, err
		}
		return driver(eng, cfg)
	}
}
