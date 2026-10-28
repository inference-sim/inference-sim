package llmd

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/engine"

	"github.com/inference-sim/blis-vllm/compat/contract"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	_ "github.com/inference-sim/blis-vllm/compat/kv"   // the KV store the engine is built with
	_ "github.com/inference-sim/blis-vllm/compat/lora" // the adapter registry it is built with

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// This package's tests build every instance's engine as `blis run` builds a 0.29/0.31 one: the
// compatibility engine behind engine.Engine (blis-vllm compat/contract), driven by a
// sim.EngineDriver. A test describes the engine with compat's own configuration
// (cengine.SimConfig, which test files may name); the helpers below are sim/cluster's test
// helpers of the same names (compat_deployment_test.go there), which a test file there cannot
// export.

// deploymentOf is a cluster deployment of numInstances instances whose engines are built from cfg.
func deploymentOf(cfg cengine.SimConfig, numInstances int) cluster.DeploymentConfig {
	return cluster.DeploymentConfig{InstanceConfig: instanceConfigOf(cfg), NumInstances: numInstances}
}

// editEngine applies f to a deployment's engine configuration, keeping the front end the test set
// on the deployment (DeploymentConfig.Frontend, a host setting).
func editEngine(d *cluster.DeploymentConfig, f func(c *cengine.SimConfig)) {
	frontend := d.Frontend
	c := engineConfigIn(d.InstanceConfig)
	f(&c)
	d.InstanceConfig = instanceConfigOf(c)
	if frontend != (sim.FrontendConfig{}) {
		d.Frontend = frontend
	}
}

func instanceConfigOf(c cengine.SimConfig) sim.InstanceConfig {
	return sim.InstanceConfig{Horizon: c.Horizon, Seed: c.Seed, GPU: c.GPU, TP: c.TP, MaxNumSeqs: c.MaxNumSeqs,
		Model: c.Model, BlockSizeTokens: c.BlockSizeTokens,
		Frontend: sim.FrontendConfig{APIServerCount: c.Frontend.APIServerCount, RendererNumWorkers: c.Frontend.RendererNumWorkers},
		Engine:   c}
}

func engineConfigIn(c sim.InstanceConfig) cengine.SimConfig {
	ec, ok := c.Engine.(cengine.SimConfig)
	if !ok {
		panic(fmt.Sprintf("test deployment: an instance configuration holds %T, not the engine's configuration", c.Engine))
	}
	ec.GPU, ec.Horizon = c.GPU, c.Horizon
	return ec
}

// newCluster builds the cluster with the compatibility engine factory its data parallelism needs,
// unless the test set an engine factory of its own.
func newCluster(d cluster.DeploymentConfig, src cluster.RequestSource, onRequestDone func(*sim.Request, int64) []*sim.Request) *cluster.ClusterSimulator {
	if d.NewEngine == nil {
		d.NewEngine = compatEngineFactory(d.DPLockstepRanks)
	}
	return cluster.NewClusterSimulator(d, src, onRequestDone)
}

// compatEngineFactory builds each instance's engine as cmd's compatEngineFactory does: one
// compat/contract.Engine per instance, or one DPGroup per run of lockstepRanks instances, whose
// rank views the instances drive, rank 0's driver stepping the group.
func compatEngineFactory(lockstepRanks int) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	driver := func(eng engine.Engine, cfg cengine.SimConfig, f sim.FrontendConfig) (*sim.EngineDriver, error) {
		dcfg := sim.EngineDriverConfig{Horizon: cfg.Horizon, ResumeGapFromLastToken: true,
			ScrapeEngineMetrics: true, ApplyAtStepStart: true}
		if m := cfg.LatencyModel; m != nil {
			dcfg.PostDecodeFixedOverhead = m.PostDecodeFixedOverhead()
			dcfg.OutputTokenProcessingTime = m.OutputTokenProcessingTime()
			dcfg.QueueingTime = func(req *sim.Request) int64 {
				return m.QueueingTime(&cengine.Request{InputTokens: make([]cengine.TokenID, req.InputLen())})
			}
		}
		return sim.NewEngineDriverWithFrontend(eng, dcfg, f)
	}
	var group *contract.DPGroup
	var leader *sim.EngineDriver
	return func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		cfg := engineConfigIn(ic)
		cfg.Frontend.APIServerCount, cfg.Frontend.RendererNumWorkers = ic.Frontend.APIServerCount, ic.Frontend.RendererNumWorkers
		if lockstepRanks <= 1 {
			eng, err := contract.New(fmt.Sprintf("instance_%d", index), cfg)
			if err != nil {
				return nil, err
			}
			return driver(eng, cfg, ic.Frontend)
		}
		k := index % lockstepRanks
		if k == 0 {
			ids := make([]string, lockstepRanks)
			for j := range ids {
				ids[j] = fmt.Sprintf("instance_%d", index+j)
			}
			g, err := contract.NewDPGroup(ids, cfg)
			if err != nil {
				return nil, err
			}
			group = g
		}
		if group == nil {
			return nil, fmt.Errorf("data-parallel rank %d of instance %d was built before its rank 0", k, index)
		}
		d, err := driver(group.View(k), cfg, ic.Frontend)
		if err != nil {
			return nil, err
		}
		if k == 0 {
			leader = d
		}
		d.JoinGroup(leader, k)
		return d, nil
	}
}

// newCompatInstance builds one instance whose engine is the compatibility engine behind the
// contract (as a cluster builds each of its instances).
func newCompatInstance(id cluster.InstanceID, ic sim.InstanceConfig) *cluster.InstanceSimulator {
	drv, err := compatEngineFactory(0)(0, ic)
	if err != nil {
		panic(fmt.Sprintf("newCompatInstance(%s): %v", id, err))
	}
	return cluster.NewEngineInstance(id, ic, drv)
}
