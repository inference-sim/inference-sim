package cluster

import (
	"fmt"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/env"

	"github.com/inference-sim/blis-vllm/compat/contract"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	_ "github.com/inference-sim/blis-vllm/compat/kv"   // the KV store the engine is built with
	_ "github.com/inference-sim/blis-vllm/compat/lora" // the adapter registry it is built with
	"github.com/inference-sim/blis-vllm/core/placement"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// This package's tests build every instance's engine as `blis run` builds a 0.29/0.31 one (cmd's
// compatEngineFactory): the compatibility engine behind engine.Engine (blis-vllm
// compat/contract), driven by a sim.EngineDriver. A test describes the engine with compat's own
// configuration (cengine.SimConfig, which test files may name), and compatDeployment turns it
// into the cluster's deployment.

// simConfig is the compatibility engine's configuration, as the tests describe an instance.
type simConfig = cengine.SimConfig

// compatDeployment is a cluster deployment of numInstances instances whose engines are built
// from cfg, each as the build site builds a compatibility release's instance. Its engine factory
// is set when the cluster is built (newCluster), so a test that edits the deployment's data
// parallelism (DPLockstepRanks) after this call gets a factory that groups the ranks.
func compatDeployment(cfg simConfig, numInstances int) DeploymentConfig {
	return DeploymentConfig{InstanceConfig: instanceConfigOf(cfg), NumInstances: numInstances}
}

// withEngines is d with the compatibility engine factory its data parallelism needs, unless the
// test set an engine factory of its own.
func withEngines(d DeploymentConfig) DeploymentConfig {
	if d.NewEngine == nil {
		d.NewEngine = compatEngineFactory(d.DPLockstepRanks)
	}
	return d
}

// withRoles gives a deployment's prefill and decode pools their own engine configurations
// (nil keeps the deployment's).
func withRoles(d DeploymentConfig, prefill, decode *simConfig) DeploymentConfig {
	if prefill != nil {
		c := instanceConfigOf(*prefill)
		d.Roles.Prefill = &c
	}
	if decode != nil {
		c := instanceConfigOf(*decode)
		d.Roles.Decode = &c
	}
	return d
}

// engineConfigOf is the compatibility engine's configuration a deployment's instances are built
// with (the global one, before any pool role's).
func engineConfigOf(d DeploymentConfig) simConfig { return engineConfigIn(d.InstanceConfig) }

// setEngine replaces a deployment's engine configuration, keeping its host settings in step. A
func setEngine(d *DeploymentConfig, cfg simConfig) {
	d.InstanceConfig = instanceConfigOf(cfg)
}

// editEngine applies f to a deployment's engine configuration.
func editEngine(d *DeploymentConfig, f func(c *simConfig)) {
	c := engineConfigOf(*d)
	f(&c)
	setEngine(d, c)
}

// instanceConfigOf is cfg as the host holds an instance's configuration: the settings the host
// reads, beside the engine's own, unopened (cmd's instanceConfigOf).
func instanceConfigOf(c simConfig) sim.InstanceConfig {
	return sim.InstanceConfig{Horizon: c.Horizon, Seed: c.Seed, GPU: c.GPU, TP: c.TP, MaxNumSeqs: c.MaxNumSeqs,
		Model: c.Model, BlockSizeTokens: c.BlockSizeTokens,
		Engine: c}
}

// engineConfigIn is the engine configuration an instance configuration holds, with the host's
// settings the cluster may have changed for the instance applied (its node pool's GPU, the run's
// horizon), as cmd's engineConfigIn applies them.
func engineConfigIn(c sim.InstanceConfig) simConfig {
	ec, ok := c.Engine.(simConfig)
	if !ok {
		panic(fmt.Sprintf("test deployment: an instance configuration holds %T, not the engine's configuration", c.Engine))
	}
	ec.GPU, ec.Horizon = c.GPU, c.Horizon
	return ec
}

// compatEngineFactory builds each instance's compatibility engine behind engine.Engine, as cmd's
// compatEngineFactory does: one compat/contract.Engine per instance, or, for a run of
// lockstepRanks consecutive instances, one DPGroup whose rank views the instances drive, rank 0's
// driver stepping the group.
func compatEngineFactory(lockstepRanks int) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	return compatEngineFactoryWith(lockstepRanks, func(block int64) env.Environment {
		return testutil.FixedPull(block, testPDTransferTime)
	})
}

// statesBlocks is the compatibility engine as an engine that states, on each pull it starts, the
// KV blocks the pull moves (engine.Transfer.Blocks), which is what blis-vllm's engine reports and
// the compatibility adapter does not: the engine's own pool pages in block-sized units.
type statesBlocks struct {
	*contract.Engine
	block int64
}

func (e statesBlocks) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.Engine.Schedule(now)
	for i := range b.TransfersStarted {
		if t := &b.TransfersStarted[i]; t.Tier == "" && t.Blocks == 0 && e.block > 0 {
			t.Blocks = int((int64(t.Tokens) + e.block - 1) / e.block)
		}
	}
	return b, err
}

// compatEngineFactoryWith is compatEngineFactory with the Environment each driver prices its
// transfers by, built for the instance's block size (the cost model moves whole blocks).
func compatEngineFactoryWith(lockstepRanks int, pull func(blockTokens int64) env.Environment) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	driver := func(eng engine.Engine, cfg simConfig) (*sim.EngineDriver, error) {
		dcfg := sim.EngineDriverConfig{Horizon: cfg.Horizon, ResumeGapFromLastToken: true,
			ScrapeEngineMetrics: true, ApplyAtStepStart: true, Environment: pull(cfg.BlockSizeTokens)}
		if m := cfg.LatencyModel; m != nil {
			dcfg.PostDecodeFixedOverhead = m.PostDecodeFixedOverhead()
			dcfg.OutputTokenProcessingTime = m.OutputTokenProcessingTime()
			// The fake latency model's admission time is all work on a server of the admission
			// queue the engine states.
			dcfg.Environment = testutil.WithHost(dcfg.Environment, func(n int) kernel.HostDemand {
				return kernel.HostDemand{Service: time.Duration(m.QueueingTime(&cengine.Request{
					InputTokens: make([]cengine.TokenID, n)})) * time.Microsecond}
			})
		}
		return sim.NewEngineDriver(eng, dcfg), nil
	}
	var group *contract.DPGroup
	var leader *sim.EngineDriver
	return func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		cfg := engineConfigIn(ic)
		if lockstepRanks <= 1 {
			eng, err := contract.New(fmt.Sprintf("instance_%d", index), cfg)
			if err != nil {
				return nil, err
			}
			// Instance N sits on node N of its pool (testInstanceEndpoint), so the end of a pull
			// the decode engine states for itself is its own node, as a deployment places it.
			if cfg.GPU != "" {
				if err := eng.SetPlacement(placement.Placement{Hardware: cfg.GPU, Fabric: "test-fabric", Tensor: max(cfg.TP, 1),
					Data: max(cfg.DP, 1), Nodes: index + 1, Node: index}); err != nil {
					return nil, err
				}
			}
			return driver(statesBlocks{Engine: eng, block: cfg.BlockSizeTokens}, cfg)
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
		d, err := driver(group.View(k), cfg)
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

// newTestCluster is NewClusterSimulator over a test deployment, with the compatibility engine
// factory its data parallelism needs (withEngines).
func newTestCluster(d DeploymentConfig, src RequestSource, onRequestDone func(*sim.Request, int64) []*sim.Request) *ClusterSimulator {
	return NewClusterSimulator(withEngines(d), src, onRequestDone)
}

// editPrefill and editDecode apply f to the prefill or decode pool's own engine configuration,
// starting from the deployment's when the pool has none yet, as the build site gives a P/D pool
// its own engine (cmd's rolesOf).
func editPrefill(d *DeploymentConfig, f func(c *simConfig)) {
	d.Roles.Prefill = editRole(d, d.Roles.Prefill, f)
}
func editDecode(d *DeploymentConfig, f func(c *simConfig)) {
	d.Roles.Decode = editRole(d, d.Roles.Decode, f)
}

func editRole(d *DeploymentConfig, role *sim.InstanceConfig, f func(c *simConfig)) *sim.InstanceConfig {
	c := engineConfigOf(*d)
	if role != nil {
		c = engineConfigIn(*role)
	}
	f(&c)
	ic := instanceConfigOf(c)
	return &ic
}

// newCompatInstance builds one instance whose engine is the compatibility engine behind the
// contract, from the host's configuration ic (as a cluster builds each of its instances).
func newCompatInstance(id InstanceID, ic sim.InstanceConfig) *InstanceSimulator {
	drv, err := compatEngineFactory(0)(0, ic)
	if err != nil {
		panic(fmt.Sprintf("newCompatInstance(%s): %v", id, err))
	}
	return NewEngineInstance(id, ic, drv)
}

// tokenIDsOf is a host request's tokens as the engine's packages type them.
func tokenIDsOf(in []sim.TokenID) []cengine.TokenID {
	out := make([]cengine.TokenID, len(in))
	for i, v := range in {
		out[i] = cengine.TokenID(v)
	}
	return out
}
