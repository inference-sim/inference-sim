package cluster

import (
	"fmt"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/env"

	cengine "github.com/inference-sim/inference-sim/engines/vllm/core/runconfig"
	"github.com/inference-sim/inference-sim/internal/vllmtest"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// This package's tests build every instance's engine as `blis run` builds a vLLM one (cmd's
// compatEngineFactory): the v0.33.0 engine (internal/vllmtest, which builds engines/vllm/core/vllm)
// behind engine.Engine, driven by a sim.EngineDriver. A test describes the engine with its own
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
		d.NewEngine = compatEngineFactory(isDisaggregated(d))
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
// compatEngineFactory does: one engine.Engine per instance, or, for a run of
// lockstepRanks consecutive instances, one DPGroup whose rank views the instances drive, rank 0's
// driver stepping the group.
func compatEngineFactory(pd bool) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	return compatEngineFactoryWith(pd, func(block int64) env.Environment {
		return testutil.FixedPull(block, testPDTransferTime)
	})
}

// compatEngineFactoryWith is compatEngineFactory with the Environment each driver prices its
// transfers by, built for the instance's block size (the cost model moves whole blocks).
func compatEngineFactoryWith(pd bool, pull func(blockTokens int64) env.Environment) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	driver := func(eng engine.Engine, cfg simConfig) (*sim.EngineDriver, error) {
		dcfg := sim.EngineDriverConfig{Horizon: cfg.Horizon,
			Environment: pull(cfg.BlockSizeTokens)}
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
	return func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		cfg := engineConfigIn(ic)
		eng, err := vllmtest.Engine(cfg, vllmtest.Options{Connector: pd, Index: index})
		if err != nil {
			return nil, err
		}
		return driver(eng, cfg)
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
	drv, err := compatEngineFactory(false)(0, ic)
	if err != nil {
		panic(fmt.Sprintf("newCompatInstance(%s): %v", id, err))
	}
	return NewEngineInstance(id, ic, drv)
}

// isDisaggregated is whether d runs prefill and decode pools: their engines carry the P/D connector.
func isDisaggregated(d DeploymentConfig) bool { return d.PrefillInstances+d.DecodeInstances > 0 }

// skipRetiredEngine marks a test whose premise is behaviour of the retired compatibility engine
// (its queueing, block accounting or request refusals) that the v0.33.0 engine, now the only
// engine, does not share. Each needs restating against the engine before it can run again.
func skipRetiredEngine(t *testing.T, why string) {
	t.Helper()
	t.Skip("premise is retired compatibility-engine behaviour: " + why)
}
