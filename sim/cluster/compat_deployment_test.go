package cluster

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/engine"

	"github.com/inference-sim/blis-vllm/compat/contract"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	_ "github.com/inference-sim/blis-vllm/compat/kv"   // the KV store the engine is built with
	_ "github.com/inference-sim/blis-vllm/compat/lora" // the adapter registry it is built with

	"github.com/inference-sim/inference-sim/sim"
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
// front end the test set on the deployment itself (DeploymentConfig.Frontend, which the cluster
// reads) is kept, as a host setting the engine configuration does not carry.
func setEngine(d *DeploymentConfig, cfg simConfig) {
	frontend := d.Frontend
	d.InstanceConfig = instanceConfigOf(cfg)
	if frontend != (sim.FrontendConfig{}) {
		d.Frontend = frontend
	}
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
		Frontend: sim.FrontendConfig{APIServerCount: c.Frontend.APIServerCount, RendererNumWorkers: c.Frontend.RendererNumWorkers},
		Engine:   c}
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
	driver := func(eng engine.Engine, cfg simConfig) (*sim.EngineDriver, error) {
		dcfg := sim.EngineDriverConfig{Horizon: cfg.Horizon, ResumeGapFromLastToken: true,
			ScrapeEngineMetrics: true, ApplyAtStepStart: true}
		if m := cfg.LatencyModel; m != nil {
			dcfg.PostDecodeFixedOverhead = m.PostDecodeFixedOverhead()
			dcfg.OutputTokenProcessingTime = m.OutputTokenProcessingTime()
			dcfg.QueueingTime = func(req *sim.Request) int64 {
				return m.QueueingTime(&cengine.Request{InputTokens: make([]cengine.TokenID, req.InputLen())})
			}
		}
		return sim.NewEngineDriverWithFrontend(eng, dcfg,
			sim.FrontendConfig{APIServerCount: cfg.Frontend.APIServerCount, RendererNumWorkers: cfg.Frontend.RendererNumWorkers})
	}
	var group *contract.DPGroup
	var leader *sim.EngineDriver
	return func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		cfg := engineConfigIn(ic)
		// The host's front end is the instance configuration's (DeploymentConfig.Frontend), which
		// the cluster also reads for a replica's shared render stage.
		cfg.Frontend.APIServerCount, cfg.Frontend.RendererNumWorkers = ic.Frontend.APIServerCount, ic.Frontend.RendererNumWorkers
		if lockstepRanks <= 1 {
			eng, err := contract.New(fmt.Sprintf("instance_%d", index), cfg)
			if err != nil {
				return nil, err
			}
			return driver(eng, cfg)
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
