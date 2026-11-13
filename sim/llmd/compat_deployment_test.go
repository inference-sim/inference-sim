package llmd

import (
	"fmt"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	"github.com/inference-sim/blis-vllm/compat/contract"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	_ "github.com/inference-sim/blis-vllm/compat/kv"   // the KV store the engine is built with
	_ "github.com/inference-sim/blis-vllm/compat/lora" // the adapter registry it is built with

	"github.com/inference-sim/blis-vllm/core/placement"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// pdTicks is the price of a KV handoff of tokens (the blocks' capacity): the setup cost plus a
// per-token term most PD tests inject, so a larger handoff takes longer and none takes less than
// a tick. The decode engine states the move and the driver's Environment prices it.
func pdTicks(tokens int64) int64 { return 50 + tokens/50 }

// instanceEndpoint places instance_N on node N of its pool.
func instanceEndpoint(id cluster.InstanceID) plan.Endpoint {
	var n int
	if _, err := fmt.Sscanf(string(id), "instance_%d", &n); err != nil {
		panic(fmt.Sprintf("instanceEndpoint: %q", id))
	}
	return plan.Endpoint{Node: n}
}

// setPDPrice makes every handoff of d take ticks(tokens) alone, and gives d the endpoints a PD
// deployment needs.
func setPDPrice(d *cluster.DeploymentConfig, ticks func(tokens int64) int64) {
	d.InstanceEndpoint = instanceEndpoint
	d.NewEngine = compatEngineFactoryWith(d.DPLockstepRanks, ticks)
}

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
	c := engineConfigIn(d.InstanceConfig)
	f(&c)
	d.InstanceConfig = instanceConfigOf(c)
}

func instanceConfigOf(c cengine.SimConfig) sim.InstanceConfig {
	return sim.InstanceConfig{Horizon: c.Horizon, Seed: c.Seed, GPU: c.GPU, TP: c.TP, MaxNumSeqs: c.MaxNumSeqs,
		Model: c.Model, BlockSizeTokens: c.BlockSizeTokens,
		Engine: c}
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
	return compatEngineFactoryWith(lockstepRanks, pdTicks)
}

// compatEngineFactoryWith is compatEngineFactory with the price each driver's Environment gives a
// handoff.
func compatEngineFactoryWith(lockstepRanks int, ticks func(tokens int64) int64) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	driver := func(eng engine.Engine, cfg cengine.SimConfig) (*sim.EngineDriver, error) {
		var pull env.Environment = testutil.FixedPull(cfg.BlockSizeTokens, func(tokens int64, _, _ plan.Endpoint) int64 { return ticks(tokens) })
		dcfg := sim.EngineDriverConfig{Horizon: cfg.Horizon, ResumeGapFromLastToken: true,
			ScrapeEngineMetrics: true, ApplyAtStepStart: true, Environment: pull}
		if m := cfg.LatencyModel; m != nil {
			dcfg.PostDecodeFixedOverhead = m.PostDecodeFixedOverhead()
			dcfg.OutputTokenProcessingTime = m.OutputTokenProcessingTime()
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
			// Instance N sits on node N of its pool, so the end of a pull the decode engine
			// states for itself is its own node.
			if cfg.GPU != "" {
				if err := eng.SetPlacement(placement.Placement{Hardware: cfg.GPU, Fabric: "test-fabric", Tensor: max(cfg.TP, 1),
					Data: max(cfg.DP, 1), Nodes: index + 1, Node: index}); err != nil {
					return nil, err
				}
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

// newCompatInstance builds one instance whose engine is the compatibility engine behind the
// contract (as a cluster builds each of its instances).
func newCompatInstance(id cluster.InstanceID, ic sim.InstanceConfig) *cluster.InstanceSimulator {
	drv, err := compatEngineFactory(0)(0, ic)
	if err != nil {
		panic(fmt.Sprintf("newCompatInstance(%s): %v", id, err))
	}
	return cluster.NewEngineInstance(id, ic, drv)
}
