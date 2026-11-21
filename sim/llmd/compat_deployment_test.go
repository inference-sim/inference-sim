package llmd

import (
	"fmt"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/env"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"

	cengine "github.com/inference-sim/inference-sim/engines/vllm/core/runconfig"

	"github.com/inference-sim/inference-sim/internal/vllmtest"

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
	d.NewEngine = compatEngineFactoryWith(isDisaggregated(*d), ticks)
}

// This package's tests build every instance's engine as `blis run` builds a vLLM one: the
// v0.33.0 engine (internal/vllmtest, which builds engines/vllm/core/vllm) behind engine.Engine,
// driven by a sim.EngineDriver. A test describes the engine with its own configuration
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
		d.NewEngine = compatEngineFactory(isDisaggregated(d))
	}
	// As cmd states it for a P/D deployment: the v0.33.0 engine needs the prefill leg to hold an output.
	d.PDPrefillSamplesFirstToken = d.PDPrefillSamplesFirstToken || isDisaggregated(d)
	return cluster.NewClusterSimulator(d, src, onRequestDone)
}

// compatEngineFactory builds each instance's engine as cmd's compatEngineFactory does: one
// engine.Engine per instance, or one DPGroup per run of lockstepRanks instances, whose
// rank views the instances drive, rank 0's driver stepping the group.
func compatEngineFactory(pd bool) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	return compatEngineFactoryWith(pd, pdTicks)
}

// compatEngineFactoryWith is compatEngineFactory with the price each driver's Environment gives a
// handoff.
func compatEngineFactoryWith(pd bool, ticks func(tokens int64) int64) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	return compatEngineFactoryStating(pd, ticks, "enable_scale_out: true\n")
}

// compatEngineFactoryStating is compatEngineFactoryWith with engine-block lines (vLLM setting
// names) the engines are built with. The default states enable_scale_out, which a router config
// with a render token-producer needs of the engines it routes to (inference-sim#2129); a test of
// the engines without it passes none.
func compatEngineFactoryStating(pd bool, ticks func(tokens int64) int64, extra string) func(int, sim.InstanceConfig) (*sim.EngineDriver, error) {
	driver := func(eng engine.Engine, cfg cengine.SimConfig) (*sim.EngineDriver, error) {
		var pull env.Environment = testutil.FixedPull(cfg.BlockSizeTokens, func(tokens int64, _, _ plan.Endpoint) int64 { return ticks(tokens) })
		dcfg := sim.EngineDriverConfig{Horizon: cfg.Horizon,
			Environment: pull}
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
	return func(index int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
		cfg := engineConfigIn(ic)
		eng, err := vllmtest.Engine(cfg, vllmtest.Options{Connector: pd, Index: index, Extra: extra})
		if err != nil {
			return nil, err
		}
		return driver(eng, cfg)
	}
}

// newCompatInstance builds one instance whose engine is the compatibility engine behind the
// contract (as a cluster builds each of its instances).
func newCompatInstance(id cluster.InstanceID, ic sim.InstanceConfig) *cluster.InstanceSimulator {
	drv, err := compatEngineFactory(false)(0, ic)
	if err != nil {
		panic(fmt.Sprintf("newCompatInstance(%s): %v", id, err))
	}
	return cluster.NewEngineInstance(id, ic, drv)
}

// isDisaggregated is whether d runs prefill and decode pools: their engines carry the P/D connector.
func isDisaggregated(d cluster.DeploymentConfig) bool {
	return d.PrefillInstances+d.DecodeInstances > 0
}

// skipRetiredEngine marks a test whose premise is behaviour or introspection of the retired
// compatibility engine that the v0.33.0 engine, now the only engine, does not share. Each needs
// restating against the engine before it can run again.
func skipRetiredEngine(t *testing.T, why string) {
	t.Helper()
	t.Skip("premise is retired compatibility-engine behaviour: " + why)
}
