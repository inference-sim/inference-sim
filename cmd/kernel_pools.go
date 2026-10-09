package cmd

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// kernelPools is a disaggregated kernel run's per-role wiring: one kernel per pool, since each
// role runs its own engine (parallelism, token budget, graph mode) and is priced by the kernel
// for that pool rather than by pool 0's copied to both.
type kernelPools struct {
	prefill, decode *kernelmodel.Model
}

// openKernelPools opens the prefill and decode pools of the run's scenario. It returns nil off
// the kernel backend or when the run is not disaggregated, and is fatal when a disaggregated
// run's scenario does not state both pools -- a P/D topology on the CLI over a colocated
// scenario would simulate engines nobody described.
func openKernelPools(catalogRoot string) *kernelPools {
	if kernelOpened == nil || (prefillInstances == 0 && decodeInstances == 0) {
		return nil
	}
	if prefillDecodeInstances > 0 || encodeInstances > 0 {
		logrus.Fatalf("--latency-model %s: --prefill-decode-instances and --encode-instances are not "+
			"supported: a blis-schemas deployment's roles are colocated, prefill and decode, so no "+
			"scenario describes the engine a shared or encode instance would run", latencyModelBackendKernel())
	}
	repos := kernelmodel.Repos{Scenarios: kernelScenarioDir, Catalog: catalogRoot, Registry: kernelRegistry}
	p := &kernelPools{}
	for _, c := range []struct {
		role deployment.Role
		dst  **kernelmodel.Model
	}{{deployment.RolePrefill, &p.prefill}, {deployment.RoleDecode, &p.decode}} {
		m, err := kernelmodel.OpenRole(kernelScenario, repos, c.role)
		if err != nil {
			logrus.Fatalf("--latency-model %s: --prefill-instances/--decode-instances describe a "+
				"disaggregated run, but %v", latencyModelBackendKernel(), err)
		}
		*c.dst = m
	}
	ps, pe := p.prefill.Settings()
	ds, de := p.decode.Settings()
	if pe != nil || de != nil {
		logrus.Fatalf("--latency-model %s: scenario %q: prefill pool: %v; decode pool: %v",
			latencyModelBackendKernel(), kernelScenario, pe, de)
	}
	// A request's KV moves block for block: the decode side reserves the blocks the prefill
	// side filled, so the two engines must page identically.
	if ps.BlockSize != ds.BlockSize {
		logrus.Fatalf("--latency-model %s: scenario %q pages the prefill pool in %d-token blocks and "+
			"the decode pool in %d; a P/D handoff moves whole blocks, so the pools must agree",
			latencyModelBackendKernel(), kernelScenario, ps.BlockSize, ds.BlockSize)
	}
	// DP-as-placement expands every pool by one replica factor (--dp), so per-pool widths that
	// differ cannot be expressed as a placement.
	if ps.DataParallel != ds.DataParallel {
		logrus.Fatalf("--latency-model %s: scenario %q runs the prefill pool at dp %d and the decode "+
			"pool at dp %d; one replica factor applies to both pools, so their dp must match",
			latencyModelBackendKernel(), kernelScenario, ps.DataParallel, ds.DataParallel)
	}
	return p
}

// overrides is each role's engine, from its own pool's kernel.
func (p *kernelPools) overrides() (prefill, decode cluster.PoolOverrides) {
	return poolOverridesOf(p.prefill), poolOverridesOf(p.decode)
}

func poolOverridesOf(m *kernelmodel.Model) cluster.PoolOverrides {
	s, err := m.Settings()
	if err != nil {
		logrus.Fatalf("--latency-model %s: %s pool: %v", latencyModelBackendKernel(), m.Role(), err)
	}
	d := m.Deployment()
	tp := d.TP
	blocks := s.KVBlocks
	seqs, toks := int64(s.MaxNumSeqs), int64(s.MaxNumBatchedTokens)
	noCache := s.PrefixCachingDisabled
	o := cluster.PoolOverrides{
		TP: &tp, GPU: d.Hardware, LatencyModel: m, TotalKVBlocks: &blocks,
		MaxNumSeqs: &seqs, MaxNumBatchedTokens: &toks, PrefixCachingDisabled: &noCache,
	}
	if s.MaxModelLen > 0 {
		ml := int64(s.MaxModelLen)
		o.MaxModelLen = &ml
	}
	return o
}

// transferTime prices a handoff with the prefill pool's kernel, between the placements of the
// two instances in their pools. Instances are numbered prefill first, then decode
// (cluster.BuildPoolMembershipFromIndices), so an instance's rank in its pool is its offset
// from the pool's first instance. It reads the instance counts after DP-as-placement, so call
// it once they are final.
func (p *kernelPools) transferTime() func(int64, cluster.InstanceID, cluster.InstanceID) int64 {
	firstDecode := prefillInstances
	rankOf := func(id cluster.InstanceID) int {
		n, err := strconv.Atoi(strings.TrimPrefix(string(id), "instance_"))
		if err != nil {
			panic(fmt.Sprintf("kernel P/D pricing: unexpected instance id %q", id))
		}
		return n
	}
	return func(tokens int64, from, to cluster.InstanceID) int64 {
		return p.prefill.PDTransferTicks(tokens,
			p.prefill.PlacementOf(rankOf(from)), p.decode.PlacementOf(rankOf(to)-firstDecode))
	}
}

// requireKernelCapacity refuses a run whose instances do not fit the scenario's pools: each
// simulated instance is one data-parallel rank of the pool's layout, and a pool holds as many
// as its nodes have GPUs for. Called once the instance counts are final.
func requireKernelCapacity(p *kernelPools) {
	if kernelOpened == nil {
		return
	}
	check := func(role string, n int, m *kernelmodel.Model) {
		if c := m.RankCapacity(); n > c {
			logrus.Fatalf("--latency-model %s: %d %s instance(s) requested, but scenario %q's %s pool "+
				"holds %d rank(s) at its layout; raise the pool's nodes in the scenario or run fewer",
				latencyModelBackendKernel(), n, role, kernelScenario, m.Role(), c)
		}
	}
	if p == nil {
		check("", numInstances, kernelOpened)
		return
	}
	check("prefill", prefillInstances, p.prefill)
	check("decode", decodeInstances, p.decode)
}

func latencyModelBackendKernel() string { return sim.LatencyBackendKernel }

// kernelCPUTier is the catalog storage device the legacy --kv-cpu-blocks tier's
// CPU↔GPU reloads cross.
const kernelCPUTier = legacyKVTransferDeviceClass

// kernelTierTicks is the kernel's price for one transfer, in whole ticks rounded up: a
// transfer is not done until its last byte lands. An unknown tier is refused rather than
// charged the kernel's unbounded sentinel, which would stall the clock.
func kernelTierTicks(tier string, toTier bool, bytes int64, inService int) int64 {
	dir := kernel.DirectionFromTier
	if toTier {
		dir = kernel.DirectionToTier
	}
	d := kernelOpened.Kernel().TierTime(tier, dir, bytes, inService)
	if d >= time.Duration(math.MaxInt64/2) {
		logrus.Fatalf("--latency-model %s: the kernel prices no storage tier %q; name a device class "+
			"from the catalog's %s", sim.LatencyBackendKernel, tier, catalogStorageDevicesRelPath)
	}
	return max(1, (d.Nanoseconds()+999)/1000)
}

// applyKernelOffloadPricing has the kernel price every KV-offload transfer, on the kernel
// backend: each secondary tier through TierTime for its catalog device class, at the depth the
// transfer station is serving, and the legacy CPU tier as one whole per-block reload charge.
// The simulator keeps the servers, queues and timing; the kernel owns the price (blis-schemas
// kernel.TierTime: the slower of the device and the host link binds). It returns the legacy
// per-block charge, 0 off the kernel or with no legacy tier. Shared by run and replay.
func applyKernelOffloadPricing(cfg *sim.KVOffloadConfig) int64 {
	if kernelOpened == nil {
		return 0
	}
	if cfg.IsEnabled() {
		for i := range cfg.Tiers {
			class := cfg.Tiers[i].DeviceClass
			if class == "" {
				logrus.Fatalf("--latency-model %s: kv_offload secondary_tiers[%d] names no device_class; "+
					"the kernel prices a tier by its catalog device, so every tier must name one",
					sim.LatencyBackendKernel, i)
			}
			kernelTierTicks(class, false, 0, 1) // refuses an unknown device up front
			cfg.Tiers[i].ServiceTime = func(write bool, bytes int64, inService int) int64 {
				return kernelTierTicks(class, write, bytes, inService)
			}
		}
	}
	if kvCPUBlocks <= 0 {
		return 0
	}
	perBlock := kernelOpened.Kernel().SequenceVariableBytes(int(blockSizeTokens))
	return kernelTierTicks(kernelCPUTier, false, perBlock, 1)
}

// refuseExplicitTierPhysics refuses, on the kernel backend, a kv_offload tier that states its
// own bandwidth or latency: the kernel prices the tier from its catalog device, and a second
// source for the same physics would need precedence rules to reconcile.
func refuseExplicitTierPhysics(block *kvOffloadBlock) {
	if kernelOpened == nil || block == nil {
		return
	}
	for i, t := range block.SecondaryTiers {
		if t.ReadBandwidth != nil || t.WriteBandwidth != nil || t.BaseLatency != nil {
			logrus.Fatalf("--latency-model %s: kv_offload secondary_tiers[%d] states read_bandwidth, "+
				"write_bandwidth or base_latency, but the kernel prices a tier from its catalog "+
				"device_class; name the device and drop the explicit physics",
				sim.LatencyBackendKernel, i)
		}
	}
}
