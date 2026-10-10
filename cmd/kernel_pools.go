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
		logrus.Fatalf("--prefill-decode-instances and --encode-instances are not " +
			"supported: a blis-schemas deployment's roles are colocated, prefill and decode, so no " +
			"scenario describes the engine a shared or encode instance would run")
	}
	repos := kernelmodel.Repos{Scenarios: kernelScenarioDir, Catalog: catalogRoot, Registry: kernelRegistry}
	p := &kernelPools{}
	for _, c := range []struct {
		role deployment.Role
		dst  **kernelmodel.Model
	}{{deployment.RolePrefill, &p.prefill}, {deployment.RoleDecode, &p.decode}} {
		m, err := kernelmodel.OpenRole(kernelScenario, repos, c.role)
		if err != nil {
			logrus.Fatalf("--prefill-instances/--decode-instances describe a "+
				"disaggregated run, but %v", err)
		}
		*c.dst = m
	}
	// For the same reason every instance must belong to a pool: one with no role would run the
	// first pool's engine with the global sizing, an engine the scenario did not describe.
	if prefillInstances+decodeInstances != numInstances {
		logrus.Fatalf("--num-instances %d, but --prefill-instances %d + --decode-instances %d "+
			"= %d: in a disaggregated run every instance is a prefill or a decode instance, so the "+
			"counts must add up", numInstances, prefillInstances, decodeInstances,
			prefillInstances+decodeInstances)
	}
	// A handoff crosses nodes: prefill and decode pools never share one. With no fabric named,
	// the kernel would price that crossing at the on-node rate rather than refuse it.
	if shape, err := kernelmodel.ShapeOf(kernelScenario, repos); err != nil {
		logrus.Fatalf("scenario %q: %v", kernelScenario, err)
	} else if shape.Fabric == "" {
		logrus.Fatalf("scenario %q is disaggregated but names no cluster.fabric; a P/D KV "+
			"handoff crosses nodes, so the run needs the inter-node fabric it crosses", kernelScenario)
	}
	ps, pe := p.prefill.Settings()
	ds, de := p.decode.Settings()
	if pe != nil || de != nil {
		logrus.Fatalf("scenario %q: prefill pool: %v; decode pool: %v",
			kernelScenario, pe, de)
	}
	// A request's KV moves block for block: the decode side reserves the blocks the prefill
	// side filled, so the two engines must page identically.
	if ps.BlockSize != ds.BlockSize {
		logrus.Fatalf("scenario %q pages the prefill pool in %d-token blocks and "+
			"the decode pool in %d; a P/D handoff moves whole blocks, so the pools must agree",
			kernelScenario, ps.BlockSize, ds.BlockSize)
	}
	// DP-as-placement expands every pool by one replica factor (--dp), so per-pool widths that
	// differ cannot be expressed as a placement.
	if ps.DataParallel != ds.DataParallel {
		logrus.Fatalf("scenario %q runs the prefill pool at dp %d and the decode "+
			"pool at dp %d; one replica factor applies to both pools, so their dp must match",
			kernelScenario, ps.DataParallel, ds.DataParallel)
	}
	// The simulator's speculative config (draft length, acceptance) is one per run, while each
	// pool's kernel prices its own draft width. Pools that disagree would advance requests by
	// one draft length and charge for another.
	if ps.SpeculativeTokens != ds.SpeculativeTokens || ps.SpeculativeMethod != ds.SpeculativeMethod {
		logrus.Fatalf("scenario %q drafts %d %q tokens in the prefill pool and %d "+
			"%q in the decode pool; one speculative configuration applies to the run, so the pools "+
			"must agree", kernelScenario, ps.SpeculativeTokens,
			ps.SpeculativeMethod, ds.SpeculativeTokens, ds.SpeculativeMethod)
	}
	return p
}

// overrides is each role's engine, from its own pool's kernel.
func (p *kernelPools) overrides() (prefill, decode cluster.PoolOverrides) {
	return poolOverridesOf(p.prefill), poolOverridesOf(p.decode)
}

func poolOverridesOf(m *kernelmodel.Model) cluster.PoolOverrides {
	s, err := m.SettingsReserving(loraReservedBytesForKV)
	if err != nil {
		logrus.Fatalf("%s pool: %v", m.Role(), err)
	}
	requireWindowFits(string(m.Role())+" pool", s)
	d := m.Deployment()
	tp := d.TP
	blocks := s.KVBlocks
	seqs, toks := int64(s.MaxNumSeqs), int64(s.MaxNumBatchedTokens)
	noCache := s.PrefixCachingDisabled
	ml := int64(s.MaxModelLen)
	return cluster.PoolOverrides{
		TP: &tp, GPU: d.Hardware, LatencyModel: m, TotalKVBlocks: &blocks, MaxModelLen: &ml,
		MaxNumSeqs: &seqs, MaxNumBatchedTokens: &toks, PrefixCachingDisabled: &noCache,
	}
}

// requireWindowFits refuses an engine whose stated window one request could not fit in, as
// vLLM does at startup, and an engine that states no window at all: unstated, vLLM derives it
// from the model's max_position_embeddings, which no document the kernel reads carries.
func requireWindowFits(what string, s kernelmodel.Settings) {
	if s.MaxModelLen <= 0 {
		logrus.Fatalf("scenario %q's %s states no engine.max_model_len; state the "+
			"window the engine serves", kernelScenario, what)
	}
	if capacity := s.KVBlocks * int64(s.BlockSize); int64(s.MaxModelLen) > capacity {
		logrus.Fatalf("scenario %q's %s states max_model_len %d, but one rank's KV "+
			"budget holds %d tokens (%d blocks of %d); vLLM refuses to start such an engine. Lower "+
			"max_model_len in the scenario", kernelScenario, what,
			s.MaxModelLen, capacity, s.KVBlocks, s.BlockSize)
	}
}

// transferTime prices a handoff with the prefill pool's kernel, between the placements of the
// two instances in their pools. It reads the instance counts after DP-as-placement, so call it
// once they are final. Every prefill-decode rank pair is priced once up front, so a pair the
// fabric cannot price (a rack boundary with no bandwidth, say) is refused now rather than
// panicking mid-run.
func (p *kernelPools) transferTime() func(int64, cluster.InstanceID, cluster.InstanceID) int64 {
	firstDecode := prefillInstances
	block := int(blockSizeTokens)
	for i := 0; i < prefillInstances; i++ {
		for j := 0; j < decodeInstances; j++ {
			from, to := p.prefill.PlacementOf(i), p.decode.PlacementOf(j)
			if d := p.prefill.Kernel().PDTransferTime(block, from, to); d < 0 || d >= time.Duration(math.MaxInt64/2) {
				logrus.Fatalf("scenario %q: the kernel cannot price a KV handoff from prefill "+
					"rank %d (%+v) to decode rank %d (%+v) -- it reports %v; the link between them "+
					"states no bandwidth -- check cluster.fabric", kernelScenario, i, from, j, to, d)
			}
		}
	}
	return func(tokens int64, from, to cluster.InstanceID) int64 {
		return p.prefill.PDTransferTicks(tokens,
			p.prefill.PlacementOf(rankInPool(from, firstDecode)),
			p.decode.PlacementOf(rankInPool(to, firstDecode)))
	}
}

// rankInPool is an instance's rank within its pool. Instances are numbered prefill first, then
// decode (cluster.BuildPoolMembershipFromIndices), so a prefill instance's rank is its number
// and a decode instance's is its offset from the first decode instance.
func rankInPool(id cluster.InstanceID, firstDecode int) int {
	n, err := strconv.Atoi(strings.TrimPrefix(string(id), "instance_"))
	if err != nil || n < 0 {
		panic(fmt.Sprintf("kernel P/D pricing: unexpected instance id %q", id))
	}
	if n >= firstDecode {
		return n - firstDecode
	}
	return n
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
			logrus.Fatalf("%d %s instance(s) requested, but scenario %q's %s pool "+
				"holds %d rank(s) at its layout; raise the pool's nodes in the scenario or run fewer",
				n, role, kernelScenario, m.Role(), c)
		}
	}
	if p == nil {
		check("", numInstances, kernelOpened)
		return
	}
	check("prefill", prefillInstances, p.prefill)
	check("decode", decodeInstances, p.decode)
}

// kernelCPUTier is the catalog storage device the legacy --kv-cpu-blocks tier's
// CPU↔GPU reloads cross: host DRAM across the PCIe/NVLink boundary.
const kernelCPUTier = "cpu_dram"

// kernelTierTicks is m's kernel's price for one transfer, in whole ticks rounded up: a
// transfer is not done until its last byte lands. An unknown tier is refused rather than
// charged the kernel's unbounded sentinel, which would stall the clock.
func kernelTierTicks(m *kernelmodel.Model, tier string, toTier bool, bytes int64, inService int) int64 {
	dir := kernel.DirectionFromTier
	if toTier {
		dir = kernel.DirectionToTier
	}
	d := m.Kernel().TierTime(tier, dir, bytes, inService)
	if d >= time.Duration(math.MaxInt64/2) {
		logrus.Fatalf("the kernel cannot price a %d-byte transfer %s storage tier %q: "+
			"either the catalog's %s defines no such device, or it states no bandwidth in that "+
			"direction", bytes, map[bool]string{true: "to", false: "from"}[toTier],
			tier, catalogStorageDevicesRelPath)
	}
	return max(1, (d.Nanoseconds()+999)/1000)
}

// applyKernelOffloadPricing has the kernel price every KV-offload transfer of a colocated
// run, from the run's one pool: see priceOffload. It returns the legacy per-block charge, 0
// off the kernel or with no legacy tier. A disaggregated run's pools are priced separately,
// each by its own kernel (kernelPools.applyOffload). Shared by run and replay.
func applyKernelOffloadPricing(cfg *sim.KVOffloadConfig) int64 {
	if kernelOpened == nil {
		return 0
	}
	return priceOffload(kernelOpened, cfg)
}

// priceOffload has m's kernel price every KV-offload transfer of m's pool: each secondary
// tier through TierTime for its catalog device class, at the depth the transfer station is
// serving, and the legacy CPU tier as one whole per-block reload charge, both at m's own
// per-block bytes (SequenceVariableBytes).
//
// TierTime's contract is a GPU↔tier transfer, while the station's secondary-tier jobs run
// CPU↔tier; the kernel's price is applied to them deliberately -- the device and the host
// link it names are the ones those jobs cross -- rather than as an exact match. The simulator
// keeps the servers, queues and timing; the kernel owns the price (blis-schemas
// kernel.TierTime: the slower of the device and the host link binds).
func priceOffload(m *kernelmodel.Model, cfg *sim.KVOffloadConfig) int64 {
	perBlock := m.Kernel().SequenceVariableBytes(int(blockSizeTokens))
	if (cfg.IsEnabled() || kvCPUBlocks > 0) && perBlock <= 0 {
		logrus.Fatalf("the kernel prices a %d-token block at %d bytes; offload "+
			"needs a block to occupy memory", blockSizeTokens, perBlock)
	}
	if cfg.IsEnabled() {
		for i := range cfg.Tiers {
			class := cfg.Tiers[i].DeviceClass
			if class == "" {
				logrus.Fatalf("kv_offload secondary_tiers[%d] names no device_class; "+
					"the kernel prices a tier by its catalog device, so every tier must name one",
					i)
			}
			if !cfg.Tiers[i].DirectIO {
				logrus.Warnf("kv_offload secondary_tiers[%d] (%s) uses buffered I/O (direct_io: "+
					"false); the kernel prices the tier at its catalog device's rates, so page-cache "+
					"effects of buffered I/O are not modelled", i, class)
			}
			// Price one real block each way up front, so an unknown device or one with no
			// bandwidth in a direction is refused now rather than at its first transfer.
			kernelTierTicks(m, class, true, perBlock, 1)
			kernelTierTicks(m, class, false, perBlock, 1)
			cfg.Tiers[i].ServiceTime = func(write bool, bytes int64, inService int) int64 {
				return kernelTierTicks(m, class, write, bytes, inService)
			}
		}
	}
	if kvCPUBlocks <= 0 {
		return 0
	}
	return kernelTierTicks(m, kernelCPUTier, false, perBlock, 1)
}

// applyOffload prices the run's KV offload once per pool, each by that pool's own kernel,
// and records it on the pool's overrides: the same tiers and capacities (cfg, the run's one
// offload description), sized and priced from each pool's layout. No-op when the run
// offloads nothing.
func (p *kernelPools) applyOffload(cfg sim.KVOffloadConfig, prefill, decode *cluster.PoolOverrides) {
	if !cfg.IsEnabled() && kvCPUBlocks <= 0 {
		return
	}
	for _, c := range []struct {
		m   *kernelmodel.Model
		dst *cluster.PoolOverrides
	}{{p.prefill, prefill}, {p.decode, decode}} {
		pool := cfg
		// Each pool gets its own tier slice: priceOffload installs a per-pool ServiceTime.
		pool.Tiers = append([]sim.KVOffloadTier(nil), cfg.Tiers...)
		// A pool's block is its own layout's slice of the KV, so pools of different widths
		// hold different bytes per block, and the tiers' block capacity and job sizes follow.
		if pool.IsEnabled() {
			pool.PerBlockBytes = c.m.Kernel().SequenceVariableBytes(int(blockSizeTokens))
		}
		ticks := priceOffload(c.m, &pool)
		c.dst.KVOffload = &pool
		c.dst.KVTransferTicksPerBlock = &ticks
	}
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
			logrus.Fatalf("kv_offload secondary_tiers[%d] states read_bandwidth, "+
				"write_bandwidth or base_latency, but the kernel prices a tier from its catalog "+
				"device_class; name the device and drop the explicit physics",
				i)
		}
	}
}

// applyKernelLoRAReservation re-sizes a kernel run's KV pool with the static LoRA adapter
// reservation set aside, once the LoRA config is known (it is resolved after the deployment
// is adopted). vLLM takes the reservation beside the weights before profiling the KV pool, so
// the kernel's budget shrinks by it. No-op off the kernel or with no reservation.
func applyKernelLoRAReservation() {
	if kernelOpened == nil || loraReservedBytesForKV <= 0 {
		return
	}
	s, err := kernelOpened.SettingsReserving(loraReservedBytesForKV)
	if err != nil {
		logrus.Fatalf("scenario %q with a %d-byte LoRA adapter reservation: %v",
			kernelScenario, loraReservedBytesForKV, err)
	}
	requireWindowFits("pool with the LoRA adapter reservation set aside", s)
	totalKVBlocks = s.KVBlocks
}
