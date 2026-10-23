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
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// kernelPools is a disaggregated kernel run's per-role wiring: one kernel per pool, since each
// role runs its own engine (parallelism, token budget, graph mode) and is priced by the kernel
// for that pool rather than by pool 0's copied to both.
type kernelPools struct {
	prefill, decode *kernelmodel.Model
	// connector is the scenario's pd_transfer.connector.
	connector string
}

// scenarioEndpointTopology uses the same role-specific kernel placements that
// price KV handoffs. Node and rack names are global within one simulated
// cluster, so replicas in different pools never get colliding host identities.
// No zone or region is invented; rack is emitted only when declared.
func scenarioEndpointTopology(model *kernelmodel.Model, pools *kernelPools, total, prefill, decode int) (map[cluster.InstanceID]cluster.EndpointTopology, error) {
	if model == nil {
		return nil, nil
	}
	out := make(map[cluster.InstanceID]cluster.EndpointTopology, total)
	for n := 0; n < total; n++ {
		m, rank := model, n
		if pools != nil {
			switch {
			case n < prefill:
				m = pools.prefill
			case n < prefill+decode:
				m, rank = pools.decode, n-prefill
			default:
				return nil, fmt.Errorf("instance_%d has no kernel pool placement", n)
			}
		}
		if rank >= m.RankCapacity() {
			return nil, fmt.Errorf("instance_%d rank %d exceeds the %s pool's %d-rank capacity", n, rank, m.Role(), m.RankCapacity())
		}
		placement, nodeCount, rackDeclared := m.EndpointPlacement(rank)
		t := cluster.EndpointTopology{NodeCount: nodeCount}
		if nodeCount == 1 {
			t.Hostname = fmt.Sprintf("blis-node-%d", placement.Node)
		}
		if rackDeclared && nodeCount == 1 {
			t.Rack = fmt.Sprintf("blis-rack-%d", placement.Rack)
		}
		out[cluster.InstanceID(fmt.Sprintf("instance_%d", n))] = t
	}
	return out, nil
}

// openKernelPools opens the prefill and decode pools of the run's scenario. It returns nil
// when no kernel is open or the run is not disaggregated, and is fatal when a disaggregated
// run's scenario does not state both pools -- a P/D topology on the CLI over a colocated
// scenario would simulate engines nobody described.
func openKernelPools(catalogRoot string) *kernelPools {
	// Checked before the early return below: a shared-only or encode+shared topology states no
	// dedicated prefill or decode instance, yet is still disaggregated, and would otherwise reach
	// cluster construction with no handoff pricer.
	if kernelOpened != nil && (prefillDecodeInstances > 0 || encodeInstances > 0) {
		logrus.Fatalf("--prefill-decode-instances and --encode-instances are not " +
			"supported: a blis-schemas deployment's roles are colocated, prefill and decode, so no " +
			"scenario describes the engine a shared or encode instance would run")
	}
	if kernelOpened == nil || (prefillInstances == 0 && decodeInstances == 0) {
		return nil
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
	shape, err := kernelmodel.ShapeOf(kernelScenario, repos)
	if err != nil {
		logrus.Fatalf("scenario %q: %v", kernelScenario, err)
	}
	p.connector = shape.PDConnector
	requireNIXLHandoff(p.connector)
	if p.connector == "MultiConnector" {
		// The schema states no MultiConnector children or their order, and the order decides
		// which connector loads a request's KV: the first, in config order, that reports
		// tokens (multi_connector.py:134-141, 429-448). BLIS assumes llm-d's composition,
		// NIXL listed first, which it cannot check.
		logrus.Warnf("scenario %q: pd_transfer.connector MultiConnector is simulated as NixlConnector "+
			"listed before OffloadingConnector (the first connector reporting tokens loads them); "+
			"blis-schemas does not state the children or their order, so this assumption is not "+
			"checked -- a deployment listing the OffloadingConnector first would load remotely "+
			"prefilled prompts from the decode CPU tier", kernelScenario)
	}
	// NIXL's prefill backoff (nixl/base_scheduler.py:385-407): the prefill stops short of the
	// last few prompt tokens and the decode side recomputes them. BLIS moves and decodes the
	// whole prompt, so a pool that backs off is refused rather than simulated without it.
	if b := p.prefill.PrefillBackoffTokens(); b > 0 {
		logrus.Fatalf("scenario %q: under NIXL this model's decode engine recomputes the last %d prompt "+
			"token(s) that the prefill engine does not compute (_prefill_backoff: a Mamba-state model, "+
			"or multi-module MTP drafting K>1); BLIS does not model that recompute", kernelScenario, b)
	}
	// For the same reason every instance must belong to a pool: one with no role would run the
	// first pool's engine with the global sizing, an engine the scenario did not describe.
	if prefillInstances+decodeInstances != numInstances {
		logrus.Fatalf("--num-instances %d, but --prefill-instances %d + --decode-instances %d "+
			"= %d: in a disaggregated run every instance is a prefill or a decode instance, so the "+
			"counts must add up", numInstances, prefillInstances, decodeInstances,
			prefillInstances+decodeInstances)
	}
	// A handoff crosses nodes (pools never share one), so it needs the inter-node fabric: the
	// kernel would price a crossing with no fabric at the on-node rate. blis-schemas already
	// refuses a multi-node cluster that names none, and a P/D deployment always spans two
	// nodes, so no check is repeated here; transferTime prices every rank pair up front.
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
	// The handoff moves the prefill pool's KV representation into the decode pool's blocks, and
	// is priced from the prefill pool's bytes: pools caching in different formats would move
	// pages the decode engine does not hold, with no conversion modelled. So the cache formats,
	// recurrent state included, must agree.
	requireSameKVCacheFormat(p.prefill, p.decode)
	// DP-as-placement expands every pool by one replica factor (the scenario's dp), so
	// per-pool widths that differ cannot be expressed as a placement.
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
	kernelPDOpened = p
	return p
}

// nixlConnectors are the pd_transfer connectors whose handoff BLIS models: vLLM v0.31.0's
// NIXL connectors (NixlConnector is an alias of NixlPullConnector, nixl/connector.py:9-12), and
// MultiConnector, which llm-d composes from a NixlConnector and an OffloadingConnector. The
// schema cannot see a MultiConnector's children, so, as the blis-schemas rules pack does
// (rules/v0_31 pack.go:964-968), it is taken to carry the NIXL one.
var nixlConnectors = map[string]bool{
	"NixlConnector": true, "NixlPullConnector": true, "NixlPushConnector": true, "MultiConnector": true,
}

// requireNIXLHandoff refuses a P/D scenario whose handoff is not NIXL's. What BLIS simulates
// around a handoff is NIXL's protocol: the decode side reserves the blocks at transfer start
// (WAITING_FOR_REMOTE_KVS), pulls only what its own prefix cache lacks, and the prefill side
// holds its blocks until the pull completes or the lease expires (INV-PD-7). Another connector
// (Mooncake, LMCache, P2P NCCL) moves and frees KV on other terms, which would be simulated as
// NIXL's without a word.
func requireNIXLHandoff(connector string) {
	if !nixlConnectors[connector] {
		logrus.Fatalf("scenario %q hands KV off with pd_transfer.connector %q; BLIS models NIXL's "+
			"handoff (NixlConnector, NixlPullConnector, NixlPushConnector, or a MultiConnector over "+
			"one), so state one of those", kernelScenario, connector)
	}
}

// holdsPrefillKV reports whether the run's prefill engines hold a finished request's KV for
// the decode pull: every NIXL handoff does (nixl/pull_scheduler.py:247-266 and
// push_scheduler.py:260-270 at v0.31.0). False for a colocated run (nil receiver).
func (p *kernelPools) holdsPrefillKV() bool {
	return p != nil && nixlConnectors[p.connector]
}

// handoffPricing returns the run's handoff price and, when contention is on, the demand behind
// it. Both nil for a colocated run, where contention is refused.
func (p *kernelPools) handoffPricing(contention bool) (
	func(int64, cluster.InstanceID, cluster.InstanceID) int64,
	func(int64, cluster.InstanceID, cluster.InstanceID) kvtransfer.Flow,
) {
	if p == nil {
		if contention {
			logrus.Fatalf("--pd-transfer-contention needs a disaggregated run: " +
				"pass --prefill-instances and --decode-instances with a P/D scenario")
		}
		return nil, nil
	}
	price := p.transferTime()
	if !contention {
		return price, nil
	}
	return price, p.transferDemand()
}

// transferDemand is the demand behind transferTime: the prefill pool kernel's
// PDTransferDemand between the two instances' placements, as payload, floor and links. The
// blis-schemas law that alone a demand costs its PDTransferTime is checked for every rank pair
// up front (a block's handoff), so a kernel breaking it is refused before the run rather than
// arbitrated.
func (p *kernelPools) transferDemand() func(int64, cluster.InstanceID, cluster.InstanceID) kvtransfer.Flow {
	firstDecode := prefillInstances
	demand := func(tokens int64, from, to kernel.Placement) kvtransfer.Flow {
		d := p.prefill.Kernel().PDTransferDemand(int(tokens), from, to)
		if err := d.Check(); err != nil {
			logrus.Fatalf("scenario %q: the kernel's demand for a %d-token handoff from %+v to %+v is "+
				"malformed: %v", kernelScenario, tokens, from, to, err)
		}
		f := kvtransfer.Flow{Bytes: d.Bytes, FloorNs: d.Floor.Nanoseconds()}
		for _, l := range d.Links {
			f.Links = append(f.Links, kvtransfer.Link{ID: string(l.Link), BytesPerSecond: l.BytesPerSecond})
		}
		return f
	}
	for i := 0; i < prefillInstances; i++ {
		for j := 0; j < decodeInstances; j++ {
			from, to := p.prefill.PlacementOf(i), p.decode.PlacementOf(j)
			if a, b := kvtransfer.AloneTicks(demand(blockSizeTokens, from, to)),
				p.prefill.PDTransferTicks(blockSizeTokens, from, to); a != b {
				logrus.Fatalf("scenario %q: the kernel's handoff demand from prefill rank %d to decode "+
					"rank %d costs %d ticks alone but it prices the handoff at %d; blis-schemas requires "+
					"PDTransferTime == PDTransferDemand.AloneTime()", kernelScenario, i, j, a, b)
			}
		}
	}
	return func(tokens int64, from, to cluster.InstanceID) kvtransfer.Flow {
		return demand(tokens, p.prefill.PlacementOf(rankInPool(from, firstDecode)),
			p.decode.PlacementOf(rankInPool(to, firstDecode)))
	}
}

// requireSameKVCacheFormat refuses a disaggregated scenario whose pools state different KV
// cache formats: the attention cache dtype and, for recurrent layers, the state dtypes and
// cache mode. An unstated field is the engine default, so it agrees only with another unstated
// one (the two pools share the model, whose default resolves alike).
func requireSameKVCacheFormat(prefill, decode *kernelmodel.Model) {
	pe, perr := prefill.Engine()
	de, derr := decode.Engine()
	if perr != nil || derr != nil {
		logrus.Fatalf("scenario %q: prefill pool: %v; decode pool: %v", kernelScenario, perr, derr)
	}
	for _, f := range []struct{ name, prefill, decode string }{
		{"cache_dtype", pe.CacheDType, de.CacheDType},
		{"mamba_cache_dtype", pe.MambaCacheDType, de.MambaCacheDType},
		{"mamba_ssm_cache_dtype", pe.MambaSSMCacheDType, de.MambaSSMCacheDType},
		{"mamba_cache_mode", pe.MambaCacheMode, de.MambaCacheMode},
	} {
		if f.prefill != f.decode {
			logrus.Fatalf("scenario %q caches KV as %s %q in the prefill pool and %q in the "+
				"decode pool; a P/D handoff moves the prefill pool's cache into the decode pool's "+
				"blocks, and no conversion between formats is modelled, so the pools must agree",
				kernelScenario, f.name, f.prefill, f.decode)
		}
	}
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
	state := s.SequenceState
	hit := prefixHitOf(string(m.Role())+" pool", m)
	logSequenceFixedCharge(string(m.Role())+" pool", s)
	o := cluster.PoolOverrides{
		TP: &tp, GPU: d.Hardware, LatencyModel: m, TotalKVBlocks: &blocks, MaxModelLen: &ml,
		MaxNumSeqs: &seqs, MaxNumBatchedTokens: &toks, PrefixCachingDisabled: &noCache,
		SequenceState: &state, PrefixHit: &hit,
	}
	poolEngineKnobs(m, &o)
	// Each pool runs its own engine's scheduling policy, unless a flag or policy bundle chose
	// one for the run, which then applies to every pool and is recorded where it departs.
	stated := enginePolicyOf(m)
	if schedulerSource != "" {
		recordPolicyOverride("scheduler", string(m.Role()), stated.scheduler, scheduler, schedulerSource)
	} else {
		o.Scheduler = stated.scheduler
	}
	if preemptionSource != "" {
		recordPolicyOverride("preemption_policy", string(m.Role()), stated.preemption, preemptionPolicy, preemptionSource)
	} else {
		o.PreemptionPolicy = stated.preemption
	}
	return o
}

// logSequenceFixedCharge reports the per-sequence term of an engine's KV capacity (#1941):
// the steady hold each decoding sequence's bounded state settles at, and the admission cap
// vLLM gates a new request on and sizes its startup capacity figure by. Silent for a model
// with no such caches, whose capacity it does not touch.
func logSequenceFixedCharge(what string, s kernelmodel.Settings) {
	if s.SequenceState.Hold == nil {
		return
	}
	atCap := min(int64(s.MaxNumSeqs)*s.SequenceSteadyBlocks, s.KVBlocks)
	logrus.Infof("KV capacity (%s): each sequence's bounded state (window rings, compressor/indexer/"+
		"recurrent state) is held step by step: %d bytes per rank (%d blocks of %d bytes) once "+
		"decoding, more while a prefill chunk is in flight; vLLM's admission gate caps it at %d "+
		"bytes (%d blocks). At max_num_seqs %d the steady hold is %d of %d blocks (%.1f%%)",
		what, s.SequenceSteadyBytes, s.SequenceSteadyBlocks, s.KVBytesPerBlock,
		s.SequenceAdmissionBytes, s.SequenceAdmissionBlocks, s.MaxNumSeqs, atCap, s.KVBlocks,
		100*float64(atCap)/float64(s.KVBlocks))
}

// requireWindowFits refuses an engine with no window at all: neither the deployment nor the
// model graph states one, and vLLM would derive it from an HF config no document the kernel
// reads carries. A window one request cannot fit in the KV cache is refused when the settings
// are sized, with vLLM's start-up message (kernelmodel.checkEnoughKVCacheMemory).
func requireWindowFits(what string, s kernelmodel.Settings) {
	if s.MaxModelLen <= 0 {
		logrus.Fatalf("scenario %q's %s states no engine.max_model_len and its model graph no "+
			"context window; state the window the engine serves", kernelScenario, what)
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

// kernelCPUTier is the catalog storage device the kernel prices the legacy --kv-cpu-blocks
// tier's CPU↔GPU reloads at: host DRAM across the PCIe/NVLink boundary.
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

// shareTiersByNode places each engine's secondary tiers on the node the kernel places its rank
// on, so a node's engines share one tier (sim.OffloadTierSharing). rankOf maps an instance to
// its rank in m's pool.
func shareTiersByNode(cfg *sim.KVOffloadConfig, m *kernelmodel.Model, rankOf func(string) int) {
	if !cfg.IsEnabled() || len(cfg.Tiers) == 0 || m == nil {
		return
	}
	cfg.Sharing = &sim.OffloadTierSharing{NodeOf: func(id string) int {
		return m.PlacementOf(rankOf(id)).Node
	}}
}

// instanceRank is a colocated run's instance index ("instance_<n>"), the rank the kernel places.
func instanceRank(id string) int { return rankInPool(cluster.InstanceID(id), math.MaxInt) }

// applyKernelOffloadPricing has the kernel price every KV-offload transfer of a colocated
// run, from the run's one pool: see priceOffload. It returns the legacy per-block charge, 0
// when no kernel is open or with no legacy tier. A disaggregated run's pools are priced separately,
// each by its own kernel (kernelPools.applyOffload). Shared by run and replay.
func applyKernelOffloadPricing(cfg *sim.KVOffloadConfig) int64 {
	if kernelOpened == nil {
		return 0
	}
	shareTiersByNode(cfg, kernelOpened, instanceRank)
	return priceOffload(kernelOpened, cfg)
}

// priceOffload has m's kernel price every KV-offload transfer of m's pool: each secondary
// tier through TierTime for its catalog device class, at the depth the transfer station is
// serving, the offload CPU tier's CPU->GPU loads at cpu_dram (INV-K7), and the legacy CPU
// tier as one whole per-block reload charge, all at m's own per-block bytes
// (SequenceVariableBytes).
//
// TierTime's contract is a GPU↔tier transfer, while the station's secondary-tier jobs run
// CPU↔tier; the kernel's price is applied to them deliberately -- the device and the host
// link it names are the ones those jobs cross -- rather than as an exact match. The simulator
// keeps the servers, queues and timing; the kernel owns the price (blis-schemas
// kernel.TierTime: the slower of the device and the host link binds).
func priceOffload(m *kernelmodel.Model, cfg *sim.KVOffloadConfig) int64 {
	perBlock := m.Kernel().SequenceVariableBytes(int(blockSizeTokens))
	if cfg.IsEnabled() {
		cfg.CPUCopies = m.OffloadCopies()
		if err := cfg.CheckFitsOneChunk(); err != nil {
			logrus.Fatalf("%v", err)
		}
	}
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
			// A catalog device's rates are O_DIRECT rates; buffered (page-cache) I/O, vLLM's
			// fallback where O_DIRECT is unsupported, is another regime the kernel does not
			// price yet (blis-latency-kernel#41), so it is refused rather than priced as O_DIRECT.
			if !cfg.Tiers[i].DirectIO {
				logrus.Fatalf("kv_offload secondary_tiers[%d] (%s) uses buffered I/O (direct_io: "+
					"false), which the kernel does not price (blis-latency-kernel#41); only "+
					"direct_io: true tiers can be simulated", i, class)
			}
			// Price one real block each way up front, so an unknown device or one with no
			// bandwidth in a direction is refused now rather than at its first transfer.
			kernelTierTicks(m, class, true, perBlock, 1)
			kernelTierTicks(m, class, false, perBlock, 1)
			cfg.Tiers[i].ServiceTime = func(write bool, bytes int64, inService int) int64 {
				return kernelTierTicks(m, class, write, bytes, inService)
			}
		}
		// vLLM's CPU->GPU loads run one at a time on the handler's stream chain
		// (kv_offload/cpu/gpu_worker.py:696-701 at v0.31.0), so each is priced alone.
		kernelTierTicks(m, kernelCPUTier, false, perBlock, 1)
		cfg.CPULoadTime = func(bytes int64) int64 {
			return kernelTierTicks(m, kernelCPUTier, false, bytes, 1)
		}
		if cfg.Simple {
			// The Simple connector's stores are device-to-host copies on their own stream
			// (vllm/v1/simple_kv_offload/worker.py:164-167, 273-295 at v0.31.0).
			kernelTierTicks(m, kernelCPUTier, true, perBlock, 1)
			cfg.CPUStoreTime = func(bytes int64) int64 {
				return kernelTierTicks(m, kernelCPUTier, true, bytes, 1)
			}
			if cfg.LazyOffload {
				// target_free: 2 x ceil(max_num_batched_tokens / block_size) for one
				// full-attention group (simple_kv_offload/manager.py:312-330).
				cfg.LazyTargetFree = 2 * ((maxNumBatchedTokens + blockSizeTokens - 1) / blockSizeTokens)
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
	// A disaggregated engine with an offload tier runs both connectors, which only a
	// MultiConnector composes: with kv_offloading_size set, vLLM overwrites kv_connector with
	// the OffloadingConnector and kv_role with kv_both (vllm/config/vllm.py:1124-1158 at
	// v0.31.0), so a plain NIXL connector and a tier cannot run together.
	if p.connector != "MultiConnector" {
		logrus.Fatalf("scenario %q offloads KV under P/D with pd_transfer.connector %q; vLLM runs a NIXL "+
			"handoff and an offload tier together only through a MultiConnector (stating "+
			"kv_offloading_size replaces kv_connector with OffloadingConnector, vllm/config/vllm.py:1124-1158), "+
			"so state pd_transfer.connector: MultiConnector", kernelScenario, p.connector)
	}
	for _, c := range []struct {
		role string
		m    *kernelmodel.Model
		dst  *cluster.PoolOverrides
	}{{"prefill", p.prefill, prefill}, {"decode", p.decode, decode}} {
		pool := cfg
		// Each pool gets its own tier slice: priceOffload installs a per-pool ServiceTime.
		pool.Tiers = append([]sim.KVOffloadTier(nil), cfg.Tiers...)
		// A pool's block is its own layout's slice of the KV, so pools of different widths
		// hold different bytes per block, and the tiers' block capacity and job sizes follow.
		if pool.IsEnabled() {
			pool.PerBlockBytes = c.m.Kernel().SequenceVariableBytes(int(blockSizeTokens))
			if err := pool.CheckFitsOneChunk(); err != nil {
				logrus.Fatalf("%s pool: %v", c.role, err)
			}
		}
		firstDecode := prefillInstances
		shareTiersByNode(&pool, c.m, func(id string) int {
			return rankInPool(cluster.InstanceID(id), firstDecode)
		})
		ticks := priceOffload(c.m, &pool)
		c.dst.KVOffload = &pool
		c.dst.KVTransferTicksPerBlock = &ticks
	}
}

// refuseExplicitTierPhysics refuses a kv_offload tier that states its own bandwidth or
// latency: the kernel prices the tier from its catalog device, and a second
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
// the kernel's budget shrinks by it. No-op when no kernel is open or with no reservation.
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
