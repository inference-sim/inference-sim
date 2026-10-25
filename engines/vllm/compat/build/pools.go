// Moved from inference-sim/cmd/kernel_pools.go at cca6cac4d1b948709fc45bab3a47d5f3151bcaff.

package build

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/compat/kvtransfer"
)

// Topology is a run's instance counts after DP-as-placement: every instance is numbered
// "instance_<n>", prefill instances first, then decode (BuildPoolMembershipFromIndices in the
// simulator's cluster).
type Topology struct {
	NumInstances, Prefill, Decode int
	// SharedOrEncode is whether the CLI asked for shared (prefill-decode) or encode
	// instances, which no blis-schemas deployment describes.
	SharedOrEncode bool
}

// Placement is where the kernel places one instance's engine, for the router's view of the
// endpoints: its node (and rack, when the fabric states racks), or how many nodes it spans.
type Placement struct {
	Hostname  string
	Rack      string
	NodeCount int
}

// Pools is a disaggregated kernel run's per-role wiring: one kernel per pool, since each
// role runs its own engine (parallelism, token budget, graph mode) and is priced by the kernel
// for that pool rather than by pool 0's copied to both.
type Pools struct {
	Prefill, Decode *kernelmodel.Model
	// Connector is the scenario's pd_transfer.connector.
	Connector string
	// Warnings are the startup warnings opening the pools produced, in order, for the caller
	// to log.
	Warnings []string
}

// EndpointPlacements uses the same role-specific kernel placements that price KV handoffs.
// Node and rack names are global within one simulated cluster, so replicas in different pools
// never get colliding host identities. No zone or region is invented; rack is emitted only
// when declared. model is the colocated run's kernel; pools, when non-nil, the P/D pools.
func EndpointPlacements(model *kernelmodel.Model, pools *Pools, total, prefill, decode int) (map[string]Placement, error) {
	if model == nil {
		return nil, nil
	}
	out := make(map[string]Placement, total)
	for n := 0; n < total; n++ {
		m, rank := model, n
		if pools != nil {
			switch {
			case n < prefill:
				m = pools.Prefill
			case n < prefill+decode:
				m, rank = pools.Decode, n-prefill
			default:
				return nil, fmt.Errorf("instance_%d has no kernel pool placement", n)
			}
		}
		if rank >= m.RankCapacity() {
			return nil, fmt.Errorf("instance_%d rank %d exceeds the %s pool's %d-rank capacity", n, rank, m.Role(), m.RankCapacity())
		}
		placement, nodeCount, rackDeclared := m.EndpointPlacement(rank)
		t := Placement{NodeCount: nodeCount}
		if nodeCount == 1 {
			t.Hostname = fmt.Sprintf("blis-node-%d", placement.Node)
		}
		if rackDeclared && nodeCount == 1 {
			t.Rack = fmt.Sprintf("blis-rack-%d", placement.Rack)
		}
		out[fmt.Sprintf("instance_%d", n)] = t
	}
	return out, nil
}

// OpenPools opens the prefill and decode pools of the run's scenario. It returns nil when no
// kernel is open (colocated, false) or the run is not disaggregated, and refuses a
// disaggregated run whose scenario does not state both pools -- a P/D topology on the CLI over
// a colocated scenario would simulate engines nobody described.
func OpenPools(scenario string, repos kernelmodel.Repos, kernelOpen bool, top Topology) (*Pools, error) {
	// Checked before the early return below: a shared-only or encode+shared topology states no
	// dedicated prefill or decode instance, yet is still disaggregated, and would otherwise reach
	// cluster construction with no handoff pricer.
	if kernelOpen && top.SharedOrEncode {
		return nil, fmt.Errorf("--prefill-decode-instances and --encode-instances are not " +
			"supported: a blis-schemas deployment's roles are colocated, prefill and decode, so no " +
			"scenario describes the engine a shared or encode instance would run")
	}
	if !kernelOpen || (top.Prefill == 0 && top.Decode == 0) {
		return nil, nil
	}
	p := &Pools{}
	for _, c := range []struct {
		role deployment.Role
		dst  **kernelmodel.Model
	}{{deployment.RolePrefill, &p.Prefill}, {deployment.RoleDecode, &p.Decode}} {
		m, err := kernelmodel.OpenRole(scenario, repos, c.role)
		if err != nil {
			return nil, fmt.Errorf("--prefill-instances/--decode-instances describe a "+
				"disaggregated run, but %v", err)
		}
		*c.dst = m
	}
	shape, err := kernelmodel.ShapeOf(scenario, repos)
	if err != nil {
		return nil, fmt.Errorf("scenario %q: %v", scenario, err)
	}
	p.Connector = shape.PDConnector
	if err := RequireNIXLHandoff(scenario, p.Connector); err != nil {
		return nil, err
	}
	if p.Connector == "MultiConnector" {
		// The schema states no MultiConnector children or their order, and the order decides
		// which connector loads a request's KV: the first, in config order, that reports
		// tokens (multi_connector.py:134-141, 429-448). BLIS assumes llm-d's composition,
		// NIXL listed first, which it cannot check.
		p.Warnings = append(p.Warnings, fmt.Sprintf("scenario %q: pd_transfer.connector MultiConnector is simulated as NixlConnector "+
			"listed before OffloadingConnector (the first connector reporting tokens loads them); "+
			"blis-schemas does not state the children or their order, so this assumption is not "+
			"checked -- a deployment listing the OffloadingConnector first would load remotely "+
			"prefilled prompts from the decode CPU tier", scenario))
	}
	// NIXL's prefill backoff (nixl/base_scheduler.py:385-407): the prefill stops short of the
	// last few prompt tokens and the decode side recomputes them. BLIS moves and decodes the
	// whole prompt, so a pool that backs off is refused rather than simulated without it.
	if b := p.Prefill.PrefillBackoffTokens(); b > 0 {
		return nil, fmt.Errorf("scenario %q: under NIXL this model's decode engine recomputes the last %d prompt "+
			"token(s) that the prefill engine does not compute (_prefill_backoff: a Mamba-state model, "+
			"or multi-module MTP drafting K>1); BLIS does not model that recompute", scenario, b)
	}
	// For the same reason every instance must belong to a pool: one with no role would run the
	// first pool's engine with the global sizing, an engine the scenario did not describe.
	if top.Prefill+top.Decode != top.NumInstances {
		return nil, fmt.Errorf("--num-instances %d, but --prefill-instances %d + --decode-instances %d "+
			"= %d: in a disaggregated run every instance is a prefill or a decode instance, so the "+
			"counts must add up", top.NumInstances, top.Prefill, top.Decode,
			top.Prefill+top.Decode)
	}
	// A handoff crosses nodes (pools never share one), so it needs the inter-node fabric: the
	// kernel would price a crossing with no fabric at the on-node rate. blis-schemas already
	// refuses a multi-node cluster that names none, and a P/D deployment always spans two
	// nodes, so no check is repeated here; TransferTime prices every rank pair up front.
	ps, pe := p.Prefill.Settings()
	ds, de := p.Decode.Settings()
	if pe != nil || de != nil {
		return nil, fmt.Errorf("scenario %q: prefill pool: %v; decode pool: %v",
			scenario, pe, de)
	}
	// A request's KV moves block for block: the decode side reserves the blocks the prefill
	// side filled, so the two engines must page identically.
	if ps.BlockSize != ds.BlockSize {
		return nil, fmt.Errorf("scenario %q pages the prefill pool in %d-token blocks and "+
			"the decode pool in %d; a P/D handoff moves whole blocks, so the pools must agree",
			scenario, ps.BlockSize, ds.BlockSize)
	}
	// The handoff moves the prefill pool's KV representation into the decode pool's blocks, and
	// is priced from the prefill pool's bytes: pools caching in different formats would move
	// pages the decode engine does not hold, with no conversion modelled. So the cache formats,
	// recurrent state included, must agree.
	if err := requireSameKVCacheFormat(scenario, p.Prefill, p.Decode); err != nil {
		return nil, err
	}
	// DP-as-placement expands every pool by one replica factor (the scenario's dp), so
	// per-pool widths that differ cannot be expressed as a placement.
	if ps.DataParallel != ds.DataParallel {
		return nil, fmt.Errorf("scenario %q runs the prefill pool at dp %d and the decode "+
			"pool at dp %d; one replica factor applies to both pools, so their dp must match",
			scenario, ps.DataParallel, ds.DataParallel)
	}
	// The simulator's speculative config (draft length, acceptance) is one per run, while each
	// pool's kernel prices its own draft width. Pools that disagree would advance requests by
	// one draft length and charge for another.
	if ps.SpeculativeTokens != ds.SpeculativeTokens || ps.SpeculativeMethod != ds.SpeculativeMethod {
		return nil, fmt.Errorf("scenario %q drafts %d %q tokens in the prefill pool and %d "+
			"%q in the decode pool; one speculative configuration applies to the run, so the pools "+
			"must agree", scenario, ps.SpeculativeTokens,
			ps.SpeculativeMethod, ds.SpeculativeTokens, ds.SpeculativeMethod)
	}
	return p, nil
}

// nixlConnectors are the pd_transfer connectors whose handoff BLIS models: vLLM v0.31.0's
// NIXL connectors (NixlConnector is an alias of NixlPullConnector, nixl/connector.py:9-12), and
// MultiConnector, which llm-d composes from a NixlConnector and an OffloadingConnector. The
// schema cannot see a MultiConnector's children, so, as the blis-schemas rules pack does
// (rules/v0_31 pack.go:964-968), it is taken to carry the NIXL one.
var nixlConnectors = map[string]bool{
	"NixlConnector": true, "NixlPullConnector": true, "NixlPushConnector": true, "MultiConnector": true,
}

// RequireNIXLHandoff refuses a P/D scenario whose handoff is not NIXL's. What BLIS simulates
// around a handoff is NIXL's protocol: the decode side reserves the blocks at transfer start
// (WAITING_FOR_REMOTE_KVS), pulls only what its own prefix cache lacks, and the prefill side
// holds its blocks until the pull completes or the lease expires (INV-PD-7). Another connector
// (Mooncake, LMCache, P2P NCCL) moves and frees KV on other terms, which would be simulated as
// NIXL's without a word.
func RequireNIXLHandoff(scenario, connector string) error {
	if !nixlConnectors[connector] {
		return fmt.Errorf("scenario %q hands KV off with pd_transfer.connector %q; BLIS models NIXL's "+
			"handoff (NixlConnector, NixlPullConnector, NixlPushConnector, or a MultiConnector over "+
			"one), so state one of those", scenario, connector)
	}
	return nil
}

// HoldsPrefillKV reports whether the run's prefill engines hold a finished request's KV for
// the decode pull: every NIXL handoff does (nixl/pull_scheduler.py:247-266 and
// push_scheduler.py:260-270 at v0.31.0). False for a colocated run (nil receiver).
func (p *Pools) HoldsPrefillKV() bool {
	return p != nil && nixlConnectors[p.Connector]
}

// Fatal reports a failure the run cannot continue from, found while it runs (a pricing
// function asked about a pair the up-front checks did not cover). The simulator's CLI passes
// its fatal exit; it never returns.
type Fatal func(format string, args ...any)

// HandoffPricing returns the run's handoff price and, when contention is on, the demand behind
// it. Both nil for a colocated run, where contention is refused. Instances are named
// "instance_<n>" (Topology).
func (p *Pools) HandoffPricing(scenario string, contention bool, top Topology, blockSizeTokens int64, fatal Fatal) (
	func(int64, string, string) int64, func(int64, string, string) kvtransfer.Flow, error,
) {
	if p == nil {
		if contention {
			return nil, nil, fmt.Errorf("--pd-transfer-contention needs a disaggregated run: " +
				"pass --prefill-instances and --decode-instances with a P/D scenario")
		}
		return nil, nil, nil
	}
	price, err := p.transferTime(scenario, top, blockSizeTokens)
	if err != nil {
		return nil, nil, err
	}
	if !contention {
		return price, nil, nil
	}
	demand, err := p.transferDemand(scenario, top, blockSizeTokens, fatal)
	if err != nil {
		return nil, nil, err
	}
	return price, demand, nil
}

// transferDemand is the demand behind transferTime: the prefill pool kernel's
// PDTransferDemand between the two instances' placements, as payload, floor and links. The
// blis-schemas law that alone a demand costs its PDTransferTime is checked for every rank pair
// up front (a block's handoff), so a kernel breaking it is refused before the run rather than
// arbitrated.
func (p *Pools) transferDemand(scenario string, top Topology, blockSizeTokens int64, fatal Fatal) (func(int64, string, string) kvtransfer.Flow, error) {
	firstDecode := top.Prefill
	demand := func(tokens int64, from, to kernel.Placement) (kvtransfer.Flow, error) {
		d := p.Prefill.Kernel().PDTransferDemand(int(tokens), from, to)
		if err := d.Check(); err != nil {
			return kvtransfer.Flow{}, fmt.Errorf("scenario %q: the kernel's demand for a %d-token handoff from %+v to %+v is "+
				"malformed: %v", scenario, tokens, from, to, err)
		}
		f := kvtransfer.Flow{Bytes: d.Bytes, FloorNs: d.Floor.Nanoseconds()}
		for _, l := range d.Links {
			f.Links = append(f.Links, kvtransfer.Link{ID: string(l.Link), BytesPerSecond: l.BytesPerSecond})
		}
		return f, nil
	}
	for i := 0; i < top.Prefill; i++ {
		for j := 0; j < top.Decode; j++ {
			from, to := p.Prefill.PlacementOf(i), p.Decode.PlacementOf(j)
			f, err := demand(blockSizeTokens, from, to)
			if err != nil {
				return nil, err
			}
			if a, b := kvtransfer.AloneTicks(f), p.Prefill.PDTransferTicks(blockSizeTokens, from, to); a != b {
				return nil, fmt.Errorf("scenario %q: the kernel's handoff demand from prefill rank %d to decode "+
					"rank %d costs %d ticks alone but it prices the handoff at %d; blis-schemas requires "+
					"PDTransferTime == PDTransferDemand.AloneTime()", scenario, i, j, a, b)
			}
		}
	}
	return func(tokens int64, from, to string) kvtransfer.Flow {
		f, err := demand(tokens, p.Prefill.PlacementOf(RankInPool(from, firstDecode)),
			p.Decode.PlacementOf(RankInPool(to, firstDecode)))
		if err != nil {
			fatal("%v", err)
		}
		return f
	}, nil
}

// requireSameKVCacheFormat refuses a disaggregated scenario whose pools state different KV
// cache formats: the attention cache dtype and, for recurrent layers, the state dtypes and
// cache mode. An unstated field is the engine default, so it agrees only with another unstated
// one (the two pools share the model, whose default resolves alike).
func requireSameKVCacheFormat(scenario string, prefill, decode *kernelmodel.Model) error {
	pe, perr := prefill.Engine()
	de, derr := decode.Engine()
	if perr != nil || derr != nil {
		return fmt.Errorf("scenario %q: prefill pool: %v; decode pool: %v", scenario, perr, derr)
	}
	for _, f := range []struct{ name, prefill, decode string }{
		{"cache_dtype", pe.CacheDType, de.CacheDType},
		{"mamba_cache_dtype", pe.MambaCacheDType, de.MambaCacheDType},
		{"mamba_ssm_cache_dtype", pe.MambaSSMCacheDType, de.MambaSSMCacheDType},
		{"mamba_cache_mode", pe.MambaCacheMode, de.MambaCacheMode},
	} {
		if f.prefill != f.decode {
			return fmt.Errorf("scenario %q caches KV as %s %q in the prefill pool and %q in the "+
				"decode pool; a P/D handoff moves the prefill pool's cache into the decode pool's "+
				"blocks, and no conversion between formats is modelled, so the pools must agree",
				scenario, f.name, f.prefill, f.decode)
		}
	}
	return nil
}

// SequenceFixedCharge is the startup note on the per-sequence term of an engine's KV capacity
// (#1941): the steady hold each decoding sequence's bounded state settles at, and the
// admission cap vLLM gates a new request on and sizes its startup capacity figure by. "" for
// a model with no such caches, whose capacity it does not touch.
func SequenceFixedCharge(what string, s kernelmodel.Settings) string {
	if s.SequenceState.Hold == nil {
		return ""
	}
	atCap := min(int64(s.MaxNumSeqs)*s.SequenceSteadyBlocks, s.KVBlocks)
	return fmt.Sprintf("KV capacity (%s): each sequence's bounded state (window rings, compressor/indexer/"+
		"recurrent state) is held step by step: %d bytes per rank (%d blocks of %d bytes) once "+
		"decoding, more while a prefill chunk is in flight; vLLM's admission gate caps it at %d "+
		"bytes (%d blocks). At max_num_seqs %d the steady hold is %d of %d blocks (%.1f%%)",
		what, s.SequenceSteadyBytes, s.SequenceSteadyBlocks, s.KVBytesPerBlock,
		s.SequenceAdmissionBytes, s.SequenceAdmissionBlocks, s.MaxNumSeqs, atCap, s.KVBlocks,
		100*float64(atCap)/float64(s.KVBlocks))
}

// RequireWindowFits refuses an engine with no window at all: neither the deployment nor the
// model graph states one, and vLLM would derive it from an HF config no document the kernel
// reads carries. A window one request cannot fit in the KV cache is refused when the settings
// are sized, with vLLM's start-up message (kernelmodel.checkEnoughKVCacheMemory).
func RequireWindowFits(scenario, what string, s kernelmodel.Settings) error {
	if s.MaxModelLen <= 0 {
		return fmt.Errorf("scenario %q's %s states no engine.max_model_len and its model graph no "+
			"context window; state the window the engine serves", scenario, what)
	}
	return nil
}

// transferTime prices a handoff with the prefill pool's kernel, between the placements of the
// two instances in their pools. It reads the instance counts after DP-as-placement, so call it
// once they are final. Every prefill-decode rank pair is priced once up front, so a pair the
// fabric cannot price (a rack boundary with no bandwidth, say) is refused now rather than
// panicking mid-run.
func (p *Pools) transferTime(scenario string, top Topology, blockSizeTokens int64) (func(int64, string, string) int64, error) {
	firstDecode := top.Prefill
	block := int(blockSizeTokens)
	for i := 0; i < top.Prefill; i++ {
		for j := 0; j < top.Decode; j++ {
			from, to := p.Prefill.PlacementOf(i), p.Decode.PlacementOf(j)
			if d := p.Prefill.Kernel().PDTransferTime(block, from, to); d < 0 || d >= time.Duration(math.MaxInt64/2) {
				return nil, fmt.Errorf("scenario %q: the kernel cannot price a KV handoff from prefill "+
					"rank %d (%+v) to decode rank %d (%+v) -- it reports %v; the link between them "+
					"states no bandwidth -- check cluster.fabric", scenario, i, from, j, to, d)
			}
		}
	}
	return func(tokens int64, from, to string) int64 {
		return p.Prefill.PDTransferTicks(tokens,
			p.Prefill.PlacementOf(RankInPool(from, firstDecode)),
			p.Decode.PlacementOf(RankInPool(to, firstDecode)))
	}, nil
}

// RankInPool is an instance's rank within its pool. Instances are numbered prefill first, then
// decode (BuildPoolMembershipFromIndices), so a prefill instance's rank is its number and a
// decode instance's is its offset from the first decode instance.
func RankInPool(id string, firstDecode int) int {
	n, err := strconv.Atoi(strings.TrimPrefix(id, "instance_"))
	if err != nil || n < 0 {
		panic(fmt.Sprintf("kernel P/D pricing: unexpected instance id %q", id))
	}
	if n >= firstDecode {
		return n - firstDecode
	}
	return n
}

// RequireCapacity refuses a run whose instances do not fit the scenario's pools: each
// simulated instance is one data-parallel rank of the pool's layout, and a pool holds as many
// as its nodes have GPUs for. Called once the instance counts are final. colocated is the
// run's kernel (nil when none is open).
func RequireCapacity(scenario string, colocated *kernelmodel.Model, p *Pools, top Topology) error {
	if colocated == nil {
		return nil
	}
	check := func(role string, n int, m *kernelmodel.Model) error {
		if c := m.RankCapacity(); n > c {
			return fmt.Errorf("%d %s instance(s) requested, but scenario %q's %s pool "+
				"holds %d rank(s) at its layout; raise the pool's nodes in the scenario or run fewer",
				n, role, scenario, m.Role(), c)
		}
		return nil
	}
	if p == nil {
		return check("", top.NumInstances, colocated)
	}
	if err := check("prefill", top.Prefill, p.Prefill); err != nil {
		return err
	}
	return check("decode", top.Decode, p.Decode)
}

// KernelCPUTier is the catalog storage device the kernel prices the legacy --kv-cpu-blocks
// tier's CPU<->GPU reloads at: host DRAM across the PCIe/NVLink boundary.
const KernelCPUTier = "cpu_dram"

// tierTicks is m's kernel's price for one transfer, in whole ticks rounded up: a transfer is
// not done until its last byte lands. An unknown tier is refused rather than charged the
// kernel's unbounded sentinel, which would stall the clock.
func tierTicks(m *kernelmodel.Model, tier string, toTier bool, bytes int64, inService int) (int64, error) {
	dir := kernel.DirectionFromTier
	if toTier {
		dir = kernel.DirectionToTier
	}
	d := m.Kernel().TierTime(tier, dir, bytes, inService)
	if d >= time.Duration(math.MaxInt64/2) {
		return 0, fmt.Errorf("the kernel cannot price a %d-byte transfer %s storage tier %q: "+
			"either the catalog's %s defines no such device, or it states no bandwidth in that "+
			"direction", bytes, map[bool]string{true: "to", false: "from"}[toTier],
			tier, CatalogStorageDevicesRelPath)
	}
	return max(1, (d.Nanoseconds()+999)/1000), nil
}

// TierTicks is tierTicks for a transfer the up-front checks priced already; an error then is a
// broken kernel mid-run, reported through fatal.
func TierTicks(m *kernelmodel.Model, tier string, toTier bool, bytes int64, inService int, fatal Fatal) int64 {
	t, err := tierTicks(m, tier, toTier, bytes, inService)
	if err != nil {
		fatal("%v", err)
	}
	return t
}

// shareTiersByNode places each engine's secondary tiers on the node the kernel places its rank
// on, so a node's engines share one tier (engine.OffloadTierSharing). rankOf maps an instance
// to its rank in m's pool.
func shareTiersByNode(cfg *engine.KVOffloadConfig, m *kernelmodel.Model, rankOf func(string) int) {
	if !cfg.IsEnabled() || len(cfg.Tiers) == 0 || m == nil {
		return
	}
	cfg.Sharing = &engine.OffloadTierSharing{NodeOf: func(id string) int {
		return m.PlacementOf(rankOf(id)).Node
	}}
}

// instanceRank is a colocated run's instance index ("instance_<n>"), the rank the kernel places.
func instanceRank(id string) int { return RankInPool(id, math.MaxInt) }

// OffloadPricing is what the offload pricing needs from the run: the GPU block size, the
// legacy CPU tier's block count, and the token budget (the Simple connector's lazy target).
type OffloadPricing struct {
	BlockSizeTokens     int64
	KVCPUBlocks         int64
	MaxNumBatchedTokens int64
}

// PriceColocatedOffload has the kernel price every KV-offload transfer of a colocated run,
// from the run's one pool: see priceOffload. It returns the legacy per-block charge, 0 when no
// kernel is open or with no legacy tier. A disaggregated run's pools are priced separately,
// each by its own kernel (Pools.PriceOffload). Shared by run and replay.
func PriceColocatedOffload(m *kernelmodel.Model, cfg *engine.KVOffloadConfig, in OffloadPricing, fatal Fatal) (int64, error) {
	if m == nil {
		return 0, nil
	}
	shareTiersByNode(cfg, m, instanceRank)
	return priceOffload(m, cfg, in, fatal)
}

// priceOffload has m's kernel price every KV-offload transfer of m's pool: each secondary
// tier through TierTime for its catalog device class, at the depth the transfer station is
// serving, the offload CPU tier's CPU->GPU loads at cpu_dram (INV-K7), and the legacy CPU
// tier as one whole per-block reload charge, all at m's own per-block bytes
// (SequenceVariableBytes).
//
// TierTime's contract is a GPU<->tier transfer, while the station's secondary-tier jobs run
// CPU<->tier; the kernel's price is applied to them deliberately -- the device and the host
// link it names are the ones those jobs cross -- rather than as an exact match. The simulator
// keeps the servers, queues and timing; the kernel owns the price (blis-schemas
// kernel.TierTime: the slower of the device and the host link binds).
func priceOffload(m *kernelmodel.Model, cfg *engine.KVOffloadConfig, in OffloadPricing, fatal Fatal) (int64, error) {
	perBlock := m.Kernel().SequenceVariableBytes(int(in.BlockSizeTokens))
	if cfg.IsEnabled() {
		cfg.CPUCopies = m.OffloadCopies()
		if err := cfg.CheckFitsOneChunk(); err != nil {
			return 0, err
		}
	}
	if (cfg.IsEnabled() || in.KVCPUBlocks > 0) && perBlock <= 0 {
		return 0, fmt.Errorf("the kernel prices a %d-token block at %d bytes; offload "+
			"needs a block to occupy memory", in.BlockSizeTokens, perBlock)
	}
	if cfg.IsEnabled() {
		for i := range cfg.Tiers {
			class := cfg.Tiers[i].DeviceClass
			if class == "" {
				return 0, fmt.Errorf("kv_offload secondary_tiers[%d] names no device_class; "+
					"the kernel prices a tier by its catalog device, so every tier must name one",
					i)
			}
			// A catalog device's rates are O_DIRECT rates; buffered (page-cache) I/O, vLLM's
			// fallback where O_DIRECT is unsupported, is another regime the kernel does not
			// price yet (blis-latency-kernel#41), so it is refused rather than priced as O_DIRECT.
			if !cfg.Tiers[i].DirectIO {
				return 0, fmt.Errorf("kv_offload secondary_tiers[%d] (%s) uses buffered I/O (direct_io: "+
					"false), which the kernel does not price (blis-latency-kernel#41); only "+
					"direct_io: true tiers can be simulated", i, class)
			}
			// Price one real block each way up front, so an unknown device or one with no
			// bandwidth in a direction is refused now rather than at its first transfer.
			for _, toTier := range []bool{true, false} {
				if _, err := tierTicks(m, class, toTier, perBlock, 1); err != nil {
					return 0, err
				}
			}
			cfg.Tiers[i].ServiceTime = func(write bool, bytes int64, inService int) int64 {
				return TierTicks(m, class, write, bytes, inService, fatal)
			}
		}
		// vLLM's CPU->GPU loads run one at a time on the handler's stream chain
		// (kv_offload/cpu/gpu_worker.py:696-701 at v0.31.0), so each is priced alone.
		if _, err := tierTicks(m, KernelCPUTier, false, perBlock, 1); err != nil {
			return 0, err
		}
		cfg.CPULoadTime = func(bytes int64) int64 {
			return TierTicks(m, KernelCPUTier, false, bytes, 1, fatal)
		}
		if cfg.Simple {
			// The Simple connector's stores are device-to-host copies on their own stream
			// (vllm/v1/simple_kv_offload/worker.py:164-167, 273-295 at v0.31.0).
			if _, err := tierTicks(m, KernelCPUTier, true, perBlock, 1); err != nil {
				return 0, err
			}
			cfg.CPUStoreTime = func(bytes int64) int64 {
				return TierTicks(m, KernelCPUTier, true, bytes, 1, fatal)
			}
			if cfg.LazyOffload {
				// target_free: 2 x ceil(max_num_batched_tokens / block_size) for one
				// full-attention group (simple_kv_offload/manager.py:312-330).
				cfg.LazyTargetFree = 2 * ((in.MaxNumBatchedTokens + in.BlockSizeTokens - 1) / in.BlockSizeTokens)
			}
		}
	}
	if in.KVCPUBlocks <= 0 {
		return 0, nil
	}
	return tierTicks(m, KernelCPUTier, false, perBlock, 1)
}

// PoolOffload is one pool's offload as its own kernel sizes and prices it.
type PoolOffload struct {
	Config        engine.KVOffloadConfig
	TicksPerBlock int64
}

// PriceOffload prices the run's KV offload once per pool, each by that pool's own kernel: the
// same tiers and capacities (cfg, the run's one offload description), sized and priced from
// each pool's layout. ok is false when the run offloads nothing.
func (p *Pools) PriceOffload(scenario string, cfg engine.KVOffloadConfig, in OffloadPricing, top Topology, fatal Fatal) (prefill, decode PoolOffload, ok bool, err error) {
	if !cfg.IsEnabled() && in.KVCPUBlocks <= 0 {
		return PoolOffload{}, PoolOffload{}, false, nil
	}
	// A disaggregated engine with an offload tier runs both connectors, which only a
	// MultiConnector composes: with kv_offloading_size set, vLLM overwrites kv_connector with
	// the OffloadingConnector and kv_role with kv_both (vllm/config/vllm.py:1124-1158 at
	// v0.31.0), so a plain NIXL connector and a tier cannot run together.
	if p.Connector != "MultiConnector" {
		return PoolOffload{}, PoolOffload{}, false, fmt.Errorf("scenario %q offloads KV under P/D with pd_transfer.connector %q; vLLM runs a NIXL "+
			"handoff and an offload tier together only through a MultiConnector (stating "+
			"kv_offloading_size replaces kv_connector with OffloadingConnector, vllm/config/vllm.py:1124-1158), "+
			"so state pd_transfer.connector: MultiConnector", scenario, p.Connector)
	}
	out := [2]PoolOffload{}
	for i, c := range []struct {
		role string
		m    *kernelmodel.Model
	}{{"prefill", p.Prefill}, {"decode", p.Decode}} {
		pool := cfg
		// Each pool gets its own tier slice: priceOffload installs a per-pool ServiceTime.
		pool.Tiers = append([]engine.KVOffloadTier(nil), cfg.Tiers...)
		// A pool's block is its own layout's slice of the KV, so pools of different widths
		// hold different bytes per block, and the tiers' block capacity and job sizes follow.
		if pool.IsEnabled() {
			pool.PerBlockBytes = c.m.Kernel().SequenceVariableBytes(int(in.BlockSizeTokens))
			if err := pool.CheckFitsOneChunk(); err != nil {
				return PoolOffload{}, PoolOffload{}, false, fmt.Errorf("%s pool: %v", c.role, err)
			}
		}
		firstDecode := top.Prefill
		shareTiersByNode(&pool, c.m, func(id string) int {
			return RankInPool(id, firstDecode)
		})
		ticks, err := priceOffload(c.m, &pool, in, fatal)
		if err != nil {
			return PoolOffload{}, PoolOffload{}, false, err
		}
		out[i] = PoolOffload{Config: pool, TicksPerBlock: ticks}
	}
	return out[0], out[1], true, nil
}
