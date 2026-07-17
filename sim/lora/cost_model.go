package lora

import (
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// CostModel derives the three Digital-Twin adapter cost terms as pure,
// deterministic deltas onto BLIS's calibrated base (data-model.md "Adapter Cost
// Model", contracts/latency-model.md):
//
//   - LoadLatency(id)        — one-time cold-load latency (µs), charged by the
//     pre-admission gate (PR3): base + ceil(footprint(rank)/bandwidth).
//   - StepOverheadFactor(b)  — multiplicative per-step compute overhead >= 1.0,
//     1 + (K6(r_max)/K7(r_max))·A_B, applied by both latency backends (PR4).
//   - FootprintBytes(id)     — per-adapter HBM footprint (footprint_bytes_per_rank
//     · rank); an input to LoadLatency, not itself a KV-reservation term.
//   - AdapterReservedBytes() — the FIXED, capacity-based static HBM reservation
//     (adapter_capacity × per-slot footprint, sized from the max declared rank)
//     subtracted from the KV budget once at startup (PR5). A constant, NOT a
//     running sum over currently-resident adapters (D2 / INV-L4).
//
// All methods are pure queries (Principle III): they never mutate the model and
// return identical results for identical inputs (INV-6, no RNG — R7). Ranks are
// resolved from the pre-declared registry built once at construction; requests
// carry only the adapter id.
//
// It satisfies sim.AdapterCost; the type stays exported so the latency backends
// consume StepOverheadFactor and the KV-capacity path consumes AdapterReservedBytes
// through that seam (each term wired as its PR landed — PR3/PR4/PR5; R13).
type CostModel struct {
	loadBaseLatencyUs     float64
	loadBandwidthBytesUs  float64 // > 0, divisor guard enforced at construction (R11)
	footprintBytesPerRank float64

	ranks map[string]int // adapter id -> declared rank (registry projection)

	// adapterCapacity and maxRank size the static HBM reservation (PR5): the
	// resident-slot count and the largest declared rank, resolved once at
	// construction. per-slot footprint = footprintBytesPerRank · maxRank, so the
	// reservation (capacity × per-slot) is a constant regardless of which adapters
	// are resident (D2 / INV-L4). Both are 0 when no capacity / no adapters are
	// declared, making the reservation 0 (INV-6 no-op).
	adapterCapacity int
	maxRank         int

	// tiers holds the per-rank compute-overhead coefficients; tierRanks is the
	// sorted key list used to clamp an out-of-envelope max rank to the nearest
	// calibrated tier (contracts/latency-model.md). Consumed by PR4.
	tiers     map[int]tierCoeff
	tierRanks []int
}

// tierCoeff is the resolved (non-pointer) per-tier coefficient pair.
type tierCoeff struct {
	k6 float64
	k7 float64 // > 0, per-tier normalization denominator (divisor guard)
}

// NewCostModel builds a CostModel from a validated LoRAConfig. It re-checks the
// cost coefficients (R3 non-negativity, R11 divisor guards) so a model built
// directly is as safe as one from a validated config, returning an error the
// caller maps to fatality (cmd/ -> logrus.Fatalf; sim/ factory -> panic).
func NewCostModel(cfg sim.LoRAConfig) (*CostModel, error) {
	// NaN/Inf slip past ordering guards (NaN <= 0 and NaN < 0 are both false; +Inf
	// passes > 0), then poison LoadLatency (int64(ceil(NaN/Inf)) is garbage and can
	// violate INV-3). Reject non-finite coefficients up front (R3/R20 degenerate
	// input). A finite base + finite positive divisor keeps LoadLatency finite.
	if cfg.LoadBaseLatencyUs == nil {
		return nil, fmt.Errorf("lora.NewCostModel: load_base_latency_us must be set")
	}
	if !isFinite(*cfg.LoadBaseLatencyUs) || *cfg.LoadBaseLatencyUs < 0 {
		return nil, fmt.Errorf("lora.NewCostModel: load_base_latency_us must be finite and >= 0, got %v", *cfg.LoadBaseLatencyUs)
	}
	if cfg.LoadBandwidthBytesUs == nil || !isFinite(*cfg.LoadBandwidthBytesUs) || *cfg.LoadBandwidthBytesUs <= 0 {
		return nil, fmt.Errorf("lora.NewCostModel: load_bandwidth_bytes_us must be finite and > 0 (divisor guard), got %v", cfg.LoadBandwidthBytesUs)
	}
	if cfg.FootprintBytesPerRank == nil || !isFinite(*cfg.FootprintBytesPerRank) || *cfg.FootprintBytesPerRank <= 0 {
		return nil, fmt.Errorf("lora.NewCostModel: footprint_bytes_per_rank must be finite and > 0, got %v", cfg.FootprintBytesPerRank)
	}

	ranks := make(map[string]int, len(cfg.Adapters))
	maxRank := 0
	for _, a := range cfg.Adapters {
		if a.ID == "" || a.Rank <= 0 {
			return nil, fmt.Errorf("lora.NewCostModel: adapter %q must have non-empty id and rank > 0", a.ID)
		}
		ranks[a.ID] = a.Rank
		if a.Rank > maxRank {
			maxRank = a.Rank
		}
	}

	// AdapterCapacity may be nil when adapters are absent (the config is inert);
	// the reservation is 0 in that case. A declared-but-non-positive capacity is
	// rejected by LoRAConfig.Validate upstream, so a set value is > 0 here.
	adapterCapacity := 0
	if cfg.AdapterCapacity != nil {
		adapterCapacity = *cfg.AdapterCapacity
	}

	tiers := make(map[int]tierCoeff, len(cfg.StepOverheadTiers))
	tierRanks := make([]int, 0, len(cfg.StepOverheadTiers))
	for rank, t := range cfg.StepOverheadTiers {
		if rank <= 0 {
			return nil, fmt.Errorf("lora.NewCostModel: step_overhead_tiers rank key must be > 0, got %d", rank)
		}
		if t.K6 == nil || !isFinite(*t.K6) || *t.K6 < 0 {
			return nil, fmt.Errorf("lora.NewCostModel: step_overhead_tiers[%d].k6 must be finite and >= 0", rank)
		}
		if t.K7 == nil || !isFinite(*t.K7) || *t.K7 <= 0 {
			return nil, fmt.Errorf("lora.NewCostModel: step_overhead_tiers[%d].k7 must be finite and > 0 (divisor guard)", rank)
		}
		tiers[rank] = tierCoeff{k6: *t.K6, k7: *t.K7}
		tierRanks = append(tierRanks, rank)
	}
	sort.Ints(tierRanks) // deterministic clamp lookup (R2)

	cm := &CostModel{
		loadBaseLatencyUs:     *cfg.LoadBaseLatencyUs,
		loadBandwidthBytesUs:  *cfg.LoadBandwidthBytesUs,
		footprintBytesPerRank: *cfg.FootprintBytesPerRank,
		ranks:                 ranks,
		adapterCapacity:       adapterCapacity,
		maxRank:               maxRank,
		tiers:                 tiers,
		tierRanks:             tierRanks,
	}

	// Fail fast if any declared adapter's rank is large enough that its load latency
	// is non-finite (±Inf) OR too large to convert to an int64 tick count. Left
	// unchecked, int64(math.Ceil(x)) for x ≥ 2^63 saturates to MaxInt64, and the
	// gate's now+loadTicks then wraps to a negative timestamp, violating INV-3
	// (R3/R20 — unbounded numeric input; rank is an unbounded int). maxLoadLatencyUs
	// bounds a single load below MaxInt64 with generous headroom for now+loadTicks.
	// Iterate the declared slice for a deterministic error (R2).
	for _, a := range cfg.Adapters {
		ll := cm.LoadLatency(a.ID)
		if !isFinite(ll) || ll > maxLoadLatencyUs {
			return nil, fmt.Errorf("lora.NewCostModel: adapter %q rank %d yields an unusable load latency (%v µs) — reduce rank or footprint_bytes_per_rank", a.ID, a.Rank, ll)
		}
	}

	// Fail fast if the static HBM reservation (capacity × footprint × maxRank) is
	// non-finite (±Inf from a huge footprint/rank) or too large to convert to an
	// int64 byte count. Its three factors are each individually bounded > 0 but
	// their product is unbounded (R3/R20 — capacity and rank are unbounded ints).
	// int64(x) for x ≥ 2^63 or ±Inf is implementation-defined (a garbage/negative
	// value on some platforms), which would corrupt the KV-capacity subtraction
	// rather than fail cleanly. maxReservedBytes bounds it below MaxInt64 with
	// generous headroom, mirroring the maxLoadLatencyUs guard above.
	if reserved := cm.AdapterReservedBytes(); !isFinite(reserved) || reserved > maxReservedBytes {
		return nil, fmt.Errorf("lora.NewCostModel: static HBM reservation (capacity %d × footprint_bytes_per_rank %g × max rank %d = %v bytes) is unusable — reduce adapter_capacity, footprint_bytes_per_rank, or the maximum declared rank", adapterCapacity, cm.footprintBytesPerRank, maxRank, reserved)
	}

	return cm, nil
}

// maxReservedBytes caps the static adapter HBM reservation well below
// math.MaxInt64 so the float64→int64 conversion at the KV-capacity boundary is
// always exact and non-negative. 1e18 bytes = 1 exabyte — orders of magnitude
// beyond any real GPU HBM, so any config exceeding it is degenerate.
const maxReservedBytes = 1e18

// maxLoadLatencyUs caps a single adapter load latency well below math.MaxInt64
// ticks (µs), leaving headroom so now+loadTicks cannot overflow int64 for any
// realistic clock. 1e15 µs ≈ 31.7 years — any config exceeding it is degenerate.
const maxLoadLatencyUs = 1e15

// RankOf returns the declared rank of an adapter id and whether it is registered.
// An empty id (base-model request) is not registered.
func (c *CostModel) RankOf(id string) (int, bool) {
	r, ok := c.ranks[id]
	return r, ok
}

// FootprintBytes returns the HBM footprint of an adapter id in bytes (>= 0):
// footprint_bytes_per_rank · rank. An empty or unregistered id has zero footprint.
func (c *CostModel) FootprintBytes(id string) float64 {
	rank, ok := c.ranks[id]
	if !ok {
		return 0
	}
	return c.footprintBytesPerRank * float64(rank)
}

// AdapterReservedBytes returns the fixed, capacity-based HBM reservation in bytes
// (>= 0): adapter_capacity × per-slot footprint, where the per-slot footprint is
// sized from the MAX declared rank (footprintBytesPerRank · maxRank) so any adapter
// fits any slot. This is the static memory model (design D2 / INV-L4): the value is
// a constant for the model's lifetime — adapters load and evict WITHIN the
// pre-reserved slots without changing it — as opposed to a dynamic running sum over
// currently-resident adapters. Returns 0 when no capacity or no adapters are
// configured (INV-6 / INV-L1 no-op). The KV-capacity module subtracts this once at
// startup beside model weights (PR5). Pure and deterministic (R7).
func (c *CostModel) AdapterReservedBytes() float64 {
	return float64(c.adapterCapacity) * c.footprintBytesPerRank * float64(c.maxRank)
}

// LoadLatency returns the one-time cold-load latency of an adapter id in µs (>= 0):
// load_base_latency_us + ceil(FootprintBytes(rank) / load_bandwidth_bytes_us). An
// empty or unregistered id carries no load cost (0) — a base-model request never
// gates. Pure and deterministic (R7).
func (c *CostModel) LoadLatency(id string) float64 {
	rank, ok := c.ranks[id]
	if !ok {
		return 0
	}
	footprint := c.footprintBytesPerRank * float64(rank)
	return c.loadBaseLatencyUs + math.Ceil(footprint/c.loadBandwidthBytesUs)
}

// StepOverheadFactor returns the multiplicative per-step compute-overhead factor
// for a batch (>= 1.0): 1 + (K6(r_max)/K7(r_max))·A_B, where A_B is the count of
// DISTINCT non-empty adapter ids in the batch and r_max is their maximum rank.
// The K6/K7 coefficients are selected by r_max's tier (rank enters here — FR-009),
// clamped to the nearest calibrated tier when out of envelope. Normalized so the
// factor is exactly 1.0 when A_B==0 for any fitted K7 (INV-6). Consumed by the
// latency backends in a later PR.
func (c *CostModel) StepOverheadFactor(batch []*sim.Request) float64 {
	seen := make(map[string]struct{}, len(batch))
	maxRank := 0
	for _, req := range batch {
		if req == nil || req.Adapter == "" {
			continue
		}
		rank, ok := c.ranks[req.Adapter]
		if !ok {
			// An unregistered id is treated as base-model — consistent with
			// LoadLatency and FootprintBytes, which both return 0 for an
			// unregistered id. It contributes to neither A_B nor r_max, so an
			// unknown-rank adapter cannot inflate the factor with a rank it never
			// declared. In practice requests carry only registered ids (validated
			// at config load); this keeps the degenerate case consistent.
			continue
		}
		if _, dup := seen[req.Adapter]; dup {
			continue
		}
		seen[req.Adapter] = struct{}{}
		if rank > maxRank {
			maxRank = rank
		}
	}
	aB := len(seen)
	if aB == 0 {
		return 1.0 // no adapters: byte-identical no-op (INV-6)
	}
	tier, ok := c.tierForRank(maxRank)
	if !ok {
		return 1.0 // no calibrated tiers: no modeled overhead
	}
	return 1.0 + (tier.k6/tier.k7)*float64(aB)
}

// tierForRank returns the coefficients for a rank, clamping to the nearest
// calibrated tier when rank falls outside the fitted envelope (below the smallest
// or above the largest calibrated rank, or between tiers). Ties in distance
// resolve to the lower calibrated rank for determinism (R2). Returns ok=false
// only when no tiers are configured.
func (c *CostModel) tierForRank(rank int) (tierCoeff, bool) {
	if len(c.tierRanks) == 0 {
		return tierCoeff{}, false
	}
	if t, exact := c.tiers[rank]; exact {
		return t, true
	}
	best := c.tierRanks[0]
	bestDist := abs(rank - best)
	for _, r := range c.tierRanks[1:] {
		if d := abs(rank - r); d < bestDist {
			best, bestDist = r, d
		}
	}
	return c.tiers[best], true
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// isFinite reports whether x is a finite float64 (not NaN, not ±Inf). Used to
// reject degenerate cost coefficients that would slip past ordering guards.
func isFinite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }
