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
//   - FootprintBytes(id)     — per-adapter HBM footprint, summed into the KV
//     reservation (PR5).
//
// All methods are pure queries (Principle III): they never mutate the model and
// return identical results for identical inputs (INV-6, no RNG — R7). Ranks are
// resolved from the pre-declared registry built once at construction; requests
// carry only the adapter id.
//
// It satisfies sim.AdapterCost (via the LoadLatency method); the type stays
// exported so PR4/PR5 can consume StepOverheadFactor / FootprintBytes directly
// while the sim/ seam interface grows only as each term is wired (R13).
type CostModel struct {
	loadBaseLatencyUs     float64
	loadBandwidthBytesUs  float64 // > 0, divisor guard enforced at construction (R11)
	footprintBytesPerRank float64

	ranks map[string]int // adapter id -> declared rank (registry projection)

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
	for _, a := range cfg.Adapters {
		if a.ID == "" || a.Rank <= 0 {
			return nil, fmt.Errorf("lora.NewCostModel: adapter %q must have non-empty id and rank > 0", a.ID)
		}
		ranks[a.ID] = a.Rank
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

	return cm, nil
}

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
