package sim

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// ScorerConfig describes a named scorer with a weight for weighted routing.
type ScorerConfig struct {
	Name   string  `yaml:"name"`
	Weight float64 `yaml:"weight"`
}

// scorerFunc computes per-instance scores in [0,1] for a scoring dimension.
// Some scorers use a sub-range by design (e.g., load-aware scores in [0, 0.5]
// per llm-d semantics); weighted combination normalizes the effective contribution.
// The req parameter provides request metadata (e.g., InputTokens for prefix matching).
// Stateless scorers may ignore it.
type scorerFunc func(req *Request, snapshots []RoutingSnapshot) map[string]float64

// cacheQueryFn maps instance IDs to functions that return the count of
// consecutive cached prefix blocks for given tokens. Used by precise
// prefix cache scoring. Nil for sim-level tests without cluster instances.
type cacheQueryFn map[string]func([]TokenID) int

// scoreVLLMDP computes per-instance scores using vLLM's data-parallel formula.
// Formula: raw = QueueDepth × 4 + BatchSize, then inverted min-max normalization.
// Matches vLLM's DPLBAsyncMPClient.get_core_engine_for_request() (core_client.py:1219).
//
// Signal freshness (R17, INV-7):
//
//	Reads: QueueDepth (Periodic when --snapshot-refresh-interval>0, else Immediate)
//	       BatchSize (Periodic when --snapshot-refresh-interval>0, else Immediate)
//
// For realistic vLLM parity, use --snapshot-refresh-interval=100000 (100ms) to match
// vLLM's default coordinator stats update interval (min_stats_update_interval_ms=100).
// The default interval=0 (Immediate mode) represents oracle routing with no signal staleness.
func scoreVLLMDP(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	// Step 1: Compute raw scores using vLLM formula (waiting×4 + running)
	rawScores := make(map[string]int, len(snapshots))
	minRaw, maxRaw := math.MaxInt, 0

	for _, snap := range snapshots {
		raw := snap.QueueDepth*4 + snap.BatchSize
		rawScores[snap.ID] = raw
		if raw < minRaw {
			minRaw = raw
		}
		if raw > maxRaw {
			maxRaw = raw
		}
	}

	// Step 2: Inverted min-max normalization
	// Lowest load (minRaw) → 1.0 (highest score, preferred by argmax)
	// Highest load (maxRaw) → 0.0 (lowest score, avoided)
	scores := make(map[string]float64, len(snapshots))
	if maxRaw == minRaw {
		// All equal → all score 1.0 (no differentiation)
		for _, snap := range snapshots {
			scores[snap.ID] = 1.0
		}
	} else {
		for _, snap := range snapshots {
			raw := rawScores[snap.ID]
			scores[snap.ID] = float64(maxRaw-raw) / float64(maxRaw-minRaw)
		}
	}

	return scores
}

// IsValidScorer returns true if name is a registered scorer. Validity is
// derived from the registry keys (single source of truth) — there is no
// separate hand-maintained name list that can drift.
func IsValidScorer(name string) bool {
	_, ok := scorerRegistry[name]
	return ok
}

// ValidScorerNames returns the registered scorer names, sorted.
func ValidScorerNames() []string { return sortedScorerNames() }

// sortedScorerNames collects the registry keys and sorts them explicitly (R2 —
// deterministic output, INV-6; never a bare range over the map feeds output).
func sortedScorerNames() []string {
	names := make([]string, 0, len(scorerRegistry))
	for n := range scorerRegistry {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// DefaultScorerConfigs returns the default scorer configuration for weighted routing.
// Default profile: precise-prefix-cache:2, queue-depth:1, kv-utilization:1 (llm-d parity).
func DefaultScorerConfigs() []ScorerConfig {
	return []ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 2.0},
		{Name: "queue-depth", Weight: 1.0},
		{Name: "kv-utilization", Weight: 1.0},
	}
}

// ParseScorerConfigs parses a comma-separated string of "name:weight" pairs.
// Returns nil for empty input. Returns error for invalid names, non-positive weights,
// NaN, Inf, or malformed input.
func ParseScorerConfigs(s string) ([]ScorerConfig, error) {
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	configs := make([]ScorerConfig, 0, len(parts))
	seen := make(map[string]bool, len(parts))
	for _, part := range parts {
		kv := strings.SplitN(strings.TrimSpace(part), ":", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid scorer config %q (expected name:weight)", strings.TrimSpace(part))
		}
		name := strings.TrimSpace(kv[0])
		if !IsValidScorer(name) {
			return nil, fmt.Errorf("unknown scorer %q; valid: %s", name, strings.Join(ValidScorerNames(), ", "))
		}
		if seen[name] {
			return nil, fmt.Errorf("duplicate scorer %q; each scorer may appear at most once", name)
		}
		seen[name] = true
		weight, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid weight for scorer %q: %w", name, err)
		}
		if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
			return nil, fmt.Errorf("scorer %q weight must be a finite positive number, got %v", name, weight)
		}
		configs = append(configs, ScorerConfig{Name: name, Weight: weight})
	}
	return configs, nil
}

// normalizeScorerWeights returns weights normalized to sum to 1.0.
// Panics if total weight is zero (should be prevented by validation).
func normalizeScorerWeights(configs []ScorerConfig) []float64 {
	total := 0.0
	for _, c := range configs {
		total += c.Weight
	}
	if total <= 0 {
		panic(fmt.Sprintf("scorer weights sum to %f; must be positive", total))
	}
	weights := make([]float64, len(configs))
	for i, c := range configs {
		weights[i] = c.Weight / total
	}
	return weights
}

// scorerConstructor builds a (scorer, observer) pair for a named scorer.
// blockSize is used by block-hash-backed scorers (prefix-affinity); cacheFn by
// cache-backed scorers (precise-prefix-cache, no-hit-lru); stateless scorers
// ignore both. The registry maps names to these constructors (B-1, #1489).
type scorerConstructor func(blockSize int, cacheFn cacheQueryFn) (scorerFunc, observerFunc)

// scorerRegistry maps scorer names to their constructors. Unexported (R8) — all
// access is via IsValidScorer / ValidScorerNames / newScorerWithObserver.
// Populated by init() in this file (the single registration site, R4).
var scorerRegistry = map[string]scorerConstructor{}

// registerScorer adds a scorer constructor under name. Panics on empty or
// duplicate name (R4 — guards double-registration; empty name would make
// IsValidScorer("") true, breaking parity).
func registerScorer(name string, c scorerConstructor) {
	if name == "" {
		panic("registerScorer: empty scorer name")
	}
	if _, dup := scorerRegistry[name]; dup {
		panic(fmt.Sprintf("registerScorer: duplicate scorer %q", name))
	}
	scorerRegistry[name] = c
}

// stateless wraps a scorer func that ignores blockSize and cacheFn (returns a
// nil observer), matching the pre-registry switch arms that returned (fn, nil).
func stateless(fn scorerFunc) scorerConstructor {
	return func(_ int, _ cacheQueryFn) (scorerFunc, observerFunc) { return fn, nil }
}

// init registers all built-in scorers. This is the single registration site
// (R4); the same file defines IsValidScorer/ValidScorerNames, so validity is
// derived from these keys (single source of truth). No other init() in sim/
// reads the registry, so registration-vs-consumption ordering is a non-issue.
func init() {
	// Stateful / param-backed scorers (preserve the exact (scorer, observer) pairing).
	registerScorer("prefix-affinity", func(blockSize int, _ cacheQueryFn) (scorerFunc, observerFunc) {
		return newPrefixAffinityScorer(blockSize)
	})
	registerScorer("precise-prefix-cache", func(_ int, cacheFn cacheQueryFn) (scorerFunc, observerFunc) {
		return newPrecisePrefixCacheScorer(cacheFn) // stateless ground-truth: (scorer, nil)
	})
	registerScorer("no-hit-lru", func(_ int, cacheFn cacheQueryFn) (scorerFunc, observerFunc) {
		return newNoHitLRUScorer(cacheFn)
	})
	// Stateless scorers.
	registerScorer("queue-depth", stateless(scoreQueueDepth))
	registerScorer("kv-utilization", stateless(scoreKVUtilization))
	registerScorer("load-balance", stateless(scoreLoadBalance))
	registerScorer("active-requests", stateless(scoreActiveRequests))
	registerScorer("running-requests", stateless(scoreRunningRequests))
	registerScorer("load-aware", stateless(scoreLoadAware))
	registerScorer("vllm-dp", stateless(scoreVLLMDP))
	registerScorer("lora-affinity", stateless(scoreLoRAAffinity))
}

// newScorerWithObserver creates a scorer function and optional observer for a named scorer.
// Returns (scorer, observer) where observer is nil for stateless scorers.
// blockSize is used by stateful scorers (e.g., prefix-affinity) for block hash computation.
// Panics on unknown name (validation should catch this before reaching here).
func newScorerWithObserver(name string, blockSize int, cacheFn cacheQueryFn) (scorerFunc, observerFunc) {
	ctor, ok := scorerRegistry[name]
	if !ok {
		panic(fmt.Sprintf("unknown scorer %q", name))
	}
	return ctor(blockSize, cacheFn)
}

// scoreQueueDepth computes per-instance queue depth scores using min-max normalization.
// Lower queue depth → higher score. All-equal depths → all score 1.0.
// Matches llm-d/GIE's queue-scorer semantics: reads QueueDepth only (WaitingQueueSize).
//
// Signal freshness (R17, INV-7):
//
//	Reads: QueueDepth (Periodic when interval>0, else Immediate).
func scoreQueueDepth(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	minDepth, maxDepth := math.MaxInt, 0
	for _, snap := range snapshots {
		depth := snap.QueueDepth
		if depth < minDepth {
			minDepth = depth
		}
		if depth > maxDepth {
			maxDepth = depth
		}
	}
	for _, snap := range snapshots {
		if maxDepth == minDepth {
			scores[snap.ID] = 1.0
		} else {
			depth := snap.QueueDepth
			scores[snap.ID] = float64(maxDepth-depth) / float64(maxDepth-minDepth)
		}
	}
	return scores
}

// scoreKVUtilization computes per-instance KV utilization scores.
// Lower utilization → higher score: score = 1 - KVUtilization.
// Matches llm-d's kv-cache-utilization-scorer semantics.
//
// Signal freshness (R17, INV-7):
//
//	Reads: KVUtilization (Periodic when interval>0, else Immediate).
//	WARNING: At high request rates with large intervals, this signal can be significantly stale.
//	Pair with load-balance (reads EffectiveLoad including synchronous InFlightRequests)
//	for staleness-critical deployments, or queue-depth for GIE parity.
//	See H3 experiment: 200x worse distribution uniformity at rate=5000.
func scoreKVUtilization(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		scores[snap.ID] = 1.0 - snap.KVUtilization
	}
	return scores
}

// scoreLoadBalance computes per-instance load balance scores using inverse transform.
// Lower effective load → higher score: score = 1/(1 + effectiveLoad).
// BLIS-native formula preserving absolute load differences (alternative to min-max).
//
// Signal freshness (R17, INV-7):
//
//	Reads: EffectiveLoad() = QueueDepth + BatchSize + InFlightRequests (synchronous + Periodic composite).
func scoreLoadBalance(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		scores[snap.ID] = 1.0 / (1.0 + float64(snap.EffectiveLoad()))
	}
	return scores
}

// scoreActiveRequests computes per-instance scores based on in-flight request count.
// Instances with zero in-flight always score 1.0. Non-zero instances use max-only
// normalization: (maxCount - count) / maxCount. When all instances have the same
// non-zero count, all score 0.0 (no differentiation) — contrast with running-requests
// which uses min-max normalization and scores 1.0 for all-equal. This asymmetry is
// intentional: it matches llm-d's active-request-scorer (active_request.go:193-230).
//
// Signal freshness (R17, INV-7):
//
//	Reads: InFlightRequests (synchronous — updated on dispatch/completion events).
func scoreActiveRequests(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	maxCount := 0
	for _, snap := range snapshots {
		if snap.InFlightRequests > maxCount {
			maxCount = snap.InFlightRequests
		}
	}
	for _, snap := range snapshots {
		if snap.InFlightRequests == 0 || maxCount == 0 {
			scores[snap.ID] = 1.0
		} else {
			scores[snap.ID] = float64(maxCount-snap.InFlightRequests) / float64(maxCount)
		}
	}
	return scores
}

// scoreRunningRequests computes per-instance scores based on running (in-batch) request count.
// Uses min-max normalization: (maxBatch - batch) / (maxBatch - minBatch). All equal = 1.0.
// Matches GIE's running-requests-size-scorer semantics (runningrequest.go:99).
//
// Signal freshness (R17, INV-7):
//
//	Reads: BatchSize (Periodic when interval>0, else Immediate).
func scoreRunningRequests(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	minBatch, maxBatch := math.MaxInt, 0
	for _, snap := range snapshots {
		if snap.BatchSize < minBatch {
			minBatch = snap.BatchSize
		}
		if snap.BatchSize > maxBatch {
			maxBatch = snap.BatchSize
		}
	}
	for _, snap := range snapshots {
		if maxBatch == minBatch {
			scores[snap.ID] = 1.0
		} else {
			scores[snap.ID] = float64(maxBatch-snap.BatchSize) / float64(maxBatch-minBatch)
		}
	}
	return scores
}

// loadAwareQueueThreshold is the default queue depth threshold for the load-aware scorer.
// Matches llm-d's QueueThresholdDefault (load_aware.go:42). Queue depths at or above
// this value score 0.0.
const loadAwareQueueThreshold = 128

// scoreLoadAware computes per-instance scores based on waiting queue depth with a
// linear threshold-capped formula. Score range: [0, 0.5].
// Empty queue scores 0.5 (maximum). Non-zero queue: 0.5 * (1 - queue/threshold),
// where queue depth is clamped to loadAwareQueueThreshold. At-or-above threshold = 0.0.
// Matches llm-d's load-aware-scorer semantics (load_aware.go:83-99).
//
// Signal freshness (R17, INV-7):
//
//	Reads: QueueDepth (Periodic when interval>0, else Immediate).
func scoreLoadAware(_ *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		if snap.QueueDepth == 0 {
			scores[snap.ID] = 0.5
		} else {
			clamped := snap.QueueDepth
			if clamped > loadAwareQueueThreshold {
				clamped = loadAwareQueueThreshold
			}
			scores[snap.ID] = 0.5 * (1.0 - float64(clamped)/float64(loadAwareQueueThreshold))
		}
	}
	return scores
}

// scoreLoRAAffinity computes per-instance scores favoring instances that already
// hold the request's LoRA adapter resident, avoiding a cold-load elsewhere (#1469,
// spec US4). Raw score is 1.0 when req.Adapter is resident on the instance, else
// 0.0; raw scores are then min-max normalized (llm-d parity), so with a mix of
// warm and cold candidates warm instances score 1.0 and cold instances 0.0.
//
// Neutral (all instances score 1.0, no routing bias) whenever raw scores are
// uniform: an empty req.Adapter (base-model request), no instance holding the
// adapter (all cold), every instance holding it (all warm), or a single candidate.
// This keeps base-model traffic and inert-LoRA deployments byte-identical to
// today (INV-6).
//
// INV-9 (oracle knowledge boundary): reads only req.Adapter and
// snapshot.ResidentAdapters — never the request's oracle output length.
//
// Signal freshness (R17, INV-7):
//
//	Reads: ResidentAdapters (Periodic when --snapshot-refresh-interval>0, else Immediate).
func scoreLoRAAffinity(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	adapter := ""
	if req != nil {
		adapter = req.Adapter
	}
	// Empty adapter (base-model request) ⇒ neutral: every instance scores 1.0.
	if adapter == "" {
		for _, snap := range snapshots {
			scores[snap.ID] = 1.0
		}
		return scores
	}
	raw := make(map[string]float64, len(snapshots))
	minRaw, maxRaw := 1.0, 0.0
	for _, snap := range snapshots {
		r := 0.0
		if snap.ResidentAdapters[adapter] {
			r = 1.0
		}
		raw[snap.ID] = r
		if r < minRaw {
			minRaw = r
		}
		if r > maxRaw {
			maxRaw = r
		}
	}
	// Uniform raw (all warm, all cold, or single instance) ⇒ neutral: all 1.0.
	if maxRaw == minRaw {
		for _, snap := range snapshots {
			scores[snap.ID] = 1.0
		}
		return scores
	}
	for _, snap := range snapshots {
		scores[snap.ID] = (raw[snap.ID] - minRaw) / (maxRaw - minRaw)
	}
	return scores
}
