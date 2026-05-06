package sim

import (
	"fmt"

	"github.com/sirupsen/logrus"
)

// DisaggregationDecision encapsulates the prefill-decode disaggregation decision for a request.
type DisaggregationDecision struct {
	Disaggregate bool // true = route to prefill pool, false = route to shared/decode pool
}

// RequestView is the read-only projection of a *Request that DisaggregationDecider
// implementations may observe. OutputTokens is deliberately absent — INV-9 (oracle
// knowledge boundary) is enforced structurally: an evolved decider cannot reach
// Request.OutputTokens at all, so policies trained against BLIS can be ported to
// llm-d's prefix-based-pd-decider (whose scheduling.InferenceRequest likewise lacks
// OutputTokens) without hidden assumptions.
//
// InputTokens is the same slice that lives on the underlying *Request; callers must
// treat it as read-only. Tokens remain addressable by index but must not be mutated.
type RequestView struct {
	ID           string
	InputTokens  []int
	MaxOutputLen int
	SLOClass     string
	// Priority mirrors the scheduling priority score set on the underlying *Request
	// (see sim/request.go). Float-typed because the control plane recomputes it each
	// step under custom priority policies.
	Priority    float64
	ArrivalTime int64
}

// NewRequestView builds a RequestView from a *Request. OutputTokens is intentionally
// not copied or referenced — this is the narrowing that enforces INV-9 structurally.
func NewRequestView(req *Request) RequestView {
	return RequestView{
		ID:           req.ID,
		InputTokens:  req.InputTokens,
		MaxOutputLen: req.MaxOutputLen,
		SLOClass:     req.SLOClass,
		Priority:     req.Priority,
		ArrivalTime:  req.ArrivalTime,
	}
}

// DisaggregationContext carries per-decision context from the cluster layer into a
// DisaggregationDecider. It is passed BY VALUE (not pointer) with unexported fields;
// deciders can only read via the accessor methods, so no mutable handle to cluster-
// owned state is ever reachable (structural analog of R9: exported mutable maps are
// forbidden).
//
// Extension point: later PRs in the #1241 tracker (e.g., pending-transfer accessor
// wiring, PR 4) add more accessors here without changing the decider interface shape.
type DisaggregationContext struct {
	decodeInstanceID string
	decodeCacheQuery func(tokens []int) int
}

// NewDisaggregationContext returns a DisaggregationContext. The `decodeCacheQuery`
// closure MUST be the same function returned by
// CachedSnapshotProvider.BuildCacheQueryFn()[decodeInstanceID] (i.e., the closure
// already consumed by the precise-prefix-cache / no-hit-lru scorers). This guarantees
// the decider observes the cache under the prevailing --cache-signal-delay regime —
// a policy evolved against a fresher signal than the runtime scorer sees could not
// be ported faithfully back to llm-d. See issue #1250 BEH-6 and CLAUDE.md INV-7.
func NewDisaggregationContext(decodeInstanceID string, decodeCacheQuery func(tokens []int) int) DisaggregationContext {
	return DisaggregationContext{
		decodeInstanceID: decodeInstanceID,
		decodeCacheQuery: decodeCacheQuery,
	}
}

// DecodeInstanceID returns the ID of the decode endpoint already selected by the
// decode routing profile.
func (c DisaggregationContext) DecodeInstanceID() string { return c.decodeInstanceID }

// DecodeCacheQuery returns the number of complete KV blocks of `tokens` already
// cached on the selected decode endpoint, matching the count that
// precise-prefix-cache would observe for the same (instance, tokens) at the same
// clock. Returns 0 when no closure is installed (disaggregation disabled path).
func (c DisaggregationContext) DecodeCacheQuery(tokens []int) int {
	if c.decodeCacheQuery == nil {
		return 0
	}
	return c.decodeCacheQuery(tokens)
}

// DisaggregationDecider decides whether a request should be disaggregated
// (sent to a dedicated prefill pool) or handled by the default routing pipeline.
// Used by ClusterSimulator's event pipeline when pool topology is configured.
//
// `view` is a narrow read-only projection of the request; OutputTokens is
// structurally unreachable (INV-9). `ctx` carries the selected decode endpoint's
// identity and its per-endpoint cache query closure.
//
// Stateful implementations may additionally implement DisaggregationObserver to
// learn from routing outcomes (e.g., GlobalPrefixThresholdDecider).
type DisaggregationDecider interface {
	Decide(view RequestView, ctx DisaggregationContext) DisaggregationDecision
}

// NeverDisaggregate always returns Disaggregate=false.
// Default decider when PD disaggregation is not configured.
type NeverDisaggregate struct{}

func (n *NeverDisaggregate) Decide(_ RequestView, _ DisaggregationContext) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: false}
}

// AlwaysDisaggregate always returns Disaggregate=true.
// Test-oriented decider for validating disaggregation pipeline wiring.
type AlwaysDisaggregate struct{}

func (a *AlwaysDisaggregate) Decide(_ RequestView, _ DisaggregationContext) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: true}
}

// NewDisaggregationDecider creates a disaggregation decider by name.
// Valid names are defined in validDisaggregationDeciders (bundle.go).
// An empty string defaults to NeverDisaggregate.
// Panics on unrecognized names or on parameterized deciders (prefix-threshold,
// prefix-based-pd-decider, global-prefix-threshold) — use the typed constructor
// for those because they require caller-supplied threshold and blockSize.
func NewDisaggregationDecider(name string) DisaggregationDecider {
	if !IsValidDisaggregationDecider(name) {
		panic(fmt.Sprintf("unknown disaggregation decider %q", name))
	}
	switch name {
	case "", "never":
		return &NeverDisaggregate{}
	case "always":
		return &AlwaysDisaggregate{}
	case "prefix-threshold", "prefix-based-pd-decider":
		panic("use NewPrefixBasedPDDecider(threshold, blockSize) to construct the prefix-based PD decider")
	case "global-prefix-threshold":
		panic("use NewGlobalPrefixThresholdDecider(threshold, blockSize) to construct the counterfactual global-prefix-threshold decider")
	default:
		panic(fmt.Sprintf(
			"disaggregation decider %q is registered in validDisaggregationDeciders (bundle.go) "+
				"but has no case in NewDisaggregationDecider; add a case here",
			name))
	}
}

// globalVirtualInstance is the single key used by GlobalPrefixThresholdDecider's
// PrefixCacheIndex to represent cluster-wide prefix knowledge. All requests update
// the same virtual instance so the decider tracks the global set of recently-seen
// prefixes. This is the BLIS pre-1250 strawman baseline, retained for counterfactual
// evaluation only (A1 protocol in #1006 Phase B §11).
const globalVirtualInstance = "__global__"

// defaultDisaggLRUCapacity is the number of block hash slots tracked in
// GlobalPrefixThresholdDecider's router-side prefix cache (LRU capacity).
const defaultDisaggLRUCapacity = 10000

// DisaggregationObserver is an optional interface for stateful DisaggregationDeciders that need
// to learn from routing decisions. ClusterSimulator calls ObserveRouting after each routing
// decision that assigns a request to an instance: after standard routing (RoutingDecisionEvent
// in cluster_event.go) and after prefill routing (PrefillRoutingEvent in pd_events.go).
// It is not called for decode routing, because the decider's prefix knowledge is based on
// input tokens, which are identical between prefill and decode sub-requests.
//
// ObserveRouting is called synchronously within the event loop immediately after each
// routing decision, so the prefix cache is always current at the next Decide() call.
//
// req is guaranteed non-nil. Implementations must treat req as read-only.
// instanceID is the routing target; implementations maintaining per-instance state should
// record it; implementations with global state (like GlobalPrefixThresholdDecider) may
// ignore it.
type DisaggregationObserver interface {
	ObserveRouting(req *Request, instanceID string)
}

// PrefixBasedPDDecider disaggregates a request when its non-cached token count on
// the selected decode pod exceeds the configured threshold. Per-pod cache state is
// read via ctx.DecodeCacheQuery (backed by CachedSnapshotProvider), matching llm-d's
// prefix-based-pd-decider behavior. Stateless — no router-side LRU.
//
// Non-cached token count: len(view.InputTokens) - cachedBlocks * blockSize. threshold
// and blockSize are in token counts, not bytes. Decision: Disaggregate = (nonCached > threshold).
//
// `threshold == 0` short-circuits to Disaggregate=false (BEH-3, matches llm-d's
// NonCachedTokens == 0 guard at decider_plugin.go:104 of parity commit e52311b7).
type PrefixBasedPDDecider struct {
	threshold int
	blockSize int
}

// NewPrefixBasedPDDecider creates a PrefixBasedPDDecider with the given threshold and
// block size. threshold must be >= 0; blockSize must be > 0. Panics otherwise (R3).
func NewPrefixBasedPDDecider(threshold, blockSize int) *PrefixBasedPDDecider {
	if threshold < 0 {
		panic(fmt.Sprintf("NewPrefixBasedPDDecider: threshold must be >= 0, got %d", threshold))
	}
	if blockSize <= 0 {
		panic(fmt.Sprintf("NewPrefixBasedPDDecider: blockSize must be > 0, got %d", blockSize))
	}
	return &PrefixBasedPDDecider{
		threshold: threshold,
		blockSize: blockSize,
	}
}

// NewPrefixThresholdDecider is retained as a backward-compatible constructor name
// that now returns the per-pod decider. Callers that previously relied on the old
// global-LRU semantics should migrate to NewGlobalPrefixThresholdDecider. This is
// a behavior change documented in PR #1255 / issue #1250 and surfaced at runtime
// by the --pd-decider=prefix-threshold warning in cmd/root.go.
func NewPrefixThresholdDecider(threshold, blockSize int) *PrefixBasedPDDecider {
	return NewPrefixBasedPDDecider(threshold, blockSize)
}

// Decide returns Disaggregate=true when non-cached token count on the selected
// decode pod exceeds the threshold. Empty input (len(InputTokens) == 0) always
// returns Disaggregate=false. `threshold == 0` always returns Disaggregate=false
// (disable-disaggregation short-circuit).
func (p *PrefixBasedPDDecider) Decide(view RequestView, ctx DisaggregationContext) DisaggregationDecision {
	if p.threshold == 0 {
		return DisaggregationDecision{Disaggregate: false}
	}
	if len(view.InputTokens) == 0 {
		return DisaggregationDecision{Disaggregate: false}
	}
	cachedBlocks := ctx.DecodeCacheQuery(view.InputTokens)
	nonCachedTokens := len(view.InputTokens) - cachedBlocks*p.blockSize
	if nonCachedTokens < 0 {
		// Defensive: cachedBlocks*blockSize should never exceed input length, but guard
		// against a misbehaving cache-query closure to avoid a negative comparison result.
		nonCachedTokens = 0
	}
	return DisaggregationDecision{Disaggregate: nonCachedTokens > p.threshold}
}

// GlobalPrefixThresholdDecider retains the pre-1250 behavior: a router-side LRU
// under a single sentinel key (globalVirtualInstance) approximates cluster-wide
// prefix knowledge. This is strictly a counterfactual baseline for experiments
// (see #1006 Phase B §11) — it does NOT mirror llm-d, so it must not be used as
// a default. Implements DisaggregationObserver so the aggregate cache is updated
// after each routing decision.
//
// Threading: cachedHashes and cachedReqID are a single-use scratchpad — Decide()
// writes them and ObserveRouting() consumes and clears them. Safe only because
// the DES event loop is single-threaded.
type GlobalPrefixThresholdDecider struct {
	threshold    int
	blockSize    int
	idx          *PrefixCacheIndex
	cachedHashes []string
	cachedReqID  string
}

// NewGlobalPrefixThresholdDecider creates a GlobalPrefixThresholdDecider with the
// given threshold and block size. threshold must be >= 0; blockSize must be > 0.
// Panics otherwise (R3).
func NewGlobalPrefixThresholdDecider(threshold, blockSize int) *GlobalPrefixThresholdDecider {
	if threshold < 0 {
		panic(fmt.Sprintf("NewGlobalPrefixThresholdDecider: threshold must be >= 0, got %d", threshold))
	}
	if blockSize <= 0 {
		panic(fmt.Sprintf("NewGlobalPrefixThresholdDecider: blockSize must be > 0, got %d", blockSize))
	}
	return &GlobalPrefixThresholdDecider{
		threshold: threshold,
		blockSize: blockSize,
		idx:       NewPrefixCacheIndex(blockSize, defaultDisaggLRUCapacity),
	}
}

// Decide returns Disaggregate=true when non-cached token count exceeds the threshold,
// using the router-side global virtual LRU to estimate cluster-wide cache hits.
// `threshold == 0` short-circuits to false to match PrefixBasedPDDecider semantics.
func (p *GlobalPrefixThresholdDecider) Decide(view RequestView, _ DisaggregationContext) DisaggregationDecision {
	if p.threshold == 0 {
		return DisaggregationDecision{Disaggregate: false}
	}
	if len(view.InputTokens) == 0 {
		return DisaggregationDecision{Disaggregate: false}
	}
	p.cachedHashes = p.idx.ComputeBlockHashes(view.InputTokens)
	p.cachedReqID = view.ID
	cachedBlocks := p.idx.MatchLength(p.cachedHashes, globalVirtualInstance)
	nonCachedTokens := len(view.InputTokens) - cachedBlocks*p.blockSize
	return DisaggregationDecision{Disaggregate: nonCachedTokens > p.threshold}
}

// ObserveRouting records the request's block hashes under globalVirtualInstance.
// Reuses hashes from Decide when the request ID matches; otherwise recomputes.
// The ID mismatch case is expected in the disaggregated path: notifyDisaggregationObserver
// is called with the prefill sub-request (ID "<parent>_prefill"), which differs from the
// parent request evaluated by Decide.
func (p *GlobalPrefixThresholdDecider) ObserveRouting(req *Request, _ string) {
	if req == nil {
		logrus.Errorf("GlobalPrefixThresholdDecider.ObserveRouting: req is nil (contract violation); skipping cache update")
		return
	}
	if len(req.InputTokens) == 0 {
		return
	}
	hashes := p.cachedHashes
	if req.ID != p.cachedReqID || p.cachedHashes == nil {
		hashes = p.idx.ComputeBlockHashes(req.InputTokens)
	}
	p.idx.RecordBlocks(hashes, globalVirtualInstance)
	p.cachedHashes = nil
	p.cachedReqID = ""
}

// Compile-time interface compliance checks.
var (
	_ DisaggregationDecider  = (*NeverDisaggregate)(nil)
	_ DisaggregationDecider  = (*AlwaysDisaggregate)(nil)
	_ DisaggregationDecider  = (*PrefixBasedPDDecider)(nil)
	_ DisaggregationDecider  = (*GlobalPrefixThresholdDecider)(nil)
	_ DisaggregationObserver = (*GlobalPrefixThresholdDecider)(nil)
)
