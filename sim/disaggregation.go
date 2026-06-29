package sim

import (
	"fmt"
)

// DisaggregationDecision encapsulates the prefill-decode disaggregation decision for a request.
//
// DecodePodOverride and PrefillPodHint let a decider reconsider pod selection made
// by the routing policy. Empty string means "no override" — the caller keeps the
// previously selected decode pod / applies normal prefill routing. Joint D+P
// policies (future work) can populate these; the three built-in deciders always
// leave them empty. PrefillPodHint is defined to pre-empt a future interface
// break and is reserved for future joint D+P policies. GAP-4 (encode pool, #1264)
// does not consume this field — encode routing threads its decision via a local
// variable and ParentRequest.EncodeInstanceID, not through DisaggregationDecision.
type DisaggregationDecision struct {
	Disaggregate      bool   // true = route to prefill pool, false = route to shared/decode pool
	DecodePodOverride string // empty = keep pre-selected decode pod
	PrefillPodHint    string // empty = normal prefill routing (reserved for future joint D+P policies)
}

// DisaggregationDecider decides whether a request should be disaggregated
// (sent to a dedicated prefill pool) or handled by the default routing pipeline.
// Used by ClusterSimulator's event pipeline when pool topology is configured.
//
// Implementations must not read Request.OutputTokens (INV-9 oracle boundary);
// use len(req.InputTokens) and req.MaxOutputLen only.
//
// req is guaranteed non-nil; implementations may assume a non-nil pointer.
// state carries the decode-pool RouterState — the same snapshots used to
// pre-select the decode pod. state.SelectedInstance holds the ID of the pod
// chosen by the decode routing policy. Implementations may read any field of
// state except Request.OutputTokens. state is guaranteed non-nil at runtime
// (executeDisaggregatedRouting returns at the empty-decode-pool guard before
// calling Decide), but implementations must still tolerate a nil pointer for
// unit-test convenience. Passing the full RouterState (rather than a single
// snapshot) lets joint D+P policies reconsider the decode pod via
// DisaggregationDecision.DecodePodOverride.
type DisaggregationDecider interface {
	Decide(req *Request, state *RouterState) DisaggregationDecision
}

// NeverDisaggregate always returns Disaggregate=false.
// Default decider when PD disaggregation is not configured.
type NeverDisaggregate struct{}

func (n *NeverDisaggregate) Decide(_ *Request, _ *RouterState) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: false}
}

// AlwaysDisaggregate always returns Disaggregate=true.
// Test-oriented decider for validating disaggregation pipeline wiring.
type AlwaysDisaggregate struct{}

func (a *AlwaysDisaggregate) Decide(_ *Request, _ *RouterState) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: true}
}

// NewDisaggregationDecider creates a disaggregation decider by name.
// Valid names are defined in validDisaggregationDeciders (bundle.go).
// An empty string defaults to NeverDisaggregate.
// Panics on unrecognized names or on "prefix-threshold" (use NewPrefixThresholdDecider
// directly, as it requires a caller-supplied threshold and cache-query map).
func NewDisaggregationDecider(name string) DisaggregationDecider {
	if !IsValidDisaggregationDecider(name) {
		panic(fmt.Sprintf("unknown disaggregation decider %q", name))
	}
	switch name {
	case "", "never":
		return &NeverDisaggregate{}
	case "always":
		return &AlwaysDisaggregate{}
	case "prefix-threshold":
		panic("use NewPrefixThresholdDecider(threshold, blockSize, cacheQuery) to construct prefix-threshold decider")
	default:
		panic(fmt.Sprintf(
			"disaggregation decider %q is registered in validDisaggregationDeciders (bundle.go) "+
				"but has no case in NewDisaggregationDecider; add a case here",
			name))
	}
}

// PrefixThresholdDecider disaggregates a request when the tokens not already
// cached on the pre-selected decode pod exceed a configured threshold.
// Matches llm-d's PrefixBasedPDDecider.disaggregate, which reads per-endpoint
// PrefixCacheMatchInfo from the selected decode endpoint.
//
// Formula:
//
//	nonCachedTokens = len(req.InputTokens) − cachedBlocks × blockSize
//	Disaggregate    = (nonCachedTokens > threshold)
//
// where cachedBlocks is obtained by calling cacheQuery[state.SelectedInstance]
// on req.InputTokens. threshold and blockSize are in token counts, not bytes.
//
// cacheQuery shares the same map used by the precise-prefix-cache scorer
// (wired in sim/cluster/cluster.go from CachedSnapshotProvider.BuildCacheQueryFn).
// Staleness is inherited from that provider — see --cache-signal-delay (INV-7).
type PrefixThresholdDecider struct {
	threshold  int
	blockSize  int
	cacheQuery map[string]func([]TokenID) int
}

// NewPrefixThresholdDecider creates a PrefixThresholdDecider with the given
// threshold, block size, and per-pod cache-query map.
// threshold must be >= 0; blockSize must be > 0. Panics otherwise (R3).
// cacheQuery may be nil (e.g. unit tests without cluster state); when nil or
// when state.SelectedInstance is missing from the map, Decide returns
// Disaggregate=false (conservative fallback, consistent with llm-d's
// nil-endpoint guard at prefix_based_pd_decider.go:108-111).
func NewPrefixThresholdDecider(threshold, blockSize int, cacheQuery map[string]func([]TokenID) int) *PrefixThresholdDecider {
	if threshold < 0 {
		panic(fmt.Sprintf("NewPrefixThresholdDecider: threshold must be >= 0, got %d", threshold))
	}
	if blockSize <= 0 {
		panic(fmt.Sprintf("NewPrefixThresholdDecider: blockSize must be > 0, got %d", blockSize))
	}
	return &PrefixThresholdDecider{
		threshold:  threshold,
		blockSize:  blockSize,
		cacheQuery: cacheQuery,
	}
}

// Decide returns Disaggregate=true when the number of input tokens not cached
// on the pre-selected decode pod exceeds the threshold.
//
// Early returns (all yielding Disaggregate=false):
//   - len(req.InputTokens) == 0
//   - state == nil
//   - state.SelectedInstance == "" (no upstream selection)
//   - cacheQuery is nil, or the selected instance is missing from the map,
//     or its closure is nil (pod not yet registered / just removed)
func (p *PrefixThresholdDecider) Decide(req *Request, state *RouterState) DisaggregationDecision {
	if req.InputLen() == 0 {
		return DisaggregationDecision{Disaggregate: false}
	}
	if state == nil || state.SelectedInstance == "" || p.cacheQuery == nil {
		return DisaggregationDecision{Disaggregate: false}
	}
	fn, ok := p.cacheQuery[state.SelectedInstance]
	if !ok || fn == nil {
		return DisaggregationDecision{Disaggregate: false}
	}
	cachedBlocks := fn(req.FullInputTokens())
	nonCachedTokens := int(req.InputLen()) - cachedBlocks*p.blockSize
	return DisaggregationDecision{Disaggregate: nonCachedTokens > p.threshold}
}

// Compile-time interface compliance checks.
var (
	_ DisaggregationDecider = (*NeverDisaggregate)(nil)
	_ DisaggregationDecider = (*AlwaysDisaggregate)(nil)
	_ DisaggregationDecider = (*PrefixThresholdDecider)(nil)
)

// ---------------------------------------------------------------------------
// E/P/D disaggregation (GAP-4, issue #1264)
// ---------------------------------------------------------------------------

// EncodeDecider decides whether a request should be routed to the encode pool
// before proceeding to prefill/decode. Modeled on llm-d's
// AlwaysDisaggMultimodalDecider.disaggregate (Permalink 3 in issue #1264):
// the single decodeInstanceID argument mirrors llm-d's
// decodeRes.TargetEndpoints[0] — a pre-selected decode endpoint, not a snapshot.
//
// Implementations must not read Request.OutputTokens (INV-9 oracle boundary);
// use input-only signals such as per-modality token counts, len(InputTokens),
// MaxOutputLen, or the decode instance ID.
type EncodeDecider interface {
	ShouldEncode(req *Request, decodeInstanceID string) bool
}

// AlwaysEncode always returns true. Test-oriented decider for validating the
// encode routing wiring end-to-end.
type AlwaysEncode struct{}

func (a *AlwaysEncode) ShouldEncode(_ *Request, _ string) bool { return true }

// NeverEncode always returns false. Default when no encode decider is
// configured, and useful for "flag enabled, decider off" wiring tests
// (analogous to NeverDisaggregate).
type NeverEncode struct{}

func (n *NeverEncode) ShouldEncode(_ *Request, _ string) bool { return false }

// MultimodalEncodeDecider triggers encoding when the request carries any
// non-text modality. Matches llm-d's AlwaysDisaggMultimodalDecider
// (always_disagg_mm_decider.go:47-49) + hasMultimodalContent
// (multimodal_helpers.go:8-21), adapted to BLIS's token-count abstraction.
type MultimodalEncodeDecider struct{}

func (m *MultimodalEncodeDecider) ShouldEncode(req *Request, _ string) bool {
	// Interface contract (parity with DisaggregationDecider): req is guaranteed
	// non-nil at the call site. No defensive nil check here — symmetric with
	// AlwaysEncode / NeverEncode.
	return req.IsMultimodal()
}

// NewEncodeDecider creates an encode decider by name. Valid names are
// defined in validEncodeDeciders (bundle.go). An empty string defaults to
// NeverEncode. Panics on unrecognized names (R6).
func NewEncodeDecider(name string) EncodeDecider {
	if !IsValidEncodeDecider(name) {
		panic(fmt.Sprintf("unknown encode decider %q", name))
	}
	switch name {
	case "", "never":
		return &NeverEncode{}
	case "always":
		return &AlwaysEncode{}
	case "multimodal":
		return &MultimodalEncodeDecider{}
	default:
		panic(fmt.Sprintf(
			"encode decider %q is registered in validEncodeDeciders (bundle.go) "+
				"but has no case in NewEncodeDecider; add a case here",
			name))
	}
}

// Compile-time interface compliance checks.
var (
	_ EncodeDecider = (*AlwaysEncode)(nil)
	_ EncodeDecider = (*NeverEncode)(nil)
	_ EncodeDecider = (*MultimodalEncodeDecider)(nil)
)
