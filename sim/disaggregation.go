package sim

import "fmt"

// DisaggregationDecision encapsulates the prefill-decode disaggregation decision for a request.
type DisaggregationDecision struct {
	Disaggregate bool // true = route to prefill pool, false = serve on selected decode instance
}

// DecodeContext carries the selected decode instance's identity and current cache state
// to the disaggregation decider. Populated by DecodeRoutingEvent before calling Decide.
type DecodeContext struct {
	InstanceID       string // ID of the decode-pool instance selected by DecodeRoutingEvent
	CachedBlockCount int    // number of complete KV blocks already cached on that instance
}

// DisaggregationDecider decides whether a request should be disaggregated (sent to a
// dedicated prefill pool) or served locally on the already-selected decode instance.
// Used by ClusterSimulator's event pipeline when pool topology is configured.
//
// In the decode-first flow, the decode instance has already been chosen by DecodeRoutingEvent
// before Decide is called. ctx carries the instance's identity and cache state, enabling
// per-instance disaggregation decisions: the same request may disaggregate on one decode
// instance (prefix not cached) but skip disaggregation on another (prefix cached).
//
// Implementations must not read Request.OutputTokens (INV-9 oracle boundary);
// use len(req.InputTokens) and req.MaxOutputLen only.
//
// req is guaranteed non-nil; implementations may assume a non-nil pointer.
type DisaggregationDecider interface {
	Decide(req *Request, ctx DecodeContext) DisaggregationDecision
}

// NeverDisaggregate always returns Disaggregate=false.
// Default decider when PD disaggregation is not configured.
type NeverDisaggregate struct{}

func (n *NeverDisaggregate) Decide(_ *Request, _ DecodeContext) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: false}
}

// AlwaysDisaggregate always returns Disaggregate=true.
// Test-oriented decider for validating disaggregation pipeline wiring.
type AlwaysDisaggregate struct{}

func (a *AlwaysDisaggregate) Decide(_ *Request, _ DecodeContext) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: true}
}

// DirectToDecodeDecider routes short prompts directly to the decode pool (Disaggregate=false)
// and long prompts through the full PD pipeline (Disaggregate=true).
// Decision: len(InputTokens) >= threshold -> disaggregate; < threshold -> direct to decode.
//
// Boundary note: this decider uses >= (disaggregate at exactly threshold tokens), while
// PrefixThresholdDecider uses strict > on the non-cached token count. A request with exactly
// threshold tokens disaggregates here but would not disaggregate under PrefixThresholdDecider
// at the same threshold value (assuming no cached tokens).
type DirectToDecodeDecider struct {
	threshold int
}

// NewDirectToDecodeDecider creates a DirectToDecodeDecider with the given input-length threshold.
// threshold must be >= 0. Panics otherwise (R3).
func NewDirectToDecodeDecider(threshold int) *DirectToDecodeDecider {
	if threshold < 0 {
		panic(fmt.Sprintf("NewDirectToDecodeDecider: threshold must be >= 0, got %d", threshold))
	}
	return &DirectToDecodeDecider{threshold: threshold}
}

// Decide returns Disaggregate=true when input length >= threshold (long prompt -> full PD pipeline),
// Disaggregate=false when input length < threshold (short prompt -> direct to decode pool).
// Empty inputs (len == 0) always return Disaggregate=false regardless of threshold.
// req must be non-nil (interface contract); panics on nil req (programming error).
// ctx is not used: this decider is purely input-length based.
func (d *DirectToDecodeDecider) Decide(req *Request, _ DecodeContext) DisaggregationDecision {
	if req == nil {
		panic("DirectToDecodeDecider.Decide: req is nil (programming error)")
	}
	if len(req.InputTokens) == 0 {
		return DisaggregationDecision{Disaggregate: false}
	}
	return DisaggregationDecision{Disaggregate: len(req.InputTokens) >= d.threshold}
}

// NewDisaggregationDecider creates a disaggregation decider by name.
// Valid names are defined in validDisaggregationDeciders (bundle.go).
// An empty string defaults to NeverDisaggregate.
// Panics on unrecognized names, on "prefix-threshold" (use NewPrefixThresholdDecider
// directly), or on "direct-to-decode" (use NewDirectToDecodeDecider directly).
// Both parameterized deciders require a caller-supplied threshold.
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
		panic("use NewPrefixThresholdDecider(threshold, blockSize) to construct prefix-threshold decider")
	case "direct-to-decode":
		panic("use NewDirectToDecodeDecider(threshold) to construct direct-to-decode decider")
	default:
		panic(fmt.Sprintf(
			"disaggregation decider %q is registered in validDisaggregationDeciders (bundle.go) "+
				"but has no case in NewDisaggregationDecider; add a case here",
			name))
	}
}

// PrefixThresholdDecider disaggregates a request when the number of non-cached input tokens
// on the selected decode instance exceeds the configured threshold.
//
// Non-cached token count: len(req.InputTokens) - ctx.CachedBlockCount * blockSize (clamped >= 0).
// Decision: Disaggregate = (nonCachedTokens > threshold).
//
// Unlike the previous implementation, this decider is stateless: it does not maintain a
// router-side prefix cache. Instead, it reads the selected decode instance's actual KV cache
// state from ctx.CachedBlockCount, which is populated by DecodeRoutingEvent via cacheQueryFn.
// This matches llm-d's prefix-based-pd-decider architecture (INV-PD-8).
//
// threshold and blockSize are in token counts, not bytes.
type PrefixThresholdDecider struct {
	threshold int
	blockSize int
}

// NewPrefixThresholdDecider creates a PrefixThresholdDecider with the given threshold and block size.
// threshold must be >= 0; blockSize must be > 0. Panics otherwise (R3).
func NewPrefixThresholdDecider(threshold, blockSize int) *PrefixThresholdDecider {
	if threshold < 0 {
		panic(fmt.Sprintf("NewPrefixThresholdDecider: threshold must be >= 0, got %d", threshold))
	}
	if blockSize <= 0 {
		panic(fmt.Sprintf("NewPrefixThresholdDecider: blockSize must be > 0, got %d", blockSize))
	}
	return &PrefixThresholdDecider{
		threshold: threshold,
		blockSize: blockSize,
	}
}

// Decide returns Disaggregate=true when the non-cached token count exceeds the threshold.
// Non-cached tokens = len(req.InputTokens) - ctx.CachedBlockCount * blockSize, clamped to 0.
// Empty requests (len(InputTokens) == 0) always return Disaggregate=false.
func (p *PrefixThresholdDecider) Decide(req *Request, ctx DecodeContext) DisaggregationDecision {
	if len(req.InputTokens) == 0 {
		return DisaggregationDecision{Disaggregate: false}
	}
	nonCachedTokens := len(req.InputTokens) - ctx.CachedBlockCount*p.blockSize
	if nonCachedTokens < 0 {
		nonCachedTokens = 0
	}
	return DisaggregationDecision{Disaggregate: nonCachedTokens > p.threshold}
}

// Compile-time interface compliance checks.
var _ DisaggregationDecider = (*DirectToDecodeDecider)(nil)
var _ DisaggregationDecider = (*PrefixThresholdDecider)(nil)
