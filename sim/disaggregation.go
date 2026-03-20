package sim

import "fmt"

// DisaggregationDecision encapsulates the prefill-decode disaggregation decision for a request.
type DisaggregationDecision struct {
	Disaggregate bool // true = route to prefill pool, false = route to shared/decode pool
}

// DisaggregationDecider decides whether a request should be disaggregated
// (sent to a dedicated prefill pool) or handled by the default routing pipeline.
// Used by ClusterSimulator's event pipeline when pool topology is configured.
//
// Implementations must not read Request.OutputTokens (INV-9 oracle boundary);
// use len(req.InputTokens) and req.MaxOutputLen only.
//
// req is guaranteed non-nil; implementations may assume a non-nil pointer.
type DisaggregationDecider interface {
	Decide(req *Request) DisaggregationDecision
}

// NeverDisaggregate always returns Disaggregate=false.
// Default decider when PD disaggregation is not configured.
type NeverDisaggregate struct{}

func (n *NeverDisaggregate) Decide(_ *Request) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: false}
}

// AlwaysDisaggregate always returns Disaggregate=true.
// Test-oriented decider for validating disaggregation pipeline wiring.
type AlwaysDisaggregate struct{}

func (a *AlwaysDisaggregate) Decide(_ *Request) DisaggregationDecision {
	return DisaggregationDecision{Disaggregate: true}
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
