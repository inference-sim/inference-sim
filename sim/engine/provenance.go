package engine

import (
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/vocab"
)

// The kernel provenance and routing records the latency kernel's adapter (sim/kernelmodel)
// produces and the host's report embeds unchanged (Metrics' --metrics-path output). They are
// the engine's output, so they are declared with the engine; package sim names them.

// KernelPoolProvenance is the evidence behind one pool's kernel (#1942): how many of the
// coefficients it priced with rest on each evidence method, the registry entries that rest on
// none (copied from another scope, or assumed), the assumptions the kernel made itself, and
// the release rules no one applied to the scenario. Entries are blis-latency-kernel's own
// kernel.CoefficientOrigin records, so their keys are that type's field names.
type KernelPoolProvenance struct {
	// MethodCounts counts the Provenance entries by evidence method; Evidenced and Total are
	// the kernel's Evidence() figures.
	MethodCounts map[vocab.Method]int `json:"method_counts"`
	Evidenced    int                  `json:"evidenced"`
	Total        int                  `json:"total"`
	// EvidenceSource says where Evidenced came from: the kernel's Evidence(), or, for a
	// kernel that does not report it, nothing (Evidenced is 0 and Total counts the entries).
	EvidenceSource string `json:"evidence_source"`
	// NotEvidenced are the registry entries whose method is neither evidenced nor
	// not_charged: copied from another scope (a fallback) or assumed.
	NotEvidenced []kernel.CoefficientOrigin `json:"fallback_or_assumed"`
	// Assumptions are the values the kernel supplied itself (Set == KernelAssumptionSet).
	Assumptions []kernel.CoefficientOrigin `json:"assumptions"`
	// SkippedRules are the scenario's release rules that neither the kernel nor BLIS
	// enforces, and SkippedRuleFindings what they report on this scenario.
	SkippedRules        []string      `json:"skipped_rules"`
	SkippedRuleFindings []RuleFinding `json:"skipped_rule_findings"`
	// Routing is the pool's resolved expert-parallel routing imbalance: which layer
	// supplied it, from which set and source, and why the default applied when it did.
	Routing *RoutingRecord `json:"routing,omitempty"`
}

// RoutingRecord is the provenance of a pool's routing-imbalance resolution
// (blis-schemas routing.Resolve), recorded the way acceptance's source layer is.
type RoutingRecord struct {
	SourceLayer string `json:"source_layer"`
	Set         string `json:"set,omitempty"`
	// Keys are the decode and prefill profiles a profile layer resolved to.
	Keys         []string `json:"keys,omitempty"`
	Source       string   `json:"source,omitempty"`
	Reason       string   `json:"reason,omitempty"`
	MeasuredOn   string   `json:"measured_on,omitempty"`
	FallbackFrom string   `json:"fallback_from,omitempty"`
	// TokenImbalance and ExpertImbalance are an inline value's ratios.
	TokenImbalance  float64 `json:"token_imbalance,omitempty"`
	ExpertImbalance float64 `json:"expert_imbalance,omitempty"`
}

// The values of KernelPoolProvenance.EvidenceSource.
const (
	EvidenceFromKernel     = "kernel"
	EvidenceFromProvenance = "provenance_entries"
)

// RuleFinding is one finding of a blis-schemas release rule.
type RuleFinding struct {
	Rule     string `json:"rule"`
	Severity string `json:"severity"`
	Path     string `json:"path,omitempty"`
	Message  string `json:"message"`
}
