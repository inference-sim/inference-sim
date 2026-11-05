// Moved from blis-latency-kernel/internal/harness/harness.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

// Package harness holds what this repository's inspection commands share: opening a kernel
// from a scenario file, and building the batch shapes they price.
//
// It is internal and it is not part of the cost model. The kernel package implements the
// interface blis-schemas defines and takes already-parsed documents; where those documents live
// on disk, and what batch a particular comparison wants, are properties of a harness rather
// than of a cost model. Putting them here keeps that boundary where the schema draws it.
//
// The reason it exists at all is that the commands had one copy each of the same resolution.
// That is four places for a loading rule to drift: one copy read the model graph and omitted
// the fabric, so a multi-node scenario would have priced its cross-node collectives at the
// on-node rate with nothing reporting it.
package harness

import (
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	vllmrules "github.com/inference-sim/blis-vllm/rules"
	"github.com/inference-sim/blis-vllm/rules/pack"

	latencykernel "github.com/inference-sim/blis-substrate/kernel"
	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
)

// Repos is the root package's Repos. An alias rather than a second type, so that a
// harness.Repos{...} literal and a latencykernel.Repos{...} literal are the same value and
// no conversion is needed at the boundary.
type Repos = latencykernel.Repos

// Open forwards to the root package's Open, which is where this sequence now lives.
//
// It stayed here as a function rather than being deleted because every command in this
// repository calls harness.Open, and the move was about making the sequence REACHABLE from
// other modules rather than about relocating callers. Keeping the name means the promotion
// is a visibility change and nothing else.
//
// It supplies the engine rules, which the root package no longer looks up: a Repos without
// a rules source gets the vLLM packs (WithVLLMRules). The commands and this repository's
// tests are the kernel's own callers, so they take the packs from the engine model's rules
// module like any other caller; the library itself never imports it.
func Open(scenario string, r Repos) (*latencykernel.Kernel, error) {
	return latencykernel.Open(scenario, WithVLLMRules(r))
}

// WithVLLMRules returns r with the vLLM rules packs (github.com/inference-sim/blis-vllm/rules)
// as its rules source, where it states none.
func WithVLLMRules(r Repos) Repos {
	if r.Rules == nil {
		r.Rules = VLLMRules
		r.RulesVersions = vllmrules.Versions
	}
	if r.EngineSettings == nil {
		r.EngineSettings = vllmsettings.OfPool
	}
	return r
}

// VLLMRules is the vLLM rules pack for a release, nil for a release with none.
var VLLMRules = latencykernel.RulesFrom(func(v string) *pack.Pack { return vllmrules.Lookup(v) })

// DecodeThreshold is the scheduled-token count above which the kernel prices a request as
// prefill, for the batches the commands build. It is this harness's cutoff, not a
// value read from the engine: vLLM's is per backend -- 1 by default, raised to
// 1 + num_speculative_tokens under speculative decoding, and 128 or 512 on the FlashMLA
// and FlashAttention-MLA backends (vllm/v1/attention/backends/utils.py, backend.py, and
// mla/flashmla.py:113, mla/flashattn_mla.py:110 at v0.31.0). cmd/overlap-probe and docgen
// price single-token decode batches, which every one of those thresholds classifies as
// decode; a prefill chunk of at most 128 tokens (FlashMLA) or 512 (FlashAttention-MLA), which
// cmd/bandprobe and cmd/worked-table can pass, is one those backends would run on their
// decode path instead.
const DecodeThreshold = 8

// DecodeBatch builds a steady-state decode batch: every resident request contributes one token
// this step and carries the given context.
//
// It is the shape a steady-state decode step has, which the figure in docs/generated and the
// overlap probe price.
func DecodeBatch(requests, context int) kernel.Batch {
	b := kernel.Batch{
		Reqs:            make([]kernel.ReqShape, requests),
		DecodeThreshold: DecodeThreshold,
	}
	for i := range b.Reqs {
		b.Reqs[i] = kernel.ReqShape{
			Scheduled: 1, Computed: context - 1, PromptLen: context,
		}
	}
	return b
}
