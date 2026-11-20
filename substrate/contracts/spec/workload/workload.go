// Package workload describes the traffic a deployment is measured against. Traffic
// comes at two fidelities. A Shape is a distribution — how long prompts are, how long
// completions are, and how much prefix requests share — because a cost model is a
// function of an actual batch and a mean hides the shape that matters: two corpora with
// the same mean input length, one tightly clustered and one spanning 52,000 to 93,000
// tokens, saturate the same deployment in different ways and at different concurrencies.
// A trace is a concrete corpus — real per-request lengths, arrival timestamps, and a
// prefix TREE — captured in an external file this package references by path.
//
// A third arm, a Replay, names a harness's agentic trace-replay scenario over a pinned
// corpus, where the harness decides when each request is sent.
//
// A Scenario's "what traffic" slot is a Binding: a sum type over those three. The slot is
// single on purpose, so a scenario names traffic once rather than carrying fields that
// could disagree.
package workload

import (
	"sort"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/spec/trace"
)

// Shape is one traffic class. Token counts are per request.
type Shape struct {
	// Name is the shape's identity, and it is a filename fact, not a file field: the
	// loader (LoadWorkload) stamps it from the path stem, so a workload is referred to by
	// the name of its file and nothing inside the file can disagree. The `yaml:"-"` tag
	// drops it from the file contract entirely — without it yaml.v3 would still bind a
	// `name` key here by the field's lowercased name, re-introducing the second source of
	// truth this removes — so under strict decoding an in-file `name` is now an unknown
	// field and is REJECTED.
	//
	// Chip.Name and Fabric.Name follow this same rule (#27): both are tagged `yaml:"-"`
	// with the stem as their only source, so spec/hardware's stated convention — "the files
	// also carry no name: identity is the filename" — and its code now agree, as they do
	// here. The three by-filename catalog entities are single-sourced identically.
	//
	// Validate still requires Name — now always loader-supplied — so a Shape reaching a
	// cost model without one is an error; it just cannot originate from the file.
	Name string `yaml:"-"`

	// PrefixTokens is the shared prefix length, which becomes the cached fraction a
	// cost model consumes. Zero means no sharing.
	PrefixTokens int `yaml:"prefix_tokens"`

	Prompt Distribution `yaml:"prompt"`
	Output Distribution `yaml:"output"`

	// Category and Thinking select the registry's acceptance profile for this traffic
	// when the deployment speculates: Category names the kind of prompts (SPEED-Bench's
	// "coding", for one), Thinking whether the model reasons. Acceptance, when set, is
	// the traffic's own measured acceptance, which outranks the registry's. All three are
	// optional; a speculative run that needs a profile and finds no category is refused
	// rather than guessed (see acceptance.Resolve).
	Category   string              `yaml:"category,omitempty"`
	Thinking   acceptance.Thinking `yaml:"thinking,omitempty"`
	Acceptance *acceptance.Value   `yaml:"acceptance,omitempty"`
	// Routing is the traffic's own measured expert-parallel routing imbalance, which
	// outranks the registry's profile for its Category (see routing.Resolve).
	Routing *routing.Value `yaml:"routing,omitempty"`
}

// UnmarshalYAML accepts the catalog's `_comment`-prefixed provenance keys and strips them
// before decoding, exactly as Chip, Fabric and StorageDevice do (#28). The catalog keeps
// each number's narrative in these keys, and workload token figures — why a chatbot prompt
// is 256 tokens with a stdev of 100, from which benchmark — are precisely the kind of number
// that wants one, so a workloads/*.yaml should be able to carry the same prose every
// hardware/*.yaml does rather than fail strict decoding on it.
//
// The custom unmarshaller carries a cost: yaml.Node.Decode does NOT honour the decoder's
// KnownFields setting, so a type that overrides UnmarshalYAML loses strict unknown-field
// checking unless it restores it. RejectUnknownKeys restores it here, and Distribution has
// its own UnmarshalYAML for the same reason — without it, a typo'd key nested under prompt:
// or output: would silently decode to a zero value, which this package rejected before #28
// and must keep rejecting.
func (s *Shape) UnmarshalYAML(node *yaml.Node) error {
	if err := validate.RejectUnknownKeys(node, Shape{}, validate.IgnoreCatalogComments); err != nil {
		return err
	}
	validate.StripCatalogComments(node)
	type shape Shape
	var raw shape
	if err := validate.DecodeNode(node, &raw); err != nil {
		return err
	}
	*s = Shape(raw)
	return nil
}

// Distribution is a token-count distribution. Mean and StdDev describe the bulk;
// Min and Max bound it, and they matter because a scheduler's behaviour at the
// tail is what sets a knee.
type Distribution struct {
	Mean   int `yaml:"tokens"`
	StdDev int `yaml:"tokens_stdev,omitempty"`
	Min    int `yaml:"tokens_min,omitempty"`
	Max    int `yaml:"tokens_max,omitempty"`
}

// UnmarshalYAML restores strict unknown-field checking at this nesting level and accepts the
// same `_comment` provenance keys, so a note can sit beside a single distribution (why this
// min, why this max) and not only at the top of the file. The hook is required, not stylistic
// symmetry with Shape: once Shape overrides UnmarshalYAML its node.Decode no longer honours
// the decoder's KnownFields setting for the values it recurses into, so without this method a
// misspelled key under prompt:/output: would silently decode to a zero value — exactly what
// RejectUnknownKeys here prevents.
func (d *Distribution) UnmarshalYAML(node *yaml.Node) error {
	if err := validate.RejectUnknownKeys(node, Distribution{}, validate.IgnoreCatalogComments); err != nil {
		return err
	}
	validate.StripCatalogComments(node)
	type shape Distribution
	var raw shape
	if err := validate.DecodeNode(node, &raw); err != nil {
		return err
	}
	*d = Distribution(raw)
	return nil
}

// Binding is a Scenario's "what traffic" slot: a sum type with three arms, exactly one
// of which is set. Shape references a catalog workload Shape — the distributional arm a
// Scenario has always carried; Trace references an external captured trace — the
// concrete arm; Replay describes a harness-driven agentic replay (see Replay). A Scenario that poses a capacity question with no traffic omits the
// binding entirely (a nil *Binding on the Scenario); a Binding that is present must
// choose one arm.
//
// The arms answer one question — what traffic a deployment is measured against — at
// different fidelities, which is why they share a single slot rather than sitting as two
// independent fields that could both be set or both omitted. A Shape is distributional:
// a cost model draws prompt and output lengths from it and shares one scalar prefix
// (Shape.PrefixTokens) across requests. A trace is concrete: real per-request lengths,
// arrival timestamps, and a prefix TREE — per-request prefix groups and lengths in the
// referenced data rows — that a single scalar cannot express. The prefix structure of a
// trace therefore lives in the referenced trace, never collapsed onto Shape.PrefixTokens.
type Binding struct {
	// Shape names a catalog workload Shape — the distributional arm. It is a name, not an
	// inline Shape, because the Shape lives in the catalog and is validated there; a
	// Scenario only identifies which one.
	Shape string `yaml:"shape,omitempty"`

	// Trace references an external captured trace — the concrete arm. The bulk
	// per-request rows stay in the file it names; only the reference and the small header
	// metadata are carried in the document.
	Trace *TraceRef `yaml:"trace,omitempty"`

	// Replay describes a named agentic trace-replay scenario over a pinned corpus -- the
	// harness-defined arm, where the harness, not a file of arrival times, decides when
	// each request is sent.
	Replay *Replay `yaml:"replay,omitempty"`
}

// TraceRef references a BLIS trace (contracts/spec/trace) by PATH, never by value. A real trace is a
// header plus a per-request data CSV that can run to millions of rows; inlining those
// rows into a Scenario would make the document unbounded and defeat the reason bulk data
// lives in files at all — the same reason the catalog and registry own their data rather
// than a scenario inlining it. So this type carries the path to the data file, optional
// integrity and provenance, and the SMALL header metadata as structured fields, and has
// no field in which per-request rows could be written.
type TraceRef struct {
	// Data is the path to the bulk per-request data CSV. It is the only reference to the
	// rows; the rows themselves are never represented in the document.
	Data string `yaml:"data"`

	// SHA256 is the optional content digest of the data file, so a consumer can detect a
	// trace that changed under a reference that did not. A 64-character hex digest when
	// present; absent when the producer recorded none.
	SHA256 string `yaml:"sha256,omitempty"`

	// Rows is the optional expected row COUNT of the data file — provenance and a cheap
	// cross-check against the file a consumer opens, NOT the rows themselves. Zero means
	// unstated.
	Rows int `yaml:"rows,omitempty"`

	// Header is the trace's header (contracts/spec/trace), copied from the trace file.
	Header trace.Header `yaml:"header"`

	// TokenBlocks is the optional sidecar of per-request token block hashes, which lets a
	// replay reproduce the prefix-cache hits of a materialized request stream rather than
	// approximate them from prefix groups.
	TokenBlocks *TokenBlocks `yaml:"token_blocks,omitempty"`

	// Category, Thinking and Acceptance state the trace's acceptance terms, as a Shape's
	// do. A trace whose server log recorded its acceptance carries it here.
	Category   string              `yaml:"category,omitempty"`
	Thinking   acceptance.Thinking `yaml:"thinking,omitempty"`
	Acceptance *acceptance.Value   `yaml:"acceptance,omitempty"`
	// Routing is the traffic's own measured expert-parallel routing imbalance, which
	// outranks the registry's profile for its Category (see routing.Resolve).
	Routing *routing.Value `yaml:"routing,omitempty"`
}

// TokenBlocks references a trace's block-hash sidecar by path and digest. The digest is
// required: a sidecar that changed under its reference would silently move every
// prefix-cache hit.
type TokenBlocks struct {
	Path   string `yaml:"path"`
	SHA256 string `yaml:"sha256"`
}

// enumerate returns the members of a closed string vocabulary, sorted, for error text.
func enumerate[T ~string](set map[T]bool) []string {
	out := make([]string, 0, len(set))
	for v := range set {
		out = append(out, string(v))
	}
	sort.Strings(out)
	return out
}

// SLODimTargets is one SLO class's latency thresholds, in milliseconds. It is the trace
// format's type, named here for the documents that state targets outside a trace.
type SLODimTargets = trace.SLODimTargets

// AcceptanceTerms returns what the shape states about acceptance, for acceptance.Resolve.
func (s *Shape) AcceptanceTerms() acceptance.Workload {
	return acceptance.Workload{Category: s.Category, Thinking: s.Thinking, Value: s.Acceptance}
}

// RoutingTerms returns what the shape states about routing, for routing.Resolve.
func (s *Shape) RoutingTerms() routing.Workload {
	return routing.Workload{Category: s.Category, Value: s.Routing}
}

// RoutingTerms returns what the trace states about routing, for routing.Resolve.
func (t *TraceRef) RoutingTerms() routing.Workload {
	return routing.Workload{Category: t.Category, Value: t.Routing}
}

// AcceptanceTerms returns what the trace states about acceptance, for acceptance.Resolve.
func (t *TraceRef) AcceptanceTerms() acceptance.Workload {
	return acceptance.Workload{Category: t.Category, Thinking: t.Thinking, Value: t.Acceptance}
}
