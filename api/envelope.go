package api

import (
	"fmt"
	"slices"
	"strings"
)

// APIVersion is the self-describing version string every BLIS document carries. It is a
// defined type rather than a bare string so a document's version can only be compared
// against, and generated from, the one constant below.
type APIVersion string

// Version is the only apiVersion this build reads or writes.
const Version APIVersion = "llm-d-perf-simulator/v1"

// Valid reports whether v is the apiVersion this build understands.
func (v APIVersion) Valid() bool { return v == Version }

// schemaConstraints pins apiVersion to exactly one value in the generated schema, so an
// external tool validating a document gets the version check for free.
func (APIVersion) schemaConstraints() map[string]any {
	return map[string]any{
		"type":        "string",
		"const":       string(Version),
		"description": "Document format version. Exactly one value is defined.",
	}
}

// Kind identifies which document this is: one of the five input kinds (named after the
// CLI verb that consumes them) or their five "…Result" output forms.
type Kind string

// The input kinds — one per CLI verb.
const (
	KindCalibrate Kind = "Calibrate"
	KindConvert   Kind = "Convert"
	KindObserve   Kind = "Observe"
	KindReplay    Kind = "Replay"
	KindRun       Kind = "Run"
)

// The output kinds — each the result form of the input kind of the same name.
const (
	KindCalibrateResult Kind = KindCalibrate + resultSuffix
	KindConvertResult   Kind = KindConvert + resultSuffix
	KindObserveResult   Kind = KindObserve + resultSuffix
	KindReplayResult    Kind = KindReplay + resultSuffix
	KindRunResult       Kind = KindRun + resultSuffix
)

// resultSuffix turns an input kind into its output kind. The two forms share a name on
// purpose: a reader can tell which verb produced a result document without a lookup table.
const resultSuffix Kind = "Result"

// inputKinds is the authoritative list of input kinds, sorted. Unexported because an
// exported slice (like an exported map, R8) is mutable by any caller; InputKinds returns
// a copy. Sorted at the source so every derived list — the schema enum, the oneOf
// partition, a diagnostic — is deterministic (INV-6) without a sort at each use.
var inputKinds = []Kind{
	KindCalibrate,
	KindConvert,
	KindObserve,
	KindReplay,
	KindRun,
}

// InputKinds returns the input kinds, sorted. Each carries a spec.
func InputKinds() []Kind { return slices.Clone(inputKinds) }

// OutputKinds returns the output kinds, sorted. Each carries a result.
func OutputKinds() []Kind {
	out := make([]Kind, 0, len(inputKinds))
	for _, k := range inputKinds {
		out = append(out, k+resultSuffix)
	}
	return out
}

// AllKinds returns every kind the envelope defines, sorted.
func AllKinds() []Kind {
	all := append(InputKinds(), OutputKinds()...)
	slices.Sort(all)
	return all
}

// IsInput reports whether k is an input kind, which carries a spec.
func IsInput(k Kind) bool { return slices.Contains(inputKinds, k) }

// IsOutput reports whether k is an output kind, which carries a result.
func IsOutput(k Kind) bool { return slices.Contains(OutputKinds(), k) }

// IsValidKind reports whether k is a kind the envelope defines.
func IsValidKind(k Kind) bool { return IsInput(k) || IsOutput(k) }

// ResultKind returns the output kind that pairs with the input kind k. The second return
// is false for any kind that is not an input kind, so a caller cannot build
// "RunResultResult" by accident.
func ResultKind(k Kind) (Kind, bool) {
	if !IsInput(k) {
		return "", false
	}
	return k + resultSuffix, true
}

// BodyField returns the name of the top-level field k's documents carry: "spec" for an
// input kind, "result" for an output kind, "" for an unknown kind.
func BodyField(k Kind) string {
	switch {
	case IsInput(k):
		return "spec"
	case IsOutput(k):
		return "result"
	default:
		return ""
	}
}

// schemaConstraints pins kind to the defined enum in the generated schema.
func (Kind) schemaConstraints() map[string]any {
	return map[string]any{
		"type":        "string",
		"enum":        kindStrings(AllKinds()),
		"description": "Which document this is. Input kinds carry a spec; …Result kinds carry a result.",
	}
}

// Body is a document body — the spec of an input document or the result of an output
// document.
//
// At this stage (API-1) it is OPAQUE: an open object whose contents are unconstrained.
// API-2 onward replace it with the typed per-verb sections, at which point the generated
// schema constrains those contents and parsing of them becomes strict (R10). Until then a
// body is accepted as long as it is an object, so this package can ship the envelope
// without freezing a single field of the bodies.
type Body map[string]any

// schemaConstraints keeps a body an OPEN object for now; see the Body doc comment.
func (Body) schemaConstraints() map[string]any {
	return map[string]any{
		"type":                 "object",
		"additionalProperties": true,
		"description": "Opaque document body. API-1 constrains envelope structure only; " +
			"the per-verb sections are defined by later PRs in epic #1855.",
	}
}

// Document is the envelope every BLIS declarative document shares. Exactly one of Spec
// and Result is populated, chosen by Kind: a spec for an input kind, a result for an
// output kind.
//
// Spec and Result are POINTERS because an empty body is a meaningful value, distinct from
// an absent one (R9): `spec: {}` is a valid minimal input document, and a plain map would
// serialize away under omitempty, turning "present and empty" into "missing" on every
// round trip.
//
// The struct is the single source of the committed JSON Schema — GenerateSchema derives
// the schema from these fields and their types, so a field added here without
// regenerating fails TestCommittedSchemaIsCurrent.
type Document struct {
	APIVersion APIVersion `json:"apiVersion" yaml:"apiVersion"`
	Kind       Kind       `json:"kind" yaml:"kind"`
	Spec       *Body      `json:"spec,omitempty" yaml:"spec,omitempty"`
	Result     *Body      `json:"result,omitempty" yaml:"result,omitempty"`
}

// NewSpec returns an input document of the given kind carrying body. It is the canonical
// construction site for an input envelope (R4): the apiVersion is never passed in, so it
// cannot be stamped wrong. An unknown or output kind is refused rather than silently
// producing a document no reader accepts.
func NewSpec(kind Kind, body Body) (Document, error) {
	if !IsInput(kind) {
		return Document{}, fmt.Errorf("kind %q is not an input kind; input kinds are %s",
			kind, strings.Join(kindStrings(InputKinds()), ", "))
	}
	if body == nil {
		body = Body{}
	}
	return Document{APIVersion: Version, Kind: kind, Spec: &body}, nil
}

// NewResult returns an output document of the given kind carrying body. Canonical
// construction site for an output envelope; see NewSpec.
func NewResult(kind Kind, body Body) (Document, error) {
	if !IsOutput(kind) {
		return Document{}, fmt.Errorf("kind %q is not an output kind; output kinds are %s",
			kind, strings.Join(kindStrings(OutputKinds()), ", "))
	}
	if body == nil {
		body = Body{}
	}
	return Document{APIVersion: Version, Kind: kind, Result: &body}, nil
}

// Validate reports whether d is a well-formed envelope, naming every problem it finds
// rather than the first: a wrong apiVersion, an undefined kind, and a missing body are
// independent mistakes, and a caller fixing them one round-trip at a time is the R1
// failure mode in slow motion.
//
// This is the Go-side twin of the committed JSON Schema. The two MUST agree, and
// TestGoValidationAgreesWithSchema holds them to it over the example corpus and the
// negative cases: a schema that rejects what the Go types accept (or the reverse) would
// make the published contract a lie.
func (d Document) Validate() error {
	var problems []string
	if !d.APIVersion.Valid() {
		problems = append(problems, fmt.Sprintf("apiVersion is %q, want %q", d.APIVersion, Version))
	}
	if !IsValidKind(d.Kind) {
		problems = append(problems, fmt.Sprintf("kind is %q, want one of %s",
			d.Kind, strings.Join(kindStrings(AllKinds()), ", ")))
	}
	if IsInput(d.Kind) {
		if d.Spec == nil {
			problems = append(problems, fmt.Sprintf("kind %q is an input kind and must carry a spec", d.Kind))
		}
		if d.Result != nil {
			problems = append(problems, fmt.Sprintf("kind %q is an input kind and must not carry a result", d.Kind))
		}
	}
	if IsOutput(d.Kind) {
		if d.Result == nil {
			problems = append(problems, fmt.Sprintf("kind %q is an output kind and must carry a result", d.Kind))
		}
		if d.Spec != nil {
			problems = append(problems, fmt.Sprintf("kind %q is an output kind and must not carry a spec", d.Kind))
		}
	}
	if len(problems) == 0 {
		return nil
	}
	return fmt.Errorf("invalid %s document: %s", Version, strings.Join(problems, "; "))
}

// kindStrings converts kinds to plain strings for schema emission and diagnostics,
// preserving the input order.
func kindStrings(kinds []Kind) []string {
	out := make([]string, len(kinds))
	for i, k := range kinds {
		out[i] = string(k)
	}
	return out
}
