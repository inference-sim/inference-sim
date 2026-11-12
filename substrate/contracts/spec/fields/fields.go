// Package fields describes an engine repository's field list: fields.json, which classifies
// every configuration field and every environment variable one release of an engine reads
// (design VLM-3), and gives each the estimation rule a scoring harness applies when a
// recorded run did not state a value the simulation needs (design §11.2).
//
// The engine repository writes the list and publishes it at its root. The accuracy dashboard
// and the configuration search read it through the simulator's `fields` command, and hold no
// table of their own (COV-3, CFG-3). This package is the list's one definition: its Go type,
// its validator, and its strict loader.
//
// A list is JSON. Parse refuses an unknown key, a duplicate key, trailing data and a number
// that does not convert exactly (SUB-12), then validates. Marshal writes the canonical bytes,
// so a repository checks its committed file by comparing it with Marshal of what Parse read.
//
// The yaml tags repeat the json tags. The list is read only as JSON; the yaml tags let the
// repository's documentation test, which reads yaml tags, check the reference page's key
// tables against these types as it checks every other document's.
package fields

// Kind is the value of FieldList.Kind.
const Kind = "FieldList"

// FieldList is one engine release's field list.
type FieldList struct {
	Kind string `json:"kind" yaml:"kind"`
	// Engine names the engine and the release whose fields the list classifies.
	Engine Engine `json:"engine" yaml:"engine"`
	// Fields holds one entry per configuration field and environment variable, sorted by
	// name and then by source, in byte order.
	Fields []Field `json:"fields" yaml:"fields"`
}

// Engine names an engine kind and one of its releases, as a scenario's engine block does.
type Engine struct {
	// Kind is the engine kind the repository registers, for example "vllm".
	Kind string `json:"kind" yaml:"kind"`
	// Version is the upstream release, for example "0.32.0".
	Version string `json:"version" yaml:"version"`
}

// Field is one configuration field or environment variable of the release.
type Field struct {
	// Name is the upstream's own name: the configuration field as the upstream spells it, or
	// the environment variable.
	Name string `json:"name" yaml:"name"`
	// Source says which kind of name Name is.
	Source Source `json:"source" yaml:"source"`
	// Status is how the engine port treats the field.
	Status Status `json:"status" yaml:"status"`
	// Evidence supports Status. Exactly one of its keys is set, the one Status requires.
	Evidence Evidence `json:"evidence" yaml:"evidence"`
	// ValueClasses are the classes of values the evidence covers (design VLM-3). Required for
	// a modelled or proven-inert field and not allowed for any other: a field with no evidence
	// covers no class.
	ValueClasses *ValueClasses `json:"value_classes,omitempty" yaml:"value_classes,omitempty"`
	// Estimation is the rule a scoring harness applies when a run did not record the field,
	// or set it to a value the port does not model (design §11.2).
	Estimation Estimation `json:"estimation" yaml:"estimation"`
}

// Source is the kind of name a field has.
type Source string

const (
	// SourceConfig is a configuration field, named as the upstream names it.
	SourceConfig Source = "config"
	// SourceEnv is an environment variable.
	SourceEnv Source = "env"
)

// Status is how the engine port treats a field (design VLM-3).
type Status string

const (
	// StatusModelled: the port reads the field and it changes a decision. Evidence.Traces
	// names the recorded traces that set it.
	StatusModelled Status = "modelled"
	// StatusProvenInert: the field changes no decision, shown by Evidence.Proof, a test or a
	// reviewed argument (design §7.4, tier T1).
	StatusProvenInert Status = "proven_inert"
	// StatusNotModelled: the port refuses the field when it is set away from its default.
	// Evidence.Issue names the issue that tracks it.
	StatusNotModelled Status = "not_modelled"
	// StatusInertUnproven: believed inert, without a proof yet. Evidence.Note says why it is
	// believed inert. Allowed only until M2 (release plan §3.3, "The field list's starting
	// point"): by then every such field has a proof or another status.
	StatusInertUnproven Status = "inert_unproven"
)

// Evidence supports a field's status. Exactly one key is set: Traces for modelled, Proof for
// proven_inert, Issue for not_modelled, Note for inert_unproven.
type Evidence struct {
	// Traces are the paths, relative to the engine repository's root, of recorded traces
	// whose scenarios set the field.
	Traces []string `json:"traces,omitempty" yaml:"traces,omitempty"`
	// Proof names the test or states the argument that shows the field inert.
	Proof string `json:"proof,omitempty" yaml:"proof,omitempty"`
	// Issue is the issue that tracks modelling the field, as "<owner>/<repo>#<n>".
	Issue string `json:"issue,omitempty" yaml:"issue,omitempty"`
	// Note says why the field is believed inert.
	Note string `json:"note,omitempty" yaml:"note,omitempty"`
}

// Rule is an estimation rule (design §11.2).
type Rule string

const (
	// RuleLeaveUnset: leave the field unset, so the engine resolves the upstream's default
	// (design X19).
	RuleLeaveUnset Rule = "leave_unset"
	// RuleSubstitute: a stated setting stands in for the unmodelled one. Estimation.Setting
	// holds it and Estimation.Reason says why it is a fair stand-in.
	RuleSubstitute Rule = "substitute"
	// RuleNone: no estimate exists. A point that needs one is not scored.
	RuleNone Rule = "none"
)

// Estimation is a field's estimation rule.
type Estimation struct {
	Rule Rule `json:"rule" yaml:"rule"`
	// Setting is the stand-in, as configuration field names and values. Only with
	// RuleSubstitute.
	Setting map[string]any `json:"setting,omitempty" yaml:"setting,omitempty"`
	// Reason says why Setting stands in for the field. Only with RuleSubstitute.
	Reason string `json:"reason,omitempty" yaml:"reason,omitempty"`
	// Preserves is a sentence naming what the upstream resolves identically when the field is
	// left unset or stated as Setting. Only with RuleLeaveUnset or RuleSubstitute, and required
	// with ValidWhen.
	Preserves string `json:"preserves,omitempty" yaml:"preserves,omitempty"`
	// ValidWhen says when the stand-in is exact: the stand-in is exact for a recorded value
	// when at least one clause holds. A value no clause covers is not stood in for. Absent,
	// the rule applies to every value. Only with RuleLeaveUnset or RuleSubstitute. A
	// consumer decides whether a clause holds from the facts it knows about the recorded value.
	ValidWhen []Clause `json:"valid_when,omitempty" yaml:"valid_when,omitempty"`
}
