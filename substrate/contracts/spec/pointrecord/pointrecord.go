// Package pointrecord defines the per-point record the scoring harness writes: one
// document per measured point it scores or declines to score (design §2.2, §11, SUB-19).
// The dashboard writes it; config search and the substrate's gates read it, so it is
// defined here.
//
// A record identifies the point and its ledger unit, grades it G1 to G5 (design §11.3),
// names every substitution the grade rests on, gives the reason a G5 point was not scored
// from the closed list of design §11.4, records the simulator build, the data and the
// harness digest that produced it (X8), and pairs each predicted figure with its measured
// counterpart under one metric identifier (DEV-8).
//
// The fields are the accuracy-dashboard's per-point output as of f87f98d
// (orchestrator/ingest/fidelity.schema.json, and the facets and points files that
// orchestrator/ingest/CONTRACT.md describes), regraded to the design's grades.
package pointrecord

import (
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// Kind is the value of Record.Kind.
const Kind = "PointRecord"

// SchemaVersion is the record schema this package reads and writes.
const SchemaVersion = 1

// Record is one scored or declined point.
type Record struct {
	Kind          string `yaml:"kind"`
	SchemaVersion int    `yaml:"schema_version"`

	// PointID is the SHA-256, as 64 lowercase hex characters, of the canonical JSON of
	// Identity: the point's identity in its source's own terms, so it never changes with a
	// BLIS release.
	PointID string `yaml:"point_id"`
	// UnitID is the ledger unit the point belongs to (design §10.2).
	UnitID string `yaml:"unit_id"`
	// Identity is the source's identifying fields.
	Identity Identity `yaml:"identity"`

	// Grade is the point's one grade, the highest-numbered whose conditions apply.
	Grade Grade `yaml:"grade"`
	// G5Reason is set exactly when Grade is G5.
	G5Reason *G5Reason `yaml:"g5_reason,omitempty"`
	// Substitutions are every departure from an exact comparison the grade rests on,
	// sorted by kind and subject. A G1 point has none.
	Substitutions []Substitution `yaml:"substitutions,omitempty"`

	// Build is the engine build that measured the point.
	Build Build `yaml:"build"`
	// Simulator is the simulator build that scored it.
	Simulator Simulator `yaml:"simulator"`
	// HarnessDigest is the digest of the scoring harness image, sha256:<64 hex>.
	HarnessDigest string `yaml:"harness_digest"`

	// Figures pair each predicted figure with the measured one, sorted by metric. Empty
	// for a G5 point.
	Figures []Figure `yaml:"figures,omitempty"`

	// Settings compare each classified engine setting the run recorded with what BLIS was
	// given, sorted by arg.
	Settings []Setting `yaml:"settings,omitempty"`
	// Resolved compare engine outcomes both sides report, such as the KV-cache size,
	// sorted by quantity. They are never passed to BLIS.
	Resolved []Resolved `yaml:"resolved,omitempty"`
	// Replay is the load generator's inputs, for an agentic point.
	Replay *Replay `yaml:"replay,omitempty"`
	// Facets are the point's labels for display and grouping.
	Facets Facets `yaml:"facets"`
	// Notes say why a reason or substitution applies, at most one per subject.
	Notes []Note `yaml:"notes,omitempty"`
	// KernelProvenance is the cost model's report of the entries it priced, verbatim.
	KernelProvenance map[string]any `yaml:"kernel_provenance,omitempty"`
}

// Identity is a point's identifying fields in its source's own terms.
type Identity struct {
	// Source names the data source, as the ledger names it, for example "inferencex".
	Source        string `yaml:"source"`
	Model         string `yaml:"model"`
	Hardware      string `yaml:"hardware"`
	Precision     string `yaml:"precision"`
	Framework     string `yaml:"framework"`
	BenchmarkType string `yaml:"benchmark_type"`
	// Recipe is the source's recipe fingerprint, or row:<id> when it has none.
	Recipe      string `yaml:"recipe"`
	Concurrency int    `yaml:"conc"`
	// MeasuredDate is the measurement's date, YYYY-MM-DD.
	MeasuredDate string `yaml:"measured_date"`
}

// Grade is a closeness grade (design §11.3).
type Grade string

const (
	// G1: the run's engine release is the one the scoring simulator runs, every setting
	// is modelled or proven inert within covered value classes, and every coefficient is
	// measured or calibrated.
	G1 Grade = "G1"
	// G2: as G1, with substitutions, each named.
	G2 Grade = "G2"
	// G3: another engine release, with none of the ported upstream functions changed.
	G3 Grade = "G3"
	// G4: another engine release with changed ported functions, or a nightly or custom
	// build.
	G4 Grade = "G4"
	// G5: not scored, for a reason of design §11.4.
	G5 Grade = "G5"
)

// G5Reason is why a point is not scored.
type G5Reason struct {
	Code G5Code `yaml:"code"`
	// Subject names what the reason is about: a field, a model, a chip or a build string.
	Subject string `yaml:"subject,omitempty"`
}

// G5Code is one of the closed list of design §11.4 (COV-5).
type G5Code string

const (
	CodeNotVisible    G5Code = "code_not_visible"
	RouterUnsupported G5Code = "router_unsupported"
	ValueNotRecorded  G5Code = "value_not_recorded"
	FieldNotModelled  G5Code = "field_not_modelled"
	FieldUnclassified G5Code = "field_unclassified"
	EngineUnsupported G5Code = "engine_unsupported"
	NotInCatalog      G5Code = "not_in_catalog"
	FieldNotInRelease G5Code = "field_not_in_release"
	BuildUnparseable  G5Code = "build_unparseable"
)

// Substitution is one departure from an exact comparison.
type Substitution struct {
	Kind SubstitutionKind `yaml:"kind"`
	// Subject names what was substituted: a field, a coefficient, a library or a build.
	Subject string `yaml:"subject"`
	// Detail says what stood in, for example the estimation rule's setting, the
	// coefficient's method, or the changed functions.
	Detail string `yaml:"detail,omitempty"`
}

// SubstitutionKind names a condition of design §11.3 that lowers a grade.
type SubstitutionKind string

const (
	// Grade G2 conditions.
	SubCoefficientMethod SubstitutionKind = "coefficient_method" // transferred, literature, vendor_spec, roofline or assumed
	SubCoefficientRange  SubstitutionKind = "coefficient_range"  // extrapolated or unidentified
	SubExecutionMode     SubstitutionKind = "execution_mode"     // an execution mode substituted
	SubKernelHash        SubstitutionKind = "kernel_hash"        // a recorded library's kernel hash differs from the one priced
	SubValueClass        SubstitutionKind = "value_class"        // a modelled field set outside every covered value class
	SubEstimated         SubstitutionKind = "estimated"          // a missing or unmodelled value replaced by the engine's estimation rule
	SubInterimStatus     SubstitutionKind = "interim_status"     // a field whose status a release plan allows only for a time
	// Grade G3 and G4 conditions.
	SubOtherRelease     SubstitutionKind = "other_release"     // the run's engine release is another one
	SubChangedFunctions SubstitutionKind = "changed_functions" // ported upstream functions changed between the releases
	SubNightlyBuild     SubstitutionKind = "nightly_or_custom" // the build is nightly or custom
)

// Build is the engine build that measured a point.
type Build struct {
	// Release is the engine release the run used, as the scoring simulator's engine names
	// releases, when it can be read.
	Release string `yaml:"release,omitempty"`
	Image   string `yaml:"image"`
	// Commit is the upstream commit, 40 lowercase hex characters, when it resolves.
	Commit string `yaml:"commit,omitempty"`
	// Source says where Commit was read: image_label, label_by_tag, server_log_banner,
	// image_tag or release_tag.
	Source string `yaml:"source,omitempty"`
	// Version is the version string the server log's banner printed.
	Version string `yaml:"version,omitempty"`
}

// Simulator is the simulator build that scored a point, and the data it read.
type Simulator struct {
	// Build is a release tag, or a commit of a branch or pull request.
	Build string `yaml:"build"`
	// Engine is the engine release the build runs.
	Engine result.Engine `yaml:"engine"`
	// Data is the catalog and registry the run read, as its Result records them.
	Data result.Data `yaml:"data"`
}

// Figure pairs a predicted and a measured figure of one metric.
type Figure struct {
	// Metric is the metric-registry identifier both sides are computed under.
	Metric    string  `yaml:"metric"`
	Predicted float64 `yaml:"predicted"`
	Measured  float64 `yaml:"measured"`
	// PredictedSamples and MeasuredSamples are the number of samples behind each.
	PredictedSamples int `yaml:"predicted_samples"`
	MeasuredSamples  int `yaml:"measured_samples"`
	// Interval is the predicted figure's replication interval, when replications ran.
	Interval *Interval `yaml:"interval,omitempty"`
}

// Interval is a confidence interval over replications (DEV-11).
type Interval struct {
	Low          float64 `yaml:"low"`
	High         float64 `yaml:"high"`
	Level        float64 `yaml:"level"`
	Replications int     `yaml:"replications"`
}

// Setting compares one engine setting the run recorded with what BLIS was given.
type Setting struct {
	Arg string `yaml:"arg"`
	// Measured is the value as the run recorded it: a string (a float as printed), an
	// integer, a boolean, or null.
	Measured any `yaml:"measured"`
	// Source says where the measured value was read.
	Source string `yaml:"source,omitempty"`
	// BLIS is the value parsed back from the scenario handed to BLIS.
	BLIS any `yaml:"blis"`
	// Field is the scenario field the value was written to, when the names differ.
	Field string `yaml:"field,omitempty"`
	Match Match  `yaml:"match"`
	// Rule names the estimation rule applied, for a value BLIS estimated.
	Rule        string `yaml:"rule,omitempty"`
	Occurrences int    `yaml:"occurrences,omitempty"`
	// Values lists every value found, for a conflict.
	Values []any `yaml:"values,omitempty"`
	Delta  any   `yaml:"delta,omitempty"`
	// Ratio is BLIS over measured, four significant figures, as a string.
	Ratio string `yaml:"ratio,omitempty"`
	Note  string `yaml:"note,omitempty"`
}

// Match is how a setting compares.
type Match string

const (
	MatchExact        Match = "exact"
	MatchMapped       Match = "mapped"
	MatchInert        Match = "inert"
	MatchEstimated    Match = "estimated"
	MatchDiffers      Match = "differs"
	MatchAbsentInBLIS Match = "absent_in_blis"
	MatchUnreadable   Match = "unreadable"
	MatchConflict     Match = "conflict"
)

// Resolved compares an engine outcome both sides report.
type Resolved struct {
	Quantity    string `yaml:"quantity"`
	Measured    any    `yaml:"measured"`
	BLIS        any    `yaml:"blis"`
	Occurrences int    `yaml:"occurrences,omitempty"`
	Delta       any    `yaml:"delta,omitempty"`
	Ratio       string `yaml:"ratio,omitempty"`
	Match       Match  `yaml:"match"`
}

// Replay is an agentic point's load-generator inputs.
type Replay struct {
	Client        string  `yaml:"client"`
	ClientVersion string  `yaml:"client_version"`
	Scenario      string  `yaml:"scenario"`
	ReplayedAs    string  `yaml:"replayed_as,omitempty"`
	Concurrency   int     `yaml:"concurrency"`
	DurationS     float64 `yaml:"duration_s,omitempty"`
	// TrajectoryStart bounds where each lane's snapshot is drawn, as fractions.
	TrajectoryStart *TrajectoryStart `yaml:"trajectory_start,omitempty"`
	Corpus          string           `yaml:"corpus"`
	CorpusSHA256    string           `yaml:"corpus_sha256"`
	Seed            int64            `yaml:"seed"`
}

// TrajectoryStart is a fractional window.
type TrajectoryStart struct {
	MinRatio *float64 `yaml:"min_ratio,omitempty"`
	MaxRatio *float64 `yaml:"max_ratio,omitempty"`
}

// Facets are a point's labels.
type Facets struct {
	// SweepID groups the points of one measured curve: one deployment, at most one point
	// per concurrency.
	SweepID string `yaml:"sweep_id"`
	// Scenario is the deployment's name, shared by every point of a sweep.
	Scenario string `yaml:"scenario"`
	// Label is the workload label, such as 8k1k or agentx.
	Label    string `yaml:"label"`
	Workload string `yaml:"workload"`
	// Model is the catalog model BLIS priced.
	Model       string      `yaml:"model"`
	GPU         string      `yaml:"gpu"`
	Precision   string      `yaml:"precision"`
	Framework   string      `yaml:"framework"`
	Serving     string      `yaml:"serving"`
	SpecMethod  string      `yaml:"spec_method"`
	Parallelism Parallelism `yaml:"parallelism"`
}

// Parallelism is a deployment's layout.
type Parallelism struct {
	TP int `yaml:"tp"`
	PP int `yaml:"pp"`
	DP int `yaml:"dp"`
	EP int `yaml:"ep"`
}

// Note says why a reason or substitution applies. Its text is fixed per subject, never
// composed per point.
type Note struct {
	Subject string `yaml:"subject"`
	Text    string `yaml:"text"`
}
