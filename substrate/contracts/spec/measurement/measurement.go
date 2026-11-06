// Package measurement describes a measured serving run as a validation case: what was
// measured, where every input came from, how each metric was defined, and how far the
// configuration BLIS would simulate is from the one that ran.
//
// It replaces evaluation.Run for scoring. A run without provenance, licence terms or
// metric definitions can be compared with a prediction, but the comparison cannot say
// whether a disagreement is the model's error or a difference in what was compared.
//
// The verdict is never read from the document. It is computed from the diff, the
// provenance rows and the licence registry (Assess), and a document that states one must
// state the computed one.
package measurement

import (
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/router"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// SchemaVersion is the MeasuredRun schema this package reads and writes.
const SchemaVersion = 1

// MeasuredRun is one measured sweep: one configuration at several loads.
type MeasuredRun struct {
	Kind          string `yaml:"kind"` // "MeasuredRun"
	Name          string `yaml:"name"`
	SchemaVersion int    `yaml:"schema_version"`
	Source        Source `yaml:"source"`

	// Model, EngineVersion and Cluster are the immutable half of the scenario BLIS
	// builds for this run, and Deployment the mutable half. Coefficients are not here:
	// they are BLIS's choice, not a fact of the measurement.
	Model         string                 `yaml:"model,omitempty"`
	EngineVersion string                 `yaml:"engine_version,omitempty"`
	Cluster       *scenario.Cluster      `yaml:"cluster,omitempty"`
	Deployment    *deployment.Deployment `yaml:"deployment,omitempty"`
	Router        *router.Router         `yaml:"router,omitempty"`
	Workload      *workload.Binding      `yaml:"workload,omitempty"`
	Client        ClientSpec             `yaml:"client"`

	MetricDefs map[string]MetricDef `yaml:"metric_defs"`
	// Points are ordered by increasing load; the first is the anchor that shape error is
	// measured from.
	Points     []Point         `yaml:"points"`
	Provenance []ProvenanceRow `yaml:"provenance,omitempty"`
	Diff       []DiffEntry     `yaml:"diff,omitempty"`

	// Verdict and Resolution are optional and checked: when present they must equal what
	// Assess computes.
	Verdict    Verdict    `yaml:"verdict,omitempty"`
	Resolution Resolution `yaml:"resolution,omitempty"`

	DuplicateOf string `yaml:"duplicate_of,omitempty"`
	Supersedes  string `yaml:"supersedes,omitempty"`
}

// Source records where a run's numbers came from and under what terms.
type Source struct {
	Kind       SourceKind `yaml:"kind"`
	Adapter    Adapter    `yaml:"adapter"`
	Inputs     []Input    `yaml:"inputs"`
	Extraction Extraction `yaml:"extraction"`
	// RetrievedUTC is when the inputs were fetched, from the fetch manifest rather than
	// the wall clock of the extraction, so a re-extraction is byte-identical.
	RetrievedUTC string `yaml:"retrieved_utc,omitempty"`
	// LicenseRef names an entry in the catalog's measurements/sources.yaml.
	LicenseRef string `yaml:"license_ref"`
}

// Adapter is the code that extracted the run, pinned by the digest of its source.
type Adapter struct {
	Name    string `yaml:"name"`
	Version string `yaml:"version"`
	SHA256  string `yaml:"sha256"`
}

// Input is one file the adapter read, or the upstream file it was cut from.
type Input struct {
	Role       InputRole `yaml:"role"`
	URL        string    `yaml:"url,omitempty"`
	Repo       string    `yaml:"repo,omitempty"`
	Commit     string    `yaml:"commit,omitempty"`
	Path       string    `yaml:"path,omitempty"`
	GitBlob    string    `yaml:"git_blob,omitempty"`
	Size       int64     `yaml:"size,omitempty"`
	SHA256     string    `yaml:"sha256"`
	FetchedUTC string    `yaml:"fetched_utc,omitempty"`
	Note       string    `yaml:"note,omitempty"`
}

// Extraction records how values were read out of the inputs.
type Extraction struct {
	Method ExtractionMethod `yaml:"method"`
	// Selectors name what was read: literal names, column names, row ids.
	Selectors []string `yaml:"selectors,omitempty"`
	// LabelVerified reports whether every metric label was checked against the
	// harness's own definition rather than taken from its name.
	LabelVerified bool `yaml:"label_verified,omitempty"`
}

// ClientSpec is the load generator, as recorded. Load that varies by point is on the
// point.
type ClientSpec struct {
	Harness        string   `yaml:"harness,omitempty"`
	HarnessVersion string   `yaml:"harness_version,omitempty"`
	ImageDigest    string   `yaml:"image_digest,omitempty"`
	Command        []string `yaml:"command,omitempty"`
	Seed           *int64   `yaml:"seed,omitempty"`
	NumPrompts     int      `yaml:"num_prompts,omitempty"`
	// Burstiness is the gamma shape of the arrival process; 1 is Poisson.
	Burstiness  *float64   `yaml:"burstiness,omitempty"`
	RampUp      *RampUp    `yaml:"ramp_up,omitempty"`
	NumWarmups  int        `yaml:"num_warmups,omitempty"`
	IgnoreEOS   *bool      `yaml:"ignore_eos,omitempty"`
	Endpoint    string     `yaml:"endpoint,omitempty"`
	Tokenizer   *Tokenizer `yaml:"tokenizer,omitempty"`
	DurationSec float64    `yaml:"duration_s,omitempty"`
	GraceSec    float64    `yaml:"grace_s,omitempty"`
}

// RampUp is a request-rate ramp from Start to End requests per second.
type RampUp struct {
	Strategy RampStrategy `yaml:"strategy"`
	Start    float64      `yaml:"start"`
	End      float64      `yaml:"end"`
}

// Tokenizer pins the tokenizer the client counted tokens with.
type Tokenizer struct {
	Repo     string `yaml:"repo"`
	Revision string `yaml:"revision,omitempty"`
	SHA256   string `yaml:"sha256,omitempty"`
	Mode     string `yaml:"mode,omitempty"`
}

// MetricDef defines one reported metric precisely enough that a scorer can refuse to
// compare it with a BLIS metric defined differently. Unit and Cite are required; every
// other field left empty is unstated, and an unstated definition keeps a run from EXACT.
type MetricDef struct {
	Formula          Formula          `yaml:"formula,omitempty"`
	Statistic        Statistic        `yaml:"statistic,omitempty"`
	Population       Population       `yaml:"population,omitempty"`
	Denominator      Denominator      `yaml:"denominator,omitempty"`
	Clock            Clock            `yaml:"clock,omitempty"`
	StartEvent       StartEvent       `yaml:"start_event,omitempty"`
	StopEvent        StopEvent        `yaml:"stop_event,omitempty"`
	OutputLenSource  OutputLenSource  `yaml:"output_len_source,omitempty"`
	ReasoningContent *bool            `yaml:"reasoning_counted_as_content,omitempty"`
	PercentileMethod PercentileMethod `yaml:"percentile_method,omitempty"`
	Unit             MetricUnit       `yaml:"unit"`
	// Cite is the harness file and line, or the report text, the definition is read from.
	Cite string `yaml:"cite"`
}

// Point is one load level.
type Point struct {
	ID            string           `yaml:"id"`
	Load          Load             `yaml:"load"`
	Metrics       map[string]Value `yaml:"metrics"`
	Completed     *int             `yaml:"completed,omitempty"`
	Failed        *int             `yaml:"failed,omitempty"`
	Realized      *Realized        `yaml:"realized,omitempty"`
	KVUtilization *float64         `yaml:"kv_utilization,omitempty"`
	PrefixHitRate *float64         `yaml:"prefix_hit_rate,omitempty"`
	Preemptions   *int             `yaml:"preemptions,omitempty"`
	Noise         *Noise           `yaml:"noise,omitempty"`
}

// Load is the offered load: a concurrency cap, a request rate, or both. An unbounded
// rate under a cap is the cap alone.
type Load struct {
	Concurrency int     `yaml:"concurrency,omitempty"`
	RequestRate float64 `yaml:"request_rate,omitempty"`
}

// Value is one measured number and where it was read.
type Value struct {
	Value         float64 `yaml:"value"`
	ReportLabel   string  `yaml:"report_label,omitempty"`
	RawKey        string  `yaml:"raw_key,omitempty"`
	LabelVerified bool    `yaml:"label_verified,omitempty"`
}

// Realized is the sequence lengths actually served.
type Realized struct {
	MeanISL float64 `yaml:"mean_isl,omitempty"`
	MeanOSL float64 `yaml:"mean_osl,omitempty"`
}

// Noise is the run-to-run spread, when repeated seeds measured it.
type Noise struct {
	Seeds int     `yaml:"seeds"`
	CV    float64 `yaml:"cv"`
}

// ProvenanceRow compares one setting across the report, the script that produced it,
// the run's own artifact, and the value BLIS would use.
type ProvenanceRow struct {
	Setting  string     `yaml:"setting"`
	Report   *Observed  `yaml:"report,omitempty"`
	Script   *Observed  `yaml:"script,omitempty"`
	Artifact *Observed  `yaml:"artifact,omitempty"`
	BLIS     *Observed  `yaml:"blis,omitempty"`
	Verdict  RowVerdict `yaml:"verdict"`
	// ResolvedBy cites the engine rule, at the pinned version, that fixes an UNSTATED
	// setting's value.
	ResolvedBy string      `yaml:"resolved_by,omitempty"`
	Material   Materiality `yaml:"material"`
	Bound      string      `yaml:"bound,omitempty"`
	Note       string      `yaml:"note,omitempty"`
}

// Observed is one source's value for a setting.
type Observed struct {
	Value  string `yaml:"value"`
	Ref    string `yaml:"ref,omitempty"`
	Commit string `yaml:"commit,omitempty"`
}

// DiffEntry is one difference between the measured configuration and what BLIS
// simulates.
type DiffEntry struct {
	Path     string   `yaml:"path"`
	Measured string   `yaml:"measured,omitempty"`
	BLIS     string   `yaml:"blis,omitempty"`
	Kind     DiffKind `yaml:"kind"`
	// Removable reports whether a change in OwnerRepo would close the difference.
	Removable bool        `yaml:"removable,omitempty"`
	OwnerRepo string      `yaml:"owner_repo,omitempty"`
	Material  Materiality `yaml:"material"`
	Bound     string      `yaml:"bound,omitempty"`
	Issue     string      `yaml:"issue,omitempty"`
}

// Case is one point of an assessed run: the unit a scorer pairs with a prediction.
type Case struct {
	Run        string
	Point      string
	Load       Load
	Anchor     bool
	Verdict    Verdict
	Resolution Resolution
}
