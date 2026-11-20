package measurement

import (
	"regexp"
	"sort"
)

// Verdict is how faithfully BLIS can reproduce a run. It is computed, never asserted.
type Verdict string

const (
	// VerdictExact: no difference, no disputed setting, and terms that permit scoring.
	// Only an EXACT run may anchor calibration.
	VerdictExact Verdict = "EXACT"
	// VerdictPartial: simulable, with recorded differences. Scored, reported apart.
	VerdictPartial Verdict = "PARTIAL"
	// VerdictNotReproducible: BLIS cannot build the run's configuration at all.
	VerdictNotReproducible Verdict = "NOT_REPRODUCIBLE"
)

// Resolution qualifies an EXACT verdict.
type Resolution string

const (
	// ResolutionStrict: every setting is stated by the run itself.
	ResolutionStrict Resolution = "STRICT"
	// ResolutionResolved: some settings are unstated and fixed by a cited engine rule.
	ResolutionResolved Resolution = "RESOLVED"
)

// SourceKind names the publisher of a run, which selects its adapter.
type SourceKind string

const (
	SourceInferenceX     SourceKind = "inferencex"
	SourceInternalLLMD   SourceKind = "internal_llmd"
	SourceLLMDBenchmark  SourceKind = "llmd_benchmark"
	SourcePrism          SourceKind = "prism"
	SourceVLLMBench      SourceKind = "vllm_bench"
	SourcePyTorchHUD     SourceKind = "pytorch_hud"
	SourceMLPerf         SourceKind = "mlperf"
	SourceAIConfigurator SourceKind = "aiconfigurator"
)

// InputRole says what an input is to the run.
type InputRole string

const (
	RoleReport      InputRole = "report"
	RoleManifest    InputRole = "manifest"
	RoleEnv         InputRole = "env"
	RoleRouter      InputRole = "router"
	RoleRawArtifact InputRole = "raw_artifact"
	RoleServerLog   InputRole = "server_log"
	RoleTokenizer   InputRole = "tokenizer"
	RoleDataset     InputRole = "dataset"
)

// ExtractionMethod is how values were read from an input. Every method parses data; none
// executes it.
type ExtractionMethod string

const (
	ExtractHTMLJSLiteral ExtractionMethod = "html_js_literal"
	ExtractHTMLTable     ExtractionMethod = "html_table"
	ExtractYAML          ExtractionMethod = "yaml"
	ExtractJSON          ExtractionMethod = "json"
	ExtractCSV           ExtractionMethod = "csv"
	ExtractLogText       ExtractionMethod = "log_text"
	// ExtractMaterialized: the request stream was regenerated with the harness's own
	// sampler, in its own image, offline.
	ExtractMaterialized ExtractionMethod = "materialized"
)

// RampStrategy is how a request-rate ramp interpolates.
type RampStrategy string

const (
	RampLinear      RampStrategy = "linear"
	RampExponential RampStrategy = "exponential"
)

// Formula is the quantity a metric measures.
type Formula string

const (
	FormulaTTFT              Formula = "ttft"
	FormulaTPOTExclFirst     Formula = "tpot_excl_first"
	FormulaITL               Formula = "itl"
	FormulaE2EL              Formula = "e2el"
	FormulaOutputThroughput  Formula = "output_throughput"
	FormulaTotalThroughput   Formula = "total_throughput"
	FormulaRequestThroughput Formula = "request_throughput"
	// FormulaInteractivity is 1000 over an ITL statistic, tokens per second per user.
	FormulaInteractivity Formula = "interactivity"
	FormulaUserTPS       Formula = "user_tps"
)

// Statistic summarizes a population. A percentile is written pNN or pNN.N, and is not a
// constant here: Valid accepts any in (0, 100).
type Statistic string

const (
	StatMean   Statistic = "mean"
	StatMedian Statistic = "median"
	StatStd    Statistic = "std"
	StatMax    Statistic = "max"
)

// Population is what a statistic is taken over.
type Population string

const (
	PopulationPerRequest   Population = "per_request"
	PopulationPooledChunks Population = "pooled_chunks"
	PopulationPooledTokens Population = "pooled_tokens"
)

// Denominator is what a throughput divides by.
type Denominator string

const (
	// DenominatorMainPhaseWall: first send after warmup to last completion.
	DenominatorMainPhaseWall   Denominator = "main_phase_wall"
	DenominatorProfilingWindow Denominator = "profiling_window"
	DenominatorPerGPU          Denominator = "per_gpu"
	DenominatorPerReplica      Denominator = "per_replica"
)

// Clock is whose timestamps a latency uses.
type Clock string

const (
	ClockClient Clock = "client"
	ClockServer Clock = "server"
	// ClockSimulated is a simulator's single event timeline: send, first token and
	// completion are events on one clock, with no client or server offset. A Result
	// uses it; a measured run cannot.
	ClockSimulated Clock = "simulated"
)

// StartEvent is when a latency's clock starts.
type StartEvent string

const (
	// StartSendAfterSemaphore: when the request leaves the client's concurrency gate.
	StartSendAfterSemaphore StartEvent = "send_after_semaphore"
	StartArrival            StartEvent = "arrival"
)

// StopEvent is when a latency's clock stops.
type StopEvent string

const (
	StopFirstContentChunk StopEvent = "first_content_chunk"
	StopLastChunk         StopEvent = "last_chunk"
)

// OutputLenSource is where an output token count comes from.
type OutputLenSource string

const (
	OutputLenUsage     OutputLenSource = "usage"
	OutputLenTokenizer OutputLenSource = "tokenizer"
	OutputLenTarget    OutputLenSource = "target"
)

// PercentileMethod is how a percentile interpolates between samples.
type PercentileMethod string

const PercentileNumpyLinear PercentileMethod = "numpy_linear"

// MetricUnit is a metric value's unit.
type MetricUnit string

const (
	UnitMs           MetricUnit = "ms"
	UnitTokensPerSec MetricUnit = "tokens_per_s"
	UnitRequestsSec  MetricUnit = "requests_per_s"
)

// RowVerdict is a provenance row's outcome.
type RowVerdict string

const (
	// RowMatch: two or more sources agree.
	RowMatch RowVerdict = "MATCH"
	// RowMismatch: two sources disagree.
	RowMismatch RowVerdict = "MISMATCH"
	// RowUnstated: at most one source states the setting.
	RowUnstated RowVerdict = "UNSTATED"
	// RowUnverifiable: the report and the script disagree and no artifact decides.
	RowUnverifiable RowVerdict = "UNVERIFIABLE"
)

// Materiality is whether a difference can move a metric.
type Materiality string

const (
	MaterialYes     Materiality = "yes"
	MaterialNo      Materiality = "no"
	MaterialUnknown Materiality = "unknown"
	// MaterialBounded: it can, by at most Bound.
	MaterialBounded Materiality = "bounded"
)

// DiffKind is why a measured setting differs from the simulated one.
type DiffKind string

const (
	// DiffNotExpressible: no BLIS document can state the measured setting.
	DiffNotExpressible DiffKind = "not_expressible"
	// DiffNotModelled: BLIS can state it but does not price it.
	DiffNotModelled  DiffKind = "not_modelled"
	DiffConflict     DiffKind = "conflict"
	DiffUnresolved   DiffKind = "unresolved"
	DiffApproximated DiffKind = "approximated"
)

// LicenseStatus is whether a source's terms were found and read.
type LicenseStatus string

const (
	LicenseVerified   LicenseStatus = "verified"
	LicenseUnverified LicenseStatus = "unverified"
	LicenseRefused    LicenseStatus = "refused"
)

// Use is something a source's terms may permit.
type Use string

const (
	UseScore              Use = "score"
	UseDeriveCoefficients Use = "derive_coefficients"
	UseRedistribute       Use = "redistribute"
)

func set[T ~string](vs ...T) map[T]bool {
	m := make(map[T]bool, len(vs))
	for _, v := range vs {
		m[v] = true
	}
	return m
}

func members[T ~string](m map[T]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, string(k))
	}
	sort.Strings(out)
	return out
}

var (
	verdicts    = set(VerdictExact, VerdictPartial, VerdictNotReproducible)
	resolutions = set(ResolutionStrict, ResolutionResolved)
	sourceKinds = set(SourceInferenceX, SourceInternalLLMD, SourceLLMDBenchmark, SourcePrism,
		SourceVLLMBench, SourcePyTorchHUD, SourceMLPerf, SourceAIConfigurator)
	inputRoles = set(RoleReport, RoleManifest, RoleEnv, RoleRouter, RoleRawArtifact,
		RoleServerLog, RoleTokenizer, RoleDataset)
	extractions = set(ExtractHTMLJSLiteral, ExtractHTMLTable, ExtractYAML, ExtractJSON,
		ExtractCSV, ExtractLogText, ExtractMaterialized)
	ramps    = set(RampLinear, RampExponential)
	formulas = set(FormulaTTFT, FormulaTPOTExclFirst, FormulaITL, FormulaE2EL,
		FormulaOutputThroughput, FormulaTotalThroughput, FormulaRequestThroughput,
		FormulaInteractivity, FormulaUserTPS)
	statistics   = set(StatMean, StatMedian, StatStd, StatMax)
	populations  = set(PopulationPerRequest, PopulationPooledChunks, PopulationPooledTokens)
	denominators = set(DenominatorMainPhaseWall, DenominatorProfilingWindow,
		DenominatorPerGPU, DenominatorPerReplica)
	clocks      = set(ClockClient, ClockServer, ClockSimulated)
	starts      = set(StartSendAfterSemaphore, StartArrival)
	stops       = set(StopFirstContentChunk, StopLastChunk)
	outputLens  = set(OutputLenUsage, OutputLenTokenizer, OutputLenTarget)
	percentiles = set(PercentileNumpyLinear)
	units       = set(UnitMs, UnitTokensPerSec, UnitRequestsSec)
	rowVerdicts = set(RowMatch, RowMismatch, RowUnstated, RowUnverifiable)
	materials   = set(MaterialYes, MaterialNo, MaterialUnknown, MaterialBounded)
	diffKinds   = set(DiffNotExpressible, DiffNotModelled, DiffConflict, DiffUnresolved,
		DiffApproximated)
	licenseStatuses = set(LicenseVerified, LicenseUnverified, LicenseRefused)
	uses            = set(UseScore, UseDeriveCoefficients, UseRedistribute)

	percentile = regexp.MustCompile(`^p([1-9][0-9]?(\.[0-9]*[1-9])?|0\.[0-9]*[1-9])$`)
)

func (v Verdict) Valid() bool          { return verdicts[v] }
func (r Resolution) Valid() bool       { return resolutions[r] }
func (k SourceKind) Valid() bool       { return sourceKinds[k] }
func (r InputRole) Valid() bool        { return inputRoles[r] }
func (m ExtractionMethod) Valid() bool { return extractions[m] }
func (s RampStrategy) Valid() bool     { return ramps[s] }
func (f Formula) Valid() bool          { return formulas[f] }
func (p Population) Valid() bool       { return populations[p] }
func (d Denominator) Valid() bool      { return denominators[d] }
func (c Clock) Valid() bool            { return clocks[c] }
func (e StartEvent) Valid() bool       { return starts[e] }
func (e StopEvent) Valid() bool        { return stops[e] }
func (o OutputLenSource) Valid() bool  { return outputLens[o] }
func (m PercentileMethod) Valid() bool { return percentiles[m] }
func (u MetricUnit) Valid() bool       { return units[u] }
func (v RowVerdict) Valid() bool       { return rowVerdicts[v] }
func (m Materiality) Valid() bool      { return materials[m] }
func (k DiffKind) Valid() bool         { return diffKinds[k] }
func (s LicenseStatus) Valid() bool    { return licenseStatuses[s] }
func (u Use) Valid() bool              { return uses[u] }

// Valid reports whether s is a named statistic or a percentile in (0, 100).
func (s Statistic) Valid() bool { return statistics[s] || s.Percentile() }

// Percentile reports whether s is a percentile, which is the case that needs a
// percentile method.
func (s Statistic) Percentile() bool { return percentile.MatchString(string(s)) }

// Throughput reports whether f is a rate over a window rather than a statistic over a
// population, so it takes a denominator and no statistic.
func (f Formula) Throughput() bool {
	return f == FormulaOutputThroughput || f == FormulaTotalThroughput ||
		f == FormulaRequestThroughput
}

// Unit returns the unit f is measured in.
func (f Formula) Unit() MetricUnit {
	switch f {
	case FormulaTTFT, FormulaTPOTExclFirst, FormulaITL, FormulaE2EL:
		return UnitMs
	case FormulaRequestThroughput:
		return UnitRequestsSec
	default:
		return UnitTokensPerSec
	}
}

// AllVerdicts and the rest expose each vocabulary for error text and for tests that
// assert a vocabulary has not silently grown.
func AllVerdicts() []string       { return members(verdicts) }
func AllResolutions() []string    { return members(resolutions) }
func AllSourceKinds() []string    { return members(sourceKinds) }
func AllInputRoles() []string     { return members(inputRoles) }
func AllExtractions() []string    { return members(extractions) }
func AllRampStrategies() []string { return members(ramps) }
func AllFormulas() []string       { return members(formulas) }
func AllStatistics() []string     { return append(members(statistics), "pNN") }
func AllPopulations() []string    { return members(populations) }
func AllDenominators() []string   { return members(denominators) }
func AllClocks() []string         { return members(clocks) }
func AllStartEvents() []string    { return members(starts) }
func AllStopEvents() []string     { return members(stops) }
func AllOutputLenSources() []string {
	return members(outputLens)
}
func AllPercentileMethods() []string { return members(percentiles) }
func AllMetricUnits() []string       { return members(units) }
func AllRowVerdicts() []string       { return members(rowVerdicts) }
func AllMaterialities() []string     { return members(materials) }
func AllDiffKinds() []string         { return members(diffKinds) }
func AllLicenseStatuses() []string   { return members(licenseStatuses) }
func AllUses() []string              { return members(uses) }
