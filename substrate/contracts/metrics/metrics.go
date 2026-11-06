// Package metrics is the metric registry (design §8.7, DEV-8): one definition, with an
// identifier, for every metric that BLIS reports, compares, ranks or uses in a
// service-level check. A definition states the metric's formula, its statistic, its
// population, its window and its unit. A Result names each figure by its identifier and
// carries no definition of its own, so the simulator, the scoring harness and config
// search read one meaning for each name.
//
// Time to first token, time per output token and inter-token latency follow the
// benchmark client BLIS is compared with: vLLM's `vllm bench serve` at v0.32.0
// (vllm/benchmarks/serve.py:624-634). Time per output token excludes
// the first token and is a statistic over requests, each request's own (latency - TTFT) /
// (output tokens - 1). Inter-token latency is pooled over every gap between consecutive
// output tokens of every request with at least two output tokens (inference-sim #1894), as
// `vllm bench serve` pools them. aiperf's inter-token latency is a per-request mean, a
// different population, so it is not cited for that family. Output tokens per second follows
// aiperf. interactivity_p50 and interactivity_p90 are defined by BLIS, as 1000 over the
// time-per-output-token percentile in milliseconds; neither benchmark client reports them.
//
// Two figures are comparable only when their definitions are identical. Compare refuses
// any other pair, and a definition is never inferred from a field's name.
package metrics

import (
	"fmt"
	"sort"
)

// Formula is the per-sample quantity a metric is computed from.
type Formula string

const (
	// FormulaTTFT is a request's time from arrival at the client to its first output
	// token.
	FormulaTTFT Formula = "ttft"
	// FormulaTPOT is a request's (E2E latency - TTFT) / (output tokens - 1): its mean
	// time per output token after the first.
	FormulaTPOT Formula = "tpot"
	// FormulaITL is one gap between two consecutive output tokens of one request.
	FormulaITL Formula = "itl"
	// FormulaE2E is a request's time from arrival at the client to its last output token.
	FormulaE2E Formula = "e2e"
	// FormulaStepInterval is the time from the end of one engine step to the end of the
	// next step that emits a token for the same request. Under speculative decoding one
	// step emits several tokens, so it is not a per-token figure.
	FormulaStepInterval Formula = "step_interval"
	// FormulaSchedulingDelay is a request's time from arrival at an engine to its first
	// scheduled step.
	FormulaSchedulingDelay Formula = "scheduling_delay"
	// FormulaGatewayQueueDelay is a request's time in the router's flow-control queue.
	FormulaGatewayQueueDelay Formula = "gateway_queue_delay"
	// FormulaOutputThroughput is output tokens emitted over the window's length.
	FormulaOutputThroughput Formula = "output_throughput"
	// FormulaTotalThroughput is prompt plus output tokens over the window's length.
	FormulaTotalThroughput Formula = "total_throughput"
	// FormulaRequestThroughput is completed requests over the window's length.
	FormulaRequestThroughput Formula = "request_throughput"
	// FormulaUserTPS is a request's (output tokens - 1) / (E2E latency - TTFT): its
	// output tokens per second after the first.
	FormulaUserTPS Formula = "user_tps"
	// FormulaInteractivity is 1000 / a TPOT percentile in milliseconds, tokens per second
	// per user.
	FormulaInteractivity Formula = "interactivity"
	// FormulaCacheHitRate is prompt tokens served from the prefix cache over prompt tokens
	// looked up, summed over admissions.
	FormulaCacheHitRate Formula = "cache_hit_rate"
	// FormulaPreemptions is the number of preemptions the engines performed.
	FormulaPreemptions Formula = "preemptions"
	// FormulaGoodput is completed requests that met every SLO target of their class, over
	// the window's length.
	FormulaGoodput Formula = "goodput"
	// FormulaSLOAttainment is completed requests that met every SLO target of their
	// class, over requests of the targeted classes that arrived.
	FormulaSLOAttainment Formula = "slo_attainment"
)

// Statistic summarises a population.
type Statistic string

const (
	StatMean Statistic = "mean"
	StatP50  Statistic = "p50"
	StatP90  Statistic = "p90"
	StatP95  Statistic = "p95"
	StatP99  Statistic = "p99"
	// StatRate is a sum over the window divided by the window's length, or by a count.
	StatRate Statistic = "rate"
	// StatTotal is a sum over the window.
	StatTotal Statistic = "total"
)

// Percentile reports whether s is a percentile.
func (s Statistic) Percentile() bool {
	switch s {
	case StatP50, StatP90, StatP95, StatP99:
		return true
	}
	return false
}

// Population is what a statistic is taken over.
type Population string

const (
	// PopulationRequests is every completed request in the window.
	PopulationRequests Population = "requests"
	// PopulationRequestsTwoTokens is every completed request in the window with at least
	// two output tokens. A request with fewer has no gap between tokens.
	PopulationRequestsTwoTokens Population = "requests_min_2_output_tokens"
	// PopulationTokens is every gap between consecutive output tokens of every request in
	// PopulationRequestsTwoTokens, pooled.
	PopulationTokens Population = "tokens"
	// PopulationSteps is every engine step in the window that emitted a token.
	PopulationSteps Population = "steps"
	// PopulationAdmissions is every admission of a request to an engine in the window.
	PopulationAdmissions Population = "admissions"
	// PopulationGatewayQueued is every request in the window that waited in the router's
	// flow-control queue.
	PopulationGatewayQueued Population = "gateway_queued_requests"
	// PopulationTargeted is every request in the window whose class has an SLO target.
	PopulationTargeted Population = "targeted_requests"
)

// Window names the interval a metric's samples are drawn from.
type Window string

const (
	// WindowMeasured is the experimental frame's measured window (design §8.6): samples
	// from requests that arrived after the warm-up and completed before the window closed.
	WindowMeasured Window = "measured"
	// WindowRun is the whole run, from its start to its stop rule, warm-up included.
	WindowRun Window = "run"
)

// Unit is a metric value's unit.
type Unit string

const (
	UnitMs             Unit = "ms"
	UnitTokensPerS     Unit = "tokens_per_s"
	UnitRequestsPerS   Unit = "requests_per_s"
	UnitTokensPerSUser Unit = "tokens_per_s_per_user"
	UnitFraction       Unit = "fraction"
	UnitCount          Unit = "count"
)

// Unfinished says how a latency statistic treats requests in flight or queued at the
// horizon (DEV-9, design §8.7). Their latency is unknown but at least what they have
// waited.
type Unfinished string

const (
	// UnfinishedExcluded: such requests give no sample; the Result reports their count
	// beside the figure.
	UnfinishedExcluded Unfinished = "excluded"
	// UnfinishedNotApplicable: the metric is not a latency, so the question does not
	// arise.
	UnfinishedNotApplicable Unfinished = "not_applicable"
)

// PercentileMethod is how a percentile interpolates between samples.
type PercentileMethod string

// PercentileNumpyLinear sorts the samples and interpolates linearly at rank
// q/100*(n-1), numpy's default and both benchmark clients' method.
const PercentileNumpyLinear PercentileMethod = "numpy_linear"

// Definition is one metric.
type Definition struct {
	// ID is the identifier a Result names the metric by, a lowercase slug.
	ID         string     `yaml:"id" json:"id"`
	Formula    Formula    `yaml:"formula" json:"formula"`
	Statistic  Statistic  `yaml:"statistic" json:"statistic"`
	Population Population `yaml:"population" json:"population"`
	Window     Window     `yaml:"window" json:"window"`
	Unit       Unit       `yaml:"unit" json:"unit"`
	// PercentileMethod is set exactly when Statistic is a percentile.
	PercentileMethod PercentileMethod `yaml:"percentile_method,omitempty" json:"percentile_method,omitempty"`
	// Unfinished says how requests unfinished at the horizon are treated.
	Unfinished Unfinished `yaml:"unfinished" json:"unfinished"`
	// Description is one sentence for a reader. It decides nothing: comparisons use the
	// fields above.
	Description string `yaml:"description" json:"description"`
	// Follows names the benchmark-client code the definition matches, file and lines at a
	// pinned release, when it follows one.
	Follows string `yaml:"follows,omitempty" json:"follows,omitempty"`
}

// sameMeaning reports whether two definitions mean the same thing. ID, Description and
// Follows are names and prose; the rest is the definition.
func (d Definition) sameMeaning(o Definition) bool {
	return d.Formula == o.Formula && d.Statistic == o.Statistic && d.Population == o.Population &&
		d.Window == o.Window && d.Unit == o.Unit && d.PercentileMethod == o.PercentileMethod &&
		d.Unfinished == o.Unfinished
}

const (
	vllmServe = "vllm v0.32.0 vllm/benchmarks/serve.py:624-634"
	aiperfTPS = "aiperf 0.12.0 src/aiperf/metrics/types/output_token_throughput_metrics.py:40-78"
)

// registry is every definition, built once by build.
var registry = build()

// latencyFamily is one latency formula reported at each of the standard statistics.
type latencyFamily struct {
	prefix     string
	formula    Formula
	population Population
	desc       string
	follows    string
}

var latencyFamilies = []latencyFamily{
	{"ttft", FormulaTTFT, PopulationRequests,
		"Time to first token", vllmServe},
	{"tpot", FormulaTPOT, PopulationRequestsTwoTokens,
		"Time per output token, excluding the first, computed per request", vllmServe},
	{"itl", FormulaITL, PopulationTokens,
		"Inter-token latency, pooled over every gap between output tokens", vllmServe},
	{"e2e", FormulaE2E, PopulationRequests,
		"Request latency, arrival to last output token", vllmServe},
	{"step_interval", FormulaStepInterval, PopulationSteps,
		"Engine step interval between token-emitting steps of a request; not per token under speculative decoding", ""},
	{"scheduling_delay", FormulaSchedulingDelay, PopulationRequests,
		"Time from arrival at an engine to the request's first scheduled step", ""},
	{"gateway_queue_delay", FormulaGatewayQueueDelay, PopulationGatewayQueued,
		"Time a request waited in the router's flow-control queue", ""},
}

var latencyStats = []Statistic{StatMean, StatP50, StatP90, StatP95, StatP99}

func build() map[string]Definition {
	out := map[string]Definition{}
	add := func(d Definition) {
		if _, dup := out[d.ID]; dup {
			panic("metrics: duplicate identifier " + d.ID)
		}
		out[d.ID] = d
	}
	for _, f := range latencyFamilies {
		for _, st := range latencyStats {
			d := Definition{ID: f.prefix + "_" + string(st) + "_ms", Formula: f.formula, Statistic: st,
				Population: f.population, Window: WindowMeasured, Unit: UnitMs, Unfinished: UnfinishedExcluded,
				Description: f.desc + ", " + string(st) + ", over the measured window.", Follows: f.follows}
			if st.Percentile() {
				d.PercentileMethod = PercentileNumpyLinear
			}
			add(d)
		}
	}
	rate := func(id string, f Formula, pop Population, u Unit, desc, follows string) {
		add(Definition{ID: id, Formula: f, Statistic: StatRate, Population: pop, Window: WindowMeasured,
			Unit: u, Unfinished: UnfinishedNotApplicable, Description: desc, Follows: follows})
	}
	rate("output_tok_s", FormulaOutputThroughput, PopulationRequests, UnitTokensPerS,
		"Output tokens of the window's completed requests over the window's length.", aiperfTPS)
	rate("output_tok_s_per_gpu", FormulaOutputThroughput, PopulationRequests, UnitTokensPerS,
		"output_tok_s over the GPUs the deployment's pools occupy.", "")
	rate("total_tok_s", FormulaTotalThroughput, PopulationRequests, UnitTokensPerS,
		"Prompt and output tokens of the window's completed requests over the window's length.", vllmServe)
	rate("request_throughput", FormulaRequestThroughput, PopulationRequests, UnitRequestsPerS,
		"Completed requests over the window's length.", vllmServe)
	rate("goodput_rps", FormulaGoodput, PopulationTargeted, UnitRequestsPerS,
		"Completed requests that met every SLO target of their class, over the window's length.", "")
	rate("slo_attainment", FormulaSLOAttainment, PopulationTargeted, UnitFraction,
		"Completed requests that met every SLO target of their class, over the targeted requests that arrived.", "")
	rate("cache_hit_rate", FormulaCacheHitRate, PopulationAdmissions, UnitFraction,
		"Prompt tokens served from the prefix cache over prompt tokens looked up.", "")
	add(Definition{ID: "interactivity_mean", Formula: FormulaUserTPS, Statistic: StatMean,
		Population: PopulationRequestsTwoTokens, Window: WindowMeasured, Unit: UnitTokensPerSUser,
		Unfinished: UnfinishedExcluded, Follows: aiperfTPS,
		Description: "Mean over requests of output tokens per second after the first."})
	for _, st := range []Statistic{StatP50, StatP90} {
		add(Definition{ID: "interactivity_" + string(st), Formula: FormulaInteractivity, Statistic: st,
			Population: PopulationRequestsTwoTokens, Window: WindowMeasured, Unit: UnitTokensPerSUser,
			PercentileMethod: PercentileNumpyLinear, Unfinished: UnfinishedExcluded,
			Description: "1000 over the " + string(st) + " of tpot in milliseconds."})
	}
	add(Definition{ID: "preemptions", Formula: FormulaPreemptions, Statistic: StatTotal,
		Population: PopulationSteps, Window: WindowRun, Unit: UnitCount, Unfinished: UnfinishedNotApplicable,
		Description: "Preemptions the engines performed over the whole run."})
	return out
}

// Lookup returns the definition with identifier id.
func Lookup(id string) (Definition, bool) {
	d, ok := registry[id]
	return d, ok
}

// All returns every definition, sorted by identifier.
func All() []Definition {
	out := make([]Definition, 0, len(registry))
	for _, d := range registry {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// IDs returns every identifier, sorted.
func IDs() []string {
	out := make([]string, 0, len(registry))
	for id := range registry {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// Figure is one reported value with the definition it was computed under.
type Figure struct {
	Definition Definition
	Value      float64
}

// Compare returns b's value minus a's when the two figures share a definition, and an
// error naming the first field that differs otherwise (DEV-8). Two figures whose
// definitions differ in any field are not comparable, whatever their identifiers say.
//
// In M1 this meets DEV-8 for the definition: formula, statistic, population, window, unit,
// percentile method and the handling of unfinished requests. DEV-8 also names scope and
// aggregate; a Definition carries neither yet, and they arrive in M2 with the registry's
// scope (design §8.7), so two figures of one definition taken at different scopes are not
// told apart until then.
func Compare(a, b Figure) (float64, error) {
	if err := Comparable(a.Definition, b.Definition); err != nil {
		return 0, err
	}
	return b.Value - a.Value, nil
}

// Comparable returns nil when a and b define the same metric, and otherwise an error
// that names the field where they differ and both values.
func Comparable(a, b Definition) error {
	if a.sameMeaning(b) {
		return nil
	}
	for _, f := range []struct {
		name   string
		av, bv string
	}{
		{"formula", string(a.Formula), string(b.Formula)},
		{"statistic", string(a.Statistic), string(b.Statistic)},
		{"population", string(a.Population), string(b.Population)},
		{"window", string(a.Window), string(b.Window)},
		{"unit", string(a.Unit), string(b.Unit)},
		{"percentile_method", string(a.PercentileMethod), string(b.PercentileMethod)},
		{"unfinished", string(a.Unfinished), string(b.Unfinished)},
	} {
		if f.av != f.bv {
			return fmt.Errorf("metrics %q and %q are not comparable: %s is %q in one and %q in the other",
				a.ID, b.ID, f.name, f.av, f.bv)
		}
	}
	return nil // unreachable: sameMeaning compares exactly these fields
}
