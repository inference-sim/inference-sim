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
// A figure is a metric at a scope: the whole cluster, one pool, or one instance (design
// §8.7). The scope selects which requests, tokens or steps form the population; the formula,
// statistic, window and unit do not change. A figure that aggregates over several scopes, such
// as the largest p99 time to first token over instances, names the scope it ranges over and the
// aggregate. A definition states the scopes it is defined at and the aggregates it admits.
//
// Design §8.7 names one aggregate, "the largest p99 time to first token over instances", and
// the plan calls the figure config search ranks on "the worst" instance. Which instance is
// worst depends on the direction of the metric: the largest figure for a latency, the smallest
// for a throughput. The enumeration is therefore the two orderings, max and min, and a
// definition states its direction so that Worst names the one that means "worst" for it. The
// design names no mean or sum of figures, and the mean of per-scope percentiles or fractions is
// neither the cluster's figure nor a weighted one, so neither is admitted; adding one is a
// design change.
//
// Two figures are comparable only when their definitions are identical and their scope,
// the scope they range over and their aggregate are the same. Compare refuses any other pair,
// and a definition is never inferred from a field's name.
package metrics

import (
	"fmt"
	"sort"
	"strings"
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

// Denominator is what a rate divides by, besides the quantity it counts. Two rates with
// the same formula, population and unit but different denominators are different metrics:
// 1000 output tokens per second over the window is not 125 per GPU.
type Denominator string

const (
	// DenominatorWindow: the length of the measured window.
	DenominatorWindow Denominator = "window"
	// DenominatorWindowAndGPUs: the length of the measured window and the number of GPUs
	// the scope occupies: the deployment's pools at the cluster, its own at a pool or an
	// instance.
	DenominatorWindowAndGPUs Denominator = "window_and_gpus"
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

// Scope is the part of a deployment a figure describes. The scope names a kind: which pool
// or which instance is the key under which a Result carries the figure.
type Scope string

const (
	// ScopeCluster is every request, token or step of the run, across all pools.
	ScopeCluster Scope = "cluster"
	// ScopePool is those of one pool of a disaggregated or multi-pool deployment.
	ScopePool Scope = "pool"
	// ScopeInstance is those of one engine instance.
	ScopeInstance Scope = "instance"
)

// scopeRank orders the scopes by how much they contain: an instance is part of a pool and a
// pool part of the cluster. Zero is not a scope.
func scopeRank(s Scope) int {
	switch s {
	case ScopeInstance:
		return 1
	case ScopePool:
		return 2
	case ScopeCluster:
		return 3
	}
	return 0
}

// Valid reports whether s is one of the scopes.
func (s Scope) Valid() bool { return scopeRank(s) != 0 }

// Aggregate picks one figure out of the figures of several narrower scopes, such as the
// largest p99 time to first token over instances. The figure it picks is one instance's (or
// pool's), not the figure of the pooled population: the largest instance p99 is not the
// cluster's p99.
type Aggregate string

const (
	// AggregateMax is the largest figure over the scopes ranged over. It is the worst for a
	// metric where lower is better, such as a latency (design §8.7's example).
	AggregateMax Aggregate = "max"
	// AggregateMin is the smallest figure. It is the worst for a metric where higher is
	// better, such as a throughput or an attainment.
	AggregateMin Aggregate = "min"
)

// Valid reports whether a is one of the aggregates.
func (a Aggregate) Valid() bool { return a == AggregateMax || a == AggregateMin }

// Direction says which way a metric improves.
type Direction string

const (
	// DirectionLowerIsBetter: a latency, a delay, a count of preemptions.
	DirectionLowerIsBetter Direction = "lower_is_better"
	// DirectionHigherIsBetter: a throughput, a hit rate, an attainment.
	DirectionHigherIsBetter Direction = "higher_is_better"
)

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
	// Denominator is set exactly when the unit is a rate per second: what the rate divides
	// by.
	Denominator Denominator `yaml:"denominator,omitempty" json:"denominator,omitempty"`
	// Scopes are the scopes the metric is defined at, in no particular order. A metric the
	// router alone observes has the cluster only.
	Scopes []Scope `yaml:"scopes" json:"scopes"`
	// Aggregates are the aggregates a figure of this metric may take over narrower scopes. A
	// metric defined at two or more scopes admits both; one defined at a single scope has
	// nothing to range over and admits none.
	Aggregates []Aggregate `yaml:"aggregates,omitempty" json:"aggregates,omitempty"`
	// Better says which way the metric improves, so that Worst can name the aggregate that
	// picks the worst instance.
	Better Direction `yaml:"better" json:"better"`
	// Description is one sentence for a reader. It decides nothing: comparisons use the
	// fields above.
	Description string `yaml:"description" json:"description"`
	// Follows names the benchmark-client code the definition matches, file and lines at a
	// pinned release, when it follows one.
	Follows string `yaml:"follows,omitempty" json:"follows,omitempty"`
}

// Worst returns the aggregate that picks the worst figure of d over several scopes: the
// largest for a metric where lower is better, the smallest otherwise. It is empty for a
// definition with no direction.
func (d Definition) Worst() Aggregate {
	switch d.Better {
	case DirectionLowerIsBetter:
		return AggregateMax
	case DirectionHigherIsBetter:
		return AggregateMin
	}
	return ""
}

// sameMeaning reports whether two definitions mean the same thing. ID, Description and
// Follows are names and prose; Scopes, Aggregates and Better say where the metric may be
// reported and which way it improves, not what it is (a figure's own scope and aggregate are compared by ComparableFigures); the rest
// is the definition.
func (d Definition) sameMeaning(o Definition) bool {
	return d.Formula == o.Formula && d.Statistic == o.Statistic && d.Population == o.Population &&
		d.Window == o.Window && d.Unit == o.Unit && d.PercentileMethod == o.PercentileMethod &&
		d.Unfinished == o.Unfinished && d.Denominator == o.Denominator
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
	// clusterOnly is set for a quantity only the router observes.
	clusterOnly bool
}

var latencyFamilies = []latencyFamily{
	{"ttft", FormulaTTFT, PopulationRequests,
		"Time to first token", vllmServe, false},
	{"tpot", FormulaTPOT, PopulationRequestsTwoTokens,
		"Time per output token, excluding the first, computed per request", vllmServe, false},
	{"itl", FormulaITL, PopulationTokens,
		"Inter-token latency, pooled over every gap between output tokens", vllmServe, false},
	{"e2e", FormulaE2E, PopulationRequests,
		"Request latency, arrival to last output token", vllmServe, false},
	{"step_interval", FormulaStepInterval, PopulationSteps,
		"Engine step interval between token-emitting steps of a request; not per token under speculative decoding", "", false},
	{"scheduling_delay", FormulaSchedulingDelay, PopulationRequests,
		"Time from arrival at an engine to the request's first scheduled step", "", false},
	{"gateway_queue_delay", FormulaGatewayQueueDelay, PopulationGatewayQueued,
		"Time a request waited in the router's flow-control queue", "", true},
}

var latencyStats = []Statistic{StatMean, StatP50, StatP90, StatP95, StatP99}

func build() map[string]Definition {
	out := map[string]Definition{}
	// add states the scopes and aggregates of a definition from its shape, so that no
	// definition is added without them, and refuses one that does not validate.
	add := func(d Definition) {
		if _, dup := out[d.ID]; dup {
			panic("metrics: duplicate identifier " + d.ID)
		}
		if d.Scopes == nil {
			d.Scopes = []Scope{ScopeCluster, ScopePool, ScopeInstance}
		}
		d.Aggregates = aggregatesFor(d)
		d.Better = direction(d)
		if err := d.Validate(); err != nil {
			panic(err.Error())
		}
		out[d.ID] = d
	}
	for _, f := range latencyFamilies {
		for _, st := range latencyStats {
			d := Definition{ID: f.prefix + "_" + string(st) + "_ms", Formula: f.formula, Statistic: st,
				Population: f.population, Window: WindowMeasured, Unit: UnitMs, Unfinished: UnfinishedExcluded,
				Description: f.desc + ", " + string(st) + ", over the measured window.", Follows: f.follows}
			if f.clusterOnly {
				d.Scopes = []Scope{ScopeCluster}
			}
			if st.Percentile() {
				d.PercentileMethod = PercentileNumpyLinear
			}
			add(d)
		}
	}
	rate := func(id string, f Formula, pop Population, u Unit, desc, follows string) {
		d := Definition{ID: id, Formula: f, Statistic: StatRate, Population: pop, Window: WindowMeasured,
			Unit: u, Unfinished: UnfinishedNotApplicable, Description: desc, Follows: follows}
		if u == UnitTokensPerS || u == UnitRequestsPerS {
			d.Denominator = DenominatorWindow
		}
		if id == "output_tok_s_per_gpu" {
			d.Denominator = DenominatorWindowAndGPUs
		}
		add(d)
	}
	rate("output_tok_s", FormulaOutputThroughput, PopulationRequests, UnitTokensPerS,
		"Output tokens of the window's completed requests over the window's length.", aiperfTPS)
	rate("output_tok_s_per_gpu", FormulaOutputThroughput, PopulationRequests, UnitTokensPerS,
		"output_tok_s over the GPUs of the scope it describes: the deployment's pools at the cluster, a pool's or an instance's own at those scopes.", "")
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

// aggregatesFor returns the aggregates a definition admits: both orderings when it is defined
// at two or more scopes, none otherwise.
func aggregatesFor(d Definition) []Aggregate {
	if len(d.Scopes) < 2 {
		return nil
	}
	return []Aggregate{AggregateMax, AggregateMin}
}

// direction states which way a metric improves, from its unit and formula: a rate, a hit rate
// or an attainment improves upward; a latency, a delay and a count of preemptions improve
// downward.
func direction(d Definition) Direction {
	switch d.Unit {
	case UnitTokensPerS, UnitRequestsPerS, UnitTokensPerSUser, UnitFraction:
		return DirectionHigherIsBetter
	}
	return DirectionLowerIsBetter
}

// Validate reports a definition that cannot be used for a figure: no scope, a scope or an
// aggregate outside the enumerations, a repeated one, no direction, and aggregates without two
// scopes to range between. Findings are joined in one error.
func (d Definition) Validate() error {
	var bad []string
	add := func(format string, a ...any) { bad = append(bad, fmt.Sprintf(format, a...)) }
	if len(d.Scopes) == 0 {
		add("states no scope; a metric is defined at the cluster, a pool or an instance")
	}
	seenScope := map[Scope]bool{}
	for _, s := range d.Scopes {
		switch {
		case !s.Valid():
			add("scope %q is not one of cluster, pool, instance", s)
		case seenScope[s]:
			add("scope %q is stated twice", s)
		}
		seenScope[s] = true
	}
	seenAgg := map[Aggregate]bool{}
	for _, a := range d.Aggregates {
		switch {
		case !a.Valid():
			add("aggregate %q is not one of max, min", a)
		case seenAgg[a]:
			add("aggregate %q is stated twice", a)
		}
		seenAgg[a] = true
	}
	if len(d.Aggregates) > 0 && len(seenScope) < 2 {
		add("admits aggregates but is defined at %d scope(s); an aggregate ranges from a narrower scope to a wider one, so it needs two", len(seenScope))
	}
	if d.Better != DirectionLowerIsBetter && d.Better != DirectionHigherIsBetter {
		add("direction %q is not lower_is_better or higher_is_better", d.Better)
	}
	if len(bad) == 0 {
		return nil
	}
	return fmt.Errorf("metrics %q: %s", d.ID, strings.Join(bad, "; "))
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

// Figure is one reported value with the definition it was computed under and the scope it
// describes.
type Figure struct {
	Definition Definition
	Value      float64
	// Scope is the scope the figure is at. A figure always has one.
	Scope Scope
	// Member names which pool or which instance the figure is of, by the name the deployment
	// gives it. Design 8.7 makes a figure's scope "the whole cluster, one pool, or one
	// instance": the scope selects the requests, tokens or steps that form the population, and
	// the population of instance 3 is not that of instance 7. So a figure at the pool or
	// instance scope names its member, and a figure at the cluster scope has none. For a
	// figure that aggregates, Member is the member of Scope, not of the scope ranged over
	// (the largest p99 over the instances of pool "decode" is Scope pool, Member "decode").
	Member string
	// Aggregate and Over are set together when the figure combines the figures of a narrower
	// scope: the aggregate, and the scope it ranges over ("max over instances" at the cluster
	// scope is Scope cluster, Aggregate max, Over instance). Both are empty for a figure
	// computed directly from the scope's own population.
	Aggregate Aggregate
	Over      Scope
}

// Validate reports a figure that does not follow its definition (DEV-8), or whose definition
// does not validate: no scope, a scope the
// metric is not defined at, an aggregate the metric does not admit, an aggregate with no scope
// to range over or one that is not narrower than the figure's own, or a scope to range over
// the metric is not defined at.
func (f Figure) Validate() error {
	d := f.Definition
	if err := d.Validate(); err != nil {
		return fmt.Errorf("figure of a malformed definition: %w", err)
	}
	var bad []string
	add := func(format string, a ...any) { bad = append(bad, fmt.Sprintf(format, a...)) }
	switch {
	case f.Scope == "":
		add("states no scope")
	case !f.Scope.Valid():
		add("scope %q is not one of cluster, pool, instance", f.Scope)
	case !hasScope(d.Scopes, f.Scope):
		add("scope %q is not one the metric is defined at (%v)", f.Scope, d.Scopes)
	}
	switch {
	case f.Scope == ScopeCluster && f.Member != "":
		add("is at the cluster scope and names member %q; only a pool or an instance has one", f.Member)
	case (f.Scope == ScopePool || f.Scope == ScopeInstance) && f.Member == "":
		add("is at the %s scope and names no member; it is the figure of one %s", f.Scope, f.Scope)
	}
	switch {
	case f.Aggregate == "" && f.Over == "":
	case f.Aggregate == "":
		add("ranges over %q but names no aggregate", f.Over)
	case f.Over == "":
		add("aggregate %q ranges over no scope", f.Aggregate)
	case !f.Aggregate.Valid():
		add("aggregate %q is not one of max, min", f.Aggregate)
	case !f.Over.Valid():
		add("ranges over %q, which is not one of cluster, pool, instance", f.Over)
	default:
		if !hasAggregate(d.Aggregates, f.Aggregate) {
			add("aggregate %q is not one the metric admits (%v)", f.Aggregate, d.Aggregates)
		}
		if !hasScope(d.Scopes, f.Over) {
			add("cannot range over %q: the metric is not defined there (%v)", f.Over, d.Scopes)
		}
		if f.Scope.Valid() && scopeRank(f.Over) >= scopeRank(f.Scope) {
			add("a figure at %q cannot range over %q, which is not narrower", f.Scope, f.Over)
		}
	}
	if len(bad) == 0 {
		return nil
	}
	return fmt.Errorf("figure of %q: %s", d.ID, strings.Join(bad, "; "))
}

func hasScope(l []Scope, s Scope) bool {
	for _, x := range l {
		if x == s {
			return true
		}
	}
	return false
}

func hasAggregate(l []Aggregate, a Aggregate) bool {
	for _, x := range l {
		if x == a {
			return true
		}
	}
	return false
}

// Compare returns b's value minus a's when the two figures are comparable, and an error
// naming the first field that differs otherwise (DEV-8). Two figures whose definitions differ
// in any field, or whose scope, aggregate or scope ranged over differ, are not comparable,
// whatever their identifiers say.
//
// The number of GPUs itself is not part of a definition: it is a fact of the deployment, and
// the denominator only says that the rate is divided by it. Which pool or instance a figure
// is of is part of its scope (design 8.7: "one pool, or one instance"), so instance 3's p99 and
// instance 7's are not comparable, and the same instance's in two runs is.
func Compare(a, b Figure) (float64, error) {
	if err := ComparableFigures(a, b); err != nil {
		return 0, err
	}
	return b.Value - a.Value, nil
}

// ComparableFigures returns nil when a and b are the same metric at the same scope with the
// same aggregate over the same scope, which includes the same member. It first refuses a
// figure that does not validate, so a figure with no scope is never comparable, even with
// another that has none. Otherwise the
// error names the field where they differ: a definition's field as Comparable names it, or
// scope, member, aggregate or over.
func ComparableFigures(a, b Figure) error {
	if err := a.Validate(); err != nil {
		return fmt.Errorf("first figure: %w", err)
	}
	if err := b.Validate(); err != nil {
		return fmt.Errorf("second figure: %w", err)
	}
	if err := Comparable(a.Definition, b.Definition); err != nil {
		return err
	}
	for _, f := range []struct {
		name   string
		av, bv string
	}{
		{"scope", string(a.Scope), string(b.Scope)},
		{"member", a.Member, b.Member},
		{"aggregate", string(a.Aggregate), string(b.Aggregate)},
		{"over", string(a.Over), string(b.Over)},
	} {
		if f.av != f.bv {
			return fmt.Errorf("metrics %q and %q are not comparable: %s is %q in one and %q in the other",
				a.Definition.ID, b.Definition.ID, f.name, f.av, f.bv)
		}
	}
	return nil
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
		{"denominator", string(a.Denominator), string(b.Denominator)},
	} {
		if f.av != f.bv {
			return fmt.Errorf("metrics %q and %q are not comparable: %s is %q in one and %q in the other",
				a.ID, b.ID, f.name, f.av, f.bv)
		}
	}
	return nil // unreachable: TestComparableAgreesWithSameMeaning holds the two lists to the same fields
}
