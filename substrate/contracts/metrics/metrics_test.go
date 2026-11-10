package metrics

import (
	"reflect"
	"regexp"
	"strings"
	"testing"
)

var slug = regexp.MustCompile(`^[a-z0-9][a-z0-9_]*$`)

// TestEveryDefinitionIsComplete: each definition states every field a comparison checks,
// and a percentile method exactly when its statistic is a percentile.
func TestEveryDefinitionIsComplete(t *testing.T) {
	for _, d := range All() {
		if !slug.MatchString(d.ID) {
			t.Errorf("%q is not a lowercase slug", d.ID)
		}
		if d.Formula == "" || d.Statistic == "" || d.Population == "" || d.Window == "" ||
			d.Unit == "" || d.Unfinished == "" || d.Description == "" {
			t.Errorf("%s leaves a field empty: %+v", d.ID, d)
		}
		if d.Statistic.Percentile() != (d.PercentileMethod != "") {
			t.Errorf("%s: statistic %s with percentile method %q", d.ID, d.Statistic, d.PercentileMethod)
		}
		if d.Unit == UnitMs && d.Unfinished != UnfinishedExcluded {
			t.Errorf("%s is a latency and must state how unfinished requests are treated", d.ID)
		}
	}
}

// TestBenchmarkClientDefinitions pins the three definitions design §8.7 names: TPOT
// excludes the first token and is per request; ITL is pooled over tokens of requests with
// at least two output tokens (inference-sim #1894).
func TestBenchmarkClientDefinitions(t *testing.T) {
	for _, c := range []struct {
		id  string
		f   Formula
		pop Population
	}{
		{"ttft_mean_ms", FormulaTTFT, PopulationRequests},
		{"tpot_mean_ms", FormulaTPOT, PopulationRequestsTwoTokens},
		{"tpot_p99_ms", FormulaTPOT, PopulationRequestsTwoTokens},
		{"itl_mean_ms", FormulaITL, PopulationTokens},
		{"itl_p90_ms", FormulaITL, PopulationTokens},
	} {
		d, ok := Lookup(c.id)
		if !ok {
			t.Fatalf("%s is not in the registry", c.id)
		}
		if d.Formula != c.f || d.Population != c.pop {
			t.Errorf("%s: formula %s over %s, want %s over %s", c.id, d.Formula, d.Population, c.f, c.pop)
		}
		if !strings.Contains(d.Follows, "vllm/benchmarks/serve.py") {
			t.Errorf("%s does not cite the benchmark client it follows", c.id)
		}
	}
}

// aiperf's inter-token latency is a mean taken per request; the registry pools gaps as
// `vllm bench serve` does, so the ITL family must not claim to follow aiperf.
func TestITLDoesNotFollowAiperf(t *testing.T) {
	for _, d := range All() {
		if d.Formula == FormulaITL && strings.Contains(d.Follows, "aiperf") {
			t.Errorf("%s follows %q, but aiperf's ITL is a different population", d.ID, d.Follows)
		}
	}
}

// TestCompareRefusesDifferentDefinitions is DEV-8's test: a comparison of two figures is
// refused unless their definitions are the same, and the refusal names the field.
func TestCompareRefusesDifferentDefinitions(t *testing.T) {
	tpot, _ := Lookup("tpot_mean_ms")
	itl, _ := Lookup("itl_mean_ms")
	if _, err := Compare(at(tpot, 10), at(itl, 12)); err == nil ||
		!strings.Contains(err.Error(), "formula") {
		t.Errorf("tpot against itl: got %v, want a refusal naming formula", err)
	}
	p90, _ := Lookup("ttft_p90_ms")
	p99, _ := Lookup("ttft_p99_ms")
	if _, err := Compare(at(p90, 1), at(p99, 2)); err == nil || !strings.Contains(err.Error(), "statistic") {
		t.Errorf("p90 against p99: got %v, want a refusal naming statistic", err)
	}
	// The same definition under another identifier is comparable: the name decides nothing.
	renamed := tpot
	renamed.ID = "my_tpot"
	if d, err := Compare(at(tpot, 10), at(renamed, 12.5)); err != nil || d != 2.5 {
		t.Errorf("same definition: got %v, %v; want 2.5, nil", d, err)
	}
	// A definition that differs only in how unfinished requests count is a different metric.
	lower := tpot
	lower.Unfinished = UnfinishedNotApplicable
	if _, err := Compare(at(tpot, 1), at(lower, 1)); err == nil || !strings.Contains(err.Error(), "unfinished") {
		t.Errorf("unfinished treatment: got %v, want a refusal", err)
	}
}

// TestEveryPreviouslyReportedMetricIsDefined lists the figures inference-sim's Result
// reported as of be78af83 (cmd/result_build.go:168-237 standardMetrics, and
// cmd/result_inputs.go:262-310 engineSection's step-interval, scheduling-delay,
// gateway-delay and cache figures); each has an identifier here.
func TestEveryPreviouslyReportedMetricIsDefined(t *testing.T) {
	for _, id := range []string{
		"output_tok_s", "output_tok_s_per_gpu", "request_throughput",
		"ttft_mean_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p95_ms", "ttft_p99_ms",
		"e2e_mean_ms", "e2e_p50_ms", "e2e_p90_ms", "e2e_p95_ms", "e2e_p99_ms",
		"tpot_mean_ms", "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms",
		"interactivity_mean", "interactivity_p50", "interactivity_p90",
		"step_interval_mean_ms", "step_interval_p50_ms", "step_interval_p90_ms", "step_interval_p95_ms", "step_interval_p99_ms",
		"scheduling_delay_p99_ms", "gateway_queue_delay_mean_ms", "gateway_queue_delay_p90_ms",
		"cache_hit_rate", "preemptions", "goodput_rps", "slo_attainment",
	} {
		if _, ok := Lookup(id); !ok {
			t.Errorf("%s is not in the registry", id)
		}
	}
}

// Output tokens per second and output tokens per second per GPU share a formula, statistic,
// population, window and unit; they differ in what the rate divides by. They are not
// comparable, and the refusal names the denominator (DEV-8).
func TestPerGPUThroughputIsNotComparableToTheAbsoluteRate(t *testing.T) {
	abs, _ := Lookup("output_tok_s")
	per, _ := Lookup("output_tok_s_per_gpu")
	if _, err := Compare(at(abs, 1000), at(per, 125)); err == nil || !strings.Contains(err.Error(), "denominator") {
		t.Fatalf("got %v, want a refusal naming denominator", err)
	}
}

// Every rate states what it divides by, and only a rate does.
func TestEveryRateStatesItsDenominator(t *testing.T) {
	for _, d := range All() {
		perSecond := d.Unit == UnitTokensPerS || d.Unit == UnitRequestsPerS
		if perSecond != (d.Denominator != "") {
			t.Errorf("%s: unit %s with denominator %q", d.ID, d.Unit, d.Denominator)
		}
	}
}

// Comparable names the field where two definitions differ for every field sameMeaning
// reads: the two lists are parallel, and a field in one and not the other would let a
// difference through unnamed or be named when it decides nothing.
func TestComparableAgreesWithSameMeaning(t *testing.T) {
	base, _ := Lookup("output_tok_s")
	variants := map[string]func(*Definition){
		"formula":           func(d *Definition) { d.Formula = FormulaTotalThroughput },
		"statistic":         func(d *Definition) { d.Statistic = StatMean },
		"population":        func(d *Definition) { d.Population = PopulationTokens },
		"window":            func(d *Definition) { d.Window = WindowRun },
		"unit":              func(d *Definition) { d.Unit = UnitCount },
		"percentile_method": func(d *Definition) { d.PercentileMethod = PercentileNumpyLinear },
		"unfinished":        func(d *Definition) { d.Unfinished = UnfinishedExcluded },
		"denominator":       func(d *Definition) { d.Denominator = DenominatorWindowAndGPUs },
	}
	for field, change := range variants {
		o := base
		change(&o)
		if base.sameMeaning(o) {
			t.Errorf("%s: sameMeaning ignores it", field)
		}
		if err := Comparable(base, o); err == nil || !strings.Contains(err.Error(), field) {
			t.Errorf("%s: Comparable gave %v, want a refusal naming it", field, err)
		}
	}
	// Names and prose decide nothing.
	o := base
	o.ID, o.Description, o.Follows = "other", "other", "other"
	if !base.sameMeaning(o) || Comparable(base, o) != nil {
		t.Error("ID, Description and Follows must not affect comparability")
	}
	// Every field of Definition is either in the variants, one of the three names, or the
	// scopes, aggregates and direction of a metric (a figure carries its own, compared by
	// ComparableFigures).
	skip := map[string]bool{"ID": true, "Description": true, "Follows": true, "Scopes": true, "Aggregates": true, "Better": true}
	rt := reflect.TypeOf(Definition{})
	if got, want := rt.NumField()-len(skip), len(variants); got != want {
		t.Errorf("Definition has %d meaning fields and the test varies %d: add the new field here, to sameMeaning and to Comparable", got, want)
	}
}

// at is a cluster-scope figure computed directly from the definition's population.
func at(d Definition, v float64) Figure { return Figure{Definition: d, Value: v, Scope: ScopeCluster} }

func mustLookup(t *testing.T, id string) Definition {
	t.Helper()
	d, ok := Lookup(id)
	if !ok {
		t.Fatalf("%s is not in the registry", id)
	}
	return d
}

// TestEveryDefinitionStatesItsScopesAndAggregates: DEV-8 and design 8.7. Every definition in
// the registry is valid, states at least one scope and a direction, and admits both aggregates
// exactly when it is defined at two or more scopes.
func TestEveryDefinitionStatesItsScopesAndAggregates(t *testing.T) {
	for _, d := range All() {
		if err := d.Validate(); err != nil {
			t.Errorf("%v", err)
		}
		want := []Aggregate(nil)
		if len(d.Scopes) >= 2 {
			want = []Aggregate{AggregateMax, AggregateMin}
		}
		if !reflect.DeepEqual(d.Aggregates, want) {
			t.Errorf("%s: scopes %v with aggregates %v, want %v", d.ID, d.Scopes, d.Aggregates, want)
		}
	}
}

// The registry states, for named metrics, exactly the scopes, aggregates and direction written
// here, not what the code that builds them derives (design 8.7): the largest p99 time to first
// token over instances; the smallest throughput for the worst instance; a router queue delay
// that exists at the cluster only.
func TestRegistryStatesTheScopesDesignNames(t *testing.T) {
	all := []Scope{ScopeCluster, ScopePool, ScopeInstance}
	both := []Aggregate{AggregateMax, AggregateMin}
	for _, c := range []struct {
		id     string
		scopes []Scope
		aggs   []Aggregate
		better Direction
		worst  Aggregate
	}{
		{"ttft_p99_ms", all, both, DirectionLowerIsBetter, AggregateMax},
		{"tpot_mean_ms", all, both, DirectionLowerIsBetter, AggregateMax},
		{"output_tok_s", all, both, DirectionHigherIsBetter, AggregateMin},
		{"output_tok_s_per_gpu", all, both, DirectionHigherIsBetter, AggregateMin},
		{"total_tok_s", all, both, DirectionHigherIsBetter, AggregateMin},
		{"request_throughput", all, both, DirectionHigherIsBetter, AggregateMin},
		{"goodput_rps", all, both, DirectionHigherIsBetter, AggregateMin},
		{"slo_attainment", all, both, DirectionHigherIsBetter, AggregateMin},
		{"cache_hit_rate", all, both, DirectionHigherIsBetter, AggregateMin},
		{"interactivity_mean", all, both, DirectionHigherIsBetter, AggregateMin},
		{"interactivity_p50", all, both, DirectionHigherIsBetter, AggregateMin},
		{"preemptions", all, both, DirectionLowerIsBetter, AggregateMax},
		{"scheduling_delay_p99_ms", all, both, DirectionLowerIsBetter, AggregateMax},
		{"gateway_queue_delay_p99_ms", []Scope{ScopeCluster}, nil, DirectionLowerIsBetter, AggregateMax},
	} {
		d := mustLookup(t, c.id)
		if !reflect.DeepEqual(d.Scopes, c.scopes) || !reflect.DeepEqual(d.Aggregates, c.aggs) ||
			d.Better != c.better || d.Worst() != c.worst {
			t.Errorf("%s: scopes %v aggregates %v better %q worst %q; want %v %v %q %q",
				c.id, d.Scopes, d.Aggregates, d.Better, d.Worst(), c.scopes, c.aggs, c.better, c.worst)
		}
	}
}

// Known-bad definitions (DEV-8): each is refused and the refusal says why.
func TestValidateRefusesBadScopesAndAggregates(t *testing.T) {
	base := mustLookup(t, "output_tok_s")
	for _, c := range []struct {
		name string
		edit func(*Definition)
		want string
	}{
		{"no scope", func(d *Definition) { d.Scopes, d.Aggregates = nil, nil }, "no scope"},
		{"scope outside the enumeration", func(d *Definition) { d.Scopes = []Scope{"rack", ScopePool} }, `scope "rack"`},
		{"scope stated twice", func(d *Definition) { d.Scopes = []Scope{ScopeCluster, ScopeCluster, ScopeInstance} }, "stated twice"},
		{"aggregate outside the enumeration", func(d *Definition) { d.Aggregates = []Aggregate{"median"} }, `aggregate "median"`},
		{"mean is not an aggregate", func(d *Definition) { d.Aggregates = []Aggregate{"mean"} }, `aggregate "mean"`},
		{"aggregate stated twice", func(d *Definition) { d.Aggregates = []Aggregate{AggregateMax, AggregateMax} }, `aggregate "max" is stated twice`},
		{"aggregate on a cluster-only metric", func(d *Definition) { d.Scopes = []Scope{ScopeCluster} }, "needs two"},
		{"aggregate on a pool-only metric", func(d *Definition) { d.Scopes = []Scope{ScopePool} }, "needs two"},
		{"aggregate on an instance-only metric", func(d *Definition) { d.Scopes = []Scope{ScopeInstance} }, "needs two"},
		{"no direction", func(d *Definition) { d.Better = "" }, "direction"},
		{"unknown direction", func(d *Definition) { d.Better = "better" }, "direction"},
	} {
		d := base
		d.Scopes = append([]Scope(nil), base.Scopes...)
		d.Aggregates = append([]Aggregate(nil), base.Aggregates...)
		c.edit(&d)
		if err := d.Validate(); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %v, want a refusal containing %q", c.name, err, c.want)
		}
	}
	if err := base.Validate(); err != nil {
		t.Errorf("the registry's own definition is refused: %v", err)
	}
	// Scope order carries no meaning.
	rev := base
	rev.Scopes = []Scope{ScopeInstance, ScopeCluster}
	if err := rev.Validate(); err != nil {
		t.Errorf("scopes in another order: %v", err)
	}
}

// A figure follows its definition: known-bad figures.
func TestFigureValidateRefusals(t *testing.T) {
	ttft := mustLookup(t, "ttft_p99_ms")
	queue := mustLookup(t, "gateway_queue_delay_p99_ms")
	ok := Figure{Definition: ttft, Value: 1, Scope: ScopeCluster, Aggregate: AggregateMax, Over: ScopeInstance}
	if err := ok.Validate(); err != nil {
		t.Fatalf("the design's example, the largest p99 over instances, is refused: %v", err)
	}
	malformed := ttft
	malformed.Scopes = nil
	for _, c := range []struct {
		name string
		f    Figure
		want string
	}{
		{"no scope", Figure{Definition: ttft}, "no scope"},
		{"unknown scope", Figure{Definition: ttft, Scope: "rack"}, `scope "rack" is not one of`},
		{"scope the metric is not defined at", Figure{Definition: queue, Scope: ScopeInstance}, "not one the metric is defined at"},
		{"aggregate with no scope to range over", Figure{Definition: ttft, Scope: ScopeCluster, Aggregate: AggregateMax}, "ranges over no scope"},
		{"scope to range over with no aggregate", Figure{Definition: ttft, Scope: ScopeCluster, Over: ScopeInstance}, "names no aggregate"},
		{"mean is not an aggregate", Figure{Definition: ttft, Scope: ScopeCluster, Aggregate: "mean", Over: ScopeInstance}, `aggregate "mean" is not one of max, min`},
		{"unknown aggregate", Figure{Definition: ttft, Scope: ScopeCluster, Aggregate: "median", Over: ScopeInstance}, `aggregate "median" is not one of max, min`},
		{"unknown scope to range over", Figure{Definition: ttft, Scope: ScopeCluster, Aggregate: AggregateMax, Over: "rack"}, `ranges over "rack", which is not one of`},
		{"aggregate the metric does not admit", Figure{Definition: func() Definition { d := ttft; d.Aggregates = []Aggregate{AggregateMax}; return d }(),
			Scope: ScopeCluster, Aggregate: AggregateMin, Over: ScopeInstance}, "not one the metric admits"},
		{"range over a scope not narrower", Figure{Definition: ttft, Scope: ScopeInstance, Aggregate: AggregateMax, Over: ScopeCluster}, "not narrower"},
		{"range over the figure's own scope", Figure{Definition: ttft, Scope: ScopePool, Aggregate: AggregateMax, Over: ScopePool}, "not narrower"},
		{"range over a scope the metric is not defined at", Figure{Definition: queue, Scope: ScopeCluster, Aggregate: AggregateMax, Over: ScopeInstance}, "the metric is not defined there"},
		{"a malformed definition", Figure{Definition: malformed, Scope: ScopeCluster}, "malformed definition"},
	} {
		if err := c.f.Validate(); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %v, want a refusal containing %q", c.name, err, c.want)
		}
	}
}

// ComparableFigures is DEV-8's three-way rule: same metric, same scope, same aggregate. Each
// field has a known-bad pair, and the refusal names the field.
func TestComparableFiguresRefusesEachDifference(t *testing.T) {
	ttft := mustLookup(t, "ttft_p99_ms")
	worst := Figure{Definition: ttft, Value: 5, Scope: ScopeCluster, Aggregate: AggregateMax, Over: ScopeInstance}
	same := worst
	same.Value = 7
	if d, err := Compare(worst, same); err != nil || d != 2 {
		t.Fatalf("identical metric, scope and aggregate: got %v, %v", d, err)
	}
	for _, c := range []struct {
		field string
		edit  func(*Figure)
	}{
		{"formula", func(f *Figure) { f.Definition = mustLookup(t, "e2e_p99_ms") }},
		{"scope", func(f *Figure) {
			f.Scope, f.Member, f.Aggregate, f.Over = ScopePool, "decode", AggregateMax, ScopeInstance
		}},
		{"aggregate", func(f *Figure) { f.Aggregate = AggregateMin }},
		{"over", func(f *Figure) { f.Scope, f.Over = ScopeCluster, ScopePool }},
	} {
		o := worst
		c.edit(&o)
		if err := ComparableFigures(worst, o); err == nil || !strings.Contains(err.Error(), c.field) {
			t.Errorf("%s: got %v, want a refusal naming it", c.field, err)
		}
		if _, err := Compare(worst, o); err == nil {
			t.Errorf("%s: Compare accepted the pair", c.field)
		}
	}
	// A figure computed directly is not comparable with the aggregate of its instances: the
	// cluster's p99 is not the largest instance p99.
	direct := Figure{Definition: ttft, Value: 5, Scope: ScopeCluster}
	if err := ComparableFigures(direct, worst); err == nil || !strings.Contains(err.Error(), "aggregate") {
		t.Errorf("direct against aggregated: got %v", err)
	}
	// A figure with no scope is refused even against another with none, and the refusal says
	// which figure it was.
	if err := ComparableFigures(Figure{Definition: ttft}, Figure{Definition: ttft}); err == nil || !strings.Contains(err.Error(), "first figure") || !strings.Contains(err.Error(), "no scope") {
		t.Errorf("two figures with no scope: got %v", err)
	}
	if err := ComparableFigures(worst, Figure{Definition: ttft}); err == nil || !strings.Contains(err.Error(), "second figure") {
		t.Errorf("a second figure with no scope: got %v", err)
	}
}

// Design 8.7 makes a figure's scope "the whole cluster, one pool, or one instance", and two
// figures compare only when their scopes are the same: instance 3's p99 is not instance 7's.
func TestFiguresOfDifferentMembersAreNotComparable(t *testing.T) {
	ttft := mustLookup(t, "ttft_p99_ms")
	at3 := Figure{Definition: ttft, Value: 5, Scope: ScopeInstance, Member: "instance-3"}
	again := at3
	again.Value = 6
	if d, err := Compare(at3, again); err != nil || d != 1 {
		t.Fatalf("the same instance in two runs: got %v, %v", d, err)
	}
	at7 := at3
	at7.Member = "instance-7"
	if err := ComparableFigures(at3, at7); err == nil || !strings.Contains(err.Error(), "member") {
		t.Errorf("instance 3 against instance 7: got %v, want a refusal naming member", err)
	}
	// An aggregate over the instances of one pool is that pool's figure.
	decode := Figure{Definition: ttft, Scope: ScopePool, Member: "decode", Aggregate: AggregateMax, Over: ScopeInstance}
	prefill := decode
	prefill.Member = "prefill"
	if err := decode.Validate(); err != nil {
		t.Fatalf("the largest p99 over the instances of one pool: %v", err)
	}
	if err := ComparableFigures(decode, prefill); err == nil || !strings.Contains(err.Error(), "member") {
		t.Errorf("two pools: got %v, want a refusal naming member", err)
	}
}

func TestFigureMemberFollowsItsScope(t *testing.T) {
	ttft := mustLookup(t, "ttft_p99_ms")
	for name, c := range map[string]struct {
		f    Figure
		want string
	}{
		"a pool figure with no member":      {Figure{Definition: ttft, Scope: ScopePool}, "names no member"},
		"an instance figure with no member": {Figure{Definition: ttft, Scope: ScopeInstance}, "names no member"},
		"a cluster figure with a member":    {Figure{Definition: ttft, Scope: ScopeCluster, Member: "decode"}, "only a pool or an instance has one"},
	} {
		if err := c.f.Validate(); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %v, want %q", name, err, c.want)
		}
	}
}
