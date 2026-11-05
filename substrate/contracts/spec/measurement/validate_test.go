package measurement

import (
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/router"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
	"gopkg.in/yaml.v3"
)

func digest(c string) string { return strings.Repeat(c, 64) }

func intp(v int) *int        { return &v }
func f64(v float64) *float64 { return &v }
func i64(v int64) *int64     { return &v }
func boolp(v bool) *bool     { return &v }

// exactRun is a run with nothing to keep it from EXACT except one UNSTATED setting the
// engine's default resolves, so it computes to EXACT/RESOLVED under a permitting licence.
func exactRun() *MeasuredRun {
	return &MeasuredRun{
		Kind: "MeasuredRun", Name: "synthetic-h100-tp4", SchemaVersion: SchemaVersion,
		Source: Source{
			Kind:    SourceVLLMBench,
			Adapter: Adapter{Name: "tools/validation/adapters/vllm_bench", Version: "1", SHA256: digest("a")},
			Inputs: []Input{
				{Role: RoleReport, URL: "https://example.org/results.json", SHA256: digest("b"),
					Size: 2048, FetchedUTC: "2026-10-15T13:58:06Z"},
				{Role: RoleManifest, Repo: "example/synthetic-fixtures",
					Commit: "0123abcd", Path: "serving-tests.json",
					SHA256: digest("c")},
			},
			Extraction:   Extraction{Method: ExtractJSON, Selectors: []string{"median_ttft_ms"}, LabelVerified: true},
			RetrievedUTC: "2026-10-15T13:58:06Z",
			LicenseRef:   "synthetic-fixture",
		},
		Model: "gpt-oss-120b", EngineVersion: "0.31.0",
		Cluster: &scenario.Cluster{Hardware: "h100", Nodes: 1, GPUsPerNode: 4},
		Deployment: &deployment.Deployment{Kind: "Deployment", Name: "tp4", Pools: []deployment.Pool{{
			Role: deployment.RoleColocated, Nodes: 1,
			Parallel: deployment.Parallelism{TP: 4, PP: 1, DP: 1},
			Engine:   engineBlock("{max_model_len: 8192}")}}},
		Router: &router.Router{Kind: router.KindDirect},
		Workload: &workload.Binding{Trace: &workload.TraceRef{Data: "trace.csv", SHA256: digest("d"),
			Header: workload.TraceHeader{Version: 2, TimeUnit: "us", Mode: "generated"}}},
		Client: ClientSpec{Harness: "vllm bench serve", HarnessVersion: "0.31.0",
			ImageDigest: "vllm/vllm-openai@sha256:" + digest("e"),
			Command:     []string{"vllm", "bench", "serve", "--seed", "0"},
			Seed:        i64(0), NumPrompts: 200, Burstiness: f64(1), IgnoreEOS: boolp(false),
			Endpoint: "completions", Tokenizer: &Tokenizer{Repo: "openai/gpt-oss-120b", Revision: "b5c939de"}},
		MetricDefs: map[string]MetricDef{
			"ttft_p50": {Formula: FormulaTTFT, Statistic: StatMedian, Population: PopulationPerRequest,
				Clock: ClockClient, StartEvent: StartSendAfterSemaphore, StopEvent: StopFirstContentChunk,
				PercentileMethod: PercentileNumpyLinear, Unit: UnitMs, Cite: "vllm/benchmarks/serve.py:572"},
			"itl_p99": {Formula: FormulaITL, Statistic: "p99", Population: PopulationPooledChunks,
				Clock: ClockClient, PercentileMethod: PercentileNumpyLinear, Unit: UnitMs,
				Cite: "vllm/benchmarks/serve.py:572"},
			"output_tps": {Formula: FormulaOutputThroughput, Population: PopulationPooledTokens,
				Denominator: DenominatorMainPhaseWall, Clock: ClockClient,
				OutputLenSource: OutputLenUsage, Unit: UnitTokensPerSec, Cite: "vllm/benchmarks/serve.py:572"},
		},
		Points: []Point{
			{ID: "c1", Load: Load{Concurrency: 1}, Completed: intp(200), Failed: intp(0),
				Metrics: map[string]Value{"ttft_p50": {Value: 100}, "itl_p99": {Value: 20},
					"output_tps": {Value: 50}},
				KVUtilization: f64(0.1), PrefixHitRate: f64(0), Noise: &Noise{Seeds: 3, CV: 0.05}},
			{ID: "c4", Load: Load{Concurrency: 4},
				Metrics: map[string]Value{"ttft_p50": {Value: 150}, "itl_p99": {Value: 25},
					"output_tps": {Value: 180}},
				Realized: &Realized{MeanISL: 1024, MeanOSL: 1024}},
		},
		Provenance: []ProvenanceRow{
			{Setting: "tensor_parallel_size", Report: &Observed{Value: "4", Ref: "log:12"},
				BLIS: &Observed{Value: "4"}, Verdict: RowMatch, Material: MaterialYes},
			{Setting: "block_size", BLIS: &Observed{Value: "16"}, Verdict: RowUnstated,
				ResolvedBy: "vllm/config/cache.py: block_size default 16 at v0.31.0", Material: MaterialYes},
		},
	}
}

func permitting() *Registry {
	return &Registry{Kind: "MeasurementSources", Sources: []License{
		{ID: "synthetic-fixture", Title: "Synthetic test fixture; every value is invented, none is measured", License: "Apache-2.0", Status: LicenseVerified,
			Permits:  []Use{UseScore, UseDeriveCoefficients, UseRedistribute},
			Evidence: []Evidence{{URL: "https://github.com/inference-sim/blis-schemas/blob/main/LICENSE", Quote: "Apache License", Location: "LICENSE:1"}}},
		{ID: "hud", Title: "PyTorch HUD", Status: LicenseUnverified, Note: "no terms found"},
	}}
}

func TestValidRunPasses(t *testing.T) {
	r := exactRun()
	if p := r.Validate(); !p.OK() {
		t.Fatalf("a valid run was rejected:\n%s", p.Error())
	}
	if p := r.ValidateWithSources(permitting()); !p.OK() {
		t.Fatalf("a valid run was rejected against its registry:\n%s", p.Error())
	}
	a := Assess(r, true)
	if a.Verdict != VerdictExact || a.Resolution != ResolutionResolved {
		t.Fatalf("computed %s/%s, want EXACT/RESOLVED: %v", a.Verdict, a.Resolution, a.Reasons)
	}
}

// TestRejects names one malformed field per case and the path the problem must be filed
// at, so a check that fires for the wrong reason does not pass the case.
func TestRejects(t *testing.T) {
	cases := []struct {
		name, path string
		mutate     func(*MeasuredRun)
	}{
		{"wrong kind", "kind", func(r *MeasuredRun) { r.Kind = "EvaluationRun" }},
		{"no name", "name", func(r *MeasuredRun) { r.Name = "" }},
		{"name not a slug", "name", func(r *MeasuredRun) { r.Name = "Has Spaces" }},
		{"wrong schema version", "schema_version", func(r *MeasuredRun) { r.SchemaVersion = 2 }},
		{"unknown source kind", "source.kind", func(r *MeasuredRun) { r.Source.Kind = "rumour" }},
		{"no adapter name", "source.adapter.name", func(r *MeasuredRun) { r.Source.Adapter.Name = "" }},
		{"no adapter version", "source.adapter.version", func(r *MeasuredRun) { r.Source.Adapter.Version = "" }},
		{"no adapter digest", "source.adapter.sha256", func(r *MeasuredRun) { r.Source.Adapter.SHA256 = "" }},
		{"no inputs", "source.inputs", func(r *MeasuredRun) { r.Source.Inputs = nil }},
		{"no report input", "source.inputs", func(r *MeasuredRun) { r.Source.Inputs[0].Role = RoleRawArtifact }},
		{"unknown input role", "source.inputs[1].role", func(r *MeasuredRun) { r.Source.Inputs[1].Role = "vibes" }},
		{"input without sha256", "source.inputs[0].sha256", func(r *MeasuredRun) { r.Source.Inputs[0].SHA256 = "" }},
		{"input with short sha256", "source.inputs[0].sha256", func(r *MeasuredRun) { r.Source.Inputs[0].SHA256 = "abc" }},
		{"input with uppercase sha256", "source.inputs[0].sha256", func(r *MeasuredRun) {
			r.Source.Inputs[0].SHA256 = strings.ToUpper(digest("b"))
		}},
		{"input with no locator", "source.inputs[1]", func(r *MeasuredRun) { r.Source.Inputs[1].Path = "" }},
		{"input with a non-http url", "source.inputs[0].url", func(r *MeasuredRun) { r.Source.Inputs[0].URL = "file:///etc/passwd" }},
		{"negative input size", "source.inputs[0].size", func(r *MeasuredRun) { r.Source.Inputs[0].Size = -1 }},
		{"local fetch time", "source.inputs[0].fetched_utc", func(r *MeasuredRun) {
			r.Source.Inputs[0].FetchedUTC = "2026-10-15T09:58:06-04:00"
		}},
		{"unknown extraction", "source.extraction.method", func(r *MeasuredRun) { r.Source.Extraction.Method = "eval" }},
		{"bad retrieval time", "source.retrieved_utc", func(r *MeasuredRun) { r.Source.RetrievedUTC = "yesterday" }},
		{"no licence ref", "source.license_ref", func(r *MeasuredRun) { r.Source.LicenseRef = "" }},
		{"invalid cluster", "cluster.gpus_per_node", func(r *MeasuredRun) { r.Cluster.GPUsPerNode = 0 }},
		{"invalid deployment", "deployment.pools[0].parallel.tp", func(r *MeasuredRun) { r.Deployment.Pools[0].Parallel.TP = 0 }},
		{"deployment overfills cluster", "deployment.pools[0].parallel", func(r *MeasuredRun) { r.Deployment.Pools[0].Parallel.TP = 8 }},
		{"invalid router", "router.kind", func(r *MeasuredRun) { r.Router.Kind = "haproxy" }},
		{"invalid workload", "workload.trace.data", func(r *MeasuredRun) { r.Workload.Trace.Data = "" }},
		{"unpinned image", "client.image_digest", func(r *MeasuredRun) { r.Client.ImageDigest = "vllm/vllm-openai:latest" }},
		{"negative prompts", "client.num_prompts", func(r *MeasuredRun) { r.Client.NumPrompts = -1 }},
		{"negative warmups", "client.num_warmups", func(r *MeasuredRun) { r.Client.NumWarmups = -1 }},
		{"zero burstiness", "client.burstiness", func(r *MeasuredRun) { r.Client.Burstiness = f64(0) }},
		{"NaN burstiness", "client.burstiness", func(r *MeasuredRun) { r.Client.Burstiness = f64(math.NaN()) }},
		{"unknown ramp", "client.ramp_up.strategy", func(r *MeasuredRun) {
			r.Client.RampUp = &RampUp{Strategy: "cubic", Start: 1, End: 2}
		}},
		{"negative ramp end", "client.ramp_up.end", func(r *MeasuredRun) {
			r.Client.RampUp = &RampUp{Strategy: RampLinear, Start: 1, End: -2}
		}},
		{"tokenizer without repo", "client.tokenizer.repo", func(r *MeasuredRun) { r.Client.Tokenizer.Repo = "" }},
		{"tokenizer with bad digest", "client.tokenizer.sha256", func(r *MeasuredRun) { r.Client.Tokenizer.SHA256 = "zz" }},
		{"negative duration", "client.duration_s", func(r *MeasuredRun) { r.Client.DurationSec = -1 }},
		{"no metric defs", "metric_defs", func(r *MeasuredRun) { r.MetricDefs = nil }},
		{"metric name not a slug", "metric_defs.TTFT", func(r *MeasuredRun) {
			r.MetricDefs["TTFT"] = r.MetricDefs["ttft_p50"]
		}},
		{"unknown formula", "metric_defs.ttft_p50.formula", func(r *MeasuredRun) { setDef(r, "ttft_p50", func(d *MetricDef) { d.Formula = "vibes" }) }},
		{"unknown statistic", "metric_defs.ttft_p50.statistic", func(r *MeasuredRun) { setDef(r, "ttft_p50", func(d *MetricDef) { d.Statistic = "p100" }) }},
		{"simulated clock", "metric_defs.ttft_p50.clock", func(r *MeasuredRun) { setDef(r, "ttft_p50", func(d *MetricDef) { d.Clock = ClockSimulated }) }},
		{"throughput with a statistic", "metric_defs.output_tps.statistic", func(r *MeasuredRun) {
			setDef(r, "output_tps", func(d *MetricDef) { d.Statistic = StatMean })
		}},
		{"latency with a denominator", "metric_defs.ttft_p50.denominator", func(r *MeasuredRun) {
			setDef(r, "ttft_p50", func(d *MetricDef) { d.Denominator = DenominatorPerGPU })
		}},
		{"percentile without a method", "metric_defs.itl_p99.percentile_method", func(r *MeasuredRun) {
			setDef(r, "itl_p99", func(d *MetricDef) { d.PercentileMethod = "" })
		}},
		{"unknown population", "metric_defs.itl_p99.population", func(r *MeasuredRun) {
			setDef(r, "itl_p99", func(d *MetricDef) { d.Population = "pooled_vibes" })
		}},
		{"unknown clock", "metric_defs.itl_p99.clock", func(r *MeasuredRun) { setDef(r, "itl_p99", func(d *MetricDef) { d.Clock = "wall" }) }},
		{"unknown unit", "metric_defs.itl_p99.unit", func(r *MeasuredRun) { setDef(r, "itl_p99", func(d *MetricDef) { d.Unit = "furlongs" }) }},
		{"unit not the formula's", "metric_defs.itl_p99.unit", func(r *MeasuredRun) {
			setDef(r, "itl_p99", func(d *MetricDef) { d.Unit = UnitTokensPerSec })
		}},
		{"no cite", "metric_defs.itl_p99.cite", func(r *MeasuredRun) { setDef(r, "itl_p99", func(d *MetricDef) { d.Cite = "" }) }},
		{"no points", "points", func(r *MeasuredRun) { r.Points = nil }},
		{"point without id", "points[1].id", func(r *MeasuredRun) { r.Points[1].ID = "" }},
		{"duplicate point id", "points[1].id", func(r *MeasuredRun) { r.Points[1].ID = "c1" }},
		{"point without load", "points[1].load", func(r *MeasuredRun) { r.Points[1].Load = Load{} }},
		{"negative rate", "points[1].load.request_rate", func(r *MeasuredRun) { r.Points[1].Load.RequestRate = -1 }},
		{"mixed load modes", "points[1].load", func(r *MeasuredRun) { r.Points[1].Load = Load{RequestRate: 4} }},
		{"points out of order", "points[1].load", func(r *MeasuredRun) { r.Points[1].Load.Concurrency = 1 }},
		{"point without metrics", "points[1].metrics", func(r *MeasuredRun) { r.Points[1].Metrics = nil }},
		{"metric without a definition", "points[1].metrics.e2e", func(r *MeasuredRun) {
			r.Points[1].Metrics["e2e"] = Value{Value: 1}
		}},
		{"negative metric", "points[0].metrics.ttft_p50.value", func(r *MeasuredRun) {
			r.Points[0].Metrics["ttft_p50"] = Value{Value: -1}
		}},
		{"infinite metric", "points[0].metrics.ttft_p50.value", func(r *MeasuredRun) {
			r.Points[0].Metrics["ttft_p50"] = Value{Value: math.Inf(1)}
		}},
		{"negative completed", "points[0].completed", func(r *MeasuredRun) { r.Points[0].Completed = intp(-1) }},
		{"negative preemptions", "points[0].preemptions", func(r *MeasuredRun) { r.Points[0].Preemptions = intp(-1) }},
		{"utilization above one", "points[0].kv_utilization", func(r *MeasuredRun) { r.Points[0].KVUtilization = f64(1.5) }},
		{"NaN hit rate", "points[0].prefix_hit_rate", func(r *MeasuredRun) { r.Points[0].PrefixHitRate = f64(math.NaN()) }},
		{"negative realized length", "points[1].realized.mean_osl", func(r *MeasuredRun) { r.Points[1].Realized.MeanOSL = -1 }},
		{"one-seed noise", "points[0].noise.seeds", func(r *MeasuredRun) { r.Points[0].Noise.Seeds = 1 }},
		{"negative noise", "points[0].noise.cv", func(r *MeasuredRun) { r.Points[0].Noise.CV = -0.1 }},
		{"row without setting", "provenance[0].setting", func(r *MeasuredRun) { r.Provenance[0].Setting = "" }},
		{"duplicate setting", "provenance[1].setting", func(r *MeasuredRun) { r.Provenance[1].Setting = r.Provenance[0].Setting }},
		{"unknown row verdict", "provenance[0].verdict", func(r *MeasuredRun) { r.Provenance[0].Verdict = "MAYBE" }},
		{"MATCH with one source", "provenance[0].verdict", func(r *MeasuredRun) { r.Provenance[0].BLIS = nil }},
		{"UNSTATED with two sources", "provenance[1].verdict", func(r *MeasuredRun) {
			r.Provenance[1].Report = &Observed{Value: "16", Ref: "a"}
			r.Provenance[1].Script = &Observed{Value: "16", Ref: "b"}
		}},
		{"UNVERIFIABLE with an artifact", "provenance[0].verdict", func(r *MeasuredRun) {
			r.Provenance[0].Verdict = RowUnverifiable
			r.Provenance[0].Script = &Observed{Value: "8", Ref: "s"}
			r.Provenance[0].Artifact = &Observed{Value: "4", Ref: "a"}
		}},
		{"observation without ref", "provenance[0].report.ref", func(r *MeasuredRun) { r.Provenance[0].Report.Ref = "" }},
		{"observation without value", "provenance[0].report.value", func(r *MeasuredRun) { r.Provenance[0].Report.Value = "" }},
		{"resolved MATCH", "provenance[0].resolved_by", func(r *MeasuredRun) { r.Provenance[0].ResolvedBy = "a rule" }},
		{"unknown materiality", "provenance[0].material", func(r *MeasuredRun) { r.Provenance[0].Material = "huge" }},
		{"bounded without bound", "provenance[0].bound", func(r *MeasuredRun) { r.Provenance[0].Material = MaterialBounded }},
		{"bound without bounded", "provenance[0].bound", func(r *MeasuredRun) { r.Provenance[0].Bound = "5%" }},
		{"diff without path", "diff[0].path", func(r *MeasuredRun) { r.Diff = []DiffEntry{{Kind: DiffNotModelled, Material: MaterialYes}} }},
		{"unknown diff kind", "diff[0].kind", func(r *MeasuredRun) { r.Diff = []DiffEntry{{Path: "x", Kind: "meh", Material: MaterialYes}} }},
		{"removable without owner", "diff[0].owner_repo", func(r *MeasuredRun) {
			r.Diff = []DiffEntry{{Path: "x", Kind: DiffNotModelled, Removable: true, Material: MaterialYes}}
		}},
		{"diff without materiality", "diff[0].material", func(r *MeasuredRun) { r.Diff = []DiffEntry{{Path: "x", Kind: DiffNotModelled}} }},
		{"unknown verdict", "verdict", func(r *MeasuredRun) { r.Verdict = "GREAT" }},
		{"unknown resolution", "resolution", func(r *MeasuredRun) { r.Verdict, r.Resolution = VerdictExact, "LOOSE" }},
		{"EXACT without resolution", "resolution", func(r *MeasuredRun) { r.Verdict = VerdictExact }},
		{"self duplicate", "duplicate_of", func(r *MeasuredRun) { r.DuplicateOf = r.Name }},
		{"self supersede", "supersedes", func(r *MeasuredRun) { r.Supersedes = r.Name }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := exactRun()
			tc.mutate(r)
			requireProblemAt(t, r.ValidateWithSources(permitting()).Errors(), tc.path)
		})
	}
}

func setDef(r *MeasuredRun, name string, f func(*MetricDef)) {
	d := r.MetricDefs[name]
	f(&d)
	r.MetricDefs[name] = d
}

func requireProblemAt[P interface{ String() string }](t *testing.T, problems []P, path string) {
	t.Helper()
	var all []string
	for _, p := range problems {
		s := p.String()
		all = append(all, s)
		if strings.Contains(s, ": "+path+": ") {
			return
		}
	}
	t.Fatalf("expected a problem at %s, got:\n%s", path, strings.Join(all, "\n"))
}

// TestSecretsAreRefused plants each kind of credential in a field a manifest or argv
// would fill, and requires the run to be refused at that field.
func TestSecretsAreRefused(t *testing.T) {
	for _, s := range []string{
		"HF_TOKEN=hf_" + strings.Repeat("x", 34),
		"--api-key hf_" + strings.Repeat("A", 34),
		"AKIA" + strings.Repeat("Q", 16),
		"ghp_" + strings.Repeat("z", 36),
		"Authorization: Bearer " + strings.Repeat("t", 40),
		"-----BEGIN RSA PRIVATE KEY-----",
		"valueFrom: {secretKeyRef: {name: hf}}",
		"--kubeconfig=/home/u/.kube/config",
		"AWS_SECRET_ACCESS_KEY=abc",
	} {
		t.Run(s[:8], func(t *testing.T) {
			r := exactRun()
			r.Client.Command = append(r.Client.Command, s)
			requireProblemAt(t, r.Validate().Errors(), "client.command[5]")
		})
	}
	r := exactRun()
	r.Provenance[0].Note = "token " + "hf_" + strings.Repeat("k", 34)
	requireProblemAt(t, r.Validate().Errors(), "provenance[0].note")
}

func TestStatisticVocabulary(t *testing.T) {
	for _, s := range []Statistic{"mean", "median", "p50", "p99", "p99.9", "p0.1", "p1"} {
		if !s.Valid() {
			t.Errorf("%s should be a statistic", s)
		}
	}
	for _, s := range []Statistic{"p0", "p100", "p099", "p99.", "p99.90", "P99", "avg", ""} {
		if s.Valid() {
			t.Errorf("%q should not be a statistic", s)
		}
	}
}

func TestRegistryRejects(t *testing.T) {
	cases := []struct {
		name, path string
		mutate     func(*Registry)
	}{
		{"wrong kind", "kind", func(r *Registry) { r.Kind = "Licences" }},
		{"no sources", "sources", func(r *Registry) { r.Sources = nil }},
		{"bad id", "sources[0].id", func(r *Registry) { r.Sources[0].ID = "Not A Slug" }},
		{"duplicate id", "sources[1].id", func(r *Registry) { r.Sources[1].ID = r.Sources[0].ID }},
		{"no title", "sources[0].title", func(r *Registry) { r.Sources[0].Title = "" }},
		{"unknown status", "sources[0].status", func(r *Registry) { r.Sources[0].Status = "probably" }},
		{"unknown use", "sources[0].permits[0]", func(r *Registry) { r.Sources[0].Permits[0] = "resell" }},
		{"duplicate use", "sources[0].permits[1]", func(r *Registry) { r.Sources[0].Permits[1] = UseScore }},
		{"verified without licence", "sources[0].license", func(r *Registry) { r.Sources[0].License = "" }},
		{"verified permitting nothing", "sources[0].permits", func(r *Registry) { r.Sources[0].Permits = nil }},
		{"verified without evidence", "sources[0].evidence", func(r *Registry) { r.Sources[0].Evidence = nil }},
		{"unverified permitting a use", "sources[1].permits", func(r *Registry) { r.Sources[1].Permits = []Use{UseScore} }},
		{"unverified without note", "sources[1].note", func(r *Registry) { r.Sources[1].Note = "" }},
		{"refused permitting a use", "sources[1].permits", func(r *Registry) {
			r.Sources[1].Status, r.Sources[1].Permits = LicenseRefused, []Use{UseRedistribute}
		}},
		{"evidence without url", "sources[0].evidence[0].url", func(r *Registry) { r.Sources[0].Evidence[0].URL = "LICENSE" }},
		{"evidence without quote", "sources[0].evidence[0].quote", func(r *Registry) { r.Sources[0].Evidence[0].Quote = "" }},
		{"evidence without location", "sources[0].evidence[0].location", func(r *Registry) { r.Sources[0].Evidence[0].Location = "" }},
	}
	if p := permitting().Validate(); !p.OK() {
		t.Fatalf("the fixture registry was rejected:\n%s", p.Error())
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reg := permitting()
			tc.mutate(reg)
			requireProblemAt(t, reg.Validate().Errors(), tc.path)
		})
	}
}

func TestRegistryPermits(t *testing.T) {
	reg := permitting()
	if !reg.Permits("synthetic-fixture", UseScore) {
		t.Error("a verified source should permit what it lists")
	}
	if reg.Permits("hud", UseScore) {
		t.Error("an unverified source permits nothing")
	}
	if reg.Permits("absent", UseScore) {
		t.Error("an unknown source permits nothing")
	}
	// An invalid registry that lists a use for an unverified source still grants nothing:
	// the gate does not depend on the registry having validated.
	reg.Sources[1].Permits = []Use{UseScore}
	if reg.Permits("hud", UseScore) {
		t.Error("an unverified source permits nothing, whatever it lists")
	}
	reg.Sources[0].Permits = []Use{UseScore}
	if reg.Permits("synthetic-fixture", UseRedistribute) {
		t.Error("a verified source permits only what it lists")
	}
}

func TestUnknownLicenceRefIsRejected(t *testing.T) {
	r := exactRun()
	r.Source.LicenseRef = "nowhere"
	requireProblemAt(t, r.ValidateWithSources(permitting()).Errors(), "source.license_ref")
}

func TestCasesTagTheAnchor(t *testing.T) {
	r := exactRun()
	cs := r.Cases(Assess(r, true))
	if len(cs) != 2 || !cs[0].Anchor || cs[1].Anchor {
		t.Fatalf("the lowest load, and only it, is the anchor: %+v", cs)
	}
	if cs[1].Verdict != VerdictExact || cs[1].Run != r.Name || cs[1].Point != "c4" {
		t.Errorf("a case carries its run's verdict and its point: %+v", cs[1])
	}
}

func TestVocabulariesAreClosed(t *testing.T) {
	cases := map[string]struct{ got, want []string }{
		"verdicts":      {AllVerdicts(), []string{"EXACT", "NOT_REPRODUCIBLE", "PARTIAL"}},
		"resolutions":   {AllResolutions(), []string{"RESOLVED", "STRICT"}},
		"row verdicts":  {AllRowVerdicts(), []string{"MATCH", "MISMATCH", "UNSTATED", "UNVERIFIABLE"}},
		"diff kinds":    {AllDiffKinds(), []string{"approximated", "conflict", "not_expressible", "not_modelled", "unresolved"}},
		"materiality":   {AllMaterialities(), []string{"bounded", "no", "unknown", "yes"}},
		"populations":   {AllPopulations(), []string{"per_request", "pooled_chunks", "pooled_tokens"}},
		"licence state": {AllLicenseStatuses(), []string{"refused", "unverified", "verified"}},
		"uses":          {AllUses(), []string{"derive_coefficients", "redistribute", "score"}},
	}
	for name, c := range cases {
		if strings.Join(c.got, ",") != strings.Join(c.want, ",") {
			t.Errorf("%s = %v, want %v", name, c.got, c.want)
		}
	}
}

// engineBlock is a pool's engine block as a document states it, held unopened.
func engineBlock(doc string) deployment.EngineBlock {
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(doc), &n); err != nil {
		panic(err)
	}
	// A block stated in flow style is written back in block style, as a document holds it.
	unflow(n.Content[0])
	return deployment.NewEngineBlock(n.Content[0])
}

func unflow(n *yaml.Node) {
	n.Style &^= yaml.FlowStyle
	for _, c := range n.Content {
		unflow(c)
	}
}

// TestValidateMetricDefPathsAreRelative: a document outside this package merges the
// problems under its own location, so each path must start at the definition's fields.
func TestValidateMetricDefPathsAreRelative(t *testing.T) {
	ok := MetricDef{Formula: FormulaTTFT, Statistic: "p99", Population: PopulationPerRequest,
		Clock: ClockSimulated, PercentileMethod: PercentileNumpyLinear, Unit: UnitMs, Cite: "x"}
	if p := ValidateMetricDef(ok); !p.OK() {
		t.Fatalf("a valid definition on the simulated clock must pass:\n%s", p.Error())
	}
	bad := ok
	bad.PercentileMethod = ""
	bad.Cite = ""
	paths := map[string]bool{}
	for _, it := range ValidateMetricDef(bad).All() {
		paths[it.Path] = true
	}
	for _, want := range []string{"percentile_method", "cite"} {
		if !paths[want] {
			t.Errorf("expected a problem at %q, got paths %v", want, paths)
		}
	}
}
