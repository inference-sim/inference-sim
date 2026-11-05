package workload

import (
	"math"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func fl(x float64) *float64 { return &x }
func bl(b bool) *bool       { return &b }

// agentX is the run issue #71 names: aiperf 0.12.0's inferencex-agentx-mvp over the 256k
// Weka corpus at a pinned revision.
func agentX() *Replay {
	return &Replay{Concurrency: 64, ReplayDefinition: agentXDefinition()}
}

func agentXDefinition() ReplayDefinition {
	return ReplayDefinition{
		Scenario: ScenarioInferenceXAgentXMVP, HarnessVersion: "0.12.0",
		Corpus: Corpus{
			Loader: "semianalysis_cc_traces_weka_062126_256k", Format: CorpusWekaTraceV7,
			Encoding: CorpusJSONL,
			Source: CorpusSource{HuggingFace: "semianalysisai/cc-traces-weka-062126-256k",
				Revision: "8fecd2fc", File: "traces.jsonl", Split: "train"},
			SHA256: "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef",
		},
		BenchmarkDurationS: 900, MaxContextLength: ContextLimitMaxModelLen,
		Category: "coding", Thinking: "on",
	}
}

func TestReplayValid(t *testing.T) {
	if p := agentX().Validate(); !p.OK() {
		t.Fatalf("a valid replay was rejected:\n%s", p.Error())
	}
	// Every locked knob stated at the value the scenario requires is still valid.
	r := agentX()
	r.CacheBust, r.SystemIdleGapCapS = CacheBustFirstTurnPrefix, fl(10)
	r.Streaming, r.IgnoreEOS, r.IgnoreTraceDelays = bl(true), bl(true), bl(false)
	r.TrajectoryStart = &TrajectoryStart{MinRatio: fl(0.25), MaxRatio: fl(0.75)}
	r.Warmup = &Warmup{Mode: WarmupCachePressure, DurationS: 600}
	r.GracePeriodS = fl(math.Inf(1))
	if p := r.Validate(); !p.OK() {
		t.Fatalf("the scenario's own values were rejected:\n%s", p.Error())
	}
	// The InferenceX fork run states the per-trace idle-gap cap and a request-bounded
	// cache-pressure warmup, both recorded version extensions of the 0.12.0 lock.
	r = agentX()
	r.TraceIdleGapCapS = fl(300)
	r.Warmup = &Warmup{Mode: WarmupCachePressure, RequestsPerLane: 10}
	if p := r.Validate(); !p.OK() {
		t.Fatalf("the fork run's extensions were rejected:\n%s", p.Error())
	}
}

// Omitted fields read as what aiperf applies for the scenario.
func TestReplayDefaults(t *testing.T) {
	r := &Replay{}
	lo, hi := r.StartWindow()
	if r.Seed() != 42 || r.Duration() != 1800 || r.Grace() != 30 || r.IdleCap() != 10 ||
		lo != 0 || hi != 1 || r.WarmupMode() != WarmupSynthesized || !r.SplitsFlattenedAgents() {
		t.Errorf("defaults: seed %d duration %v grace %v idle %v window [%v, %v] warmup %s split %v",
			r.Seed(), r.Duration(), r.Grace(), r.IdleCap(), lo, hi, r.WarmupMode(), r.SplitsFlattenedAgents())
	}
}

func TestReplayRefuses(t *testing.T) {
	cases := []struct {
		name   string
		field  string
		mutate func(*Replay)
	}{
		{"no scenario", "scenario", func(r *Replay) { r.Scenario = "" }},
		{"unknown scenario", "scenario", func(r *Replay) { r.Scenario = "agentx-full" }},
		{"no harness version", "harness_version", func(r *Replay) { r.HarnessVersion = "" }},
		{"untranscribed harness version", "harness_version", func(r *Replay) { r.HarnessVersion = "0.13.0" }},
		{"no lanes", "concurrency", func(r *Replay) { r.Concurrency = 0 }},
		{"duration under the floor", "benchmark_duration_s", func(r *Replay) { r.BenchmarkDurationS = 600 }},
		{"negative duration", "benchmark_duration_s", func(r *Replay) { r.BenchmarkDurationS = -1 }},
		{"NaN duration", "benchmark_duration_s", func(r *Replay) { r.BenchmarkDurationS = math.NaN() }},
		{"negative grace", "grace_period_s", func(r *Replay) { r.GracePeriodS = fl(-1) }},
		{"other cache bust", "cache_bust", func(r *Replay) { r.CacheBust = "every_turn" }},
		{"other idle cap", "system_idle_gap_cap_s", func(r *Replay) { r.SystemIdleGapCapS = fl(300) }},
		{"start ratio above one", "trajectory_start.max_ratio", func(r *Replay) {
			r.TrajectoryStart = &TrajectoryStart{MaxRatio: fl(1.5)}
		}},
		{"inverted start window", "trajectory_start", func(r *Replay) {
			r.TrajectoryStart = &TrajectoryStart{MinRatio: fl(0.8), MaxRatio: fl(0.2)}
		}},
		{"unknown warmup", "warmup.mode", func(r *Replay) { r.Warmup = &Warmup{Mode: "ten_per_lane"} }},
		{"cache pressure with neither bound", "warmup", func(r *Replay) {
			r.Warmup = &Warmup{Mode: WarmupCachePressure}
		}},
		{"cache pressure with both bounds", "warmup", func(r *Replay) {
			r.Warmup = &Warmup{Mode: WarmupCachePressure, DurationS: 600, RequestsPerLane: 10}
		}},
		{"negative warmup duration", "warmup.duration_s", func(r *Replay) {
			r.Warmup = &Warmup{Mode: WarmupCachePressure, DurationS: -1}
		}},
		{"NaN warmup duration", "warmup.duration_s", func(r *Replay) {
			r.Warmup = &Warmup{Mode: WarmupCachePressure, DurationS: math.NaN()}
		}},
		{"Inf warmup duration", "warmup.duration_s", func(r *Replay) {
			r.Warmup = &Warmup{Mode: WarmupCachePressure, DurationS: math.Inf(1)}
		}},
		{"non-positive per-lane count", "warmup.requests_per_lane", func(r *Replay) {
			r.Warmup = &Warmup{Mode: WarmupCachePressure, RequestsPerLane: -1}
		}},
		{"synthesized with duration", "warmup.duration_s", func(r *Replay) {
			r.Warmup = &Warmup{Mode: WarmupSynthesized, DurationS: 60}
		}},
		{"synthesized with per-lane count", "warmup.requests_per_lane", func(r *Replay) {
			r.Warmup = &Warmup{Mode: WarmupSynthesized, RequestsPerLane: 10}
		}},
		{"no streaming", "streaming", func(r *Replay) { r.Streaming = bl(false) }},
		{"eos honoured", "ignore_eos", func(r *Replay) { r.IgnoreEOS = bl(false) }},
		{"trace delays ignored", "ignore_trace_delays", func(r *Replay) { r.IgnoreTraceDelays = bl(true) }},
		{"input truncation", "synthesis_max_isl", func(r *Replay) { r.SynthesisMaxISL = 131072 }},
		{"negative per-trace idle cap", "trace_idle_gap_cap_s", func(r *Replay) { r.TraceIdleGapCapS = fl(-1) }},
		{"zero per-trace idle cap", "trace_idle_gap_cap_s", func(r *Replay) { r.TraceIdleGapCapS = fl(0) }},
		{"NaN per-trace idle cap", "trace_idle_gap_cap_s", func(r *Replay) { r.TraceIdleGapCapS = fl(math.NaN()) }},
		{"per-turn delay cap", "inter_turn_delay_cap_s", func(r *Replay) { r.InterTurnDelayCapS = fl(60) }},
		{"malformed category", "category", func(r *Replay) { r.Category = "Coding" }},
		{"unknown thinking", "thinking", func(r *Replay) { r.Thinking = "high" }},
		{"no loader", "corpus.loader", func(r *Replay) { r.Corpus.Loader = "" }},
		{"loader the scenario refuses", "corpus.loader", func(r *Replay) { r.Corpus.Loader = "sharegpt" }},
		{"unpinned local corpus", "corpus.loader", func(r *Replay) { r.Corpus.Loader = "weka_trace" }},
		{"weka_hf on another repository", "corpus.source.huggingface", func(r *Replay) { r.Corpus.Loader = "weka_hf" }},
		{"alias on another repository", "corpus.source.huggingface", func(r *Replay) {
			r.Corpus.Source.HuggingFace = "semianalysisai/cc-traces-weka-062126"
		}},
		{"unknown format", "corpus.format", func(r *Replay) { r.Corpus.Format = "weka-trace-v6" }},
		{"unknown encoding", "corpus.encoding", func(r *Replay) { r.Corpus.Encoding = "parquet" }},
		{"no repository", "corpus.source.huggingface", func(r *Replay) { r.Corpus.Source.HuggingFace = "" }},
		{"no revision", "corpus.source.revision", func(r *Replay) { r.Corpus.Source.Revision = "" }},
		{"no digest", "corpus.sha256", func(r *Replay) { r.Corpus.SHA256 = "" }},
		{"short digest", "corpus.sha256", func(r *Replay) { r.Corpus.SHA256 = "e39cd2ff" }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := agentX()
			c.mutate(r)
			p := r.Validate()
			for _, it := range p.Errors() {
				if it.Path == c.field {
					return
				}
			}
			t.Errorf("no error at %s:\n%s", c.field, p.Error())
		})
	}
}

// weka_hf is accepted when it names the one repository the scenario allows.
func TestReplayWekaHFPinnedRepository(t *testing.T) {
	r := agentX()
	r.Corpus.Loader, r.Corpus.Source.HuggingFace = "weka_hf", "semianalysisai/cc-traces-weka-062126"
	if p := r.Validate(); !p.OK() {
		t.Fatalf("weka_hf on the allowed repository was rejected:\n%s", p.Error())
	}
}

func TestContextLimit(t *testing.T) {
	for _, c := range []struct {
		yaml string
		want ContextLimit
		ok   bool
	}{
		{"262144", 262144, true},
		{"max_model_len", ContextLimitMaxModelLen, true},
		{"0", 0, false},
		{"-5", 0, false},
		{"auto", 0, false},
		{"[1]", 0, false},
	} {
		var got struct {
			L ContextLimit `yaml:"l"`
		}
		err := yaml.Unmarshal([]byte("l: "+c.yaml), &got)
		if (err == nil) != c.ok || (c.ok && got.L != c.want) {
			t.Errorf("%s: got %v, %v", c.yaml, got.L, err)
		}
	}
	if n, ok, err := ContextLimitMaxModelLen.Resolve(1048576); n != 1048576 || !ok || err != nil {
		t.Errorf("max_model_len resolved to %d, %v, %v", n, ok, err)
	}
	if _, _, err := ContextLimitMaxModelLen.Resolve(0); err == nil {
		t.Error("max_model_len resolved with no deployment length")
	}
	if n, ok, _ := ContextLimit(0).Resolve(1 << 20); ok || n != 0 {
		t.Error("an unstated limit filtered")
	}
	if n, ok, _ := ContextLimit(4096).Resolve(1 << 20); !ok || n != 4096 {
		t.Error("a stated count did not apply as stated")
	}
	out, err := yaml.Marshal(struct {
		L ContextLimit `yaml:"l"`
	}{ContextLimitMaxModelLen})
	if err != nil || !strings.Contains(string(out), "max_model_len") {
		t.Errorf("marshal: %s %v", out, err)
	}
}

// A binding takes exactly one of its three arms.
func TestBindingReplayArm(t *testing.T) {
	if p := (&Binding{Replay: agentX()}).Validate(); !p.OK() {
		t.Fatalf("replay arm rejected:\n%s", p.Error())
	}
	for name, b := range map[string]*Binding{
		"shape and replay": {Shape: "chatbot", Replay: agentX()},
		"trace and replay": {Trace: traceRef(), Replay: agentX()},
		"invalid replay":   {Replay: &Replay{}},
	} {
		if p := b.Validate(); p.OK() {
			t.Errorf("%s: accepted", name)
		}
	}
}

// A reference states the catalog name, the lane count and run choices only; Resolve lays
// the run choices over the entry's definition.
func TestReplayReference(t *testing.T) {
	entry := &CatalogReplay{Name: "agentx", ReplayDefinition: agentXDefinition()}
	if p := entry.Validate(); !p.OK() {
		t.Fatalf("a catalog replay was rejected:\n%s", p.Error())
	}
	ref := &Replay{Workload: "agentx", Concurrency: 32,
		ReplayDefinition: ReplayDefinition{GracePeriodS: fl(1800), RandomSeed: func() *int64 { s := int64(7); return &s }()}}
	if p := ref.Validate(); !p.OK() {
		t.Fatalf("a reference was rejected:\n%s", p.Error())
	}
	got, err := ref.Resolve(entry)
	if err != nil {
		t.Fatal(err)
	}
	if got.Workload != "" || got.Concurrency != 32 || got.Grace() != 1800 || got.Seed() != 7 ||
		got.Corpus.Source.Revision != "8fecd2fc" || got.Duration() != 900 {
		t.Errorf("resolved to %+v", got)
	}
	if entry.Seed() != 42 {
		t.Error("resolving changed the catalog entry")
	}

	for name, c := range map[string]struct {
		ref   *Replay
		entry *CatalogReplay
	}{
		"no lanes":           {&Replay{Workload: "agentx"}, entry},
		"fixed field stated": {&Replay{Workload: "agentx", Concurrency: 1, ReplayDefinition: ReplayDefinition{CacheBust: CacheBustFirstTurnPrefix}}, entry},
		"short window":       {&Replay{Workload: "agentx", Concurrency: 1, ReplayDefinition: ReplayDefinition{BenchmarkDurationS: 60}}, entry},
		"entry not supplied": {&Replay{Workload: "agentx", Concurrency: 1}, nil},
		"another entry":      {&Replay{Workload: "agentx", Concurrency: 1}, &CatalogReplay{Name: "other", ReplayDefinition: agentXDefinition()}},
		"invalid merged":     {&Replay{Workload: "agentx", Concurrency: 1}, &CatalogReplay{Name: "agentx"}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := c.ref.Resolve(c.entry); err == nil {
				t.Error("resolved")
			}
		})
	}
	if p := (&Replay{Workload: "agentx", Concurrency: 1, ReplayDefinition: ReplayDefinition{Scenario: ScenarioInferenceXAgentXMVP}}).Validate(); p.OK() ||
		!strings.Contains(p.Error(), "scenario") {
		t.Errorf("a reference restating the scenario validated: %s", p.Error())
	}
}

// A catalog entry is one form or the other.
func TestCatalogWorkloadForms(t *testing.T) {
	shape := &CatalogWorkload{Shape: chatbot()}
	replay := &CatalogWorkload{Replay: &CatalogReplay{Name: "agentx", ReplayDefinition: agentXDefinition()}}
	if !shape.Validate().OK() || shape.Name() != "chatbot" || !replay.Validate().OK() || replay.Name() != "agentx" {
		t.Error("a valid entry was rejected or misnamed")
	}
	if (&CatalogWorkload{}).Validate().OK() || (&CatalogWorkload{Shape: chatbot(), Replay: replay.Replay}).Validate().OK() {
		t.Error("an entry with neither or both forms validated")
	}
}

// A rolling alias validates, with a warning that it tracks whatever the default is.
func TestReplayRollingAliasWarns(t *testing.T) {
	r := agentX()
	r.Corpus.Loader = "semianalysis_cc_traces_weka_with_subagents_256k"
	p := r.Validate()
	if !p.OK() {
		t.Fatalf("a rolling alias on its repository was rejected:\n%s", p.Error())
	}
	for _, it := range p.All() {
		if it.Path == "corpus.loader" && it.Severity == "warning" && strings.Contains(it.Message, "rolling alias") {
			return
		}
	}
	t.Errorf("no rolling-alias warning:\n%s", p.Error())
}
