package workload

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// constantLoadYAML is the run the aiperf-constant-load scenario exists for: aiperf 0.12.0
// without --scenario over the 256k Weka corpus, 256 live sessions, a 900 s warmup and a 900 s
// measured phase.
const constantLoadYAML = `
scenario: aiperf-constant-load
harness_version: 0.12.0
corpus:
  loader: semianalysis_cc_traces_weka_062126_256k
  format: weka-trace-v7
  encoding: jsonl
  source:
    huggingface: semianalysisai/cc-traces-weka-062126-256k
    revision: 8fecd2fc
    file: traces.jsonl
    split: train
  sha256: e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef
benchmark_duration_s: 900
grace_period_s: 900
category: coding
thinking: "on"
load:
  warmup_duration_s: 900
  measured_sessions: 1200
  trace_idle_gap_cap_s: 1
`

func decodeReplay(t *testing.T, doc string) *Replay {
	t.Helper()
	var r Replay
	if err := yaml.Unmarshal([]byte(doc), &r); err != nil {
		t.Fatalf("decoding: %v", err)
	}
	return &r
}

func TestConstantLoadValid(t *testing.T) {
	r := decodeReplay(t, constantLoadYAML+"concurrency: 256\n")
	if p := r.Validate(); !p.OK() {
		t.Fatalf("the constant-load run was rejected:\n%s", p.Error())
	}
	if c, ok := r.TraceIdleCap(); !ok || c != 1 {
		t.Errorf("per-trace idle cap = %v, %v; want 1, true", c, ok)
	}
	if r.SessionRecycling() != RecyclingSequential {
		t.Errorf("an unstated recycling rule read as %q, want the loader's sequential", r.SessionRecycling())
	}
	if r.Load.MeasuredSessions != 1200 || r.Load.WarmupDurationS != 900 || r.Duration() != 900 {
		t.Errorf("load block read as %+v", r.Load)
	}
}

func TestConstantLoadRefuses(t *testing.T) {
	for name, c := range map[string]struct{ edit, want string }{
		"no load block":      {"load:\n  warmup_duration_s: 900\n  measured_sessions: 1200\n  trace_idle_gap_cap_s: 1\n", "load"},
		"no measured window": {"benchmark_duration_s: 900\n", "benchmark_duration_s"},
		"unknown recycling":  {"  measured_sessions: 1200\n", "load.recycling"},
		"negative warmup":    {"  warmup_duration_s: 900\n", "load.warmup_duration_s"},
		"zero idle cap":      {"  trace_idle_gap_cap_s: 1\n", "load.trace_idle_gap_cap_s"},
		"cap stated twice":   {"  measured_sessions: 1200\n", "trace_idle_gap_cap_s"},
		"agentic warmup":     {"  measured_sessions: 1200\n", "warmup"},
		"snapshot window":    {"  measured_sessions: 1200\n", "trajectory_start"},
		"global idle cap":    {"  measured_sessions: 1200\n", "system_idle_gap_cap_s"},
		"negative sessions":  {"  measured_sessions: 1200\n", "load.measured_sessions"},
		"unknown cache bust": {"category: coding\n", "cache_bust"},
	} {
		t.Run(name, func(t *testing.T) {
			doc := constantLoadYAML
			switch name {
			case "no load block", "no measured window":
				doc = strings.Replace(doc, c.edit, "", 1)
			case "unknown recycling":
				doc = strings.Replace(doc, c.edit, c.edit+"  recycling: lifo\n", 1)
			case "negative warmup":
				doc = strings.Replace(doc, c.edit, "  warmup_duration_s: -5\n", 1)
			case "zero idle cap":
				doc = strings.Replace(doc, c.edit, "  trace_idle_gap_cap_s: 0\n", 1)
			case "cap stated twice":
				doc += "trace_idle_gap_cap_s: 1\n"
			case "agentic warmup":
				doc += "warmup:\n  mode: synthesized\n"
			case "snapshot window":
				doc += "trajectory_start:\n  min_ratio: 0.25\n"
			case "global idle cap":
				doc += "system_idle_gap_cap_s: 10\n"
			case "negative sessions":
				doc = strings.Replace(doc, c.edit, "  measured_sessions: -1\n", 1)
			case "unknown cache bust":
				doc += "cache_bust: every_turn\n"
			}
			p := decodeReplay(t, doc+"concurrency: 256\n").Validate()
			if p.OK() {
				t.Fatalf("validated:\n%s", doc)
			}
			if !strings.Contains(p.Error(), c.want) {
				t.Errorf("the refusal does not name %q:\n%s", c.want, p.Error())
			}
		})
	}
}

// The scenario's locks belong to inferencex-agentx-mvp: a load block on it is refused, and
// the agentic scenario keeps refusing a window under its 900 s floor.
func TestAgenticScenarioKeepsItsLocks(t *testing.T) {
	r := agentX()
	r.Load = &ConstantLoad{WarmupDurationS: 60}
	if p := r.Validate(); p.OK() || !strings.Contains(p.Error(), "load") {
		t.Errorf("a load block on inferencex-agentx-mvp validated: %s", p.Error())
	}
	r = agentX()
	r.BenchmarkDurationS = 600
	if p := r.Validate(); p.OK() {
		t.Error("a 600 s window validated under inferencex-agentx-mvp")
	}
}

// A constant-load window has no 900 s floor, and a reference to a constant-load entry may
// state the measured session count of the run it reproduces, which the entry does not fix.
func TestConstantLoadReference(t *testing.T) {
	entryDoc := strings.Replace(constantLoadYAML, "  measured_sessions: 1200\n", "", 1)
	entry := &CatalogReplay{Name: "constant", ReplayDefinition: decodeReplay(t, entryDoc+"concurrency: 1\n").ReplayDefinition}
	if p := entry.Validate(); !p.OK() {
		t.Fatalf("the catalog entry was rejected:\n%s", p.Error())
	}
	ref := &Replay{Workload: "constant", Concurrency: 512, ReplayDefinition: ReplayDefinition{
		BenchmarkDurationS: 300, Load: &ConstantLoad{MeasuredSessions: 2400}}}
	if p := ref.Validate(); !p.OK() {
		t.Fatalf("the reference was rejected:\n%s", p.Error())
	}
	got, err := ref.Resolve(entry)
	if err != nil {
		t.Fatal(err)
	}
	if got.Concurrency != 512 || got.Duration() != 300 || got.Load.MeasuredSessions != 2400 ||
		got.Load.WarmupDurationS != 900 || got.SessionRecycling() != RecyclingSequential {
		t.Errorf("resolved to %+v with load %+v", got, got.Load)
	}
	if entry.Load.MeasuredSessions != 0 {
		t.Error("resolving changed the catalog entry")
	}

	fixed := &Replay{Workload: "constant", Concurrency: 1, ReplayDefinition: ReplayDefinition{
		Load: &ConstantLoad{TraceIdleGapCapS: fl(5)}}}
	if p := fixed.Validate(); p.OK() {
		t.Error("a reference restated the entry's idle cap")
	}
	if _, err := (&Replay{Workload: "agentx", Concurrency: 1, ReplayDefinition: ReplayDefinition{
		Load: &ConstantLoad{MeasuredSessions: 5}}}).Resolve(&CatalogReplay{Name: "agentx", ReplayDefinition: agentXDefinition()}); err == nil {
		t.Error("a load block was silently dropped from an agentic entry")
	}
}
