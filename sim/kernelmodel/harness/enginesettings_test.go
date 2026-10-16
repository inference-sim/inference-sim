package harness

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/kernelmodel/internal/artifacts"
)

// settingsPath is the engine settings extracted from InferenceX's launch logs, under
// BLIS_MEASUREMENTS.
func settingsPath(t testing.TB) string {
	return artifacts.Measurement(t, "inferencex_engine_settings.json")
}

// No scored vLLM point may be configured from a value this project invented. Each must be
// either MEASURED from the run's own command line or RESOLVED as vLLM itself resolves it; the
// scenario's assumed numbers are a fallback for engines whose logs do not exist at all.
//
// The distinction is not cosmetic. On h200 vLLM resolves max_num_seqs to 1024 where the
// scenario files carry 256, and the sequence cap decides whether a request waits, so a point
// configured from the scenario simulates a deployment nobody ran.
//
// Two tiers exist in this corpus and the test reports both: gpt-oss and minimax carry a
// measured command line on every point, and llama-3.1-70B carries none -- no row for it in
// the InferenceX dump has a server_log_id, under either framework -- so it resolves.
func TestNoScoredVLLMPointUsesAnInventedSetting(t *testing.T) {
	set, err := LoadEngineSettings(settingsPath(t))
	if err != nil {
		t.Fatalf("engine settings: %v", err)
	}
	c, err := LoadCorpus(
		artifacts.Measurement(t, "aisimulate_e2e.json"))
	if err != nil {
		t.Fatal(err)
	}
	// The settings are decided before any simulation (pointSettings, which Run calls), so every
	// scored vLLM point on every chip is checked without simulating one.
	cfg := Config{Repos: testRepos(), EngineSettings: set}
	byTier := map[SettingSource]int{}
	perModel := map[string]map[SettingSource]int{}
	for i := range c.Sweeps {
		sw := c.Sweeps[i]
		if sw.Framework != "vllm" {
			continue
		}
		for _, p := range sw.Points {
			pc, _, err := pointSettings(sw, p.Concurrency, cfg)
			if err != nil {
				t.Fatalf("%s c=%d: %v", sw.Scenario, p.Concurrency, err)
			}
			byTier[pc.SeqsFrom]++
			if perModel[sw.Model] == nil {
				perModel[sw.Model] = map[SettingSource]int{}
			}
			perModel[sw.Model][pc.SeqsFrom]++
			if pc.SeqsFrom == SourceScenario {
				t.Errorf("%s c=%d: max_num_seqs came from the scenario, which is a value "+
					"this project chose rather than one the engine used",
					sw.Scenario, p.Concurrency)
			}
		}
	}
	if byTier[SourceMeasured] == 0 {
		t.Error("no point used a measured setting; the measured tier is unexercised")
	}
	if byTier[SourceResolved] == 0 {
		t.Error("no point used a resolved setting; the resolved tier is unexercised and " +
			"this test would not notice the fallback regressing")
	}
	for m, tiers := range perModel {
		t.Logf("%-26s measured=%d resolved=%d scenario=%d", m,
			tiers[SourceMeasured], tiers[SourceResolved], tiers[SourceScenario])
	}
}

// The settings and the absolute latencies must come from the SAME InferenceX run. If they
// drift apart the comparison silently stops being apples to apples: a point would be
// configured from one run and scored against another.
func TestSettingsAndAbsolutesShareOneRun(t *testing.T) {
	set, err := LoadEngineSettings(settingsPath(t))
	if err != nil {
		t.Fatalf("engine settings: %v", err)
	}
	abs, err := LoadAbsolutes(absolutesPath(t))
	if err != nil {
		t.Fatalf("absolutes: %v", err)
	}
	byKey := map[string]*Absolutes{}
	for i := range abs.Anchors {
		byKey[absKey(abs.Anchors[i].Scenario, abs.Anchors[i].Label)] = &abs.Anchors[i]
	}
	checked := 0
	for i := range set.Settings {
		s := &set.Settings[i]
		a := byKey[absKey(s.Scenario, s.Label)]
		if a == nil {
			t.Errorf("%s %s: settings exist with no matching absolutes", s.Scenario, s.Label)
			continue
		}
		if s.RunDate != a.RunDate || s.ConfigID != a.ConfigID {
			t.Errorf("%s %s: settings come from run %s/%d but latencies from %s/%d",
				s.Scenario, s.Label, s.RunDate, s.ConfigID, a.RunDate, a.ConfigID)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("nothing checked")
	}
	t.Logf("%d sweeps share one run between settings and latencies", checked)
}

// A setting the run did not pass must be ABSENT, not filled in. The distinction is the point:
// filling it in would make a resolved default indistinguishable from a measurement, and the
// provenance a report prints would be a lie.
func TestUnpassedSettingsStayAbsent(t *testing.T) {
	set, err := LoadEngineSettings(settingsPath(t))
	if err != nil {
		t.Fatalf("engine settings: %v", err)
	}
	unpassedSeqs, passedSeqs := 0, 0
	for i := range set.Settings {
		for _, p := range set.Settings[i].ByConcurrency {
			if p.Passed.MaxNumSeqs == nil {
				unpassedSeqs++
			} else {
				passedSeqs++
			}
		}
	}
	// Both regimes must be present, or the fixture no longer exercises the distinction and
	// the resolution path below is untested by this corpus.
	if unpassedSeqs == 0 {
		t.Error("no point leaves max_num_seqs unpassed; the resolved-default path is " +
			"unexercised and this test proves nothing")
	}
	if passedSeqs == 0 {
		t.Error("no point passes max_num_seqs; the measured path is unexercised")
	}
	t.Logf("max_num_seqs: %d points passed it, %d left it to the engine",
		passedSeqs, unpassedSeqs)
}

// Provenance must be reported per field, and a measured value must win over a resolved one.
// A run that passed max_num_seqs must be simulated with THAT number, whatever vLLM would
// have resolved.
func TestMeasuredSettingsWinOverResolvedDefaults(t *testing.T) {
	set, err := LoadEngineSettings(settingsPath(t))
	if err != nil {
		t.Fatalf("engine settings: %v", err)
	}
	c, err := LoadCorpus(
		artifacts.Measurement(t, "aisimulate_e2e.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{Repos: testRepos(), EngineSettings: set}
	for i := range c.Sweeps {
		sw := c.Sweeps[i]
		if sw.Framework != "vllm" || sw.Scenario != "gpt-oss-120b-h200-fp4-vllm-tp4.yaml" ||
			sw.Label != "1k1k" {
			continue
		}
		rec := set.For(sw)
		for _, p := range sw.Points {
			want := rec.At(p.Concurrency)
			if want == nil || want.Passed.MaxNumSeqs == nil {
				continue
			}
			pc, _, err := pointSettings(sw, p.Concurrency, cfg)
			if err != nil {
				t.Fatalf("c=%d: %v", p.Concurrency, err)
			}
			if pc.MaxNumSeqs != *want.Passed.MaxNumSeqs {
				t.Errorf("c=%d: runs with max_num_seqs %d, but the run passed %d",
					p.Concurrency, pc.MaxNumSeqs, *want.Passed.MaxNumSeqs)
			}
			if pc.SeqsFrom != SourceMeasured {
				t.Errorf("c=%d: provenance says %q for a value the run passed",
					p.Concurrency, pc.SeqsFrom)
			}
		}
		return
	}
	t.Skip("reference sweep not in the corpus")
}

// With no settings file supplied at all, a vLLM deployment must still take vLLM's resolution
// rather than the scenario's assumption. The engine resolved those values whether or not this
// project captured a log, so reproducing the resolution is the closer answer; falling back to
// a number chosen here would be the only case in the table that describes no real deployment.
func TestWithoutASettingsFileAVLLMSweepStillResolves(t *testing.T) {
	sw := testSweep(t)
	pc, _, err := pointSettings(sw, 8, Config{Repos: testRepos()})
	if err != nil {
		t.Fatalf("settings: %v", err)
	}
	for name, got := range map[string]SettingSource{
		"seqs": pc.SeqsFrom, "tokens": pc.TokensFrom,
		"block": pc.BlockFrom, "prefix": pc.PrefixFrom,
	} {
		if got != SourceResolved {
			t.Errorf("with no settings file, %s on a vLLM sweep should report %q, got %q",
				name, SourceResolved, got)
		}
	}
	// And the value must be vLLM's, not the scenario's.
	if pc.MaxNumSeqs == 256 {
		t.Error("max_num_seqs is 256, the scenario's assumed value; vLLM resolves 1024 on " +
			"every chip in this corpus")
	}
}

// A NON-vLLM sweep must NOT take vLLM's resolution. sglang and trtllm resolve their own
// defaults, and substituting one engine's for another's would be a different deployment.
func TestANonVLLMSweepDoesNotTakeVLLMDefaults(t *testing.T) {
	c, err := LoadCorpus(
		artifacts.Measurement(t, "aisimulate_e2e.json"))
	if err != nil {
		t.Fatal(err)
	}
	var sw Sweep
	for i := range c.Sweeps {
		if c.Sweeps[i].Framework == "sglang" {
			sw = c.Sweeps[i]
			break
		}
	}
	if sw.Scenario == "" {
		t.Skip("no sglang sweep in the corpus")
	}
	pc, _, err := pointSettings(sw, sw.Points[0].Concurrency, Config{Repos: testRepos()})
	if err != nil {
		t.Fatalf("%s: %v", sw.Scenario, err)
	}
	if pc.SeqsFrom != SourceScenario {
		t.Errorf("an sglang sweep should report %q, got %q -- vLLM's device-memory "+
			"resolution is not sglang's", SourceScenario, pc.SeqsFrom)
	}
}

// The three run settings added for #1920 follow the same order as the admission four: a value
// the run passed wins and reads "measured"; an unpassed one takes vLLM's own default where one
// is defined (expert parallelism off; gpu_memory_utilization 0.9 before v0.20.0 and 0.92 from
// it) and reads "resolved"; with no default the scenario's value stands and says so.
func TestTheWindowMemoryAndExpertParallelSettingsResolveLikeTheOthers(t *testing.T) {
	sw := committedSweep("vllm")
	m, err := kernelmodel.Open(sw.Scenario, testRepos())
	if err != nil {
		t.Fatal(err)
	}
	eng, err := m.Engine()
	if err != nil {
		t.Fatal(err)
	}
	at := func(ps *PointSettings) PointConfig {
		set := &EngineSettingSet{Settings: []SweepSettings{{Scenario: sw.Scenario, Label: sw.Label,
			ByConcurrency: map[string]*PointSettings{"4": ps}}}}
		return resolveAdmission(sw, 4, eng, m.Deployment(), set)
	}
	ml, gmu, ep := 2304, 0.95, true
	got := at(&PointSettings{Passed: PassedSettings{MaxModelLen: &ml, GPUMemoryUtilization: &gmu,
		EnableExpertParallel: &ep}, Resolved: ResolvedSettings{EngineVersion: "0.22.0"}})
	if got.MaxModelLen != ml || got.GPUMemoryUtilization != gmu || !got.EnableExpertParallel ||
		got.WindowFrom != SourceMeasured || got.MemoryFrom != SourceMeasured || got.EPFrom != SourceMeasured {
		t.Errorf("passed settings not taken as measured: %+v", got)
	}
	for _, tc := range []struct {
		version string
		want    float64
		from    SettingSource
	}{
		{"0.15.1", 0.9, SourceResolved}, {"0.18.0", 0.9, SourceResolved}, {"0.19.1", 0.9, SourceResolved},
		{"0.20.0", 0.92, SourceResolved}, {"0.25.0", 0.92, SourceResolved},
		{"", eng.GPUMemoryUtilization, SourceScenario},
	} {
		got := at(&PointSettings{Resolved: ResolvedSettings{EngineVersion: tc.version}})
		if got.GPUMemoryUtilization != tc.want || got.MemoryFrom != tc.from {
			t.Errorf("engine %q: gpu_memory_utilization %v (%s), want %v (%s)", tc.version,
				got.GPUMemoryUtilization, got.MemoryFrom, tc.want, tc.from)
		}
		if got.EnableExpertParallel || got.EPFrom != SourceResolved {
			t.Errorf("engine %q: unpassed expert parallelism %v (%s), want vLLM's default false (resolved)",
				tc.version, got.EnableExpertParallel, got.EPFrom)
		}
		if got.MaxModelLen != int(eng.MaxModelLen) || got.WindowFrom != SourceScenario {
			t.Errorf("engine %q: unpassed max_model_len %d (%s), want the scenario's %d", tc.version,
				got.MaxModelLen, got.WindowFrom, eng.MaxModelLen)
		}
	}
}
