package harness

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// A var rather than a const: the path is resolved at run time from the vendored default or
// an environment override, and a const cannot call a function.
var settingsPath = filepath.Join(
	kernelmodel.DefaultMeasurements(), "inferencex_engine_settings.json")

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
	set, err := LoadEngineSettings(settingsPath)
	if err != nil {
		t.Skipf("engine settings unavailable: %v", err)
	}
	c, err := LoadCorpus(
		filepath.Join(kernelmodel.DefaultMeasurements(), "aisimulate_e2e.json"))
	if err != nil {
		t.Skip(err)
	}
	cfg := Config{Repos: hopperRepos(), Admission: AdmissionKernelKV, Seed: 42,
		EngineSettings: set}
	byTier := map[SettingSource]int{}
	perModel := map[string]map[SettingSource]int{}
	for i := range c.Sweeps {
		sw := c.Sweeps[i]
		if sw.Framework != "vllm" || sw.GPU != "h200" {
			continue // one chip keeps the test quick; the resolution is chip-independent
		}
		for _, p := range sw.Points {
			obs, err := Run(sw, p.Concurrency, cfg)
			if err != nil {
				t.Fatalf("%s c=%d: %v", sw.Scenario, p.Concurrency, err)
			}
			byTier[obs.Settings.SeqsFrom]++
			if perModel[sw.Model] == nil {
				perModel[sw.Model] = map[SettingSource]int{}
			}
			perModel[sw.Model][obs.Settings.SeqsFrom]++
			if obs.Settings.SeqsFrom == SourceScenario {
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
	set, err := LoadEngineSettings(settingsPath)
	if err != nil {
		t.Skipf("engine settings unavailable: %v", err)
	}
	abs, err := LoadAbsolutes(absolutesPath)
	if err != nil {
		t.Skipf("absolutes unavailable: %v", err)
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
	set, err := LoadEngineSettings(settingsPath)
	if err != nil {
		t.Skipf("engine settings unavailable: %v", err)
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
	set, err := LoadEngineSettings(settingsPath)
	if err != nil {
		t.Skipf("engine settings unavailable: %v", err)
	}
	c, err := LoadCorpus(
		filepath.Join(kernelmodel.DefaultMeasurements(), "aisimulate_e2e.json"))
	if err != nil {
		t.Skip(err)
	}
	cfg := Config{Repos: hopperRepos(), Admission: AdmissionKernelKV, Seed: 42,
		EngineSettings: set}
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
			obs, err := Run(sw, p.Concurrency, cfg)
			if err != nil {
				t.Fatalf("c=%d: %v", p.Concurrency, err)
			}
			if obs.Settings.MaxNumSeqs != *want.Passed.MaxNumSeqs {
				t.Errorf("c=%d: ran with max_num_seqs %d, but the run passed %d",
					p.Concurrency, obs.Settings.MaxNumSeqs, *want.Passed.MaxNumSeqs)
			}
			if obs.Settings.SeqsFrom != SourceMeasured {
				t.Errorf("c=%d: provenance says %q for a value the run passed",
					p.Concurrency, obs.Settings.SeqsFrom)
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
	cfg := Config{Repos: hopperRepos(), Admission: AdmissionKernelKV, Seed: 42}
	obs, err := Run(sw, 8, cfg)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	for name, got := range map[string]SettingSource{
		"seqs": obs.Settings.SeqsFrom, "tokens": obs.Settings.TokensFrom,
		"block": obs.Settings.BlockFrom, "prefix": obs.Settings.PrefixFrom,
	} {
		if got != SourceResolved {
			t.Errorf("with no settings file, %s on a vLLM sweep should report %q, got %q",
				name, SourceResolved, got)
		}
	}
	// And the value must be vLLM's, not the scenario's.
	if obs.Settings.MaxNumSeqs == 256 {
		t.Error("max_num_seqs is 256, the scenario's assumed value; vLLM resolves 1024 on " +
			"every chip in this corpus")
	}
}

// A NON-vLLM sweep must NOT take vLLM's resolution. sglang and trtllm resolve their own
// defaults, and substituting one engine's for another's would be a different deployment.
func TestANonVLLMSweepDoesNotTakeVLLMDefaults(t *testing.T) {
	c, err := LoadCorpus(
		filepath.Join(kernelmodel.DefaultMeasurements(), "aisimulate_e2e.json"))
	if err != nil {
		t.Skip(err)
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
	cfg := Config{Repos: hopperRepos(), Admission: AdmissionKernelKV, Seed: 42}
	obs, err := Run(sw, sw.Points[0].Concurrency, cfg)
	if err != nil {
		t.Skipf("%s: %v", sw.Scenario, err)
	}
	if obs.Settings.SeqsFrom != SourceScenario {
		t.Errorf("an sglang sweep should report %q, got %q -- vLLM's device-memory "+
			"resolution is not sglang's", SourceScenario, obs.Settings.SeqsFrom)
	}
}
