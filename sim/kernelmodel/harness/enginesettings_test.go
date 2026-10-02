package harness

import "testing"

const settingsPath = "/Users/sri/Documents/Projects/blis-latency-kernel/testdata/measurements/inferencex_engine_settings.json"

// Every vLLM point this corpus scores must have a measured setting record. Without that the
// comparison configures some points from the run and others from an assumption, and reports
// one number over both.
func TestEveryScoredVLLMPointHasMeasuredSettings(t *testing.T) {
	set, err := LoadEngineSettings(settingsPath)
	if err != nil {
		t.Skipf("engine settings unavailable: %v", err)
	}
	c, err := LoadCorpus(
		"/Users/sri/Documents/Projects/blis-latency-kernel/testdata/measurements/aisimulate_e2e.json")
	if err != nil {
		t.Skip(err)
	}
	points, missing := 0, 0
	for i := range c.Sweeps {
		sw := c.Sweeps[i]
		if sw.Framework != "vllm" {
			continue
		}
		s := set.For(sw)
		if s == nil {
			t.Errorf("%s %s: no settings record at all", sw.Scenario, sw.Label)
			continue
		}
		for _, p := range sw.Points {
			points++
			if s.At(p.Concurrency) == nil {
				missing++
				t.Errorf("%s %s: no settings at concurrency %d",
					sw.Scenario, sw.Label, p.Concurrency)
			}
		}
	}
	if points == 0 {
		t.Fatal("no vLLM points found")
	}
	t.Logf("%d vLLM points, %d without measured settings", points, missing)
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
		"/Users/sri/Documents/Projects/blis-latency-kernel/testdata/measurements/aisimulate_e2e.json")
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

// With no settings supplied, every point must fall back to the scenario and SAY so. This is
// the INV-6 half: a caller that passes nothing gets the previous behaviour, not a silent
// substitution.
func TestWithoutSettingsEveryFieldReportsTheScenario(t *testing.T) {
	c, err := LoadCorpus(
		"/Users/sri/Documents/Projects/blis-latency-kernel/testdata/measurements/aisimulate_e2e.json")
	if err != nil {
		t.Skip(err)
	}
	sw := testSweep(t)
	cfg := Config{Repos: hopperRepos(), Admission: AdmissionKernelKV, Seed: 42}
	_ = c
	obs, err := Run(sw, 8, cfg)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	for name, got := range map[string]SettingSource{
		"seqs": obs.Settings.SeqsFrom, "tokens": obs.Settings.TokensFrom,
		"block": obs.Settings.BlockFrom, "prefix": obs.Settings.PrefixFrom,
	} {
		if got != SourceScenario {
			t.Errorf("with no settings supplied, %s should report %q, got %q",
				name, SourceScenario, got)
		}
	}
}

// Every arm must run one point with the SAME configuration. Only the step-time model may
// differ; if an arm got different admission settings the five-estimator table would be
// comparing deployments rather than timing models, and the difference would look like
// accuracy.
func TestEveryArmSharesOneMeasuredConfiguration(t *testing.T) {
	set, err := LoadEngineSettings(settingsPath)
	if err != nil {
		t.Skipf("engine settings unavailable: %v", err)
	}
	sw := testSweep(t)
	base := Config{Repos: hopperRepos(), Admission: AdmissionKernelKV, Seed: 42,
		EngineSettings: set, Backends: hopperBackends()}

	var first PointConfig
	var firstArm Estimator
	for i, e := range []Estimator{EstimatorKernel, EstimatorRoofline, EstimatorTrainedPhysics} {
		cfg := base
		cfg.Estimator = e
		obs, err := Run(sw, 16, cfg)
		if err != nil {
			t.Fatalf("%s: %v", e, err)
		}
		if i == 0 {
			first, firstArm = obs.Settings, e
			if first.SeqsFrom != SourceMeasured {
				t.Fatalf("the fixture point does not use a measured max_num_seqs (%q), so "+
					"this test would not detect a divergence", first.SeqsFrom)
			}
			continue
		}
		if obs.Settings != first {
			t.Errorf("%s ran with a different configuration than %s:\n  %+v\n  %+v",
				e, firstArm, obs.Settings, first)
		}
		if obs.KVBlocks == 0 {
			t.Errorf("%s got no KV budget", e)
		}
	}
	t.Logf("all arms ran with max_num_seqs %d (%s), tokens %d (%s), prefixOff %v (%s)",
		first.MaxNumSeqs, first.SeqsFrom, first.MaxNumBatchedTokens, first.TokensFrom,
		first.PrefixCachingDisabled, first.PrefixFrom)
}

// The measured settings must actually change the simulation. If supplying them moved nothing
// anywhere, the lookup would be decoration and the scenario's assumed values would still be
// in force.
//
// The fixture is deliberately the b300 ep4-dp2 sweep rather than a Hopper one. On gpt-oss the
// runs passed max_num_seqs EQUAL to the client concurrency, so the cap never binds in a closed
// loop that holds exactly that many in flight, and TTFT is identical either way -- correctly.
// On this sweep the run passed nothing, vLLM resolves 1024 against the scenario's assumed 256,
// and with dp 2 that is the difference between a cap of 512 and one of 2048 at concurrency
// 1024. Measured: TTFT falls from 13,825,369 us to 221,779 us, a 62x change.
func TestMeasuredSettingsChangeTheSimulation(t *testing.T) {
	set, err := LoadEngineSettings(settingsPath)
	if err != nil {
		t.Skipf("engine settings unavailable: %v", err)
	}
	c, err := LoadCorpus(
		"/Users/sri/Documents/Projects/blis-latency-kernel/testdata/measurements/aisimulate_e2e.json")
	if err != nil {
		t.Skip(err)
	}
	var sw Sweep
	for i := range c.Sweeps {
		if c.Sweeps[i].Scenario == "minimax-m2.5-b300-fp4-vllm-tp2-ep4-dp2.yaml" &&
			c.Sweeps[i].Label == "1k1k" {
			sw = c.Sweeps[i]
			break
		}
	}
	if sw.Scenario == "" {
		t.Skip("the b300 ep4-dp2 sweep is not in the corpus")
	}

	without := Config{Repos: hopperRepos(), Admission: AdmissionKernelKV, Seed: 42}
	with := without
	with.EngineSettings = set

	// Concurrency 1024: the first point where the scenario's assumed cap binds and the
	// measured one does not.
	const conc = 1024
	a, err := Run(sw, conc, without)
	if err != nil {
		t.Fatalf("without: %v", err)
	}
	b, err := Run(sw, conc, with)
	if err != nil {
		t.Fatalf("with: %v", err)
	}

	if a.Settings.MaxNumSeqs == b.Settings.MaxNumSeqs {
		t.Fatalf("the scenario and the engine agree on max_num_seqs (%d), so this fixture "+
			"cannot show the lookup taking effect", a.Settings.MaxNumSeqs)
	}
	if b.Settings.SeqsFrom != SourceResolved {
		t.Errorf("this sweep's run passed no max_num_seqs, so the value should be %q, got %q",
			SourceResolved, b.Settings.SeqsFrom)
	}
	if a.MeanTTFTUs == b.MeanTTFTUs {
		t.Error("supplying the measured settings changed nothing; the lookup is not reaching " +
			"the simulation")
	}
	// The assumed cap made requests queue behind whole generations. The measured one does
	// not, so TTFT must fall by a wide margin rather than drift.
	if b.MeanTTFTUs >= a.MeanTTFTUs/10 {
		t.Errorf("TTFT went from %.0f us to %.0f us; the assumed cap bound hard at this "+
			"concurrency, so the measured cap should be far lower, not comparable",
			a.MeanTTFTUs, b.MeanTTFTUs)
	}
	t.Logf("assumed cap %d -> TTFT %.0f us; resolved cap %d -> TTFT %.0f us (%.0fx)",
		a.Settings.MaxNumSeqs, a.MeanTTFTUs, b.Settings.MaxNumSeqs, b.MeanTTFTUs,
		a.MeanTTFTUs/b.MeanTTFTUs)
}
