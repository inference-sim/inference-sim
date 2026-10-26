// Original to blis-vllm.

package build

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/core/vllm"
)

// A run needs its scenario, scenario directory and registry, and the refusal names every one
// that is missing.
func TestAdopt_NeedsTheScenarioAndItsRoots(t *testing.T) {
	_, err := Adopt(AdoptInputs{})
	if err == nil {
		t.Fatal("no scenario must be refused")
	}
	for _, flag := range []string{"--scenario", "--scenarios", "--registry"} {
		if !strings.Contains(err.Error(), flag) {
			t.Errorf("refusal %q must name %s", err, flag)
		}
	}
}

// A P/D scenario adopts as disaggregated and refuses to open without a P/D topology; with one
// it opens the first pool, sized by the kernel, at the catalog the locator names.
func TestAdopt_DisaggregatedNeedsAPDTopology(t *testing.T) {
	repos := vendoredRepos(t)
	in := AdoptInputs{Scenario: "glm-5-h200-3p1d-ib.yaml", ScenariosDir: repos.Scenarios, Registry: repos.Registry,
		Catalog: func() (string, error) { return repos.Catalog, nil }}
	a, err := Adopt(in)
	if err != nil || !a.Disaggregated || len(a.Shape.Roles) != 2 {
		t.Fatalf("adopt: %+v, %v", a, err)
	}
	if err := OpenFirstPool(in, &a); err == nil || !strings.Contains(err.Error(), "needs a P/D topology") {
		t.Errorf("no topology: %v", err)
	}
	in.PDTopology = true
	opened := false
	in.BeforeOpen = func(kernelmodel.Shape) error { opened = true; return nil }
	if err := OpenFirstPool(in, &a); err != nil || a.Model == nil || a.Settings.KVBlocks <= 0 || a.CatalogRoot != repos.Catalog || !opened {
		t.Fatalf("with a topology: model %v, %d blocks, root %q, hook %v, err %v", a.Model != nil, a.Settings.KVBlocks, a.CatalogRoot, opened, err)
	}
	if a.Deployment.Model == "" || a.Deployment.TP < 1 {
		t.Errorf("deployment %+v, want the first pool's model and parallelism", a.Deployment)
	}
	in.Catalog = func() (string, error) { return "", errors.New("no catalog located") }
	if err := OpenFirstPool(in, &a); err == nil || !strings.Contains(err.Error(), "no catalog located") {
		t.Errorf("no catalog: %v", err)
	}
	in.Catalog = func() (string, error) { return repos.Catalog, nil }
	in.BeforeOpen = func(kernelmodel.Shape) error { return errors.New("bad routing flags") }
	if err := OpenFirstPool(in, &a); err == nil || !strings.Contains(err.Error(), "bad routing flags") {
		t.Errorf("a failing hook: %v", err)
	}
}

// A colocated scenario adopts as colocated and opens with no topology.
func TestAdopt_Colocated(t *testing.T) {
	repos := vendoredRepos(t)
	in := AdoptInputs{Scenario: "glm-5-h200-tp8-mtp3.yaml", ScenariosDir: repos.Scenarios, Registry: repos.Registry,
		Catalog: func() (string, error) { return repos.Catalog, nil }}
	a, err := Adopt(in)
	if err != nil || a.Disaggregated {
		t.Fatalf("adopt: %+v, %v", a, err)
	}
	if err := OpenFirstPool(in, &a); err != nil || a.Model == nil {
		t.Fatalf("open: %v", err)
	}
	if _, err := OpenModel("missing.yaml", repos.Scenarios, repos.Catalog, repos.Registry); err == nil ||
		!strings.Contains(err.Error(), `"missing.yaml"`) {
		t.Errorf("a missing scenario: %v", err)
	}
}

// Dense data and expert parallelism are refused; an MoE model runs both.
func TestCheckParallelism(t *testing.T) {
	for _, c := range []struct {
		dp       int
		ep, moe  bool
		wantFrag string
	}{
		{0, false, true, "it must be >= 1"},
		{2, false, false, "DP placement is modelled for MoE"},
		{1, true, false, "which vLLM refuses"},
		{2, true, true, ""},
		{1, false, false, ""},
	} {
		err := CheckParallelism("s.yaml", "m", c.dp, c.ep, c.moe)
		if c.wantFrag == "" {
			if err != nil {
				t.Errorf("dp %d ep %v moe %v: %v", c.dp, c.ep, c.moe, err)
			}
			continue
		}
		if err == nil || !strings.Contains(err.Error(), c.wantFrag) {
			t.Errorf("dp %d ep %v moe %v: %v, want %q", c.dp, c.ep, c.moe, err, c.wantFrag)
		}
	}
}

// A scenario stating two colocated pools is refused: a run simulates one colocated engine,
// and the second pool would be dropped without a word.
func TestAdopt_RefusesTwoColocatedPools(t *testing.T) {
	repos := vendoredRepos(t)
	raw, err := os.ReadFile(filepath.Join(repos.Scenarios, "glm-5-h200-3p1d-ib.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	two := strings.ReplaceAll(strings.ReplaceAll(string(raw), "role: prefill", "role: colocated"), "role: decode", "role: colocated")
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "two.yaml"), []byte(two), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = Adopt(AdoptInputs{Scenario: "two.yaml", ScenariosDir: dir, Registry: repos.Registry})
	if err == nil || !strings.Contains(err.Error(), "states 2 colocated pools") {
		t.Errorf("two colocated pools: %v", err)
	}
}

// A scenario stating a release the compatibility engine does not model is refused, naming the
// release and the ones it does; the releases it models adopt. One build models one release
// (#2021 B6), so a 0.32.0 deployment is never run as a 0.29 or 0.31 one.
func TestAdopt_RefusesAReleaseTheEngineDoesNotModel(t *testing.T) {
	repos := vendoredRepos(t)
	raw, err := os.ReadFile(filepath.Join(repos.Scenarios, "glm-5-h200-tp8-mtp3.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		version string
		ok      bool
	}{{"0.29.0", true}, {"0.31.0", true}, {"0.32.0", false}, {"0.30.0", false}, {"", false}} {
		body := strings.Replace(string(raw), `engine_version: "0.29.0"`, `engine_version: "`+c.version+`"`, 1)
		if !strings.Contains(body, `engine_version: "`+c.version+`"`) {
			t.Fatalf("fixture has no engine_version to edit")
		}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "v.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := Adopt(AdoptInputs{Scenario: "v.yaml", ScenariosDir: dir, Registry: repos.Registry})
		switch {
		case c.ok && err != nil:
			t.Errorf("engine_version %s: %v, want it adopted", c.version, err)
		case !c.ok && (err == nil || !strings.Contains(err.Error(), statedVersion(c.version)) ||
			!strings.Contains(err.Error(), "0.29.0 and 0.31.0")):
			t.Errorf("engine_version %q: %v, want a refusal naming it and the modelled releases", c.version, err)
		}
	}
}

// A 0.29/0.31 scenario that states an engine setting the compatibility engine does not model is
// refused, naming the setting and the pool, rather than run with the setting silently dropped:
// the setting came with the v0.32.0 engine (blis-schemas#102), and a 0.32.0 scenario runs on that
// engine (core/vllm) once it is wired in (inference-sim#2027). The same scenario without the
// setting adopts, and so does one stating prefix_cache_retention_interval at vLLM's default 0,
// which is what the compatibility engine models.
func TestAdopt_RefusesEngineSettingsTheCompatibilityEngineDoesNotModel(t *testing.T) {
	repos := vendoredRepos(t)
	raw, err := os.ReadFile(filepath.Join(repos.Scenarios, "glm-5-h200-tp8-mtp3.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const marker = "      gpu_memory_utilization: 0.9\n"
	if !strings.Contains(string(raw), marker) {
		t.Fatal("fixture has no engine block to edit")
	}
	adopt := func(t *testing.T, extra string) error {
		t.Helper()
		body := strings.Replace(string(raw), marker, marker+extra, 1)
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "v.yaml"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := Adopt(AdoptInputs{Scenario: "v.yaml", ScenariosDir: dir, Registry: repos.Registry})
		return err
	}
	if err := adopt(t, ""); err != nil {
		t.Fatalf("the fixture itself must adopt: %v", err)
	}
	if err := adopt(t, "      prefix_cache_retention_interval: 0\n"); err != nil {
		t.Errorf("prefix_cache_retention_interval 0 is the default the compatibility engine models: %v", err)
	}
	for _, c := range []struct{ yaml, name string }{
		{"      max_num_scheduled_tokens: 4096\n", "max_num_scheduled_tokens"},
		{"      max_num_active_seqs: 64\n", "max_num_active_seqs"},
		{"      stream_interval: 4\n", "stream_interval"},
		{"      prefill_schedule_interval: 2\n", "prefill_schedule_interval"},
		{"      kv_cache_memory_bytes: 1000000000\n", "kv_cache_memory_bytes"},
		{"      disable_hybrid_kv_cache_manager: true\n", "disable_hybrid_kv_cache_manager"},
		{"      max_loras: 2\n", "max_loras"},
		{"      watermark: 0.02\n", "watermark"},
		{"      prefix_match_unit: 16\n", "prefix_match_unit"},
		{"      long_prefill_token_threshold_adaptive: true\n", "long_prefill_token_threshold_adaptive"},
		{"      prefix_cache_retention_interval: 16\n", "prefix_cache_retention_interval"},
		{"      kv_events:\n        enable_kv_cache_events: true\n", "kv_events"},
		{"      kv_events:\n        enable_kv_cache_events: false\n", "kv_events"},
		{"      max_cpu_loras: 4\n", "max_cpu_loras"},
	} {
		err := adopt(t, c.yaml)
		if err == nil {
			t.Errorf("%s: adopted, want a refusal", c.name)
			continue
		}
		for _, frag := range []string{"engine." + c.name, "colocated pool", "compatibility engine", "0.32.0", "inference-sim#2027"} {
			if !strings.Contains(err.Error(), frag) {
				t.Errorf("%s: refusal %q must name %q", c.name, err, frag)
			}
		}
	}
	// The drafter of a speculative block: speculative.draft_model.
	body := strings.Replace(string(raw), "        num_spec_tokens: 3\n", "        num_spec_tokens: 3\n        draft_model: some-drafter\n", 1)
	if !strings.Contains(body, "draft_model: some-drafter") {
		t.Fatal("fixture has no speculative block to edit")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Adopt(AdoptInputs{Scenario: "v.yaml", ScenariosDir: dir, Registry: repos.Registry}); err == nil ||
		!strings.Contains(err.Error(), "engine.speculative.draft_model") {
		t.Errorf("speculative.draft_model: %v, want a refusal naming it", err)
	}
}

// statedVersion is how a refusal names the release a scenario states: an empty one is named as
// none stated, not as the release "".
func statedVersion(v string) string {
	if v == "" {
		return "states no engine_version"
	}
	return `states engine_version "` + v + `"`
}

// The compatibility engine's releases are the ones core/vllm's factory hands to it.
func TestCompatVersionsAgreeWithTheFactory(t *testing.T) {
	if strings.Join(CompatVersions(), ",") != strings.Join(vllm.CompatVersions, ",") {
		t.Errorf("compat/build models %v, core/vllm hands the compatibility engine %v", CompatVersions(), vllm.CompatVersions)
	}
}
