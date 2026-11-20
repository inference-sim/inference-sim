package workload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// These tests re-derive the replay descriptor's aiperf constants from an aiperf 0.12.0
// source tree at AIPERF_SOURCE (its src/aiperf directory), as the rules packs' tests do
// from VLLM_SOURCE. Without one they skip and say so; AIPERF_SOURCE_REQUIRED turns the
// skip into a failure.

func aiperfSource(t *testing.T) string {
	t.Helper()
	root := os.Getenv("AIPERF_SOURCE")
	skip := t.Skipf
	if os.Getenv("AIPERF_SOURCE_REQUIRED") != "" {
		skip = t.Fatalf
	}
	if root == "" {
		skip("AIPERF_SOURCE is unset: the replay descriptor's aiperf constants were NOT checked against aiperf 0.12.0")
	}
	if _, err := os.Stat(filepath.Join(root, "common", "scenario")); err != nil {
		skip("AIPERF_SOURCE=%s has no common/scenario: the aiperf constants were NOT checked", root)
	}
	return root
}

func readAiperf(t *testing.T, root string, parts ...string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(append([]string{root}, parts...)...))
	if err != nil {
		t.Fatalf("reading %v: %v (the aiperf layout may have moved)", parts, err)
	}
	return string(b)
}

// Every alias the table names reads the repository plugins.yaml fixes for it, and the
// rolling set is exactly the aliases plugins.yaml calls rolling.
func TestAgentXLoaderReposMatchSource(t *testing.T) {
	root := aiperfSource(t)
	var plugins struct {
		Loaders map[string]struct {
			Description string `yaml:"description"`
			Metadata    struct {
				HFDatasetName string `yaml:"hf_dataset_name"`
			} `yaml:"metadata"`
		} `yaml:"public_dataset_loader"`
	}
	if err := yaml.Unmarshal([]byte(readAiperf(t, root, "plugin", "plugins.yaml")), &plugins); err != nil {
		t.Fatal(err)
	}
	for alias, repo := range AgentXLoaders {
		if alias == loaderWekaHF || alias == loaderWekaTrace {
			continue
		}
		l, ok := plugins.Loaders[alias]
		if !ok {
			t.Errorf("%s is not a public_dataset_loader in plugins.yaml", alias)
			continue
		}
		if l.Metadata.HFDatasetName != repo {
			t.Errorf("%s reads %s in plugins.yaml; the table says %s", alias, l.Metadata.HFDatasetName, repo)
		}
		rolling := strings.Contains(strings.Join(strings.Fields(l.Description), " "), "rolling alias that tracks the current default")
		if rolling != AgentXRollingLoaders[alias] {
			t.Errorf("%s: rolling in plugins.yaml %v, in the table %v", alias, rolling, AgentXRollingLoaders[alias])
		}
	}
}

// The scenario's declaration, the lock that applies it, and the CLI and environment
// defaults the descriptor fills in, as aiperf 0.12.0 states them.
func TestAgentXConstantsMatchSource(t *testing.T) {
	root := aiperfSource(t)
	mvp := readAiperf(t, root, "common", "scenario", "inferencex_agentx_mvp.py")
	for _, want := range []string{
		`name="` + string(ScenarioInferenceXAgentXMVP) + `"`,
		"timing_mode=TimingMode.AGENTIC_REPLAY",
		"require_ignore_eos=True",
		"require_streaming=True",
		"forbid_ignore_trace_delays=True",
		"forbid_input_truncation=True",
		"min_benchmark_duration_seconds=900",
		"default_benchmark_duration_seconds=1800",
		"default_trajectory_start_min_ratio=0.0",
		"default_trajectory_start_max_ratio=1.0",
		"system_idle_gap_cap_seconds=10.0",
		"forbid_trace_idle_gap_cap=True",
		"forbid_inter_turn_delay_cap=True",
		"require_cache_bust=CacheBustTarget.FIRST_TURN_PREFIX",
	} {
		if !strings.Contains(mvp, want) {
			t.Errorf("inferencex_agentx_mvp.py no longer states %s", want)
		}
	}
	// require_loader is exactly the loader table's keys.
	start := strings.Index(mvp, "require_loader=(")
	end := strings.Index(mvp[start:], "),")
	if start < 0 || end < 0 {
		t.Fatal("require_loader tuple not found")
	}
	declared := map[string]bool{}
	for _, f := range strings.Split(mvp[start+len("require_loader=("):start+end], ",") {
		if name := strings.Trim(strings.TrimSpace(f), `"`); name != "" {
			declared[name] = true
		}
	}
	for name := range AgentXLoaders {
		if !declared[name] {
			t.Errorf("%s is in the loader table but not in require_loader", name)
		}
	}
	for name := range declared {
		if _, ok := AgentXLoaders[name]; !ok {
			t.Errorf("require_loader names %s, which the loader table lacks", name)
		}
	}

	enums := readAiperf(t, root, "common", "enums", "enums.py")
	if !strings.Contains(enums, `FIRST_TURN_PREFIX = "`+string(CacheBustFirstTurnPrefix)+`"`) {
		t.Errorf("CacheBustTarget.FIRST_TURN_PREFIX is no longer %q", CacheBustFirstTurnPrefix)
	}
	val := readAiperf(t, root, "common", "scenario", "validator.py")
	if !strings.Contains(val, `_AGENTX_WEKA_HF_REPO = "`+agentXWekaHFRepo+`"`) {
		t.Errorf("validator.py no longer pins weka_hf to %s", agentXWekaHFRepo)
	}
	cli := readAiperf(t, root, "config", "flags", "cli_config.py")
	grace := strings.Index(cli, "benchmark_grace_period: Annotated[")
	if grace < 0 || !strings.Contains(cli[grace:grace+700], "] = 30.0") {
		t.Error("--benchmark-grace-period no longer defaults to 30.0")
	}
	env := readAiperf(t, root, "common", "environment.py")
	split := strings.Index(env, "WEKA_SPLIT_FLATTENED_AGENTS: bool = Field(")
	if split < 0 || !strings.Contains(env[split:split+120], "default=True") {
		t.Error("WEKA_SPLIT_FLATTENED_AGENTS no longer defaults to True")
	}
	if r := (&Replay{}); r.Duration() != 1800 || r.Grace() != 30 || r.IdleCap() != 10 || !r.SplitsFlattenedAgents() {
		t.Error("the descriptor's defaults no longer match the pinned values")
	}
}
