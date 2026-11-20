package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// resolveEngine is the one place a scenario's engine is chosen (SIM-1). Each row is a scenario's
// engine fields and what selection does with them: a scenario names its engine by kind alone,
// the release and commit are the build's (LIF-6), and the refusals name the field and the value
// (SIM-2).
func TestResolveEngine(t *testing.T) {
	built := builtEngines()["vllm"]
	if built.Commit == "" || built.Version == "" {
		t.Fatalf("this build records no vllm release and commit: %+v", built)
	}
	cases := []struct {
		name string
		in   scenarioEngineFields
		want engineRef
		err  []string
	}{
		{name: "engine: {kind}", in: scenarioEngineFields{Engine: &scenarioEngineBlock{Kind: "vllm"}}, want: built},
		{name: "engine.version at the release this build runs",
			in:  scenarioEngineFields{Engine: &scenarioEngineBlock{Kind: "vllm", Version: built.Version}},
			err: []string{"field engine.version", "not found in type scenario.Engine", built.Commit}},
		{name: "engine.version at another release",
			in:  scenarioEngineFields{Engine: &scenarioEngineBlock{Kind: "vllm", Version: "0.29.0"}},
			err: []string{`engine.version "0.29.0"`, "the inference-sim release that runs it"}},
		{name: "engine_version", in: scenarioEngineFields{EngineVersion: built.Version},
			err: []string{"field engine_version not found", "engine: {kind: vllm}"}},
		{name: "a kind this build does not register", in: scenarioEngineFields{Engine: &scenarioEngineBlock{Kind: "tensorrt-llm"}},
			err: []string{`engine kind "tensorrt-llm"`, "this build runs the kinds sglang, vllm"}},
		{name: "no kind", in: scenarioEngineFields{Engine: &scenarioEngineBlock{}},
			err: []string{"must state the kind"}},
		{name: "no engine", in: scenarioEngineFields{},
			err: []string{"names no engine", "engine: {kind: vllm}"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := resolveEngine("s.yaml", c.in)
			if len(c.err) > 0 {
				if err == nil {
					t.Fatalf("selected %v, want a refusal", got)
				}
				for _, w := range c.err {
					if !strings.Contains(err.Error(), w) {
						t.Errorf("refusal %q does not contain %q", err, w)
					}
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != c.want {
				t.Errorf("got %+v, want %v", got, c.want)
			}
		})
	}
}

// selectEngine reads a scenario file's first document only: the deployment that follows it may
// state keys of its own.
func TestSelectEngine_ReadsTheScenarioDocument(t *testing.T) {
	path := filepath.Join(t.TempDir(), "s.yaml")
	body := "kind: Scenario\nengine: {kind: vllm}\n---\nkind: Deployment\nengine_version: \"0.29.0\"\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := selectEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != builtEngines()["vllm"] {
		t.Errorf("got %+v", got)
	}
}

// Every kind this build registers carries the full 40-digit commit it runs, which the Result
// records.
func TestBuiltEnginesRecordCommits(t *testing.T) {
	for kind, e := range builtEngines() {
		if len(e.Commit) != 40 || e.Version == "" {
			t.Errorf("%s: version %q commit %q, want a release and a 40-digit commit", kind, e.Version, e.Commit)
		}
	}
}
