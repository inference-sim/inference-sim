package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// resolveEngine is the one place a scenario's engine is chosen (SIM-1). Each row is a scenario's
// engine fields and what selection does with them: the refusals name the field and the value
// (SIM-2, LIF-6), and each refusal has a closest accepted row (SIM-9).
func TestResolveEngine(t *testing.T) {
	ref := func(k, v string) *engineRef { return &engineRef{Kind: k, Version: v} }
	cases := []struct {
		name       string
		in         scenarioEngineFields
		want       engineRef
		deprecated bool
		err        []string
	}{
		{name: "engine: {kind, version} at the release this build runs", in: scenarioEngineFields{Engine: ref("vllm", "0.32.0")},
			want: engineRef{"vllm", "0.32.0"}},
		{name: "engine_version, deprecated, names a vLLM release", in: scenarioEngineFields{EngineVersion: "0.32.0"},
			want: engineRef{"vllm", "0.32.0"}, deprecated: true},
		{name: "vllm 0.29.0", in: scenarioEngineFields{EngineVersion: "0.29.0"},
			err: []string{"names engine vllm 0.29.0", "this build runs vllm 0.32.0", "newest inference-sim release that runs it is v0.15.0"}},
		{name: "vllm 0.31.0", in: scenarioEngineFields{Engine: ref("vllm", "0.31.0")},
			err: []string{"names engine vllm 0.31.0", "this build runs vllm 0.32.0", "is v0.15.0"}},
		{name: "a release no inference-sim release runs", in: scenarioEngineFields{Engine: ref("vllm", "0.33.0")},
			err: []string{"names engine vllm 0.33.0", "no inference-sim release runs it"}},
		{name: "a kind this build does not register", in: scenarioEngineFields{Engine: ref("sglang", "0.5.23")},
			err: []string{`engine kind "sglang"`, "this build runs the kinds vllm"}},
		{name: "both spellings", in: scenarioEngineFields{Engine: ref("vllm", "0.32.0"), EngineVersion: "0.32.0"},
			err: []string{"states both engine: {kind, version} and engine_version"}},
		{name: "a kind with no version", in: scenarioEngineFields{Engine: ref("vllm", "")},
			err: []string{"must state both kind and version"}},
		{name: "no engine", in: scenarioEngineFields{},
			err: []string{"names no engine", `engine: {kind: vllm, version: "0.32.0"}`}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := resolveEngine("s.yaml", c.in)
			if len(c.err) > 0 {
				if err == nil {
					t.Fatalf("selected %v, want a refusal", got.Ref)
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
			if got.Ref != c.want || got.Deprecated != c.deprecated {
				t.Errorf("got %+v, want %v deprecated=%v", got, c.want, c.deprecated)
			}
		})
	}
}

// selectEngine reads a scenario file's first document only: the deployment that follows it may
// state keys of its own.
func TestSelectEngine_ReadsTheScenarioDocument(t *testing.T) {
	path := filepath.Join(t.TempDir(), "s.yaml")
	body := "kind: Scenario\nengine:\n  kind: vllm\n  version: \"0.32.0\"\n---\nkind: Deployment\nengine_version: \"0.29.0\"\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := selectEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Ref != (engineRef{"vllm", "0.32.0"}) || got.Deprecated {
		t.Errorf("got %+v", got)
	}
}

// Every engine release this build refuses with a named successor is a release of a kind it
// registers, at a release other than the one it runs: the table cannot name a release this build
// runs as one it does not.
func TestLastReleaseRunning_NamesOnlyReleasesThisBuildDoesNotRun(t *testing.T) {
	built := builtEngineReleases()
	for ref := range lastReleaseRunning() {
		runs, ok := built[ref.Kind]
		if !ok || runs == ref.Version {
			t.Errorf("%v: this build registers kind %v at %q", ref, ok, runs)
		}
	}
}
