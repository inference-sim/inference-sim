package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// resolveEngine is the one place a scenario's engine is chosen (SIM-1). Each row is a scenario's
// engine fields and what selection does with them: the refusals name the field and the value
// (SIM-2, LIF-6), and each refusal has a closest accepted row (SIM-9). A scenario that states
// engine_version, alone or beside engine, is refused with a message naming engine: {kind,
// version}; the closest accepted row states the same release in engine.
func TestResolveEngine(t *testing.T) {
	ref := func(k, v string) *engineRef { return &engineRef{Kind: k, Version: v} }
	cases := []struct {
		name string
		in   scenarioEngineFields
		want engineRef
		err  []string
	}{
		{name: "engine: {kind, version} at the release this build runs", in: scenarioEngineFields{Engine: ref("vllm", "0.33.0")},
			want: engineRef{"vllm", "0.33.0"}},
		{name: "engine_version alone, at the release this build runs", in: scenarioEngineFields{EngineVersion: "0.33.0"},
			err: []string{`states engine_version "0.33.0"`, "was removed in " + flagsRemovedIn, `state engine: {kind: vllm, version: "0.33.0"} instead`}},
		{name: "engine_version alone, at a release this build does not run", in: scenarioEngineFields{EngineVersion: "0.29.0"},
			err: []string{`states engine_version "0.29.0"`, `state engine: {kind: vllm, version: "0.29.0"} instead`}},
		{name: "vllm 0.29.0", in: scenarioEngineFields{Engine: ref("vllm", "0.29.0")},
			err: []string{"names engine vllm 0.29.0", "this build runs vllm 0.33.0", "newest inference-sim release that runs it is v0.15.0"}},
		{name: "vllm 0.31.0", in: scenarioEngineFields{Engine: ref("vllm", "0.31.0")},
			err: []string{"names engine vllm 0.31.0", "this build runs vllm 0.33.0", "is v0.15.0"}},
		{name: "vllm 0.32.0, last run by v0.17.0", in: scenarioEngineFields{Engine: ref("vllm", "0.32.0")},
			err: []string{"names engine vllm 0.32.0", "this build runs vllm 0.33.0", "newest inference-sim release that runs it is v0.17.0"}},
		{name: "a release no inference-sim release runs", in: scenarioEngineFields{Engine: ref("vllm", "0.34.0")},
			err: []string{"names engine vllm 0.34.0", "no inference-sim release runs it"}},
		{name: "a kind this build does not register", in: scenarioEngineFields{Engine: ref("tensorrt-llm", "0.18.0")},
			err: []string{`engine kind "tensorrt-llm"`, "this build runs the kinds sglang, vllm"}},
		{name: "both spellings", in: scenarioEngineFields{Engine: ref("vllm", "0.33.0"), EngineVersion: "0.33.0"},
			err: []string{`states engine_version "0.33.0"`, "was removed", "state engine: {kind, version} alone instead"}},
		{name: "a kind with no version", in: scenarioEngineFields{Engine: ref("vllm", "")},
			err: []string{"must state both kind and version"}},
		{name: "no engine", in: scenarioEngineFields{},
			err: []string{"names no engine", `engine: {kind: vllm, version: "0.33.0"}`}},
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
	body := "kind: Scenario\nengine:\n  kind: vllm\n  version: \"0.33.0\"\n---\nkind: Deployment\nengine_version: \"0.29.0\"\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := selectEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != (engineRef{"vllm", "0.33.0"}) {
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
