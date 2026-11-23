package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// Engine selection (design SIM-1, LIF-6). A scenario names the engine it runs by kind alone,
// in engine: {kind}. It states no release of an engine: this build runs one upstream commit of
// each kind it registers (stack.json), and every Result records it. A scenario that states
// engine.version, or the older engine_version, is refused as an unknown field.

// engineRef is the engine a run uses: its kind, and the upstream release and commit this build
// runs for it.
type engineRef struct {
	Kind    string
	Version string
	Commit  string
}

func (e engineRef) String() string { return e.Kind + " " + e.Version }

// engineKindVLLM is the vLLM engine kind.
const engineKindVLLM = "vllm"

// builtEngines maps each engine kind this build registers to the upstream release and commit
// it runs, as stack.json records them (LIF-6); one per kind.
func builtEngines() map[string]engineRef {
	out := map[string]engineRef{}
	for _, u := range EngineReleases() {
		out[u.Kind] = engineRef{Kind: u.Kind, Version: u.Version, Commit: u.Commit}
	}
	return out
}

// scenarioEngineBlock is the engine block of a scenario as the selector reads it. Version is
// read only to refuse it.
type scenarioEngineBlock struct {
	Kind    string `yaml:"kind"`
	Version string `yaml:"version"`
}

// scenarioEngineFields is the part of a scenario file the selector reads. EngineVersion is read
// only to refuse it.
type scenarioEngineFields struct {
	Engine        *scenarioEngineBlock `yaml:"engine"`
	EngineVersion string               `yaml:"engine_version"`
}

// selectEngine reads the engine a scenario file names and refuses a scenario that states an
// engine release or a kind this build does not run. It returns an error and never exits.
func selectEngine(path string) (engineRef, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return engineRef{}, fmt.Errorf("scenario %s: %w", path, err)
	}
	var f scenarioEngineFields
	// The scenario is decoded strictly elsewhere (the substrate's scenario loader); this
	// reads two fields of its first document and ignores the rest.
	if err := yaml.Unmarshal(firstDocument(data), &f); err != nil {
		return engineRef{}, fmt.Errorf("scenario %s: %w", path, err)
	}
	name := filepath.Base(path)
	return resolveEngine(name, f)
}

// firstDocument is the first YAML document of a multi-document file.
func firstDocument(data []byte) []byte {
	s := strings.TrimPrefix(string(data), "---\n")
	if i := strings.Index(s, "\n---"); i >= 0 {
		s = s[:i+1]
	}
	return []byte(s)
}

// resolveEngine is selectEngine over the decoded fields of the scenario called name.
func resolveEngine(name string, f scenarioEngineFields) (engineRef, error) {
	built := builtEngines()
	if f.EngineVersion != "" {
		return engineRef{}, fmt.Errorf("scenario %q: field engine_version not found in type scenario.Scenario: "+
			"a scenario states no engine release, so state engine: {kind: %s} and no version (LIF-6)",
			name, engineKindVLLM)
	}
	if f.Engine == nil {
		return engineRef{}, fmt.Errorf("scenario %q names no engine: state engine: {kind}, for example engine: {kind: %s}",
			name, engineKindVLLM)
	}
	if f.Engine.Version != "" {
		b := built[f.Engine.Kind]
		runs := "this build does not run that kind"
		if b.Kind != "" {
			runs = fmt.Sprintf("this build runs %s at commit %s; a user who needs another upstream commit "+
				"uses the inference-sim release that runs it", b, b.Commit)
		}
		return engineRef{}, fmt.Errorf("scenario %q: field engine.version %q not found in type scenario.Engine: "+
			"a scenario states no engine release, the build fixes it (stack.json) and the Result records it; %s",
			name, f.Engine.Version, runs)
	}
	if f.Engine.Kind == "" {
		return engineRef{}, fmt.Errorf("scenario %q: engine: {kind} must state the kind", name)
	}
	ref, ok := built[f.Engine.Kind]
	if !ok {
		return engineRef{}, fmt.Errorf("scenario %q names engine kind %q; this build runs the kinds %s",
			name, f.Engine.Kind, strings.Join(sortedKinds(built), ", "))
	}
	return ref, nil
}

// sortedKinds is the kinds of built, sorted.
func sortedKinds(built map[string]engineRef) []string {
	out := make([]string, 0, len(built))
	for k := range built {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
