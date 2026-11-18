package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// Engine selection (design SIM-1, LIF-6). A scenario names the engine it runs by kind and
// upstream release, in engine: {kind, version}. This build runs one release of each kind it
// registers; a scenario that names another release is refused with the message LIF-6 asks for.
// A scenario that still states engine_version, the field engine: {kind, version} replaced, is
// refused with a message naming the replacement: v0.16.0 deprecated it, and flagsRemovedIn
// removed it (LIF-7).

// engineRef is the engine a scenario names: its kind and the upstream release it models.
type engineRef struct {
	Kind    string `yaml:"kind"`
	Version string `yaml:"version"`
}

func (e engineRef) String() string { return e.Kind + " " + e.Version }

// engineKindVLLM is the vLLM engine kind.
const engineKindVLLM = "vllm"

// builtEngineReleases maps each engine kind this build registers to the upstream release it
// runs. It is the list stack.json records (LIF-6); one release per kind.
func builtEngineReleases() map[string]string {
	return map[string]string{engineKindVLLM: "0.33.0", engineKindSGLang: "0.5.23"}
}

// lastReleaseRunning maps an engine release this build does not run to the newest
// inference-sim release that runs it. v0.15.0 is the last release that runs vLLM 0.29.0 and
// 0.31.0 (release plan §3.3, "Removing blis-vllm's 0.29 and 0.31 code").
func lastReleaseRunning() map[engineRef]string {
	return map[engineRef]string{
		{Kind: engineKindVLLM, Version: "0.29.0"}: "v0.15.0",
		{Kind: engineKindVLLM, Version: "0.31.0"}: "v0.15.0",
		// v0.17.0 is the last release that runs vLLM 0.32.0 (its stack.json), before the move to 0.33.0.
		{Kind: engineKindVLLM, Version: "0.32.0"}: "v0.17.0",
	}
}

// scenarioEngineFields is the part of a scenario file the selector reads. EngineVersion is read
// only to refuse it by name.
type scenarioEngineFields struct {
	Engine        *engineRef `yaml:"engine"`
	EngineVersion string     `yaml:"engine_version"`
}

// selectEngine reads the engine a scenario file names and refuses one this build does not
// run. It returns an error and never exits.
func selectEngine(path string) (engineRef, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return engineRef{}, fmt.Errorf("scenario %s: %w", path, err)
	}
	var f scenarioEngineFields
	// The scenario is decoded strictly elsewhere (blis-substrate's scenario loader); this
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
	example := fmt.Sprintf("engine: {kind: %s, version: %q}", engineKindVLLM, builtEngineReleases()[engineKindVLLM])
	if f.EngineVersion != "" {
		suggest := fmt.Sprintf("engine: {kind: %s, version: %q}", engineKindVLLM, f.EngineVersion)
		if f.Engine != nil {
			suggest = "engine: {kind, version} alone"
		}
		return engineRef{}, fmt.Errorf("scenario %q states engine_version %q, which was removed in %s: "+
			"state %s instead", name, f.EngineVersion, flagsRemovedIn, suggest)
	}
	if f.Engine == nil {
		return engineRef{}, fmt.Errorf("scenario %q names no engine: state engine: {kind, version}, for example %s",
			name, example)
	}
	ref := *f.Engine
	if ref.Kind == "" || ref.Version == "" {
		return engineRef{}, fmt.Errorf("scenario %q: engine: {kind: %q, version: %q} must state both kind and version",
			name, ref.Kind, ref.Version)
	}
	built := builtEngineReleases()
	runs, ok := built[ref.Kind]
	if !ok {
		return engineRef{}, fmt.Errorf("scenario %q names engine kind %q; this build runs the kinds %s",
			name, ref.Kind, strings.Join(sortedKinds(built), ", "))
	}
	if ref.Version != runs {
		newest := "no inference-sim release runs it"
		if r, ok := lastReleaseRunning()[ref]; ok {
			newest = "the newest inference-sim release that runs it is " + r
		}
		return engineRef{}, fmt.Errorf("scenario %q names engine %s; this build runs %s %s, and %s",
			name, ref, ref.Kind, runs, newest)
	}
	return ref, nil
}

// sortedKinds is the kinds of built, sorted.
func sortedKinds(built map[string]string) []string {
	out := make([]string, 0, len(built))
	for k := range built {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
