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
// upstream release. This build runs one release of each kind it registers; a scenario that
// names another release is refused with the message LIF-6 asks for.
//
// The scenario type this build reads (blis-schemas v0.10.0 spec/scenario) has only the field
// engine_version, with no kind. Until the scenario type in blis-substrate contracts/spec
// carries engine: {kind, version}, the selector reads both spellings from the scenario file
// itself: engine: {kind, version} is the field, and engine_version is its deprecated form,
// accepted with a warning until M4 (release plan §3.3, "Relaxing the deprecation rule"). A
// scenario that states engine_version states no kind; it names a vLLM release, since vLLM is
// the only engine any scenario written with that field ran.

// engineRef is the engine a scenario names: its kind and the upstream release it models.
type engineRef struct {
	Kind    string `yaml:"kind"`
	Version string `yaml:"version"`
}

func (e engineRef) String() string { return e.Kind + " " + e.Version }

// engineKindVLLM is the engine kind a scenario stating only engine_version names.
const engineKindVLLM = "vllm"

// builtEngineReleases maps each engine kind this build registers to the upstream release it
// runs. It is the list stack.json records (LIF-6); one release per kind.
func builtEngineReleases() map[string]string {
	return map[string]string{engineKindVLLM: "0.32.0"}
}

// lastReleaseRunning maps an engine release this build does not run to the newest
// inference-sim release that runs it. v0.15.0 is the last release that runs vLLM 0.29.0 and
// 0.31.0 (release plan §3.3, "Removing blis-vllm's 0.29 and 0.31 code").
func lastReleaseRunning() map[engineRef]string {
	return map[engineRef]string{
		{Kind: engineKindVLLM, Version: "0.29.0"}: "v0.15.0",
		{Kind: engineKindVLLM, Version: "0.31.0"}: "v0.15.0",
	}
}

// scenarioEngineFields is the part of a scenario file the selector reads.
type scenarioEngineFields struct {
	Engine        *engineRef `yaml:"engine"`
	EngineVersion string     `yaml:"engine_version"`
}

// selectedEngine is the engine a scenario selects, and whether it named it with the
// deprecated engine_version.
type selectedEngine struct {
	Ref        engineRef
	Deprecated bool
}

// selectEngine reads the engine a scenario file names and refuses one this build does not
// run. It returns an error and never exits.
func selectEngine(path string) (selectedEngine, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return selectedEngine{}, fmt.Errorf("scenario %s: %w", path, err)
	}
	var f scenarioEngineFields
	// The scenario is decoded strictly elsewhere (blis-schemas LoadBundle); this reads two
	// fields of its first document and ignores the rest.
	if err := yaml.Unmarshal(firstDocument(data), &f); err != nil {
		return selectedEngine{}, fmt.Errorf("scenario %s: %w", path, err)
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
func resolveEngine(name string, f scenarioEngineFields) (selectedEngine, error) {
	var sel selectedEngine
	switch {
	case f.Engine != nil && f.EngineVersion != "":
		return sel, fmt.Errorf("scenario %q states both engine: {kind, version} and engine_version %q; "+
			"state engine: {kind, version} only", name, f.EngineVersion)
	case f.Engine != nil:
		if f.Engine.Kind == "" || f.Engine.Version == "" {
			return sel, fmt.Errorf("scenario %q: engine: {kind: %q, version: %q} must state both kind and version",
				name, f.Engine.Kind, f.Engine.Version)
		}
		sel.Ref = *f.Engine
	case f.EngineVersion != "":
		sel.Ref = engineRef{Kind: engineKindVLLM, Version: f.EngineVersion}
		sel.Deprecated = true
	default:
		return sel, fmt.Errorf("scenario %q names no engine: state engine: {kind, version}, for example "+
			"engine: {kind: vllm, version: %q}", name, builtEngineReleases()[engineKindVLLM])
	}
	built := builtEngineReleases()
	runs, ok := built[sel.Ref.Kind]
	if !ok {
		return sel, fmt.Errorf("scenario %q names engine kind %q; this build runs the kinds %s",
			name, sel.Ref.Kind, strings.Join(sortedKinds(built), ", "))
	}
	if sel.Ref.Version != runs {
		newest := "no inference-sim release runs it"
		if r, ok := lastReleaseRunning()[sel.Ref]; ok {
			newest = "the newest inference-sim release that runs it is " + r
		}
		return sel, fmt.Errorf("scenario %q names engine %s; this build runs %s %s, and %s",
			name, sel.Ref, sel.Ref.Kind, runs, newest)
	}
	return sel, nil
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

// engineDeprecationWarning is the warning a scenario stating engine_version gets.
func engineDeprecationWarning(name string, ref engineRef) string {
	return fmt.Sprintf("scenario %q states engine_version %q, which is deprecated and removed in M4: "+
		"state engine: {kind: %s, version: %q} instead", name, ref.Version, ref.Kind, ref.Version)
}
