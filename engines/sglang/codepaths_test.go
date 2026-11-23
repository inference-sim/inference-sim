// Original to blis-sglang.

package blissglang

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/codepaths"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
)

// code_paths.json is the identity of the engine code the host, scheduler and graph-launch costs
// depend on (Table DC). This test holds the committed file to the document type blis-substrate
// defines for it:
//
//   - it loads through the substrate's strict loader, which recomputes every CodeID from the
//     function hashes: the generator's Python restatement of the rule cannot drift from the Go rule;
//   - it is for the release this build models (Upstream);
//   - it is in canonical form, the bytes the substrate's Marshal writes;
//   - it is the file this build embeds;
//   - it lists the functions code_paths/spec.json names, no more and no fewer.
func TestTheCodePathTableLoadsAndNamesThisRelease(t *testing.T) {
	tab, err := codepaths.Load("code_paths.json")
	if err != nil {
		t.Fatal(err)
	}
	if tab.Engine.Kind != Kind || tab.Engine.Version != Upstream.Version || tab.Engine.Commit != Upstream.Commit {
		t.Errorf("code_paths.json is for %+v; this build models %s %s at %s", tab.Engine, Kind, Upstream.Version, Upstream.Commit)
	}
	raw, err := os.ReadFile("code_paths.json")
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := tab.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, canonical) {
		t.Error("code_paths.json is not in the substrate's canonical form; regenerate it with substrate/tools/codepaths/regen.py")
	}
	if !bytes.Equal(CodePathsJSON, raw) {
		t.Error("CodePathsJSON differs from code_paths.json")
	}
	if _, err := CodePaths(); err != nil {
		t.Errorf("the embedded table does not load: %v", err)
	}
}

// Each path states a CodeID, and the host and graph paths state different ones, so a coefficient
// keyed by one is never matched by the other.
func TestEachCodePathStatesItsOwnCodeID(t *testing.T) {
	host, graphs := CodeIDOf(coefficient.CodePathHost), CodeIDOf(coefficient.CodePathGraphs)
	if len(host) != 64 || len(graphs) != 64 {
		t.Fatalf("CodeIDs are not SHA-256 digests: %q, %q", host, graphs)
	}
	if host == graphs {
		t.Error("the host and graph paths state one CodeID")
	}
}

// The spec lists the functions the table lists: a function added to the spec and not regenerated,
// or hashed and no longer named, fails here before CI's regeneration does.
func TestTheTableListsTheFunctionsTheSpecNames(t *testing.T) {
	checkSpecMatchesTable(t, "code_paths/spec.json", "code_paths.json")
}

// The registry keys its engine-path coefficients by the CodeIDs in registry/inputs/engine_code_ids.json
// (the substrate reads no file of an engine module, so the CodeIDs reach it there, written by
// tools/codepaths/export_code_ids.py). This engine's entry must be the CodeIDs it states, or
// the registry would key a coefficient to code that is not the code this build runs.
func TestTheRegistryInputStatesTheCodeIDsThisBuildStates(t *testing.T) {
	raw, err := os.ReadFile("../../substrate/registry/inputs/engine_code_ids.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Engines map[string]struct {
			Version string `json:"version"`
			Commit  string `json:"commit"`
			Host    string `json:"host"`
			Graphs  string `json:"graphs"`
		} `json:"engines"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	got, ok := doc.Engines[Kind]
	if !ok {
		t.Fatalf("registry/inputs/engine_code_ids.json lists no %s", Kind)
	}
	if got.Version != Upstream.Version || got.Commit != Upstream.Commit {
		t.Errorf("the registry input is for %s at %s, this build models %s at %s", got.Version, got.Commit, Upstream.Version, Upstream.Commit)
	}
	if got.Host != CodeIDOf(coefficient.CodePathHost) || got.Graphs != CodeIDOf(coefficient.CodePathGraphs) {
		t.Errorf("registry/inputs/engine_code_ids.json states host %s, graphs %s; this build states host %s, graphs %s; "+
			"run tools/codepaths/export_code_ids.py", got.Host, got.Graphs, CodeIDOf(coefficient.CodePathHost), CodeIDOf(coefficient.CodePathGraphs))
	}
}
