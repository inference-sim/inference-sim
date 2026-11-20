package contracts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

const catalogReplay = `_comment: the 256k AgentX corpus
replay:
  scenario: inferencex-agentx-mvp
  harness_version: 0.12.0
  corpus:
    loader: semianalysis_cc_traces_weka_062126_256k
    format: weka-trace-v7
    encoding: jsonl
    source: {huggingface: semianalysisai/cc-traces-weka-062126-256k, revision: 8fecd2fc}
    sha256: e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef
  benchmark_duration_s: 900
`

// LoadCatalogWorkload reads either form of a catalog workload, by the top-level replay:
// key, and LoadWorkload keeps reading shapes only.
func TestLoadCatalogWorkload(t *testing.T) {
	w, err := LoadCatalogWorkload(write(t, "agentx.yaml", catalogReplay))
	if err != nil {
		t.Fatal(err)
	}
	if w.Shape != nil || w.Replay == nil || w.Replay.Name != "agentx" ||
		w.Replay.Corpus.Source.Revision != "8fecd2fc" || !w.Validate().OK() {
		t.Fatalf("replay entry loaded as %+v", w)
	}
	if _, err := LoadWorkload(write(t, "agentx.yaml", catalogReplay)); err == nil ||
		!strings.Contains(err.Error(), "LoadCatalogWorkload") {
		t.Errorf("LoadWorkload accepted a replay: %v", err)
	}
	shape, err := LoadCatalogWorkload("../catalog/workloads/chatbot.yaml")
	if err != nil || shape.Shape == nil || shape.Name() != "chatbot" {
		t.Fatalf("shape entry: %+v, %v", shape, err)
	}
	for name, body := range map[string]string{
		"nested typo":         strings.Replace(catalogReplay, "revision:", "revison:", 1),
		"unknown top-level":   catalogReplay + "prompt: {tokens: 1}\n",
		"lane count in entry": catalogReplay + "  concurrency: 8\n",
		"empty replay":        "replay:\n",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadCatalogWorkload(write(t, "agentx.yaml", body)); err == nil {
				t.Error("loaded")
			}
		})
	}
}

func replayCatalog(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for name, body := range map[string]string{
		"agentx.yaml": catalogReplay, "chatbot.yaml": "prompt: {tokens: 256}\noutput: {tokens: 256}\n",
	} {
		p := filepath.Join(root, "workloads", name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// A binding's replay reference resolves against the catalog to the merged replay, and
// every way it can fail to is reported where the reference is written.
func TestResolveReplayReference(t *testing.T) {
	root := replayCatalog(t)
	ref := &workload.Replay{Workload: "agentx", Concurrency: 16}
	got, err := ResolveReplay(root, ref)
	if err != nil {
		t.Fatal(err)
	}
	if got.Concurrency != 16 || got.Scenario != workload.ScenarioInferenceXAgentXMVP || got.Duration() != 900 {
		t.Errorf("resolved to %+v", got)
	}
	if inline, _ := ResolveReplay("", got); inline != got {
		t.Error("an inline replay did not resolve to itself")
	}

	s := scenarioRef()
	s.Workload = &workload.Binding{Replay: ref}
	for _, p := range ValidateAgainstCatalog(Bundle{Scenario: s}, root, "").Problems() {
		if strings.HasPrefix(p.Path, "references.workload") {
			t.Errorf("a resolvable reference was reported: %s", p)
		}
	}
	for name, c := range map[string]struct {
		binding *workload.Binding
		path    string
		has     string
	}{
		"missing entry":         {&workload.Binding{Replay: &workload.Replay{Workload: "nope", Concurrency: 1}}, "references.workload.replay.workload", "looked for"},
		"shape named as replay": {&workload.Binding{Replay: &workload.Replay{Workload: "chatbot", Concurrency: 1}}, "references.workload.replay.workload", "is a shape"},
		"replay named as shape": {&workload.Binding{Shape: "agentx"}, "references.workload.shape", "is a replay workload"},
		"invalid run choice": {&workload.Binding{Replay: &workload.Replay{Workload: "agentx", Concurrency: 1,
			ReplayDefinition: workload.ReplayDefinition{BenchmarkDurationS: 60}}}, "references.workload.replay.workload", "900"},
	} {
		t.Run(name, func(t *testing.T) {
			s := scenarioRef()
			s.Workload = c.binding
			for _, p := range ValidateAgainstCatalog(Bundle{Scenario: s}, root, "").Problems() {
				if p.Path == c.path && strings.Contains(p.Message, c.has) {
					return
				}
			}
			t.Errorf("no problem at %s containing %q", c.path, c.has)
		})
	}
}
