package cmd

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A replay descriptor controls the corpus, the phase windows, the warm-up and the acceptance
// provenance, so an unknown field fails in each supported form and names the key, where it once
// fell back to a default and ran a different workload.
func TestLoadReplaySpec_RefusesUnknownFieldsInEveryForm(t *testing.T) {
	const body = "concurrency: 4\n"
	indent := func(s, pad string) string {
		var b strings.Builder
		for _, l := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
			b.WriteString(pad + l + "\n")
		}
		return b.String()
	}
	forms := map[string]func(extra string) string{
		"bare":            func(extra string) string { return body + extra },
		"replay:":         func(extra string) string { return "replay:\n" + indent(body+extra, "  ") },
		"workload.replay": func(extra string) string { return "workload:\n  replay:\n" + indent(body+extra, "    ") },
	}
	write := func(doc string) string {
		p := filepath.Join(t.TempDir(), "spec.yaml")
		if err := os.WriteFile(p, []byte(doc), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	for name, form := range forms {
		t.Run(name, func(t *testing.T) {
			if _, err := loadReplaySpec(write(form(""))); err != nil {
				t.Fatalf("a correct document is refused: %v", err)
			}
			_, err := loadReplaySpec(write(form("benchmark_duraton_s: 900\n")))
			if err == nil || !strings.Contains(err.Error(), "benchmark_duraton_s") {
				t.Errorf("a misspelled field: %v, want an error naming benchmark_duraton_s", err)
			}
		})
	}
	wrapped := map[string]string{
		"replay: beside a stray key":  "replay:\n  concurrency: 4\nbenchmark_duraton_s: 900\n",
		"workload.replay beside one":  "workload:\n  replay:\n    concurrency: 4\n  stray: 1\n",
		"workload.replay under a key": "stray: 1\nworkload:\n  replay:\n    concurrency: 4\n",
	}
	for name, doc := range wrapped {
		t.Run(name, func(t *testing.T) {
			_, err := loadReplaySpec(write(doc))
			if err == nil || !strings.Contains(err.Error(), "stray") && !strings.Contains(err.Error(), "benchmark_duraton_s") {
				t.Errorf("an unknown key beside the wrapper: %v, want an error naming it", err)
			}
		})
	}
}

// The substrate's own replay documents carry _comment keys at the top level beside replay:, and a
// bare descriptor may carry them too. They name no field, so the strict loader accepts them; a
// valid document that loaded before must still load. The files are the substrate's example and
// its catalog workload, read from the pinned module.
func TestLoadReplaySpec_AcceptsTheSubstratesCommentKeys(t *testing.T) {
	dir := func(rel string) string {
		out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "github.com/inference-sim/blis-substrate").Output()
		if err != nil {
			t.Fatal(err)
		}
		return filepath.Join(strings.TrimSpace(string(out)), rel)
	}
	for _, rel := range []string{"docs/contracts/examples/workload-replay.yaml", "catalog/workloads/agentx-mvp.yaml"} {
		t.Run(filepath.Base(rel), func(t *testing.T) {
			r, err := loadReplaySpec(dir(rel))
			if err != nil {
				t.Fatalf("%s: %v", rel, err)
			}
			if r.Corpus.Loader == "" {
				t.Errorf("%s loaded with no corpus: the descriptor was not read", rel)
			}
		})
	}
	t.Run("a bare descriptor with a comment key", func(t *testing.T) {
		p := filepath.Join(t.TempDir(), "spec.yaml")
		if err := os.WriteFile(p, []byte("_comment: why\nconcurrency: 4\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := loadReplaySpec(p); err != nil {
			t.Errorf("a bare descriptor with _comment: %v", err)
		}
	})
}
