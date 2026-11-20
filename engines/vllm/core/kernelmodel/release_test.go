package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A scenario names its engine by kind alone; the release is the build's (LIF-6, SIM-1). It
// reads as that release and opens; a scenario stating a version, or another engine kind, is
// refused before anything opens.
func TestTheEngineFieldNamesTheKind(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "scenarios", "glm-5-h200-tp8-mtp3.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const old = "engine: {kind: vllm}\n"
	if !strings.Contains(string(raw), old) {
		t.Fatal("fixture has no engine block to edit")
	}
	for _, c := range []struct {
		name, stated, release, refusal string
	}{
		{"kind only", old, "0.33.0", ""},
		{"a version stated", "engine: {kind: vllm, version: \"0.33.0\"}\n", "", "field version not found"},
		{"another kind", "engine: {kind: sglang}\n", "", `engine.kind is "sglang"`},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := repos()
			r.Scenarios = t.TempDir()
			body := strings.Replace(string(raw), old, c.stated, 1)
			if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
			sh, err := ShapeOf("v.yaml", r)
			if c.refusal != "" {
				if err == nil || !strings.Contains(err.Error(), c.refusal) {
					t.Errorf("ShapeOf: %v, want a refusal containing %q", err, c.refusal)
				}
				return
			}
			if err != nil || sh.EngineVersion != c.release {
				t.Fatalf("ShapeOf: release %q (%v), want %q", sh.EngineVersion, err, c.release)
			}
			if _, err := Open("v.yaml", r); err != nil {
				t.Errorf("Open: %v", err)
			}
		})
	}
}
