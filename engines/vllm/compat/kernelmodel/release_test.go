package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A scenario that names its release as engine: {kind, version}, with no engine_version, opens
// and reads as that release; engine_version alone still does until M4; the two stated unequal,
// or another engine kind, are refused before anything opens.
func TestTheEngineFieldNamesTheRelease(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "scenarios", "glm-5-h200-tp8-mtp3.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const old = `engine_version: "0.29.0"` + "\n"
	if !strings.Contains(string(raw), old) {
		t.Fatal("fixture has no engine_version to edit")
	}
	for _, c := range []struct {
		name, stated, release, refusal string
	}{
		{"engine only", "engine: {kind: vllm, version: \"0.31.0\"}\n", "0.31.0", ""},
		{"engine_version only", `engine_version: "0.31.0"` + "\n", "0.31.0", ""},
		{"both, equal", "engine: {kind: vllm, version: \"0.31.0\"}\n" + `engine_version: "0.31.0"` + "\n", "0.31.0", ""},
		{"both, unequal", "engine: {kind: vllm, version: \"0.31.0\"}\n" + `engine_version: "0.29.0"` + "\n", "",
			`engine_version is "0.29.0" but engine.version is "0.31.0"`},
		{"another kind", "engine: {kind: sglang, version: \"0.31.0\"}\n", "", `engine.kind is "sglang"`},
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
