package decision

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// copyArtifact copies the fixture artifact into a temporary directory and returns it, with the
// artifact.json as a map the test edits and writes back.
func copyArtifact(t *testing.T) (dir string, doc map[string]any) {
	t.Helper()
	dir = t.TempDir()
	src := filepath.Join("testdata", "artifact")
	for _, f := range append([]string{"artifact.json"}, ModelFile(FrontXI), ModelFile(FrontXT), ModelFile(FrontIT), ModelFile(FrontXIT)) {
		b, err := os.ReadFile(filepath.Join(src, f))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, f), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	b, _ := os.ReadFile(filepath.Join(dir, "artifact.json"))
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	return dir, doc
}

func writeArtifact(t *testing.T, dir string, doc map[string]any) {
	t.Helper()
	b, _ := json.Marshal(doc)
	if err := os.WriteFile(filepath.Join(dir, "artifact.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestTheFixtureArtifactLoads(t *testing.T) {
	s, a, err := LoadStage(filepath.Join("testdata", "artifact"), DefaultEpsilon)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Models) != 4 || a.RoleCounts.Fit < 1 || a.Epsilon != DefaultEpsilon {
		t.Errorf("stage %+v artifact %+v", s, a)
	}
}

// DEC-5: an artifact is refused when its epsilon differs from the one in use, and when any
// recorded field is wrong; the message names the field.
func TestAnArtifactWithADifferentEpsilonOrABadFieldIsRefused(t *testing.T) {
	for _, c := range []struct {
		name string
		edit func(m map[string]any)
		eps  float64
		want string
	}{
		{"another epsilon", func(m map[string]any) { m["epsilon"] = 0.1 }, DefaultEpsilon, "epsilon 0.1"},
		{"the command asks for another epsilon", func(m map[string]any) {}, 0.1, "fitted with epsilon 0.05 and this command uses 0.1"},
		{"epsilon out of range", func(m map[string]any) { m["epsilon"] = 0.0 }, 0.0, "not in (0, 1)"},
		{"no hash", func(m map[string]any) { m["training_data_hash"] = "" }, DefaultEpsilon, "training_data_hash"},
		{"short commit", func(m map[string]any) { m["writer_commit"] = "abc" }, DefaultEpsilon, "writer_commit"},
		{"no fit units", func(m map[string]any) { m["role_counts"] = map[string]any{"fit": 0, "select": 0, "validate": 0} }, DefaultEpsilon, "role_counts"},
		{"a model identifier as input", func(m map[string]any) {
			m["features"] = append(m["features"].([]any), "model_id")
		}, DefaultEpsilon, `"model_id"`},
		{"a repeated feature", func(m map[string]any) {
			m["features"] = append(m["features"].([]any), "dX")
		}, DefaultEpsilon, "listed twice"},
		{"wrong kind", func(m map[string]any) { m["kind"] = "Other" }, DefaultEpsilon, "kind"},
		{"an unknown field", func(m map[string]any) { m["extra"] = 1 }, DefaultEpsilon, "unknown field"},
		{"feature count differs from the models'", func(m map[string]any) {
			m["features"] = m["features"].([]any)[:53]
		}, DefaultEpsilon, "takes 54 inputs"},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir, doc := copyArtifact(t)
			c.edit(doc)
			writeArtifact(t, dir, doc)
			_, _, err := LoadStage(dir, c.eps)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("got %v, want an error containing %q", err, c.want)
			}
		})
	}
}

func TestAnArtifactMissingAModelIsRefused(t *testing.T) {
	dir, _ := copyArtifact(t)
	if err := os.Remove(filepath.Join(dir, ModelFile(FrontIT))); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadStage(dir, DefaultEpsilon); err == nil || !strings.Contains(err.Error(), "model-IT.json") {
		t.Errorf("got %v", err)
	}
}
