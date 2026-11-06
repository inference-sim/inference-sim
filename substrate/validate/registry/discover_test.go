package registry

import (
	"os"
	"path/filepath"
	"testing"
)

func mk(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, "coefficients", rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// SetPaths finds .yaml and .yml at any depth, case-insensitively, and ignores other
// files — the predicate that keeps this gate from under-covering what blis-registry's own
// validator (validate.py) sees, in deterministic sorted order.
func TestSetPathsDiscoversYamlYmlNestedAndUppercase(t *testing.T) {
	root := t.TempDir()
	mk(t, root, "a.yaml", "x")
	mk(t, root, "b.yml", "x")
	mk(t, root, filepath.Join("sub", "c.yaml"), "x")
	mk(t, root, "D.YAML", "x")
	mk(t, root, "notes.txt", "x")
	mk(t, root, "config.json", "x")

	got, err := SetPaths(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 {
		t.Fatalf("found %d sets, want 4 (.txt/.json excluded): %v", len(got), got)
	}
	for i := 1; i < len(got); i++ {
		if got[i-1] > got[i] {
			t.Errorf("paths not sorted: %v", got)
		}
	}
}

// Two names for one file — a symlink and its target — resolve to one path and are counted
// once, so the same bytes are not validated twice.
func TestSetPathsDeduplicatesSymlinkAliases(t *testing.T) {
	root := t.TempDir()
	mk(t, root, "real.yaml", "x")
	coeff := filepath.Join(root, "coefficients")
	if err := os.Symlink(filepath.Join(coeff, "real.yaml"), filepath.Join(coeff, "alias.yaml")); err != nil {
		t.Skipf("symlinks unavailable on this filesystem: %v", err)
	}
	got, err := SetPaths(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("found %d paths, want 1 (symlink alias must de-dup): %v", len(got), got)
	}
}

// An absent coefficients/ directory is "no sets", not an error, so a caller can decide
// what an empty root means.
func TestSetPathsAbsentDirIsEmpty(t *testing.T) {
	got, err := SetPaths(t.TempDir())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("want no paths for an absent coefficients/, got %v", got)
	}
}

// AcceptancePaths applies SetPaths' predicate to acceptance/, and an absent directory is
// no sets rather than an error.
func TestAcceptancePaths(t *testing.T) {
	root := t.TempDir()
	if got, err := AcceptancePaths(root); err != nil || len(got) != 0 {
		t.Fatalf("absent acceptance/: got %v, %v", got, err)
	}
	for _, rel := range []string{"a.yaml", "B.YML", filepath.Join("sub", "c.yaml"), "notes.txt"} {
		p := filepath.Join(root, "acceptance", rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := AcceptancePaths(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Errorf("found %v, want the three YAML files", got)
	}
	if sets, _ := SetPaths(root); len(sets) != 0 {
		t.Errorf("SetPaths found acceptance files: %v", sets)
	}
}
