package main

import (
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The command checks the committed ledger and lints the repository's workflows, and exits 2
// on misuse.
func TestRun(t *testing.T) {
	for _, c := range []struct {
		args []string
		want int
	}{
		{[]string{"check", "../../../catalog/ledger"}, 0},
		{[]string{"lint-workflows", "../../../.github/workflows"}, 0},
		{[]string{"check", "../../../catalog"}, 1},
		{[]string{"bogus"}, 2},
		{[]string{"check"}, 2},
		{nil, 2},
	} {
		var out, errb bytes.Buffer
		if got := run(c.args, &out, &errb); got != c.want {
			t.Errorf("ledger %v = %d, want %d: %s", c.args, got, c.want, errb.String())
		}
	}
}

// `check -release` is the release gate on a tree: a ledger with no published lists loads (exit 0)
// and is not releasable (exit 1, naming the missing lists). The copy has any published/
// directory removed, so the test keeps its meaning once the lists are committed.
func TestCheckReleaseRefusesATreeWithoutLists(t *testing.T) {
	src := "../../../catalog/ledger"
	dst := t.TempDir()
	err := filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		if rel == "published" {
			return filepath.SkipDir
		}
		if d.IsDir() {
			return os.MkdirAll(filepath.Join(dst, rel), 0o755)
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(dst, rel), b, 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if got := run([]string{"check", dst}, &out, &errb); got != 0 {
		t.Fatalf("check on a tree with no lists = %d: %s", got, errb.String())
	}
	out.Reset()
	errb.Reset()
	if got := run([]string{"check", "-release", dst}, &out, &errb); got != 1 || !strings.Contains(errb.String(), "not releasable") || !strings.Contains(errb.String(), "anchor list is not published") {
		t.Errorf("check -release on a tree with no lists = %d: %s", got, errb.String())
	}
}
