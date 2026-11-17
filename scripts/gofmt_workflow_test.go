package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The gofmt job's step fails on an unformatted Go file anywhere in the tree and names it, passes
// on a formatted tree, and fails on a tree with no Go file (X18). The script the job runs is run
// in a throwaway git repository.
func TestGofmtWorkflowStepFailsOnAnUnformattedFile(t *testing.T) {
	script, err := filepath.Abs("gofmt-check.sh")
	if err != nil {
		t.Fatal(err)
	}
	run := func(files map[string]string) (string, error) {
		dir := t.TempDir()
		for name, body := range files {
			p := filepath.Join(dir, name)
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		for _, args := range [][]string{{"init", "-q"}, {"add", "-A"}} {
			c := exec.Command("git", args...)
			c.Dir = dir
			if out, err := c.CombinedOutput(); err != nil {
				t.Fatalf("git %v: %v\n%s", args, err, out)
			}
		}
		c := exec.Command("bash", script)
		c.Dir = dir
		out, err := c.CombinedOutput()
		return string(out), err
	}
	clean := "package a\n\nfunc F() {}\n"
	dirty := "package b\nfunc  G( ) {}\n"

	if out, err := run(map[string]string{"a.go": clean, "sub/dir/c.go": clean}); err != nil {
		t.Errorf("a formatted tree failed: %v\n%s", err, out)
	}
	out, err := run(map[string]string{"a.go": clean, "sub/dir/b.go": dirty})
	if err == nil || !strings.Contains(out, "sub/dir/b.go") {
		t.Errorf("a nested unformatted file: err %v, output %q; want a failure naming sub/dir/b.go", err, out)
	}
	if out, err := run(map[string]string{"README.md": "x"}); err == nil {
		t.Errorf("a tree with no Go file passed:\n%s", out)
	}
}
