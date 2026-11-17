package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A change that moves a byte-gate golden must carry a statement of what moved and why (#2107 I7).
// The check is run in a throwaway repository on commits that do and do not move a golden.
func TestGoldenChangeCheck(t *testing.T) {
	script, err := filepath.Abs("golden-change-check.sh")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		c := exec.Command("git", append([]string{"-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"}, args...)...)
		c.Dir = dir
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	write := func(rel, body string) {
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git("init", "-q")
	write("testdata/bytegate/golden/a.json", "{}\n")
	write("code.go", "package a\n")
	git("add", "-A")
	git("commit", "-q", "-m", "base")
	base := git("rev-parse", "HEAD")

	write("code.go", "package a\n// changed\n")
	git("commit", "-qam", "code only")
	codeOnly := git("rev-parse", "HEAD")

	write("testdata/bytegate/golden/a.json", `{"moved":true}`+"\n")
	git("commit", "-qam", "golden")
	golden := git("rev-parse", "HEAD")

	check := func(head, body string) (string, error) {
		c := exec.Command("bash", script, base, head)
		c.Dir = dir
		c.Stdin = strings.NewReader(body)
		out, err := c.CombinedOutput()
		return string(out), err
	}
	if out, err := check(codeOnly, ""); err != nil {
		t.Errorf("a change that leaves the goldens alone needs no statement: %v\n%s", err, out)
	}
	for name, body := range map[string]string{
		"no statement":       "What changes: x\n",
		"an empty statement": "Golden change:\n\nWhy: because\n",
		"a bare mention":     "Golden change: yes\n",
	} {
		out, err := check(golden, body)
		if err == nil || !strings.Contains(out, "a.json") {
			t.Errorf("%s: err %v, output %q; want a failure naming a.json", name, err, out)
		}
	}
	for name, body := range map[string]string{
		"one line": "Golden change: a.json gains moved:true because the router now reports it.\n",
		"wrapped":  "Intro\n\nGolden change:\n  a.json gains moved:true,\n  because the router now reports it.\n\nMore\n",
	} {
		if out, err := check(golden, body); err != nil {
			t.Errorf("%s: a stated change was refused: %v\n%s", name, err, out)
		}
	}
}
