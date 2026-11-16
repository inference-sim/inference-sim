package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// scripts/deadcode-check.sh is LIF-7's check: no function that no shipped main package reaches,
// unless it is allow-listed with a reason, and no allow-list entry that has gone stale. These
// tests run the script over small fixture modules, never over this repository's own source.

const fixtureModule = "example.com/deadfixture"

// fixtureDir writes a module root holding go.mod and the given files, and returns it.
func fixtureDir(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	files["go.mod"] = "module " + fixtureModule + "\n\ngo 1.26\n"
	for name, body := range files {
		p := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// runDeadcodeCheck runs the script in dir with the allow-list text; report, when not empty, is
// deadcode's output, read instead of running deadcode. It returns the exit code and the output.
func runDeadcodeCheck(t *testing.T, dir, allow, report string) (int, string) {
	t.Helper()
	script, err := filepath.Abs("deadcode-check.sh")
	if err != nil {
		t.Fatal(err)
	}
	allowPath := filepath.Join(t.TempDir(), "allow.txt")
	if err := os.WriteFile(allowPath, []byte(allow), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("bash", script, allowPath)
	cmd.Dir = dir
	cmd.Env = os.Environ()
	if report != "" {
		reportPath := filepath.Join(t.TempDir(), "report.txt")
		if err := os.WriteFile(reportPath, []byte(report), 0o644); err != nil {
			t.Fatal(err)
		}
		cmd.Env = append(cmd.Env, "DEADCODE_REPORT="+reportPath)
	}
	out, err := cmd.CombinedOutput()
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode(), string(out)
	}
	if err != nil {
		t.Fatalf("running deadcode-check.sh: %v\n%s", err, out)
	}
	return 0, string(out)
}

// A report naming one dead function in a sub-package and one in the root package.
const twoDead = fixtureModule + "/lib Dead lib/lib.go:3:6\n" +
	fixtureModule + " server.close main.go:9:18\n"

func TestDeadcodeCheckRules(t *testing.T) {
	dir := fixtureDir(t, map[string]string{})
	for _, c := range []struct {
		name, allow, report string
		pass                bool
		want                []string // substrings of the output
	}{
		{name: "an unlisted dead function fails, naming it",
			allow: "lib Dead stays: reached through reflection\n", report: twoDead,
			want: []string{"main.go:9:18", "\". server.close\"", "not on the allow-list"}},
		{name: "a listed dead function passes",
			allow:  "# comment\n\nlib Dead reached through reflection\n. server.close an interface method the type must have\n",
			report: twoDead, pass: true},
		{name: "an entry without a reason fails",
			allow:  "lib Dead\n. server.close an interface method the type must have\n",
			report: twoDead, want: []string{"line 1 (lib Dead) gives no reason"}},
		{name: "a stale entry fails",
			allow: "lib Dead reached through reflection\n. server.close an interface method\n" +
				"lib Gone was dead once\n",
			report: twoDead, want: []string{"line 3 names \"lib Gone\"", "remove the entry"}},
		{name: "an entry listed twice fails",
			allow:  "lib Dead one reason\nlib Dead another reason\n. server.close an interface method\n",
			report: twoDead, want: []string{"line 2 lists lib Dead again"}},
		{name: "contracts/ is exempt",
			allow:  "",
			report: fixtureModule + "/contracts/engine Engine.Step contracts/engine/engine.go:4:2\n",
			pass:   true},
	} {
		t.Run(c.name, func(t *testing.T) {
			code, out := runDeadcodeCheck(t, dir, c.allow, c.report)
			if c.pass != (code == 0) {
				t.Fatalf("exit %d, want pass=%v\n%s", code, c.pass, out)
			}
			for _, w := range c.want {
				if !strings.Contains(out, w) {
					t.Errorf("output does not contain %q:\n%s", w, out)
				}
			}
		})
	}
}

// End to end, with the pinned deadcode: a function only a test calls counts as unreachable, a
// function the main package calls does not, and a package no main package imports is checked.
func TestDeadcodeCheckRunsDeadcode(t *testing.T) {
	dir := fixtureDir(t, map[string]string{
		"main.go":         "package main\n\nimport \"" + fixtureModule + "/lib\"\n\nfunc main() { lib.Live() }\n",
		"lib/lib.go":      "package lib\n\nfunc Live() {}\n\nfunc TestOnly() {}\n",
		"lib/lib_test.go": "package lib\n\nimport \"testing\"\n\nfunc TestX(t *testing.T) { TestOnly() }\n",
		"orphan/o.go":     "package orphan\n\nfunc Unlinked() {}\n",
	})
	code, out := runDeadcodeCheck(t, dir, "", "")
	if code == 0 {
		t.Fatalf("an empty allow-list passed a module with dead functions:\n%s", out)
	}
	for _, w := range []string{"\"lib TestOnly\"", "\"orphan Unlinked\""} {
		if !strings.Contains(out, w) {
			t.Errorf("output does not name %s:\n%s", w, out)
		}
	}
	if strings.Contains(out, "Live") {
		t.Errorf("a function main calls was reported:\n%s", out)
	}
	allow := "lib TestOnly only its package's test calls it\norphan Unlinked no main imports it\n"
	if code, out := runDeadcodeCheck(t, dir, allow, ""); code != 0 {
		t.Errorf("exit %d with both functions allow-listed:\n%s", code, out)
	}
}
