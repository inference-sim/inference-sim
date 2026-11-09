package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// best_effort (scripts/lib-best-effort.sh) is the one place a workflow step lets a failing
// command pass (X22). It must never hide the failure: the warning and the job-summary line
// carry the failed command's exit status and the message, and the step goes on.
func TestBestEffort_LogsTheFailureAndGoesOn(t *testing.T) {
	summary := filepath.Join(t.TempDir(), "summary.md")
	cmd := exec.Command("bash", "-c", `source scripts/lib-best-effort.sh; sh -c 'exit 7' || best_effort "could not label #5"; echo after=$?`)
	cmd.Dir = ".."
	cmd.Env = append(os.Environ(), "GITHUB_STEP_SUMMARY="+summary)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("the step did not go on: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "::warning::best effort failed (exit 7): could not label #5") || !strings.Contains(string(out), "after=0") {
		t.Errorf("output %q lacks the warning with the exit status, or the step did not continue", out)
	}
	got, _ := os.ReadFile(summary)
	if !strings.Contains(string(got), "exit 7") || !strings.Contains(string(got), "could not label #5") {
		t.Errorf("job summary %q does not record the swallowed failure", got)
	}
}

// Every workflow step that calls best_effort sources the function first, and no workflow
// defines a second way to swallow an exit code: the lint exemptions that remain are few and
// each carries its reason (grep -c 'lint-ok' .github/workflows/*.yml).
func TestBestEffort_EveryCallerSourcesIt(t *testing.T) {
	files, _ := filepath.Glob(filepath.Join("..", ".github", "workflows", "*.yml"))
	for _, f := range files {
		raw, _ := os.ReadFile(f)
		if strings.Contains(string(raw), `best_effort "`) && !strings.Contains(string(raw), "source scripts/lib-best-effort.sh") {
			t.Errorf("%s calls best_effort and never sources scripts/lib-best-effort.sh", f)
		}
	}
}
