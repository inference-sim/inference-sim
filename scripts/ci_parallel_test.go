package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The oracle workflows run the parts of a shard side by side in one job (scripts/ci-parallel.sh).
// The job must fail when any part fails, even one that is not the last to finish, and every
// part's output must reach the log.
func TestCIParallelRunsEveryPartAndFailsIfAnyFails(t *testing.T) {
	run := func(count, command string) (int, string) {
		t.Helper()
		cmd := exec.Command("bash", "ci-parallel.sh", count, command)
		out, err := cmd.CombinedOutput()
		if err == nil {
			return 0, string(out)
		}
		ee, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("ci-parallel.sh: %v", err)
		}
		return ee.ExitCode(), string(out)
	}

	dir := t.TempDir()
	code, out := run("3", `echo "part $1 says hi"; touch "`+dir+`/ran-$1"`)
	if code != 0 {
		t.Fatalf("all parts pass, yet exit %d:\n%s", code, out)
	}
	for _, p := range []string{"0", "1", "2"} {
		if !strings.Contains(out, "part "+p+" says hi") || !strings.Contains(out, "::group::part "+p+" of 3: exit 0") {
			t.Errorf("part %s's output or status is missing:\n%s", p, out)
		}
		if _, err := os.Stat(filepath.Join(dir, "ran-"+p)); err != nil {
			t.Errorf("part %s did not run: %v", p, err)
		}
	}

	// Part 0 fails first and parts 1 and 2 still pass: the whole fails, and the later parts ran.
	code, out = run("3", `if [ "$1" = 0 ]; then echo "part 0 broke"; exit 7; fi; sleep 0.2; echo "part $1 done"`)
	if code == 0 {
		t.Fatalf("part 0 failed, yet exit 0:\n%s", out)
	}
	for _, want := range []string{"part 0 broke", "::group::part 0 of 3: exit 7", "part 1 done", "part 2 done"} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}

	for _, bad := range []string{"0", "x", ""} {
		if code, out := run(bad, "true"); code != 2 {
			t.Errorf("COUNT %q: exit %d, want 2 (usage):\n%s", bad, code, out)
		}
	}
}
