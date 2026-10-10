package cmd

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

var (
	builtBlisOnce sync.Once
	builtBlisPath string
	builtBlisErr  error
)

// builtBlis is the real blis binary built once per test process from this checkout's main.go.
// The re-exec'd test binary cannot stand in for it here: only main installs the compiled-in
// defaults.yaml (cmd.SetBundledDefaults). Built into the user cache directory and moved into
// place atomically, as the harness tests build theirs, so concurrent test processes never see
// a half-written binary.
func builtBlis(t *testing.T) string {
	t.Helper()
	builtBlisOnce.Do(func() {
		cache, err := os.UserCacheDir()
		if err != nil {
			builtBlisErr = err
			return
		}
		dir := filepath.Join(cache, "inference-sim", "cmd-test")
		if builtBlisErr = os.MkdirAll(dir, 0o755); builtBlisErr != nil {
			return
		}
		tmp, err := os.CreateTemp(dir, "blis-*")
		if err != nil {
			builtBlisErr = err
			return
		}
		_ = tmp.Close()
		root, err := filepath.Abs("..")
		if err != nil {
			builtBlisErr = err
			return
		}
		build := exec.Command("go", "build", "-o", tmp.Name(), ".")
		build.Dir = root
		if out, err := build.CombinedOutput(); err != nil {
			_ = os.Remove(tmp.Name())
			builtBlisErr = &buildFailure{err: err, out: string(out)}
			return
		}
		builtBlisPath = filepath.Join(dir, "blis")
		builtBlisErr = os.Rename(tmp.Name(), builtBlisPath)
	})
	if builtBlisErr != nil {
		t.Fatalf("building blis: %v", builtBlisErr)
	}
	return builtBlisPath
}

type buildFailure struct {
	err error
	out string
}

func (e *buildFailure) Error() string { return e.err.Error() + "\n" + e.out }

// The shipped binary carries its defaults: run from a directory with no defaults.yaml and no
// --defaults-filepath, a run completes on the compiled-in copy (and says so at info); an
// explicit --defaults-filepath naming a missing file is refused naming it, never masked by
// the bundled copy.
func TestBlisBinary_UsesItsEmbeddedDefaults(t *testing.T) {
	blis := builtBlis(t)
	scenarios, catalog, registry := kernelRepos(t)
	work := t.TempDir()
	run := func(extra ...string) (string, string, error) {
		args := append([]string{"run", "--scenario", kernelTestScenario, "--scenarios", scenarios,
			"--catalog", catalog, "--registry", registry, "--num-requests", "4", "--rate", "2", "--log", "info"}, extra...)
		c := exec.Command(blis, args...)
		c.Dir = work
		var out, errBuf bytes.Buffer
		c.Stdout, c.Stderr = &out, &errBuf
		err := c.Run()
		return out.String(), errBuf.String(), err
	}
	if _, err := os.Stat(filepath.Join(work, "defaults.yaml")); err == nil {
		t.Fatal("the working directory has a defaults.yaml; the case would not exercise the bundled copy")
	}
	out, stderr, err := run()
	if err != nil || !strings.Contains(out, `"completed_requests": 4`) {
		t.Fatalf("a run with no defaults file did not complete: %v\n%s", err, lastLines(stderr, 3))
	}
	if !strings.Contains(stderr, "using the defaults compiled into blis") {
		t.Errorf("the run did not report using the compiled-in defaults:\n%s", lastLines(stderr, 5))
	}

	missing := filepath.Join(work, "no-such-defaults.yaml")
	if _, stderr, err := run("--defaults-filepath", missing); err == nil || !strings.Contains(stderr, missing) {
		t.Errorf("an explicit missing --defaults-filepath was not refused naming it: %v\n%s", err, lastLines(stderr, 3))
	}
}
