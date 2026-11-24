package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// test-selected-deps.sh fetches its modules from the module proxy with an empty cache (X25), so a
// dropped connection there is a transient fault that must be retried, not reported as a test
// failure (#2680). This drives the script with a fake `go` whose router-module prefetch fails once
// with the proxy's stream error, and checks the retry carries it through. Modeled on
// TestCIShardRetriesAFailedFetch. The fake stands in for every `go` the script runs, so no network.
func TestSelectedDepsRetriesAFailedFetch(t *testing.T) {
	if testing.Short() {
		t.Skip("runs test-selected-deps.sh with a stubbed go")
	}
	script, err := filepath.Abs("test-selected-deps.sh")
	if err != nil {
		t.Fatal(err)
	}
	bin := t.TempDir()
	marker := filepath.Join(t.TempDir(), "prefetch-failed-once")
	// The prefetch (`go -C "$router_dir" list -deps -test ./...`, run from the router module's own
	// directory) fails once with the proxy's stream error, then succeeds; `list -m -f {{.Dir}}`
	// yields the router dir it runs in, and every other `go` the script runs is answered minimally.
	fake := `#!/usr/bin/env bash
args="$*"
case "$args" in
  "mod edit"*) exit 0 ;;
  "list -m -f {{.Dir}} "*) echo /fake/router; exit 0 ;;
  "-C "*"list -deps -test "*)
    if [ ! -e "$FAKE_GO_MARKER" ]; then : > "$FAKE_GO_MARKER"
      echo "go: downloading: stream error: stream ID 71; INTERNAL_ERROR; received from peer" >&2; exit 1; fi
    exit 0 ;;
  "list -deps "*) echo "github.com/llm-d/llm-d-router/pkg/embed"; exit 0 ;;
  "list -m"*) echo "github.com/llm-d/llm-d-router => github.com/inference-sim/llm-d-router v1.0.0-blis.6"; exit 0 ;;
  "get "*) exit 0 ;;
  "test "*) echo "ok  github.com/llm-d/llm-d-router/pkg/embed"; exit 0 ;;
  *) echo "fake go: unhandled args: $args" >&2; exit 3 ;;
esac`
	if err := os.WriteFile(filepath.Join(bin, "go"), []byte(fake), 0o755); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command("bash", script, "-count=1")
	cmd.Env = append(os.Environ(),
		"PATH="+bin+string(os.PathListSeparator)+os.Getenv("PATH"),
		"CI_SHARD_FETCH_RETRY_DELAY=0",
		"FAKE_GO_MARKER="+marker,
		"GOFLAGS=", "GOTOOLCHAIN=local")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("the script failed after one dropped prefetch (want the retry to carry it): %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "retrying") {
		t.Fatalf("the script did not retry the failed prefetch\n%s", out)
	}
	if !marked(marker) {
		t.Fatalf("the fake go's prefetch was never called\n%s", out)
	}
}

func marked(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
