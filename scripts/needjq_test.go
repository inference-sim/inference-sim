package scripts_test

import (
	"os"
	"os/exec"
	"testing"
)

// needJQ skips a test that runs a jq script when jq is not installed, as a developer's machine may
// lack it, and fails it instead under REQUIRE_JQ, which CI sets: a runner without jq must not pass
// these tests by skipping them (X16).
func needJQ(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("jq"); err != nil {
		if os.Getenv("REQUIRE_JQ") != "" {
			t.Fatalf("REQUIRE_JQ is set and jq is not installed: %v", err)
		}
		t.Skip("jq is not on PATH; CI runs this with jq installed (REQUIRE_JQ=1)")
	}
}
