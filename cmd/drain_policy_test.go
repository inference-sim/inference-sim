package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// instance_lifecycle.drain_policy names how a scale-down drains an instance (#910). A name that is
// not a policy is refused with the valid ones; before the field existed in the bundle, the file was
// refused as having an unknown field, and the cluster ignored any drain policy it was given.
func TestRunCmd_DrainPolicyIsValidated(t *testing.T) {
	dir := writeColocatedVariant(t)
	for _, c := range []struct{ policy, want string }{
		{"BOGUS", "IMMEDIATE, WAIT, REDIRECT"},
		{"REDIRECT", ""},
	} {
		path := filepath.Join(t.TempDir(), "bundle.yaml")
		if err := os.WriteFile(path, []byte("instance_lifecycle:\n  drain_policy: "+c.policy+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		_, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", "variant.yaml",
			"--num-requests", "4", "--rate", "1", "--policy-config", path)
		switch {
		case c.want != "" && (err == nil || !strings.Contains(stderr, c.want)):
			t.Errorf("drain_policy %s: want a refusal naming %q, got err=%v\n%s", c.policy, c.want, err, lastLines(stderr, 3))
		case c.want == "" && err != nil:
			t.Errorf("drain_policy %s: %v\n%s", c.policy, err, lastLines(stderr, 3))
		}
	}
}
