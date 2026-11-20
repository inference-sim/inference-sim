package cmd

import (
	"bytes"
	"os"
	"regexp"
	"strings"
	"testing"
)

func withBundledStack(t *testing.T) {
	t.Helper()
	b, err := os.ReadFile("../stack.json")
	if err != nil {
		t.Fatal(err)
	}
	prev := bundledStack
	SetBundledStack(b)
	t.Cleanup(func() { SetBundledStack(prev) })
}

// The version command prints the commit of every engine this build runs, the llm-d commit and
// the substrate, which a scenario no longer states (LIF-6).
func TestVersionCommandPrintsTheCommitsTheBuildRuns(t *testing.T) {
	withBundledStack(t)
	var out bytes.Buffer
	versionCmd.SetOut(&out)
	if err := versionCmd.RunE(versionCmd, nil); err != nil {
		t.Fatal(err)
	}
	for kind, e := range builtEngines() {
		if want := "engine " + kind + " " + e.Version + " " + e.Commit; !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
	if !regexp.MustCompile(`(?m)^llm-d \S+ [0-9a-f]{40} \(router fork `).MatchString(out.String()) ||
		!strings.Contains(out.String(), "substrate ") || !strings.HasPrefix(out.String(), "inference-sim ") {
		t.Errorf("output lacks the simulator, substrate or llm-d line:\n%s", out.String())
	}
}

// Every Result records the llm-d commit beside the engine commits.
func TestRevisionsRecordTheLLMDCommit(t *testing.T) {
	withBundledStack(t)
	if got := revisionsFrom(nil).LLMD; !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(got) {
		t.Errorf("revisions.llm_d = %q, want the 40-hex llm-d commit from stack.json", got)
	}
}
