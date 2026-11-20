package latencykernel

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The mutation check (X13): every behaviour change has a test that fails without it. This
// breaks each guarded property in the source, one at a time, and requires the test that guards
// it to fail. The break is applied with go test -overlay, which replaces a file's content for
// one build and leaves the tree alone; the sweep runs in CI as the "mutants" job, and the
// registry names this test as the known-bad input of X13's mutation check, so the nightly
// known-bad sweep (X18) runs it too. To guard a property, add a row: the file, the exact text
// to break, its replacement, and the test that must then fail.
var mutants = []struct {
	name, file, old, new, killedBy string
}{
	{"mHC form on all tokens under sequence parallelism (#66)",
		"prefill_forms.go", "rows = s.spRows", "rows = s.tokensF",
		"TestAnMHCNodeRunsOnItsSequenceParallelShard"},
	{"drafter width from decode rows only (#66)",
		"draft.go", "if r.Computed+r.Scheduled > 0 {\n\t\t\treqs++",
		"if r.Computed+r.Scheduled > 0 && r.Scheduled <= b.DecodeThreshold {\n\t\t\treqs++",
		"TestTheDrafterCeilingCountsPrefillRows"},
	{"pipeline plan priced at a defaulted gpu_memory_utilization (#76)",
		"pipeline.go", "if util <= 0 {\n\t\treturn fmt.Errorf(", "if util <= 0 {\n\t\tutil = 0.92\n\t}\n\tif false {\n\t\treturn fmt.Errorf(",
		"TestAZeroMemoryUtilizationIsRefusedNotDefaulted"},
}

func TestGuardedPropertiesKillTheirMutants(t *testing.T) {
	if testing.Short() {
		t.Skip("each mutant rebuilds the package")
	}
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range mutants {
		t.Run(m.killedBy, func(t *testing.T) {
			path := filepath.Join(dir, m.file)
			src, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if n := strings.Count(string(src), m.old); n != 1 {
				t.Fatalf("%s: the text to break occurs %d times in %s, want once; the mutant is stale",
					m.name, n, m.file)
			}
			tmp := t.TempDir()
			mutated := filepath.Join(tmp, m.file)
			if err := os.WriteFile(mutated, []byte(strings.Replace(string(src), m.old, m.new, 1)), 0o644); err != nil {
				t.Fatal(err)
			}
			overlay := filepath.Join(tmp, "overlay.json")
			js, _ := json.Marshal(map[string]map[string]string{"Replace": {path: mutated}})
			if err := os.WriteFile(overlay, js, 0o644); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command("go", "test", "-count=1", "-overlay="+overlay, "-run", "^"+m.killedBy+"$", "-v", ".")
			cmd.Dir = dir
			out, err := cmd.CombinedOutput()
			if err == nil {
				t.Fatalf("%s: %s passed with the property broken, so it does not guard it", m.name, m.killedBy)
			}
			if !strings.Contains(string(out), "--- FAIL: "+m.killedBy) {
				t.Fatalf("%s: the run failed, but not in %s (a build error, not a kill):\n%s", m.name, m.killedBy, out)
			}
		})
	}
}
