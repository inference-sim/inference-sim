package ledger_test

import (
	"os/exec"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/ledger"
	"github.com/inference-sim/blis-substrate/ledger/inferencex"
)

// The committed ledger loads and passes every check that needs no source data: the layout,
// D3's pinned revisions, the cut-points, each unit's unit_id and role key, one unit per
// measurement, and the legacy table against the unit list.
func TestCommittedLedger(t *testing.T) {
	l, err := ledger.Load("../catalog/ledger", map[string]ledger.RoleKeyFunc{
		inferencex.SourceID: inferencex.RoleKey,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(l.Units[inferencex.SourceID]) == 0 {
		t.Fatal("the InferenceX unit list is empty (X22)")
	}
	if l.Legacy == nil || len(l.Legacy.Units) == 0 {
		t.Fatal("legacy-map.yaml is missing or empty")
	}
}

// No tracked file outside the published lists and the legacy table pairs a unit with a role
// (D4).
func TestNoCommittedRoles(t *testing.T) {
	out, err := exec.Command("git", "-C", "..", "ls-files").Output()
	if err != nil {
		t.Fatalf("git ls-files: %v", err)
	}
	var files []string
	for _, f := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		for _, ext := range []string{".yaml", ".yml", ".json", ".jsonl", ".jsonl.gz", ".csv", ".tsv"} {
			if strings.HasSuffix(f, ext) && !strings.HasPrefix(f, ".github/") {
				files = append(files, f)
				break
			}
		}
	}
	if len(files) == 0 {
		t.Fatal("no files to scan (X22)")
	}
	if err := ledger.CheckNoRoles("..", files, nil); err != nil {
		t.Fatal(err)
	}
}
