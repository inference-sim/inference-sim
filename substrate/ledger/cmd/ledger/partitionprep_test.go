package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
)

// The join report and the proposal, end to end on the committed ledger and synthetic reads. The
// reads name configiq units by their committed (non-secret) ids; no seed is read.
func TestPartitionPrepDryRun(t *testing.T) {
	const dir = "../../../catalog/ledger"
	l, err := ledger.Load(dir, roleKeys)
	if err != nil {
		t.Fatal(err)
	}
	if l.Config.PartitionVersion != 1 {
		t.Skip("the tree is past partition version 1")
	}
	units := l.Units["configiq"]
	if len(units) < 10 {
		t.Fatalf("configiq has %d units", len(units))
	}
	rec := ledger.Reads{Kind: "UnitReads", Consumer: "fit-step", Revision: "abc"}
	for i := 0; i < 10; i++ {
		rec.Reads = append(rec.Reads, ledger.UnitRead{Source: "configiq", UnitID: units[i].UnitID, Role: ledger.RoleFit})
	}
	raw, _ := json.Marshal(rec)
	tmp := t.TempDir()
	reads := filepath.Join(tmp, "reads.json")
	if err := os.WriteFile(reads, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	report, proposal := filepath.Join(tmp, "report.yaml"), filepath.Join(tmp, "ledger.yaml")

	var out, errb bytes.Buffer
	args := []string{"partition-prep", "-require", "fit-step", "-out", report, "-propose", proposal, dir, reads}
	if got := run(args, &out, &errb); got != 0 {
		t.Fatalf("exit %d: %s", got, errb.String())
	}
	if !strings.Contains(out.String(), "10 read under a training role (left out)") {
		t.Errorf("summary %q", out.String())
	}
	b, err := os.ReadFile(report)
	if err != nil || !strings.Contains(string(b), units[0].UnitID) {
		t.Errorf("the report does not name a left-out unit: %v", err)
	}
	pb, err := os.ReadFile(proposal)
	if err != nil || !strings.Contains(string(pb), "partition_version: 2") || !strings.Contains(string(pb), units[9].UnitID) {
		t.Errorf("proposal lacks version 2 or a left-out unit: %v", err)
	}

	// A required consumer with no record, and a ledger already at version 2, are refused.
	for name, a := range map[string][]string{
		"missing consumer": {"partition-prep", "-require", "fit-step,error", "-out", report, dir, reads},
		"no output":        {"partition-prep", dir, reads},
		"other source":     {"partition-prep", "-source", "aisimulate", "-out", report, dir, reads},
	} {
		out.Reset()
		errb.Reset()
		if got := run(a, &out, &errb); got == 0 {
			t.Errorf("%s: exit 0", name)
		}
	}
	v2 := filepath.Join(tmp, "v2")
	if err := os.MkdirAll(v2, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := copyTree(dir, v2); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Join(v2, "published")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(v2, "ledger.yaml"), pb, 0o644); err != nil {
		t.Fatal(err)
	}
	if got := run([]string{"partition-prep", "-out", report, "-propose", filepath.Join(tmp, "x.yaml"), v2, reads}, &out, &errb); got == 0 || !strings.Contains(errb.String(), "partition version 2") {
		t.Errorf("-propose on a version 2 ledger: exit %d, %q", got, errb.String())
	}
}

// There is no operator-level held-out gate: aggregate does not pair any job with held.
func TestAggregateHasNoHeldOutGate(t *testing.T) {
	var out, errb bytes.Buffer
	if got := run([]string{"aggregate", "-job", "gate", "-role", "held", "x"}, &out, &errb); got != 2 {
		t.Errorf("aggregate -job gate -role held = %d, want 2: %s", got, errb.String())
	}
}

func copyTree(src, dst string) error {
	return filepath.Walk(src, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		if info.IsDir() {
			return os.MkdirAll(filepath.Join(dst, rel), 0o755)
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(dst, rel), b, 0o644)
	})
}
