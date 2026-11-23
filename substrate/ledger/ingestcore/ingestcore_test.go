package ingestcore_test

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/ingestcore"
)

func cand(rec, file, rk string, v int) ingestcore.Candidate {
	return ingestcore.Candidate{Record: rec, Identity: map[string]any{"source": "s", "file": file},
		RoleKey: map[string]any{"source": "s", "role": rk}, Measured: map[string]any{"v": v}, Facet: "f"}
}

func ingest(t *testing.T, records int, cs []ingestcore.Candidate, left map[string][]string) ([]ledger.Unit, *ingestcore.Report) {
	t.Helper()
	u, r, err := ingestcore.Ingest("s", "rev", "in", "facet", records, cs, left)
	if err != nil {
		t.Fatal(err)
	}
	return u, r
}

// Every record is accounted for: a record is a measurement of a unit, a duplicate, a merged
// record or left out, whatever the mix (§10.2).
func TestEveryRecordIsAccountedFor(t *testing.T) {
	cs := []ingestcore.Candidate{
		cand("r0", "a", "k1", 1),
		cand("r1", "a", "k1", 1), // duplicate of r0
		cand("r2", "a", "k1", 2), // same unit, another measurement
		cand("r3", "b", "k1", 1), // r0's measurement under another identity, same role key: merged
		cand("r4", "c", "k2", 7),
		cand("r5", "d", "k3", 7), // r4's measurement under another role key: both left out
	}
	units, rep := ingest(t, 8, cs, map[string][]string{"empty": {"x", "y"}})
	n := 0
	for _, u := range units {
		for _, m := range u.Measurements {
			n += 1 + len(m.Duplicates) + len(m.Merged)
		}
	}
	for _, ids := range rep.LeftOut {
		n += len(ids)
	}
	if n != rep.Records {
		t.Fatalf("%d records accounted for, want %d", n, rep.Records)
	}
	if rep.Duplicates != 1 || rep.Merged != 1 || rep.JoinedUnits != 1 || rep.JoinedRecords != 4 {
		t.Errorf("duplicates %d merged %d joined %d/%d, want 1 1 1/4", rep.Duplicates, rep.Merged, rep.JoinedUnits, rep.JoinedRecords)
	}
	if got := rep.LeftOut[ingestcore.SameMeasurementTwoRoleKeys]; !reflect.DeepEqual(got, []string{"r4", "r5"}) {
		t.Errorf("left out for two role keys: %v", got)
	}
	if rep.Units != 1 || rep.RoleKeys != 1 {
		t.Errorf("units %d role keys %d, want 1 1", rep.Units, rep.RoleKeys)
	}
}

// Near-copies share a role key, and a record's role never depends on what else was ingested:
// a unit's id is a function of its identity alone.
func TestUnitIDIsOfTheIdentityAlone(t *testing.T) {
	a, _ := ingest(t, 1, []ingestcore.Candidate{cand("r0", "a", "k1", 1)}, nil)
	b, _ := ingest(t, 3, []ingestcore.Candidate{cand("q", "z", "k9", 5), cand("r0", "a", "k1", 9), cand("r1", "a", "k1", 1)}, nil)
	var got string
	for _, u := range b {
		if u.Identity["file"] == "a" {
			got = u.UnitID
			if len(u.Measurements) != 2 {
				t.Errorf("%d measurements, want 2", len(u.Measurements))
			}
		}
	}
	if got != a[0].UnitID {
		t.Errorf("unit id changed with the other records: %s, %s", got, a[0].UnitID)
	}
}

// A record that does not reconcile is refused rather than dropped (R1).
func TestUnaccountedRecordsAreRefused(t *testing.T) {
	_, _, err := ingestcore.Ingest("s", "rev", "in", "facet", 5, []ingestcore.Candidate{cand("r0", "a", "k", 1)}, nil)
	if err == nil {
		t.Fatal("5 records with 1 kept and none left out were accepted")
	}
}

func TestIngestIsDeterministic(t *testing.T) {
	cs := []ingestcore.Candidate{cand("r0", "a", "k1", 1), cand("r1", "b", "k2", 2), cand("r2", "a", "k1", 3)}
	u1, r1 := ingest(t, 3, cs, nil)
	u2, r2 := ingest(t, 3, cs, nil)
	if !reflect.DeepEqual(u1, u2) || !reflect.DeepEqual(r1, r2) {
		t.Error("two runs differ")
	}
}
