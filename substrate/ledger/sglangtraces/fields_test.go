package sglangtraces_test

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/sglangtraces"
)

// Every field of a unit is checked against the listing line it came from (requirements §2.2):
// the identity is the path as listed, the measured values are that line's blob SHA-1 and size,
// and the facet is the phase named in the file name. Each file has a blob and a size of its own.
func TestUnitFieldsMatchTheirRecord(t *testing.T) {
	dec := base + "bs-8-il-1024-1.5-TP-0-DECODE.trace.json.gz"
	ext := base + "bs-2-il-4096-1.5-TP-0-DP-0-EP-0-EXTEND.trace.json.gz"
	l := read(t, listing(file(ext, "b", 222), file(dec, "a", 111)))
	units, rep, err := sglangtraces.Ingest(l, commit, "in")
	if err != nil {
		t.Fatal(err)
	}
	if len(units) != 2 || rep.Records != 2 {
		t.Fatalf("units %d records %d, want 2 2", len(units), rep.Records)
	}
	for _, c := range []struct {
		path, blob, phase string
		size              int64
	}{{dec, sha("a"), "DECODE", 111}, {ext, sha("b"), "EXTEND", 222}} {
		id := map[string]any{"source": "sglang-traces", "file": c.path}
		uid, _ := ledger.ID(id)
		var u *ledger.Unit
		for i := range units {
			if units[i].UnitID == uid {
				u = &units[i]
			}
		}
		if u == nil {
			t.Errorf("%s: no unit has the identity of its listing line", c.path)
			continue
		}
		if !reflect.DeepEqual(u.Identity, id) || u.RoleKey != uid {
			t.Errorf("%s: identity %v role key %s, want %v and its hash", c.path, u.Identity, u.RoleKey, id)
		}
		want, _ := ledger.MeasuredID(map[string]any{"git_blob_sha1": c.blob, "size": c.size})
		if m := u.Measurements[0]; m.Measured != want || m.Record != c.path {
			t.Errorf("%s: measurement %s of %s, want %s of the path", c.path, m.Measured, m.Record, want)
		}
	}
	if rep.UnitsBy["DECODE"] != 1 || rep.UnitsBy["EXTEND"] != 1 {
		t.Errorf("units by phase %v, want one of each", rep.UnitsBy)
	}
}
