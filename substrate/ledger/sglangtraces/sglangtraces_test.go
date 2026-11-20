package sglangtraces_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/sglangtraces"
)

const commit = "2085ec76442096a48a19f5f17f0aff864209f2ee"

func sha(c string) string { return strings.Repeat(c, 40) }

func listing(files ...string) string {
	var b strings.Builder
	fmt.Fprintf(&b, `{"repo":"o/ci-data","commit":%q,"tree":%q}`+"\n", commit, sha("1"))
	for _, f := range files {
		b.WriteString(f + "\n")
	}
	return b.String()
}

func file(path, blob string, size int) string {
	return fmt.Sprintf(`{"path":%q,"sha":%q,"size":%d}`, path, sha(blob), size)
}

const base = "traces/1/Model_TP8_1/100.5/"

func read(t *testing.T, s string) *sglangtraces.Listing {
	t.Helper()
	l, err := sglangtraces.ReadListing(strings.NewReader(s))
	if err != nil {
		t.Fatal(err)
	}
	return l
}

// A unit is one rank-0 trace file and its role key is the file: every other file, whatever its
// reason, is left out and counted, and nothing is lost (Table 1.4).
func TestIngestClassifiesEveryFile(t *testing.T) {
	l := read(t, listing(
		file(base+"bs-1-il-4096-1.1-TP-0-DECODE.trace.json.gz", "a", 10),
		file(base+"bs-1-il-4096-1.1-TP-0-EXTEND.trace.json.gz", "b", 10),
		file(base+"bs-1-il-4096-1.1-TP-1-DECODE.trace.json.gz", "c", 10),
		file(base+"bs-2-il-4096-1.1-TP-0-DECODE.trace.json.gz", "d", 0),
		file(base+"bs-3-il-4096-1.1-TP-0-DP-0-EP-0-DECODE.trace.json.gz", "f", 10),
		file(base+"notes.txt", "e", 10),
	))
	units, rep, err := sglangtraces.Ingest(l, commit, "in")
	if err != nil {
		t.Fatal(err)
	}
	if rep.Records != 6 || rep.Units != 3 || rep.RoleKeys != 3 {
		t.Errorf("records %d units %d role keys %d, want 6 3 3", rep.Records, rep.Units, rep.RoleKeys)
	}
	for why, n := range map[string]int{sglangtraces.LeftOutNotRankZero: 1, sglangtraces.LeftOutEmpty: 1, sglangtraces.LeftOutNotTraceFile: 1} {
		if len(rep.LeftOut[why]) != n {
			t.Errorf("left out as %s: %v, want %d", why, rep.LeftOut[why], n)
		}
	}
	if rep.UnitsBy["DECODE"] != 2 || rep.UnitsBy["EXTEND"] != 1 {
		t.Errorf("units by phase: %v", rep.UnitsBy)
	}
	for _, u := range units {
		rk, err := sglangtraces.RoleKey(u.Identity)
		if err != nil {
			t.Fatal(err)
		}
		if id, _ := ledger.ID(rk); id != u.RoleKey {
			t.Errorf("unit %s: role key %s is not the file's", u.UnitID, u.RoleKey)
		}
		if _, ok := u.Identity["measured"]; ok || len(u.Measurements) != 1 {
			t.Errorf("unit %s: want one measurement and no measured value in the identity", u.UnitID)
		}
	}
}

// The same blob under two files is the same measurement under two role keys: neither file is
// a unit, since nothing says which one the content belongs to.
func TestSameBlobUnderTwoFilesIsLeftOut(t *testing.T) {
	l := read(t, listing(
		file(base+"bs-1-il-4096-1.1-TP-0-DECODE.trace.json.gz", "a", 10),
		file(base+"bs-2-il-4096-1.1-TP-0-DECODE.trace.json.gz", "a", 10),
		file(base+"bs-4-il-4096-1.1-TP-0-DECODE.trace.json.gz", "b", 10),
	))
	_, rep, err := sglangtraces.Ingest(l, commit, "in")
	if err != nil {
		t.Fatal(err)
	}
	if rep.Units != 1 || len(rep.LeftOut["same_measurement_two_role_keys"]) != 2 {
		t.Errorf("units %d, left out %v; want 1 unit and 2 files left out", rep.Units, rep.LeftOut)
	}
}

func TestReadListingIsStrict(t *testing.T) {
	good := file(base+"a-TP-0-DECODE.trace.json.gz", "a", 1)
	for name, in := range map[string]string{
		"no files":           listing(),
		"out of order":       listing(file("traces/2/a", "a", 1), file("traces/1/a", "a", 1)),
		"repeated":           listing(good, good),
		"short blob":         listing(`{"path":"traces/1/a","sha":"abc","size":1}`),
		"unknown field":      listing(`{"path":"traces/1/a","sha":"` + sha("a") + `","size":1,"extra":1}`),
		"negative size":      listing(`{"path":"traces/1/a","sha":"` + sha("a") + `","size":-1}`),
		"header lacks tree":  `{"repo":"o/r","commit":"` + commit + `"}` + "\n" + good + "\n",
		"header unknown key": `{"repo":"o/r","commit":"x","tree":"y","z":1}` + "\n" + good + "\n",
	} {
		if _, err := sglangtraces.ReadListing(strings.NewReader(in)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// D3: the listing is of the pinned commit of the pinned repository.
func TestCheckRevision(t *testing.T) {
	l := read(t, listing(file(base+"a-TP-0-DECODE.trace.json.gz", "a", 1)))
	src := func(url, ref string) *ledger.Source {
		return &ledger.Source{ID: sglangtraces.SourceID, Revision: ledger.Revision{URL: url, Ref: ref}}
	}
	if err := sglangtraces.CheckRevision(src("https://github.com/o/ci-data/tree/"+commit, commit), l); err != nil {
		t.Errorf("the pinned listing was refused: %v", err)
	}
	if err := sglangtraces.CheckRevision(src("https://github.com/o/ci-data/tree/"+commit, sha("9")), l); err == nil {
		t.Error("a listing of another commit was accepted")
	}
	if err := sglangtraces.CheckRevision(src("https://github.com/o/other/tree/"+commit, commit), l); err == nil {
		t.Error("a listing of another repository was accepted")
	}
}

func TestRoleKeyRefusesOtherIdentities(t *testing.T) {
	for name, id := range map[string]map[string]any{
		"extra field":  {"source": sglangtraces.SourceID, "file": "f", "x": 1},
		"no file":      {"source": sglangtraces.SourceID},
		"other source": {"source": "other", "file": "f"},
	} {
		if _, err := sglangtraces.RoleKey(id); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
