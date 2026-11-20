package inferencex

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
)

// spec is one synthetic record: its numbers are invented (X12).
type spec struct {
	id, run, conc int64
	date, ttft    string
}

func (s spec) json() json.RawMessage {
	return json.RawMessage(strings.Replace(string(apiJSON(s.id, s.run, s.conc, 4, s.ttft)),
		`"date":"2026-11-09"`, `"date":"`+s.date+`"`, 1))
}

func rowsOf(t *testing.T, specs []spec) []Row {
	t.Helper()
	var rows []Row
	for _, s := range specs {
		r, err := RowFromAPI(s.json())
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, r)
	}
	return rows
}

func feedOf(t *testing.T, specs []spec) *FeedRead {
	t.Helper()
	var recs []json.RawMessage
	for _, s := range specs {
		recs = append(recs, s.json())
	}
	body, _ := json.Marshal(recs)
	f, err := FetchFeed([]string{"https://api.test/feed"}, func(string) ([]byte, error) { return body, nil })
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// D3's test (requirements 3A.9): a run whose records straddle the cutoff keeps one
// measured_date, every unit of the run carries it, and the units the dump already made keep
// their ids when the feed joins.
func TestStraddlingRunKeepsOneMeasuredDate(t *testing.T) {
	for _, tc := range []struct {
		name        string
		dump, feed  []spec
		wantDate    map[int64]string
		wantSides   map[string]int
		wantStraddl []int64
	}{
		{
			name: "feed records dated after the dump's",
			dump: []spec{{5, 42, 1, "2026-09-20", "1.5"}, {6, 42, 8, "2026-09-20", "2.5"}},
			feed: []spec{{20, 42, 1, "2026-10-02", "1.7"}, {21, 42, 16, "2026-10-02", "3.5"}},
			// conc 1 is measured on both sides of the cutoff and is one unit with two measurements.
			wantDate:    map[int64]string{42: "2026-09-20"},
			wantSides:   map[string]int{"dump": 1, "api": 1, "both": 1},
			wantStraddl: []int64{42},
		},
		{
			name: "feed record dated before the dump's: the frozen dump decides",
			dump: []spec{{5, 42, 1, "2026-09-20", "1.5"}},
			feed: []spec{{20, 42, 1, "2026-09-10", "1.7"}},

			wantDate:    map[int64]string{42: "2026-09-20"},
			wantSides:   map[string]int{"both": 1},
			wantStraddl: []int64{42},
		},
		{
			name: "a run only the feed holds takes the earliest date of its own records",
			dump: []spec{{5, 42, 1, "2026-09-20", "1.5"}},
			feed: []spec{{20, 77, 1, "2026-10-03", "1.7"}, {21, 77, 8, "2026-10-01", "1.9"}},

			wantDate:  map[int64]string{42: "2026-09-20", 77: "2026-10-01"},
			wantSides: map[string]int{"dump": 1, "api": 2},
		},
		{
			name: "a run only the dump holds is unchanged by the feed",
			dump: []spec{{5, 42, 1, "2026-09-20", "1.5"}, {6, 43, 1, "2026-09-25", "2.5"}},
			feed: []spec{{20, 77, 1, "2026-10-03", "1.7"}},

			wantDate:  map[int64]string{42: "2026-09-20", 43: "2026-09-25", 77: "2026-10-03"},
			wantSides: map[string]int{"dump": 2, "api": 1},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dump := rowsOf(t, tc.dump)
			dumpUnits, _, err := Ingest(dump, "rev", "in")
			if err != nil {
				t.Fatal(err)
			}
			units, rep, err := IngestWithFeed(dump, feedOf(t, tc.feed), "rev", "in")
			if err != nil {
				t.Fatal(err)
			}
			got := map[int64]map[string]bool{}
			for _, u := range units {
				if err := u.Check(RoleKey); err != nil {
					t.Fatal(err)
				}
				run := u.Identity["benchmark_run"].(int64)
				if got[run] == nil {
					got[run] = map[string]bool{}
				}
				got[run][u.Identity["measured_date"].(string)] = true
			}
			for run, want := range tc.wantDate {
				if len(got[run]) != 1 || !got[run][want] {
					t.Errorf("run %d has measured dates %v, want only %s", run, got[run], want)
				}
			}
			have := map[string]bool{}
			for _, u := range units {
				have[u.UnitID] = true
			}
			for _, u := range dumpUnits {
				if !have[u.UnitID] {
					t.Errorf("unit %s, made from the dump alone, is not among the units made with the feed", u.UnitID)
				}
			}
			if !reflect.DeepEqual(rep.Feed.UnitsBySourceRevision, tc.wantSides) {
				t.Errorf("units by source revision %v, want %v", rep.Feed.UnitsBySourceRevision, tc.wantSides)
			}
			if !reflect.DeepEqual(rep.Feed.StraddlingRuns, tc.wantStraddl) {
				t.Errorf("straddling runs %v, want %v", rep.Feed.StraddlingRuns, tc.wantStraddl)
			}
		})
	}
}

// The dump owns the ids up to its last and the feed those above: a feed record at or below the
// cutoff is never pulled again, whether the dump holds it or not, and one whose content differs
// from the dump's copy is reported while the dump's copy stays.
func TestFeedOwnsOnlyTheIdsAboveTheDump(t *testing.T) {
	dump := []spec{{5, 42, 1, "2026-09-20", "1.5"}, {6, 42, 8, "2026-09-20", "2.5"}}
	feed := []spec{
		{3, 41, 1, "2026-09-01", "9.9"},  // below the cutoff, not in the dump: not pulled
		{5, 42, 1, "2026-09-20", "1.5"},  // the dump's record, printed the same
		{6, 42, 8, "2026-09-20", "2.75"}, // the dump's record, printed differently
		{20, 42, 16, "2026-10-02", "3.5"},
	}
	units, rep, err := IngestWithFeed(rowsOf(t, dump), feedOf(t, feed), "rev", "in")
	if err != nil {
		t.Fatal(err)
	}
	f := rep.Feed
	if f.Cutoff != 6 || f.Served != 4 || f.OwnedByDump != 3 || f.APIRecords != 1 || !reflect.DeepEqual(f.RowDiffers, []int64{6}) {
		t.Errorf("feed report %+v", f)
	}
	if rep.Records != 3 {
		t.Errorf("%d records ingested, want the dump's 2 and the feed's 1", rep.Records)
	}
	dumpOnly, _, _ := Ingest(rowsOf(t, dump), "rev", "in")
	have := map[string]ledger.Unit{}
	for _, u := range units {
		have[u.UnitID] = u
	}
	for _, d := range dumpOnly {
		u, ok := have[d.UnitID]
		if !ok || !reflect.DeepEqual(u.Measurements, d.Measurements) {
			t.Errorf("unit %s changed when the feed joined: %+v, was %+v", d.UnitID, u.Measurements, d.Measurements)
		}
	}
}

func TestReconcileNeedsADump(t *testing.T) {
	if _, _, err := Reconcile(nil, nil); err == nil {
		t.Error("a feed with no dump has no cutoff, and was reconciled")
	}
}

// A record the ingest cannot read is left out and counted, never guessed; a record two URLs
// serve differently stops the read.
func TestFetchFeedLeavesOutWhatItCannotRead(t *testing.T) {
	good := spec{20, 42, 1, "2026-10-02", "1.5"}.json()
	noRun := json.RawMessage(strings.Replace(string(spec{21, 42, 1, "2026-10-02", "1.5"}.json()),
		`"run_url":"https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42/attempts/1"`, `"run_url":null`, 1))
	body, _ := json.Marshal([]json.RawMessage{good, noRun})
	f, err := FetchFeed([]string{"https://a.test/x"}, func(string) ([]byte, error) { return body, nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(f.Records) != 1 || f.Records[0].ID != 20 || !reflect.DeepEqual(f.Unreadable, map[string][]int64{"run_url": {21}}) {
		t.Errorf("records %+v, unreadable %v", f.Records, f.Unreadable)
	}

	other, _ := json.Marshal([]json.RawMessage{spec{20, 42, 1, "2026-10-02", "1.6"}.json()})
	one, _ := json.Marshal([]json.RawMessage{good})
	_, err = FetchFeed([]string{"https://a.test/x", "https://b.test/y"}, func(u string) ([]byte, error) {
		if strings.HasPrefix(u, "https://a") {
			return one, nil
		}
		return other, nil
	})
	if err == nil || !strings.Contains(err.Error(), "20") {
		t.Errorf("two URLs serving record 20 differently: %v", err)
	}
}

// The proposal pins each new record once, by run and attempt, and skips what a supplement
// already pins and what the dump owns.
func TestProposeSupplements(t *testing.T) {
	f := feedOf(t, []spec{
		{4, 41, 1, "2026-09-01", "9.9"},
		{20, 42, 1, "2026-10-02", "1.5"}, {21, 42, 8, "2026-10-02", "1.6"}, {22, 77, 1, "2026-10-03", "1.7"},
	})
	got := ProposeSupplements(f.Records, 6, map[int64]bool{21: true})
	if len(got) != 2 || got[0].ID != "run-42-attempt-1" || len(got[0].Records) != 1 || got[0].Records[0].ID != 20 ||
		got[1].ID != "run-77-attempt-1" || got[1].Run != "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/77/attempts/1" {
		t.Errorf("proposed %+v", got)
	}
}

func TestMergePinnedRefusesAChangedRecord(t *testing.T) {
	f := feedOf(t, []spec{{20, 42, 1, "2026-10-02", "1.5"}})
	pin := FeedRecord{ID: 20, SHA256: strings.Repeat("a", 64)}
	if err := MergePinned(f, []FeedRecord{pin}); err == nil || !strings.Contains(err.Error(), "20") {
		t.Errorf("a feed record whose content is not its pin: %v", err)
	}
	extra := FeedRecord{ID: 30, SHA256: strings.Repeat("b", 64)}
	if err := MergePinned(f, []FeedRecord{extra}); err != nil || len(f.Records) != 2 || f.Records[1].ID != 30 {
		t.Errorf("a pinned record the feed does not serve: %v, %+v", err, f.Records)
	}
}
