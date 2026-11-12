package inferencex

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/ledger"
)

// apiRecord returns a synthetic API record. Its numbers are invented; no InferenceX
// measurement is committed (X12). It carries fields the ingest does not read, as the API does.
func apiJSON(id int64, run int64, conc int64, tp int64, ttft string) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"id":"%d","model":"m","hardware":"h200","precision":"fp4","framework":"vllm",`+
		`"benchmark_type":"agentic_traces","conc":%d,"date":"2026-11-09","run_url":"https://github.com/SemiAnalysisAI/InferenceX/actions/runs/%d/attempts/1",`+
		`"recipe_fingerprint":"fp%d","image":"vllm/vllm-openai:v0.33.0@sha256:abc","isl":null,"osl":null,"offload_mode":"off",`+
		`"spec_method":"mtp","disagg":false,"is_multinode":false,"prefill_tp":%d,"prefill_ep":1,"prefill_dp_attention":false,`+
		`"prefill_num_workers":0,"num_prefill_gpu":%d,"decode_tp":%d,"decode_ep":1,"decode_dp_attention":false,`+
		`"decode_num_workers":0,"num_decode_gpu":%d,"power_audit":{"gpus":["a","b"]},"workers":[{"role":"agg"}],`+
		`"metrics":{"mean_ttft":%s,"mean_tpot":0.0345}}`, id, conc, run, tp, tp, tp, tp, tp, ttft))
}

// served returns a fetcher that serves records, and the supplement that pins them.
func served(t *testing.T, recs ...json.RawMessage) (Fetcher, ledger.Supplement) {
	t.Helper()
	body, _ := json.Marshal(recs)
	sup := ledger.Supplement{ID: "s", Kind: ledger.SupplementKindAPI, URL: "https://example.test/api",
		Run: "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/9/attempts/1"}
	for _, r := range recs {
		id, _ := recordID(r)
		h, err := RecordSHA256(r)
		if err != nil {
			t.Fatal(err)
		}
		sup.Records = append(sup.Records, ledger.PinnedRecord{ID: id, SHA256: h})
	}
	return func(string) ([]byte, error) { return body, nil }, sup
}

func fetched(t *testing.T, recs ...json.RawMessage) *SupplementData {
	t.Helper()
	f, sup := served(t, recs...)
	d, err := FetchSupplement(sup, f)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func dumpRows(t *testing.T) []Row {
	t.Helper()
	rows, err := ReadRows(strings.NewReader(row(5, 10, 1, 4, `{"ttft":0.1}`) + "\n" + row(6, 10, 1, 8, `{"ttft":0.2}`) + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	return rows
}

// The reviewer's counterexample: units made for records outside the pinned dump carried the
// all-zero role key, and the scoring path accepted them. A supplement's units carry the
// canonical role key of design §10.2, which Unit.Check recomputes from the identity, so a
// unit with any other key is refused.
func TestSupplementUnitsCarryTheCanonicalRoleKey(t *testing.T) {
	d := fetched(t, apiJSON(900, 42, 1, 4, "382.0"), apiJSON(901, 42, 8, 4, "400.5"))
	units, _, err := IngestWithSupplements(dumpRows(t), []SupplementRows{{ID: d.ID, Rows: d.Rows}}, "rev", "in")
	if err != nil {
		t.Fatal(err)
	}
	if len(units) != 4 {
		t.Fatalf("%d units, want the dump's 2 and the supplement's 2", len(units))
	}
	zero := strings.Repeat("0", 64)
	for _, u := range units {
		if u.RoleKey == zero {
			t.Errorf("unit %s has the placeholder role key", u.UnitID)
		}
		if err := u.Check(RoleKey); err != nil {
			t.Errorf("unit %s does not pass the ledger's check: %v", u.UnitID, err)
		}
	}
	forged := units[0]
	forged.RoleKey = zero
	if err := forged.Check(RoleKey); err == nil {
		t.Error("a unit with the all-zero role key passed the check")
	}
}

// A record served by the API makes the unit the same record exported from the dump would
// make: same identity, unit_id and role key. The two sources of a record cannot disagree about
// who it is.
func TestAPIRecordAndExportedRowMakeOneUnit(t *testing.T) {
	d := fetched(t, apiJSON(900, 42, 8, 4, "382.0"))
	exported := edit(row(900, 42, 1, 8, `{"mean_ttft":382.0,"mean_tpot":0.0345}`), func(m map[string]any) {
		m["model"], m["precision"], m["benchmark_type"] = "m", "fp4", "agentic_traces"
		m["date"], m["recipe_fingerprint"] = "2026-11-09", "fp4"
		m["image"] = "vllm/vllm-openai:v0.33.0@sha256:abc"
		m["isl"], m["osl"], m["spec_method"] = nil, nil, "mtp"
		m["prefill_tp"], m["decode_tp"], m["num_prefill_gpu"], m["num_decode_gpu"] = 4, 4, 4, 4
	})
	viaDump, _ := ingest(t, exported)
	viaAPI, _, err := IngestWithSupplements(nil, []SupplementRows{{ID: d.ID, Rows: d.Rows}}, "rev", "in")
	if err != nil {
		t.Fatal(err)
	}
	a, b := viaDump[0], viaAPI[0]
	if a.UnitID != b.UnitID || a.RoleKey != b.RoleKey {
		t.Errorf("the API record makes unit %s/%s, the export %s/%s", b.UnitID, b.RoleKey, a.UnitID, a.RoleKey)
	}
}

// Keyed assignment (D4): every concurrency of one deployment draws one role, whatever the
// seed, and the two deployments are drawn by their own role keys.
func TestSupplementSweepSharesOneRole(t *testing.T) {
	var recs []json.RawMessage
	id := int64(900)
	for _, tp := range []int64{4, 8} {
		for _, conc := range []int64{1, 2, 4, 8, 16, 32, 64, 128} {
			recs = append(recs, apiJSON(id, 42, conc, tp, fmt.Sprintf("%d.5", 300+id)))
			id++
		}
	}
	d := fetched(t, recs...)
	units, rep, err := IngestWithSupplements(dumpRows(t), []SupplementRows{{ID: d.ID, Rows: d.Rows}}, "rev", "in")
	if err != nil {
		t.Fatal(err)
	}
	if rep.RoleKeys != 3 { // the dump's rows share one key; the supplement adds one per tensor-parallel size
		t.Errorf("role keys %d, want 3", rep.RoleKeys)
	}
	cfg, err := ledger.LoadConfig("../../catalog/ledger")
	if err != nil {
		t.Fatal(err)
	}
	for _, seed := range []string{strings.Repeat("5a", 32), strings.Repeat("c3", 32)} {
		raw, _ := hex.DecodeString(seed)
		sum := sha256.Sum256(raw)
		c := *cfg
		c.Seed = ledger.SeedRecord{SHA256: hex.EncodeToString(sum[:]), GeneratedBy: "test", GeneratedOn: "2026-11-12", RotationOwner: "test"}
		a, err := ledger.NewAssigner(&c, nil, map[string]ledger.RoleKeyFunc{SourceID: RoleKey}, seed)
		if err != nil {
			t.Fatal(err)
		}
		roleOf := map[string]ledger.Role{} // by TP: the deployments
		for i := range units {
			u := units[i]
			if u.Identity["benchmark_type"] != "agentic_traces" {
				continue
			}
			r, err := a.Role(SourceID, &u)
			if err != nil {
				t.Fatal(err)
			}
			key := u.RoleKey
			if prev, ok := roleOf[key]; ok && prev != r {
				t.Errorf("seed %s: one deployment's sweep drew %s and %s", seed[:4], prev, r)
			}
			roleOf[key] = r
		}
		if len(roleOf) != 2 {
			t.Errorf("%d role keys among the supplement's units, want one per deployment shape (2)", len(roleOf))
		}
	}
}

// The unit list is extended without the dump: the units of the dump and the supplement
// together are exactly the dump's units and the supplement's alone, so a committed list can be
// checked, and extended, by whoever has either part. This is what lets the 16 new units be
// added to the committed list without a 34 GB restore, and the nightly check-generated
// confirm it against the dump.
func TestSupplementUnitsAreIndependentOfTheDump(t *testing.T) {
	d := fetched(t, apiJSON(900, 42, 1, 4, "382.0"), apiJSON(901, 42, 8, 4, "400.5"))
	sr := []SupplementRows{{ID: d.ID, Rows: d.Rows}}
	both, brep, err := IngestWithSupplements(dumpRows(t), sr, "rev", "in")
	if err != nil {
		t.Fatal(err)
	}
	dumpOnly, drep, _ := Ingest(dumpRows(t), "rev", "in")
	suppOnly, srep, err := IngestWithSupplements(nil, sr, "rev", "in")
	if err != nil {
		t.Fatal(err)
	}
	want, _ := ledger.UnitListBytes(append(append([]ledger.Unit{}, dumpOnly...), suppOnly...))
	got, _ := ledger.UnitListBytes(both)
	if !bytes.Equal(got, want) {
		t.Error("ingesting the dump and the supplement together differs from the union of ingesting each")
	}
	if brep.Records != drep.Records+srep.Records || brep.Units != drep.Units+srep.Units {
		t.Errorf("report counts %d/%d, want the sums %d/%d", brep.Records, brep.Units, drep.Records+srep.Records, drep.Units+srep.Units)
	}
}

// D3: a record is pinned by the SHA-256 of its content. A record the API now serves with
// another value, or no longer serves, stops the ingest and names the record; nothing is
// ingested from a partly matching supplement.
func TestSupplementRefusesChangedOrMissingRecords(t *testing.T) {
	good := []json.RawMessage{apiJSON(900, 42, 1, 4, "382.0"), apiJSON(901, 42, 8, 4, "400.5")}
	_, sup := served(t, good...)

	changed, _ := json.Marshal([]json.RawMessage{good[0], apiJSON(901, 42, 8, 4, "401.5")})
	_, err := FetchSupplement(sup, func(string) ([]byte, error) { return changed, nil })
	if err == nil || !strings.Contains(err.Error(), "record 901") || strings.Contains(err.Error(), "record 900") {
		t.Errorf("a changed metric: %v", err)
	}

	gone, _ := json.Marshal([]json.RawMessage{good[0]})
	_, err = FetchSupplement(sup, func(string) ([]byte, error) { return gone, nil })
	if err == nil || !strings.Contains(err.Error(), "no longer serves") {
		t.Errorf("a record no longer served: %v", err)
	}

	// A change to a field the ingest does not read still changes the content hash.
	other := strings.Replace(string(good[1]), `"gpus":["a","b"]`, `"gpus":["a"]`, 1)
	changed, _ = json.Marshal([]json.RawMessage{good[0], json.RawMessage(other)})
	if _, err = FetchSupplement(sup, func(string) ([]byte, error) { return changed, nil }); err == nil {
		t.Error("a change outside the metrics was accepted")
	}
}

// Records the API serves that the supplement does not pin are not read: the ledger gains units
// only for what a ledger owner pinned.
func TestSupplementReadsOnlyPinnedRecords(t *testing.T) {
	pinned := apiJSON(900, 42, 1, 4, "382.0")
	extra := apiJSON(950, 43, 1, 4, "999.0")
	_, sup := served(t, pinned)
	body, _ := json.Marshal([]json.RawMessage{pinned, extra})
	d, err := FetchSupplement(sup, func(string) ([]byte, error) { return body, nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Rows) != 1 || d.Rows[0].ID != 900 {
		t.Errorf("rows %v, want only record 900", d.Rows)
	}
}

// A record the pinned dump already holds is not a supplement: the two copies could print a
// number differently and become two measurements of one unit.
func TestSupplementRefusesARecordTheDumpHolds(t *testing.T) {
	d := fetched(t, apiJSON(6, 42, 1, 4, "382.0"))
	_, _, err := IngestWithSupplements(dumpRows(t), []SupplementRows{{ID: d.ID, Rows: d.Rows}}, "rev", "in")
	if err == nil || !strings.Contains(err.Error(), "increasing id order") {
		t.Errorf("a record at or below the dump's last id: %v", err)
	}
}

// The attempt is read from the run URL and never guessed; a record with no run is refused.
func TestRowFromAPIRefusesWhatItCannotRead(t *testing.T) {
	good := string(apiJSON(900, 42, 1, 4, "382.0"))
	r, err := RowFromAPI(json.RawMessage(good))
	if err != nil {
		t.Fatal(err)
	}
	if r.RunAttempt != 1 || r.ID != 900 || r.Conc != 1 {
		t.Errorf("row %+v", r)
	}
	for name, bad := range map[string]string{
		"no run":        strings.Replace(good, `"run_url":"https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42/attempts/1"`, `"run_url":null`, 1),
		"no attempt":    strings.Replace(good, "/attempts/1", "", 1),
		"bad date":      strings.Replace(good, `"2026-11-09"`, `"last week"`, 1),
		"fractional id": strings.Replace(good, `"id":"900"`, `"id":"9.5"`, 1),
	} {
		if _, err := RowFromAPI(json.RawMessage(bad)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
