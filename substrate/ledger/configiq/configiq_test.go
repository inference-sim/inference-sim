package configiq_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/ledger"
	"github.com/inference-sim/blis-substrate/ledger/configiq"
)

// row returns a synthetic exported row with every column. Identifying columns get plain
// values; measured columns get distinct numbers derived from m, so rows with the same m hold
// the same measured values.
func row(file string, i int64, m float64, set map[string]any) configiq.Row {
	cols := map[string]any{
		"pair_id": "openai_gpt_oss_120b__b200", "model": "openai/gpt-oss-120b", "model_id": "openai/gpt-oss-120b",
		"accelerator": "B200", "backend": "vllm", "version": "vLLM-0.32.0", "backend_version": "vLLM-0.32.0",
		"tp": 1.0, "isl": 1000.0, "osl": 1000.0, "concurrency": 32.0, "turns": 1.0,
		"prefix_tokens": nil, "prefix_count": nil, "prefix_caching": "yes", "request_type": "/v1/completions",
		"spec_decoding": "", "gpu_memory_utilization": "0.92", "max_num_seqs": nil, "max_seq_len": nil,
		"chunked_prefill_enabled": nil, "precision": nil, "weight_quantization": "unknown", "kv_quantization": "unknown",
	}
	for j, c := range configiq.MeasuredColumns {
		cols[c] = m + float64(j)/8
	}
	cols["successful_requests"] = 100.0
	for k, v := range set {
		cols[k] = v
	}
	raw := map[string]json.RawMessage{}
	for k, v := range cols {
		b, _ := json.Marshal(v)
		raw[k] = b
	}
	return configiq.Row{File: file, Row: i, Columns: raw}
}

const f = "pairs/openai_gpt_oss_120b__b200/ground_truth.parquet"

func ingest(t *testing.T, rows ...configiq.Row) ([]ledger.Unit, *configiq.Report) {
	t.Helper()
	for i := range rows {
		rows[i].File, rows[i].Row = f, int64(i)
	}
	units, rep, err := configiq.Ingest(rows, strings.Repeat("a", 40), strings.Repeat("b", 64))
	if err != nil {
		t.Fatal(err)
	}
	for i := range units {
		if err := units[i].Check(configiq.RoleKey); err != nil {
			t.Fatal(err)
		}
	}
	return units, rep
}

// Re-measurements of one configuration are one unit holding each measurement, and a row that
// repeats another's identity and values is a duplicate of it (design §10.2, COV-10).
func TestReMeasurementsAreOneUnit(t *testing.T) {
	units, rep := ingest(t, row(f, 0, 10, nil), row(f, 0, 20, nil), row(f, 0, 10, nil), row(f, 0, 30, nil))
	if len(units) != 1 {
		t.Fatalf("%d units, want 1", len(units))
	}
	u := units[0]
	if len(u.Measurements) != 3 {
		t.Fatalf("%d measurements, want 3", len(u.Measurements))
	}
	if u.Measurements[0].Record != f+":0" || fmt.Sprint(u.Measurements[0].Duplicates) != "["+f+":2]" {
		t.Errorf("first measurement %s with duplicates %v, want %s:0 with [%s:2]", u.Measurements[0].Record, u.Measurements[0].Duplicates, f, f)
	}
	if rep.Duplicates != 1 || rep.JoinedUnits != 1 || rep.JoinedRecords != 4 || rep.Units != 1 {
		t.Errorf("report %+v", rep)
	}
}

// A sweep shares one role: rows that differ only in concurrency are two units with one role
// key, and a row that differs in TP has another role key.
func TestRoleKeyLeavesOutOnlyConcurrency(t *testing.T) {
	units, rep := ingest(t, row(f, 0, 10, nil), row(f, 0, 20, map[string]any{"concurrency": 64.0}),
		row(f, 0, 30, map[string]any{"tp": 2.0}))
	if len(units) != 3 || rep.RoleKeys != 2 {
		t.Fatalf("%d units and %d role keys, want 3 and 2", len(units), rep.RoleKeys)
	}
	keys := map[float64]string{}
	for _, u := range units {
		keys[float64(u.Identity["tp"].(int64))*1000+float64(u.Identity["concurrency"].(int64))] = u.RoleKey
	}
	if keys[1032] != keys[1064] || keys[1032] == keys[2032] {
		t.Errorf("role keys by tp and concurrency: %v", keys)
	}
}

// An identifying number is written as an integer whatever the file's column type, so a file
// that stores tp as 1 and one that stores it as 1.0 give one identity; a fraction is refused.
func TestIdentityNumbersAreCanonical(t *testing.T) {
	a, _ := ingest(t, row(f, 0, 10, nil))
	b, _ := ingest(t, row(f, 0, 10, map[string]any{"tp": 1, "isl": 1000, "concurrency": 32}))
	if a[0].UnitID != b[0].UnitID {
		t.Errorf("tp 1.0 and tp 1 give units %s and %s", a[0].UnitID, b[0].UnitID)
	}
	r := row(f, 0, 10, map[string]any{"tp": 1.5})
	if _, _, err := configiq.Ingest([]configiq.Row{r}, strings.Repeat("a", 40), ""); err == nil || !strings.Contains(err.Error(), "tp") {
		t.Errorf("a fractional tp: err %v, want a refusal naming tp", err)
	}
}

// A row with no successful request is not a measurement: it is left out and counted.
func TestNoSuccessfulRequestIsLeftOut(t *testing.T) {
	units, rep := ingest(t, row(f, 0, 10, nil), row(f, 0, 0, map[string]any{"successful_requests": 0.0, "concurrency": 1.0}))
	if len(units) != 1 || fmt.Sprint(rep.LeftOut[configiq.LeftOutNoSuccess]) != "["+f+":1]" {
		t.Errorf("%d units, left out %v", len(units), rep.LeftOut)
	}
}

// The same measured values under two identities with two role keys are left out, so one
// measurement cannot draw two roles; with one role key they are one unit.
func TestSameMeasurementUnderTwoIdentities(t *testing.T) {
	units, rep := ingest(t, row(f, 0, 10, nil), row(f, 0, 10, map[string]any{"tp": 2.0}), row(f, 0, 50, nil))
	if len(units) != 1 || len(rep.LeftOut["same_measurement_two_role_keys"]) != 2 {
		t.Errorf("two role keys: %d units, left out %v", len(units), rep.LeftOut)
	}
	units, rep = ingest(t, row(f, 0, 10, nil), row(f, 0, 10, map[string]any{"concurrency": 64.0}))
	if len(units) != 1 || rep.Merged != 1 || units[0].Measurements[0].Merged[0].Record != f+":1" {
		t.Errorf("one role key: %d units, merged %d", len(units), rep.Merged)
	}
}

// RoleKey refuses an identity of another source or with a field missing or added.
func TestRoleKeyRefusesForeignIdentities(t *testing.T) {
	r := row(f, 0, 10, nil)
	id, err := r.Identity()
	if err != nil {
		t.Fatal(err)
	}
	other := map[string]any{}
	for k, v := range id {
		other[k] = v
	}
	other["source"] = "inferencex"
	missing := map[string]any{}
	for k, v := range id {
		if k != "tp" {
			missing[k] = v
		}
	}
	extra := map[string]any{"pp": 1}
	for k, v := range id {
		extra[k] = v
	}
	for name, bad := range map[string]map[string]any{"another source": other, "no tp": missing, "an extra field": extra} {
		if _, err := configiq.RoleKey(bad); err == nil {
			t.Errorf("%s: RoleKey accepted it", name)
		}
	}
}

func exportLines(rows ...configiq.Row) string {
	var b strings.Builder
	for _, r := range rows {
		line, _ := json.Marshal(r)
		b.Write(line)
		b.WriteByte('\n')
	}
	return b.String()
}

// The export is read strictly: an unknown or missing column, and rows out of order, are
// refused (SUB-12).
func TestReadRowsIsStrict(t *testing.T) {
	good := exportLines(row(f, 0, 1, nil), row(f, 1, 2, nil), row("pairs/z/ground_truth.parquet", 0, 3, nil))
	if rows, err := configiq.ReadRows(strings.NewReader(good)); err != nil || len(rows) != 3 {
		t.Fatalf("good export: %d rows, err %v", len(rows), err)
	}
	unknown := row(f, 0, 1, nil)
	unknown.Columns["pp"] = json.RawMessage("1")
	missing := row(f, 0, 1, nil)
	delete(missing.Columns, "tp")
	for name, text := range map[string]string{
		"unknown column": exportLines(unknown),
		"missing column": exportLines(missing),
		"skipped row":    exportLines(row(f, 0, 1, nil), row(f, 2, 2, nil)),
		"files reversed": exportLines(row("pairs/z/ground_truth.parquet", 0, 1, nil), row(f, 0, 2, nil)),
		"empty":          "",
	} {
		if _, err := configiq.ReadRows(strings.NewReader(text)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// writeRevision writes a synthetic revision: one Parquet stand-in and a manifest naming it with
// n records, and returns the directory and the source entry that pins both.
func writeRevision(t *testing.T, n int) (string, *ledger.Source) {
	t.Helper()
	dir := t.TempDir()
	write := func(name string, b []byte) string {
		p := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		s := sha256.Sum256(b)
		return hex.EncodeToString(s[:])
	}
	ps := write(f, []byte("synthetic parquet"))
	man, _ := json.Marshal(map[string]any{"pairs": []any{map[string]any{"path": f, "file_sha256": ps, "records": n}}})
	ms := write("manifest.json", man)
	src := &ledger.Source{ID: configiq.SourceID, Revision: ledger.Revision{Assets: []ledger.Asset{{Name: f, SHA256: ps}, {Name: "manifest.json", SHA256: ms}}}}
	return dir, src
}

// D3: the ingest tool reads only the revision sources.yaml pins. A file whose bytes differ
// from its pin, and an export whose rows are not the manifest's, are refused.
func TestCheckRevision(t *testing.T) {
	rows := []configiq.Row{row(f, 0, 1, nil), row(f, 1, 2, nil)}
	dir, src := writeRevision(t, 2)
	if err := configiq.CheckRevision(src, dir, rows); err != nil {
		t.Fatalf("pinned revision: %v", err)
	}
	if err := configiq.CheckRevision(src, dir, rows[:1]); err == nil || !strings.Contains(err.Error(), "the export has 1 rows") {
		t.Errorf("a short export: err %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, filepath.FromSlash(f)), []byte("changed"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := configiq.CheckRevision(src, dir, rows); err == nil || !strings.Contains(err.Error(), "sources.yaml pins") {
		t.Errorf("a changed file: err %v", err)
	}
}
