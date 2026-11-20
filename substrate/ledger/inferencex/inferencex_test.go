package inferencex

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/ledger"
)

// row returns a synthetic exported row. Its numbers are invented; no InferenceX measurement is
// committed (X12).
func row(id int64, run int64, attempt int64, conc int64, metrics string) string {
	r := map[string]any{
		"id": id, "github_run_id": run, "run_url": fmt.Sprintf("https://github.com/SemiAnalysisAI/InferenceX/actions/runs/%d/attempts/%d", run, attempt),
		"run_attempt": attempt, "run_conclusion": "success", "config_id": 7,
		"model": "m", "hardware": "h200", "precision": "fp8", "framework": "vllm", "benchmark_type": "single_turn",
		"conc": conc, "error": nil, "date": "2026-01-02", "recipe_fingerprint": nil, "image": "vllm/vllm-openai:v0.32.0",
		"isl": 1024, "osl": 1024, "offload_mode": "off", "spec_method": "none", "disagg": false, "is_multinode": false,
		"prefill_tp": 8, "prefill_ep": 1, "decode_tp": 8, "decode_ep": 1, "prefill_dp_attention": false,
		"decode_dp_attention": false, "prefill_num_workers": 0, "decode_num_workers": 0, "num_prefill_gpu": 8,
		"num_decode_gpu": 8, "metrics": json.RawMessage(metrics),
	}
	b, _ := json.Marshal(r)
	return string(b)
}

func edit(line string, f func(m map[string]any)) string {
	var m map[string]any
	d := json.NewDecoder(strings.NewReader(line))
	d.UseNumber()
	if err := d.Decode(&m); err != nil {
		panic(err)
	}
	f(m)
	b, _ := json.Marshal(m)
	return string(b)
}

func ingest(t *testing.T, lines ...string) ([]ledger.Unit, *Report) {
	t.Helper()
	rows, err := ReadRows(strings.NewReader(strings.Join(lines, "\n") + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	units, rep, err := Ingest(rows, "rev", "input")
	if err != nil {
		t.Fatal(err)
	}
	for i := range units {
		if err := units[i].Check(RoleKey); err != nil {
			t.Fatal(err)
		}
	}
	return units, rep
}

// The configuration hash covers all 17 fields, with nulls kept, and drops the image digest.
func TestConfigurationHash(t *testing.T) {
	rows, _ := ReadRows(strings.NewReader(row(1, 10, 1, 4, `{"a":1}`) + "\n" +
		edit(row(2, 10, 1, 4, `{"a":1}`), func(m map[string]any) { m["image"] = "vllm/vllm-openai:v0.32.0@sha256:abc" }) + "\n" +
		edit(row(3, 10, 1, 4, `{"a":1}`), func(m map[string]any) { m["isl"] = nil }) + "\n" +
		edit(row(4, 10, 1, 4, `{"a":1}`), func(m map[string]any) { m["decode_dp_attention"] = true }) + "\n"))
	h := func(r *Row) string { id, _ := r.Identity(); return id["configuration_hash"].(string) }
	if h(&rows[0]) != h(&rows[1]) {
		t.Error("the image digest changed the configuration hash")
	}
	if h(&rows[0]) == h(&rows[2]) {
		t.Error("a null isl hashed as the stated isl")
	}
	if h(&rows[0]) == h(&rows[3]) {
		t.Error("decode_dp_attention is not in the configuration hash")
	}
	if len(rows[0].Configuration()) != 17 {
		t.Errorf("configuration has %d fields, want 17", len(rows[0].Configuration()))
	}
	if _, d := SplitImage(rows[1].Image); d == nil || *d != "sha256:abc" {
		t.Errorf("digest %v", d)
	}
}

// The benchmark run is the run id without its attempt, so two attempts are one unit; a run
// with no URL has a null run.
func TestBenchmarkRun(t *testing.T) {
	u := "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/123/attempts/4"
	if r := BenchmarkRun(&u); r == nil || *r != 123 {
		t.Errorf("run %v", r)
	}
	if BenchmarkRun(nil) != nil {
		t.Error("a missing run_url gave a run")
	}
	bad := "https://example.com/x"
	if BenchmarkRun(&bad) != nil {
		t.Error("a URL naming no run gave a run")
	}
}

// Duplicates: two attempts with the same metrics are one unit with one measurement, keeping
// the smallest row id, and the other listed as its duplicate. A field only the duplicate
// records (the image digest) comes from it.
func TestDuplicatesKeepSmallestRow(t *testing.T) {
	a := row(5, 10, 2, 4, `{"ttft":0.1}`)
	b := edit(row(9, 10, 3, 4, `{"ttft":0.1}`), func(m map[string]any) { m["image"] = "vllm/vllm-openai:v0.32.0@sha256:d1" })
	units, rep := ingest(t, a, b)
	if len(units) != 1 || len(units[0].Measurements) != 1 {
		t.Fatalf("units %+v", units)
	}
	m := units[0].Measurements[0]
	if m.Record != "5" || len(m.Duplicates) != 1 || m.Duplicates[0] != "9" || rep.Duplicates != 1 {
		t.Errorf("kept %s, duplicates %v", m.Record, m.Duplicates)
	}
	if m.Recorded["image_digest"] != "sha256:d1" {
		t.Errorf("image_digest %v, want the duplicate's", m.Recorded["image_digest"])
	}
	if fmt.Sprint(m.Recorded["run_attempt"]) != "2" {
		t.Errorf("run_attempt %v, want the kept record's", m.Recorded["run_attempt"])
	}
}

// Joined: one identity with different metrics is one unit holding each measurement.
func TestDifferentMeasurementsJoin(t *testing.T) {
	units, rep := ingest(t, row(1, 10, 1, 4, `{"ttft":0.1}`), row(2, 10, 2, 4, `{"ttft":0.2}`))
	if len(units) != 1 || len(units[0].Measurements) != 2 || rep.JoinedUnits != 1 || rep.JoinedRecords != 2 {
		t.Fatalf("units %d, measurements %d, report %+v", len(units), len(units[0].Measurements), rep)
	}
}

// The same metrics under two identities: with one role key they merge under the smallest row
// id's identity; with two role keys every record is left out.
func TestSameMeasurementTwoIdentities(t *testing.T) {
	a := row(1, 10, 1, 4, `{"ttft":0.1}`)
	b := edit(row(2, 10, 1, 4, `{"ttft":0.1}`), func(m map[string]any) { m["recipe_fingerprint"] = "fp" })
	units, rep := ingest(t, a, b)
	if len(units) != 1 || rep.Merged != 1 || len(units[0].Measurements[0].Merged) != 1 {
		t.Fatalf("one role key: units %d, merged %d", len(units), rep.Merged)
	}
	if units[0].Identity["recipe_fingerprint"] != nil {
		t.Error("the unit took the larger row id's identity")
	}
	c := edit(row(3, 10, 1, 4, `{"ttft":0.1}`), func(m map[string]any) { m["decode_tp"] = 4 })
	units, rep = ingest(t, a, c, row(4, 11, 1, 8, `{"ttft":0.3}`))
	if len(units) != 1 || len(rep.LeftOut["same_measurement_two_role_keys"]) != 2 {
		t.Fatalf("two role keys: units %d, left out %v", len(units), rep.LeftOut)
	}
}

// Not a measurement: an error, or empty metrics, is left out and counted; the run's conclusion
// never removes a record.
func TestNotAMeasurement(t *testing.T) {
	failed := edit(row(2, 10, 1, 8, `{"ttft":0.2}`), func(m map[string]any) { m["run_conclusion"] = "failure" })
	errored := edit(row(3, 10, 1, 16, `{"ttft":0.3}`), func(m map[string]any) { m["error"] = "oom" })
	units, rep := ingest(t, row(1, 10, 1, 4, `{"ttft":0.1}`), failed, errored, row(4, 10, 1, 32, `{}`))
	if len(units) != 2 || len(rep.LeftOut["error_reported"]) != 1 || len(rep.LeftOut["empty_metrics"]) != 1 {
		t.Fatalf("units %d, left out %v", len(units), rep.LeftOut)
	}
}

// A sweep and its re-measurement share one role key; another configuration does not.
func TestRoleKeyLeavesOutSweepFields(t *testing.T) {
	units, rep := ingest(t, row(1, 10, 1, 4, `{"a":1}`), row(2, 10, 1, 8, `{"a":2}`),
		edit(row(3, 11, 1, 4, `{"a":3}`), func(m map[string]any) { m["date"] = "2026-02-03"; m["recipe_fingerprint"] = "fp" }),
		edit(row(4, 12, 1, 4, `{"a":4}`), func(m map[string]any) { m["prefill_num_workers"] = 2 }))
	if len(units) != 4 || rep.RoleKeys != 2 {
		t.Fatalf("units %d, role keys %d, want 4 and 2", len(units), rep.RoleKeys)
	}
}

// Known-bad exports: rows out of order, an unknown field, a run with two dates, and an empty
// export.
func TestReadRowsRefuses(t *testing.T) {
	for name, body := range map[string]string{
		"order":   row(2, 10, 1, 4, `{"a":1}`) + "\n" + row(1, 10, 1, 8, `{"a":2}`),
		"unknown": edit(row(1, 10, 1, 4, `{"a":1}`), func(m map[string]any) { m["extra"] = 1 }),
		"empty":   "",
	} {
		if _, err := ReadRows(strings.NewReader(body)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// The measured date is the earliest date of the benchmark run, for every attempt; a record
// with no run id keeps its own.
func TestMeasuredDateIsTheRunsEarliest(t *testing.T) {
	later := edit(row(1, 10, 2, 4, `{"a":1}`), func(m map[string]any) { m["date"] = "2026-01-05" })
	earlier := edit(row(2, 10, 1, 8, `{"a":2}`), func(m map[string]any) { m["date"] = "2026-01-03" })
	norun := edit(row(3, 0, 1, 16, `{"a":3}`), func(m map[string]any) { m["run_url"] = nil; m["date"] = "2026-01-09" })
	units, _ := ingest(t, later, earlier, norun)
	dates := map[string]bool{}
	for _, u := range units {
		dates[fmt.Sprintf("%v %v", u.Identity["conc"], u.Identity["measured_date"])] = true
	}
	for _, want := range []string{"4 2026-01-03", "8 2026-01-03", "16 2026-01-09"} {
		if !dates[want] {
			t.Errorf("no unit at %s; units %v", want, dates)
		}
	}
}

// The ingest is a function of its input: two runs give the same bytes (D4).
func TestIngestIsDeterministic(t *testing.T) {
	lines := []string{row(1, 10, 1, 4, `{"a":1}`), row(2, 10, 2, 4, `{"a":1}`), row(3, 10, 1, 8, `{"a":2}`)}
	u1, _ := ingest(t, lines...)
	u2, _ := ingest(t, lines...)
	b1, _ := ledger.UnitListBytes(u1)
	b2, _ := ledger.UnitListBytes(u2)
	if !bytes.Equal(b1, b2) {
		t.Error("two ingests of one input differ")
	}
}

// The legacy point identifier reproduces build_inventory.py's for a row: the value Python
// gives for github_run_id 18117598309, attempt 1, config 16, single_turn, 1024/1024, no image,
// offload off, concurrency 8.
func TestLegacyPoint(t *testing.T) {
	r := edit(row(1, 18117598309, 1, 8, `{"a":1}`), func(m map[string]any) {
		m["config_id"] = 16
		m["image"] = nil
	})
	rows, _ := ReadRows(strings.NewReader(r))
	if got := LegacyPoint(&rows[0]); got != "2532419df3822a0a" {
		t.Errorf("LegacyPoint = %s, want 2532419df3822a0a", got)
	}
	// An agentic row has no isl or osl; Python writes them as None.
	r = edit(r, func(m map[string]any) { m["benchmark_type"] = "agentic_traces"; m["isl"] = nil; m["osl"] = nil })
	rows, _ = ReadRows(strings.NewReader(r))
	if got := LegacyPoint(&rows[0]); got != "6f5f6a3667e9144b" {
		t.Errorf("LegacyPoint = %s, want 6f5f6a3667e9144b", got)
	}
}
