package vllmperf_test

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/vllmperf"
)

// jsonFields returns the JSON names of a struct's fields.
func jsonFields(t reflect.Type) map[string]bool {
	out := map[string]bool{}
	for i := 0; i < t.NumField(); i++ {
		out[strings.Split(t.Field(i).Tag.Get("json"), ",")[0]] = true
	}
	return out
}

// Every field of a unit is checked against the record it came from (requirements §2.2). The
// row and the recipe give each field a value of its own, so a field read from the wrong place
// or two fields swapped fails; the expected unit is written here from those values and not
// from the writer's field lists.
func TestUnitFieldsMatchTheirRecord(t *testing.T) {
	const (
		commit = "0123456789abcdef0123456789abcdef01234567"
		repo   = "registry.example/vllm-release-repo"
	)
	metrics := []string{"tput_per_gpu", "input_tput_per_gpu", "output_tput_per_gpu",
		"mean_ttft", "mean_tpot", "mean_itl", "mean_e2el",
		"p99_ttft", "p99_tpot", "p99_itl", "p99_e2el",
		"median_ttft", "median_tpot", "median_itl", "median_e2el"}
	rowFields := map[string]string{"date": "2026-07-01 12:34:56", "model": "org/model-x", "device": "b200", "tp": "4", "conc": "64",
		"isl": "2048", "osl": "256", "precision": "fp8", "image": repo + ":" + commit}
	wantMeasured := map[string]any{}
	for i, m := range metrics {
		v := fmt.Sprintf("%d.5", 10+i)
		rowFields[m] = v
		wantMeasured[m] = json.Number(v)
	}
	got := jsonFields(reflect.TypeOf(vllmperf.Row{}))
	for f := range got {
		if _, ok := rowFields[f]; !ok {
			t.Fatalf("row field %q has no value in this test", f)
		}
	}
	if len(got) != len(rowFields) {
		t.Fatalf("Row has %d fields but the test gives %d a value", len(got), len(rowFields))
	}
	rowJSON, _ := json.Marshal(rowFields)
	rc := vllmperf.Recipe{Workload: "wl-b200", GPU: "B200", Model: "org/model-x", ServeArgs: "--tp 4 --kv-cache-dtype fp8",
		Benchmark: "2k-in-256-out", InputLen: 2048, OutputLen: 256, MaxConcurrency: 64}
	if n := len(jsonFields(reflect.TypeOf(rc))); n != 8 {
		t.Fatalf("Recipe has %d fields, the test accounts for 8", n)
	}
	rcJSON, _ := json.Marshal(rc)

	units, rep, err := vllmperf.Ingest(input(t, []string{string(rowJSON)}, []string{string(rcJSON)}), "rev")
	if err != nil {
		t.Fatal(err)
	}
	if len(units) != 1 || len(units[0].Measurements) != 1 {
		t.Fatalf("units %d, want 1 with one measurement; left out %v", len(units), rep.LeftOut)
	}
	u, m := units[0], units[0].Measurements[0]

	cfg, _ := ledger.ID(map[string]any{"tp": "4", "conc": "64", "isl": "2048", "osl": "256", "precision": "fp8",
		"image_repository": repo, "serve_args": "--tp 4 --kv-cache-dtype fp8"})
	wantIdentity := map[string]any{"source": "vllm-ci-perf", "commit": commit, "model": "org/model-x", "hardware": "b200",
		"benchmark": "2k-in-256-out", "config_sha256": cfg}
	if !reflect.DeepEqual(u.Identity, wantIdentity) {
		t.Errorf("identity %v, want %v", u.Identity, wantIdentity)
	}
	wantID, _ := ledger.ID(wantIdentity)
	if u.UnitID != wantID {
		t.Errorf("unit id %s, want %s", u.UnitID, wantID)
	}
	if u.RoleKey != wantID {
		t.Errorf("role key %s, want the identity's hash %s: every identity field is in it", u.RoleKey, wantID)
	}
	wantM, _ := ledger.MeasuredID(wantMeasured)
	if m.Measured != wantM || m.Record != "row:0" {
		t.Errorf("measurement %s of %s, want %s of row:0: a metric is read from another field", m.Measured, m.Record, wantM)
	}
	if want := (map[string]any{"date": "2026-07-01 12:34:56", "workload": "wl-b200"}); !reflect.DeepEqual(m.Recorded, want) {
		t.Errorf("recorded %v, want %v", m.Recorded, want)
	}
	if rep.UnitsBy["b200"] != 1 {
		t.Errorf("units by hardware %v, want the row's device", rep.UnitsBy)
	}
}

// A null metric stays null in the measured values: the writer does not fill a value the record
// does not hold (requirements §2.2).
func TestNullMetricIsNotFilled(t *testing.T) {
	rows := []string{row("2026-06-01 00:00:00", "m", "h200", "128", img(c1), "1.5")}
	units, _, err := vllmperf.Ingest(input(t, rows, []string{recipe("w", "H200", "m", "b", 128)}), "rev")
	if err != nil {
		t.Fatal(err)
	}
	want, _ := ledger.MeasuredID(map[string]any{"tput_per_gpu": json.Number("1.5"), "input_tput_per_gpu": nil, "output_tput_per_gpu": nil,
		"mean_ttft": json.Number("0.5"), "mean_tpot": nil, "mean_itl": nil, "mean_e2el": nil,
		"p99_ttft": nil, "p99_tpot": nil, "p99_itl": nil, "p99_e2el": nil,
		"median_ttft": nil, "median_tpot": nil, "median_itl": nil, "median_e2el": nil})
	if got := units[0].Measurements[0].Measured; got != want {
		t.Errorf("measured %s, want %s", got, want)
	}
}
