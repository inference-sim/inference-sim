package sglangnightly_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/ledger/sglangnightly"
)

func jsonFields(t reflect.Type) map[string]bool {
	out := map[string]bool{}
	for i := 0; i < t.NumField(); i++ {
		out[strings.Split(t.Field(i).Tag.Get("json"), ",")[0]] = true
	}
	return out
}

// Every field of a unit is checked against the record it came from (requirements §2.2). Each
// field of the artifact has a value of its own, so a field read from the wrong place or two
// fields swapped fails; the expected unit is written here from those values. A field no unit
// keeps is named, so a field added to the artifact without a test fails.
func TestUnitFieldsMatchTheirRecord(t *testing.T) {
	// Fields of the artifact and of a result that reach a unit, and the ones that do not.
	kept := []string{"run_id", "run_date", "commit_sha", "results", "gpu_config", "partition", "model", "variant", "server_args", "benchmarks"}
	notKept := []string{"branch", "benchmarks_by_io_len"}
	want := map[string]bool{}
	for _, f := range append(kept, notKept...) {
		want[f] = true
	}
	for f := range jsonFields(reflect.TypeOf(sglangnightly.Consolidated{})) {
		if !want[f] {
			t.Fatalf("artifact field %q is not accounted for", f)
		}
	}
	for f := range jsonFields(reflect.TypeOf(sglangnightly.Result{})) {
		if !want[f] {
			t.Fatalf("result field %q is not accounted for", f)
		}
	}

	const commit = "89abcdef0123456789abcdef0123456789abcdef"
	res := func(byIOLen string) string {
		return `{"gpu_config":"8-gpu-b200","partition":3,"model":"org/model-y","variant":"tp8-mtp","server_args":["--tp","8","--speculative-algorithm","EAGLE"],` +
			`"benchmarks":[{"batch_size":4,"input_len":1000,"output_len":100,"latency":1.25,"last_ttft":0.5},{"batch_size":8,"input_len":2000,"output_len":200,"latency":2.5,"last_ttft":0.75}],` +
			`"benchmarks_by_io_len":` + byIOLen + `}`
	}
	ingest := func(branch, byIOLen string) ledger.Unit {
		name := "consolidated-metrics-777"
		js := []byte(`{"run_id":"777","run_date":"2026-11-02T03:04:05+00:00","commit_sha":"` + commit + `","branch":"` + branch + `","results":[` + res(byIOLen) + `]}`)
		a, z := pinBytes(t, "777", 2, commit, name, js)
		src, dir := write(t, []ledger.Artifact{a}, [][]byte{z})
		arts, input, err := sglangnightly.ReadArtifacts(src, dir)
		if err != nil {
			t.Fatal(err)
		}
		units, _, err := sglangnightly.Ingest(arts, commit, input)
		if err != nil || len(units) != 1 || len(units[0].Measurements) != 1 {
			t.Fatalf("units %d, %v; want 1 with one measurement", len(units), err)
		}
		return units[0]
	}
	u := ingest("main", "{}")

	args, _ := ledger.ID([]any{"--tp", "8", "--speculative-algorithm", "EAGLE"})
	wantIdentity := map[string]any{"source": "sglang-nightly", "run_id": "777", "commit": commit, "test": "tp8-mtp", "model": "org/model-y",
		"hardware": "8-gpu-b200", "partition": int64(3), "server_args_sha256": args}
	if !reflect.DeepEqual(u.Identity, wantIdentity) {
		t.Errorf("identity %v, want %v", u.Identity, wantIdentity)
	}
	if id, _ := ledger.ID(wantIdentity); u.UnitID != id {
		t.Errorf("unit id %s, want %s", u.UnitID, id)
	}
	wantRK, _ := ledger.ID(map[string]any{"source": "sglang-nightly", "commit": commit, "test": "tp8-mtp", "model": "org/model-y", "hardware": "8-gpu-b200"})
	if u.RoleKey != wantRK {
		t.Errorf("role key %s, want %s: it is the commit, test, model and hardware", u.RoleKey, wantRK)
	}
	rows := []any{
		map[string]any{"batch_size": json.Number("4"), "input_len": json.Number("1000"), "output_len": json.Number("100"), "latency": json.Number("1.25"), "last_ttft": json.Number("0.5")},
		map[string]any{"batch_size": json.Number("8"), "input_len": json.Number("2000"), "output_len": json.Number("200"), "latency": json.Number("2.5"), "last_ttft": json.Number("0.75")},
	}
	wantM, _ := ledger.MeasuredID(map[string]any{"benchmarks": rows})
	m := u.Measurements[0]
	if m.Measured != wantM || m.Record != "777/2:0" {
		t.Errorf("measurement %s of %s, want %s of 777/2:0", m.Measured, m.Record, wantM)
	}
	if want := (map[string]any{"attempt": 2, "run_date": "2026-11-02T03:04:05+00:00"}); !reflect.DeepEqual(m.Recorded, want) {
		t.Errorf("recorded %v, want %v", m.Recorded, want)
	}
	if other := ingest("release", `{"x":1}`); !reflect.DeepEqual(other, u) {
		t.Error("the branch or the regrouped benchmarks changed the unit: a field that reaches no unit reached one")
	}
}
