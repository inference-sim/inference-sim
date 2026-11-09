package workload

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// v3Fixture is testdata/trace-v3: the trace v0.15.0's `blis run --trace-output` wrote for the
// byte gate's moe-prefix case (trace format version 3), and v4Golden the one this build
// writes for the same run (testdata/bytegate/trace).
func v3Fixture(t *testing.T) (header, data string) {
	t.Helper()
	root := repoRootFromHere(t)
	return filepath.Join(root, "testdata", "trace-v3", "moe-prefix.yaml"), filepath.Join(root, "testdata", "trace-v3", "moe-prefix.csv")
}

func v4Golden(t *testing.T) (header, data string) {
	t.Helper()
	root := repoRootFromHere(t)
	return filepath.Join(root, "testdata", "bytegate", "trace", "moe-prefix.yaml"), filepath.Join(root, "testdata", "bytegate", "trace", "moe-prefix.csv")
}

func repoRootFromHere(t *testing.T) string {
	t.Helper()
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("no caller information")
	}
	return filepath.Dir(filepath.Dir(filepath.Dir(self)))
}

// Version 3 to version 4: the version 3 trace of a run converts to exactly the bytes this
// build writes for the same run, and send_time_us, which carried a value on every row, is
// reported dropped without changing the replay (it equals arrival_time_us on every row).
func TestConvertTraceV3_ToTheTraceThisBuildWrites(t *testing.T) {
	hp, dp := v3Fixture(t)
	tr, drops, err := ConvertTraceV3(hp, dp)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	oh, od := filepath.Join(dir, "t.yaml"), filepath.Join(dir, "t.csv")
	if err := ExportTraceV2(&tr.Header, tr.Records, oh, od); err != nil {
		t.Fatal(err)
	}
	gh, gd := v4Golden(t)
	for _, pair := range [][2]string{{oh, gh}, {od, gd}} {
		got, _ := os.ReadFile(pair[0])
		want, err := os.ReadFile(pair[1])
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("the conversion of %s differs from %s", filepath.Base(pair[1]), pair[1])
		}
	}
	if len(drops) != 1 || drops[0].Field != "send_time_us" || drops[0].Rows != len(tr.Records) || drops[0].ChangesReplay {
		t.Errorf("drops %+v, want send_time_us on every row, not changing the replay", drops)
	}
}

// Version 4 to version 3 and back: every field version 4 has survives a version 3 file, and
// each field version 4 dropped is reported, named, with whether it changed what v0.15.0's
// replay simulated.
func TestConvertTraceV3_EveryDroppedFieldIsNamed(t *testing.T) {
	gh, gd := v4Golden(t)
	want, err := LoadTraceV2(gh, gd)
	if err != nil {
		t.Fatal(err)
	}
	// The golden as version 3: the v3 header keys and columns, each carrying a value.
	head, _ := os.ReadFile(gh)
	h3 := strings.Replace(string(head), "trace_version: 4\n", "trace_version: 3\ncreated_at: \"2026-01-01T00:00:00Z\"\n"+
		"warm_up_requests: 2\nworkload_spec: w.yaml\nserver:\n  type: vllm\nnetwork:\n  measured_rtt_ms: 2.5\n"+
		"observed_kv_metrics:\n  hit_rate: 0.5\n", 1)
	data, _ := os.ReadFile(gd)
	lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
	lines[0] += ",server_input_tokens,send_time_us,num_chunks,error_message,finish_reason,vllm_priority,x_request_id"
	for i := 1; i < len(lines); i++ {
		// Row 1: a send time after the arrival, and row 2 a server-reported length on a row with
		// no prefix group: the two values v0.15.0's replay read. (Every golden row has prefix
		// group "shared", which made v0.15.0 ignore the server-reported length, so row 2's
		// group is cleared here and in the expectation.)
		extra := ",0,0,0,,,,"
		switch i {
		case 1:
			extra = ",0,99999999,3,boom,stop,1,x-1"
		case 2:
			lines[i] = strings.Replace(lines[i], ",shared,", ",,", 1)
			want.Records[i-1].PrefixGroup = ""
			extra = ",9999,0,0,,,,"
		}
		lines[i] += extra
	}
	dir := t.TempDir()
	hp, dp := filepath.Join(dir, "v3.yaml"), filepath.Join(dir, "v3.csv")
	if err := os.WriteFile(hp, []byte(h3), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dp, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, drops, err := ConvertTraceV3(hp, dp)
	if err != nil {
		t.Fatal(err)
	}
	var b1, b2 bytes.Buffer
	if err := writeTraceHeader(&b1, &want.Header); err != nil {
		t.Fatal(err)
	}
	if err := writeTraceHeader(&b2, &got.Header); err != nil {
		t.Fatal(err)
	}
	if b1.String() != b2.String() || len(got.Records) != len(want.Records) {
		t.Fatalf("the round trip changed the header or the row count:\n%s\n%s", b1.String(), b2.String())
	}
	for i := range want.Records {
		if !recordsEqual(want.Records[i], got.Records[i]) {
			t.Errorf("record %d: %+v, want %+v", i, got.Records[i], want.Records[i])
		}
	}
	named := map[string]TraceV3Drop{}
	for _, d := range drops {
		named[d.Field] = d
	}
	for _, f := range []string{"created_at", "warm_up_requests", "workload_spec", "server", "network",
		"observed_kv_metrics", "server_input_tokens", "send_time_us", "num_chunks", "error_message",
		"finish_reason", "vllm_priority", "x_request_id"} {
		d, ok := named[f]
		if !ok || d.Rows != 1 {
			t.Errorf("%s: dropped %+v, want it named with 1 row", f, d)
		}
		if replay := f == "server_input_tokens" || f == "send_time_us"; d.ChangesReplay != replay {
			t.Errorf("%s: ChangesReplay %v, want %v", f, d.ChangesReplay, replay)
		}
	}
	if len(named) != 13 {
		t.Errorf("drops %+v, want the 13 fields version 4 dropped", drops)
	}
}

// A mode-real header (a capture by blis observe) becomes converted; a version 4 trace and an
// unknown version are refused.
func TestConvertTraceV3_ModeAndVersion(t *testing.T) {
	_, dp := v3Fixture(t)
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	tr, drops, err := ConvertTraceV3(write("real.yaml", "trace_version: 3\ntime_unit: microseconds\nmode: real\nwarm_up_requests: 0\n"), dp)
	if err != nil || tr.Header.Mode != TraceModeConverted {
		t.Fatalf("mode real: %v, %+v", err, tr)
	}
	if drops[0].Field != "mode: real (now converted)" {
		t.Errorf("drops %+v, want the mode named first", drops)
	}
	for body, want := range map[string]string{
		"trace_version: 4\ntime_unit: microseconds\nmode: generated\n": "already version 4",
		"trace_version: 7\ntime_unit: microseconds\nmode: generated\n": "not version 2 or 3",
	} {
		if _, _, err := ConvertTraceV3(write("h.yaml", body), dp); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%q: want %q, got %v", body, want, err)
		}
	}
}

// LoadTraceV2 refuses a version 3 trace before reading its keys, naming the conversion.
func TestLoadTraceV2_RefusesVersion3NamingTheConversion(t *testing.T) {
	hp, dp := v3Fixture(t)
	_, err := LoadTraceV2(hp, dp)
	if err == nil || !strings.Contains(err.Error(), "trace format version 3") ||
		!strings.Contains(err.Error(), "blis convert trace --header") {
		t.Errorf("want a refusal naming version 3 and blis convert trace, got %v", err)
	}
}

func recordsEqual(a, b TraceRecord) bool {
	eq := func(x, y *int64) bool { return (x == nil) == (y == nil) && (x == nil || *x == *y) }
	if !eq(a.ThinkTimeUs, b.ThinkTimeUs) || !eq(a.InputTokensReset, b.InputTokensReset) {
		return false
	}
	a.ThinkTimeUs, a.InputTokensReset, b.ThinkTimeUs, b.InputTokensReset = nil, nil, nil, nil
	return a == b
}

// A version 3 file with an slo_target_us column, written by v0.15.0's writer (testdata/trace-v3/
// slo-target.*): v0.15.0's writer placed every column correctly, and its reader misread
// session_id, prefix_group and model on such a file (one column to the right). The converter
// reads by column name, so it keeps the values as written; the file needed a correct reader,
// not a repair.
func TestConvertTraceV3_ReadsAnSLOTargetFileByName(t *testing.T) {
	root := repoRootFromHere(t)
	tr, _, err := ConvertTraceV3(filepath.Join(root, "testdata", "trace-v3", "slo-target.yaml"),
		filepath.Join(root, "testdata", "trace-v3", "slo-target.csv"))
	if err != nil {
		t.Fatal(err)
	}
	want := []struct {
		session, group, model string
		slo, deadline         int64
	}{{"s0", "g", "m", 150000, 900000}, {"s1", "", "", 0, 0}}
	for i, w := range want {
		r := tr.Records[i]
		if r.SessionID != w.session || r.PrefixGroup != w.group || r.Model != w.model || r.SLOTargetUs != w.slo || r.DeadlineUs != w.deadline {
			t.Errorf("record %d: %+v, want session %q group %q model %q slo %d deadline %d", i, r, w.session, w.group, w.model, w.slo, w.deadline)
		}
	}
}
