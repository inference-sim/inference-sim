package trace

import (
	"bytes"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func i64(v int64) *int64 { return &v }

func sample() *Trace {
	return &Trace{
		Header: Header{Version: Version, TimeUnit: TimeUnitMicrosecondsLong, Mode: ModeGenerated,
			WorkloadSeed:      i64(0),
			GoodputSLOTargets: map[string]SLODimTargets{"critical": {TTFTMs: 500}},
			KVOffload: &KVOffload{CPUBytesToUse: 1 << 30, PerBlockBytes: 4096, BlockSize: 16, BlocksPerChunk: 4,
				TokensPerHash: 16, EvictionPolicy: "lru",
				Tiers: []KVOffloadTier{{Type: "nvme", RootDir: "/x", ReadBandwidth: 3000, SaturationQueueDepth: 1, SingleTransferFraction: 1}}},
			Router: &Router{EPPFlags: []string{"--pool-name=p"}, Seed: i64(7)},
		},
		Records: []Record{
			{RequestID: 0, ClientID: "c", SessionID: "s", RoundIndex: 0, PrefixGroup: "g", PrefixLength: 8,
				Streaming: true, InputTokens: 10, OutputTokens: 4, ReasonRatio: 0.25, ArrivalTimeUs: 0,
				FirstChunkTimeUs: 100, LastChunkTimeUs: 160, Status: StatusOK, ThinkTimeUs: i64(0)},
			{RequestID: 1, SessionID: "s", RoundIndex: 1, InputTokens: 3, OutputTokens: 2, ArrivalTimeUs: 500,
				DeadlineUs: 9000, Status: StatusIncomplete, Adapter: "lora-a", SLOTargetUs: 300, InputTokensReset: i64(9)},
		},
	}
}

func write(t *testing.T, tr *Trace) (header, data []byte) {
	t.Helper()
	var h, d bytes.Buffer
	if err := WriteHeader(&h, &tr.Header); err != nil {
		t.Fatal(err)
	}
	if err := WriteRecords(&d, tr.Records); err != nil {
		t.Fatal(err)
	}
	return h.Bytes(), d.Bytes()
}

// TestRoundTrip: what the writer writes, the reader reads back equal, and writing it again
// gives the same bytes (the run-export-replay round trip rests on this, DEV-7).
func TestRoundTrip(t *testing.T) {
	tr := sample()
	h, d := write(t, tr)
	back, err := Read(bytes.NewReader(h), bytes.NewReader(d))
	if err != nil {
		t.Fatalf("Read refused what the writer wrote: %v\n%s\n%s", err, h, d)
	}
	if !reflect.DeepEqual(back, tr) {
		t.Errorf("read back differs:\n got %+v\nwant %+v", back, tr)
	}
	h2, d2 := write(t, back)
	if !bytes.Equal(h, h2) || !bytes.Equal(d, d2) {
		t.Error("writing the read-back trace gave other bytes")
	}
}

// TestOptionalColumnsOnlyWhenUsed: a trace whose records carry no optional value has
// exactly the required columns, so adding an optional column changes nothing else.
func TestOptionalColumnsOnlyWhenUsed(t *testing.T) {
	var d bytes.Buffer
	if err := WriteRecords(&d, []Record{{RequestID: 0, Status: StatusOK}}); err != nil {
		t.Fatal(err)
	}
	first := strings.SplitN(d.String(), "\n", 2)[0]
	if first != strings.Join(Columns, ",") {
		t.Errorf("header row %q, want the required columns only", first)
	}
	// A recorded zero think time is a value, so it writes the column (the pointer is the
	// presence signal), and an unrecorded one in another row is an empty cell.
	d.Reset()
	if err := WriteRecords(&d, []Record{{RequestID: 0, Status: StatusOK, ThinkTimeUs: i64(0)}, {RequestID: 1, Status: StatusOK}}); err != nil {
		t.Fatal(err)
	}
	recs, err := ReadRecords(&d)
	if err != nil {
		t.Fatal(err)
	}
	if recs[0].ThinkTimeUs == nil || *recs[0].ThinkTimeUs != 0 || recs[1].ThinkTimeUs != nil {
		t.Errorf("think_time_us presence did not survive: %v, %v", recs[0].ThinkTimeUs, recs[1].ThinkTimeUs)
	}
}

// TestReaderIsStrict: every malformed input names what is wrong (SUB-12).
func TestReaderIsStrict(t *testing.T) {
	head := strings.Join(Columns, ",")
	row := "0,,,,,0,,0,true,1,1,0,0,0,0,0,,0,0,0,0,ok"
	for _, c := range []struct {
		name, data, want string
	}{
		{"empty", "", "empty file"},
		{"unknown column", head + ",send_time_us\n" + row + ",5\n", `"send_time_us" is not a trace column`},
		{"observe's vllm_priority", head + ",vllm_priority\n" + row + ",0\n", `"vllm_priority" is not a trace column`},
		{"repeated column", head + ",adapter,adapter\n" + row + ",a,b\n", `"adapter" appears twice`},
		{"missing column", strings.TrimSuffix(head, ",status") + "\n" + strings.TrimSuffix(row, ",ok") + "\n", `"status" is missing`},
		{"short row", head + "\n0,1\n", "wrong number of fields"},
		{"non-integer", head + "\n" + strings.Replace(row, "true,1,1", "true,1.5,1", 1) + "\n", "input_tokens"},
		{"non-boolean", head + "\n" + strings.Replace(row, "true", "yes", 1) + "\n", "streaming"},
		{"non-number", head + "\n" + strings.Replace(row, ",0,,0,0", ",x,,0,0", 1) + "\n", "reason_ratio"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := ReadRecords(strings.NewReader(c.data))
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("got %v, want an error containing %q", err, c.want)
			}
		})
	}
	for _, c := range []struct{ name, yaml, want string }{
		{"unknown key", "trace_version: 4\ntime_unit: us\nmode: generated\nserver: {}\n", "server"},
		{"observe's created_at", "trace_version: 4\ntime_unit: us\nmode: generated\ncreated_at: x\n", "created_at"},
		{"duplicate key", "trace_version: 4\ntrace_version: 4\ntime_unit: us\nmode: generated\n", "already defined"},
		{"two documents", "trace_version: 4\ntime_unit: us\nmode: generated\n---\na: 1\n", "more than one"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := ReadHeader(strings.NewReader(c.yaml))
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("got %v, want an error containing %q", err, c.want)
			}
		})
	}
}

// TestEachFailureModeIsRefused: one case per validation rule, each at its path.
func TestEachFailureModeIsRefused(t *testing.T) {
	nan := math.NaN()
	cases := []struct {
		name   string
		break_ func(*Trace)
		path   string
		msg    string
	}{
		{"old version", func(t *Trace) { t.Header.Version = 3 }, "header.trace_version", "must be 4"},
		{"no time unit", func(t *Trace) { t.Header.TimeUnit = "" }, "header.time_unit", "required"},
		{"unknown time unit", func(t *Trace) { t.Header.TimeUnit = "fortnights" }, "header.time_unit", "is not one of"},
		{"no mode", func(t *Trace) { t.Header.Mode = "" }, "header.mode", "required"},
		{"mode real", func(t *Trace) { t.Header.Mode = "real" }, "header.mode", "converted"},
		{"unknown mode", func(t *Trace) { t.Header.Mode = "simulated" }, "header.mode", "is not one of"},
		{"unknown context growth", func(t *Trace) { t.Header.SessionContextGrowth = "independent" }, "header.session_context_growth", "omit it"},
		{"empty slo class", func(t *Trace) { t.Header.GoodputSLOTargets[""] = SLODimTargets{TTFTMs: 1} }, "header.goodput_slo_targets", "must not be empty"},
		{"class sets no target", func(t *Trace) { t.Header.GoodputSLOTargets["critical"] = SLODimTargets{} }, "header.goodput_slo_targets.critical", "sets no target"},
		{"non-finite target", func(t *Trace) { t.Header.GoodputSLOTargets["critical"] = SLODimTargets{TTFTMs: nan} }, "header.goodput_slo_targets.critical.ttft_ms", "finite"},
		{"negative offload bytes", func(t *Trace) { t.Header.KVOffload.CPUBytesToUse = -1 }, "header.kv_offload.cpu_bytes_to_use", "negative"},
		{"tier without type", func(t *Trace) { t.Header.KVOffload.Tiers[0].Type = "" }, "header.kv_offload.secondary_tiers[0].type", "required"},
		{"non-finite tier bandwidth", func(t *Trace) { t.Header.KVOffload.Tiers[0].ReadBandwidth = nan }, "header.kv_offload.secondary_tiers[0].read_bandwidth", "finite"},
		{"span of one", func(t *Trace) { t.Header.MaxNodesSpanned = 1 }, "header.max_nodes_spanned", "omit"},
		{"negative span", func(t *Trace) { t.Header.MaxNodesSpanned = -2 }, "header.max_nodes_spanned", "negative"},
		{"router without flags", func(t *Trace) { t.Header.Router.EPPFlags = nil }, "header.router.epp_flags", "required"},
		{"converted with offload", func(t *Trace) { t.Header.Mode = ModeConverted; t.Header.Router = nil }, "header.kv_offload", "no BLIS deployment"},
		{"converted with a router", func(t *Trace) { t.Header.Mode = ModeConverted; t.Header.KVOffload = nil }, "header.router", "no BLIS deployment"},
		{"negative input", func(t *Trace) { t.Records[0].InputTokens = -1 }, "records[0].input_tokens", "negative"},
		{"negative arrival", func(t *Trace) { t.Records[0].ArrivalTimeUs = -1 }, "records[0].arrival_time_us", "negative"},
		{"reason ratio above one", func(t *Trace) { t.Records[0].ReasonRatio = 1.5 }, "records[0].reason_ratio", "[0, 1]"},
		{"deadline before arrival", func(t *Trace) { t.Records[1].DeadlineUs = 100 }, "records[1].deadline_us", "before arrival"},
		{"last chunk before first", func(t *Trace) { t.Records[0].LastChunkTimeUs = 50 }, "records[0].last_chunk_time_us", "before first"},
		{"unknown status", func(t *Trace) { t.Records[0].Status = "aborted" }, "records[0].status", "is not one of"},
		{"negative think time", func(t *Trace) { t.Records[0].ThinkTimeUs = i64(-1) }, "records[0].think_time_us", "negative"},
		{"negative reset", func(t *Trace) { t.Records[1].InputTokensReset = i64(-1) }, "records[1].input_tokens_reset", "negative"},
		{"repeated request id", func(t *Trace) { t.Records[1].RequestID = 0 }, "records[1].request_id", "twice"},
		{"repeated round", func(t *Trace) { t.Records[1].RoundIndex = 0 }, "records[1].round_index", "twice"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tr := sample()
			c.break_(tr)
			p := tr.Validate()
			for _, it := range p.Errors() {
				if it.Path == c.path && strings.Contains(it.Message, c.msg) {
					return
				}
			}
			t.Errorf("expected a problem at %q containing %q, got:\n%s", c.path, c.msg, p.Error())
		})
	}
}

// TestRecordsDeployment: inference-sim #2067. A converted trace records no deployment, so a
// replay may take deployment options from the scenario; a BLIS-written one keeps its header.
func TestRecordsDeployment(t *testing.T) {
	for m, want := range map[Mode]bool{ModeGenerated: true, ModeReplayed: true, ModeConverted: false} {
		if got := (&Header{Mode: m}).RecordsDeployment(); got != want {
			t.Errorf("%s: RecordsDeployment = %v, want %v", m, got, want)
		}
	}
}

type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) { return 0, errors.New("disk full") }

// TestWritersReturnTheirErrors: a write that fails is reported, not dropped
// (inference-sim #661).
func TestWritersReturnTheirErrors(t *testing.T) {
	tr := sample()
	if err := WriteRecords(failingWriter{}, tr.Records); err == nil || !strings.Contains(err.Error(), "disk full") {
		t.Errorf("WriteRecords: got %v", err)
	}
	if err := WriteHeader(failingWriter{}, &tr.Header); err == nil || !strings.Contains(err.Error(), "disk full") {
		t.Errorf("WriteHeader: got %v", err)
	}
}
