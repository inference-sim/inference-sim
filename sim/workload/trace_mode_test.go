package workload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/trace"
)

// writeTrace writes a header and a data file into a new directory and returns their paths.
func writeTrace(t *testing.T, header, data string) (string, string) {
	t.Helper()
	dir := t.TempDir()
	hp, dp := filepath.Join(dir, "h.yaml"), filepath.Join(dir, "d.csv")
	if err := os.WriteFile(hp, []byte(header), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dp, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	return hp, dp
}

// traceDataHeaderRow is the CSV header row of a trace with no optional column.
var traceDataHeaderRow = strings.Join(traceColumns, ",") + "\n"

// Each mode a BLIS command writes loads; mode "real", which only blis observe wrote, is refused
// naming the value and the removal, and any other value is refused naming the allowed set.
func TestLoadTraceV2_Mode(t *testing.T) {
	for _, mode := range []trace.Mode{TraceModeGenerated, TraceModeReplayed, TraceModeConverted} {
		hp, dp := writeTrace(t, "trace_version: 4\ntime_unit: microseconds\nmode: "+string(mode)+"\n", traceDataHeaderRow)
		if _, err := LoadTraceV2(hp, dp); err != nil {
			t.Errorf("mode %s: %v", mode, err)
		}
	}
	for mode, want := range map[string]string{
		"real":      `trace header mode "real" is not supported: it was written by blis observe, which v0.16.0 removed`,
		"synthetic": `trace header mode "synthetic" is not one of converted, generated, replayed`,
		"":          `trace header mode "" is not one of converted, generated, replayed`,
	} {
		hp, dp := writeTrace(t, "trace_version: 4\ntime_unit: microseconds\nmode: \""+mode+"\"\n", traceDataHeaderRow)
		if _, err := LoadTraceV2(hp, dp); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("mode %q: want an error containing %q, got %v", mode, want, err)
		}
	}
}

// The header keys only blis observe wrote are refused by strict parsing, naming the key.
func TestLoadTraceV2_ObserveHeaderKeysAreRefused(t *testing.T) {
	for _, key := range []string{
		"created_at: \"2026-01-01T00:00:00Z\"",
		"warm_up_requests: 0",
		"workload_spec: w.yaml",
		"server:\n  type: vllm",
		"network:\n  measured_rtt_ms: 2.5",
		"observed_kv_metrics:\n  source: tiered\n  hit_rate: 0.5\n  block_hits: 5\n  block_queries: 10",
	} {
		name := strings.SplitN(key, ":", 2)[0]
		hp, dp := writeTrace(t, "trace_version: 4\ntime_unit: microseconds\nmode: converted\n"+key+"\n", traceDataHeaderRow)
		if _, err := LoadTraceV2(hp, dp); err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("header key %s: want a refusal naming it, got %v", name, err)
		}
	}
}

// The data columns version 4 dropped are refused by name, pointing at the conversion.
// vllm_priority sat after slo_class, so a loader that skipped it would read every later
// column one place off.
func TestLoadTraceV2_V3DataColumnsAreRefused(t *testing.T) {
	header := "trace_version: 4\ntime_unit: microseconds\nmode: converted\n"
	for _, col := range []string{"vllm_priority", "x_request_id", "server_input_tokens", "send_time_us",
		"num_chunks", "error_message", "finish_reason"} {
		hp, dp := writeTrace(t, header, strings.TrimSuffix(traceDataHeaderRow, "\n")+","+col+"\n")
		_, err := LoadTraceV2(hp, dp)
		if err == nil || !strings.Contains(err.Error(), `column "`+col+`" belongs to trace format version 3`) ||
			!strings.Contains(err.Error(), "blis convert trace") {
			t.Errorf("column %s: want a refusal naming it, got %v", col, err)
		}
	}
	// The closest file that loads: the same header row with a column the loader reads.
	hp, dp := writeTrace(t, header, strings.TrimSuffix(traceDataHeaderRow, "\n")+",adapter\n")
	if _, err := LoadTraceV2(hp, dp); err != nil {
		t.Errorf("an adapter column must load: %v", err)
	}
}

// A converted trace records no deployment; one blis run or blis replay wrote does.
func TestTraceHeader_RecordsDeployment(t *testing.T) {
	for mode, want := range map[trace.Mode]bool{TraceModeGenerated: true, TraceModeReplayed: true, TraceModeConverted: false} {
		if got := (&TraceHeader{Mode: mode}).RecordsDeployment(); got != want {
			t.Errorf("mode %s: RecordsDeployment() = %v, want %v", mode, got, want)
		}
	}
}
