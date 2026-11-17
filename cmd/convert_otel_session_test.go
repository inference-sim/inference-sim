package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// #2130: `blis convert otel` writes the sessions in the order the client started them, gives the
// sessions that opened with one system message a shared prefix group on round 0, and records each
// follow-up's think time from the spans' end times when asked.
func TestConvertOtel_SessionOrderSharedPrefixAndThink(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "in")
	if err := os.MkdirAll(in, 0o755); err != nil {
		t.Fatal(err)
	}
	// The file names sort a, b, c; the sessions started c, a, b.
	starts := map[string]int{"a": 20, "b": 40, "c": 10}
	for name, at := range starts {
		sys := "You are a careful agent."
		if name == "b" {
			sys = "A different system message."
		}
		body := fmt.Sprintf(`{"spans":[
		 {"span_id":"1","name":"chat m","start_time":"2026-01-01T00:00:%02d.000000","end_time":"2026-01-01T00:00:%02d.500000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":400,"gen_ai.usage.output_tokens":10,"gen_ai.input.messages":"[{\"role\":\"system\",\"content\":\"%s\"},{\"role\":\"user\",\"content\":\"%s\"}]"},"trace_id":"sess-%s"},
		 {"span_id":"2","name":"chat m","start_time":"2026-01-01T00:00:%02d.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":450,"gen_ai.usage.output_tokens":10,"gen_ai.input.messages":"[]"},"trace_id":"sess-%s"}]}`,
			at, at, sys, "a question that is about as long as the system message", name, at+3, name)
		if err := os.WriteFile(filepath.Join(in, name+".json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	out := filepath.Join(dir, "out")
	if err := runConvertOtel(in, out, workload.OTelConvertOptions{ContextGrowth: "accumulate", MinRounds: 1}); err != nil {
		t.Fatal(err)
	}
	tr, err := workload.LoadTraceV2(out+".yaml", out+".csv")
	if err != nil {
		t.Fatal(err)
	}
	var order []string
	group := map[string]string{}
	for _, r := range tr.Records {
		if r.RoundIndex == 0 {
			order = append(order, r.SessionID)
			group[r.SessionID] = r.PrefixGroup
			continue
		}
		if r.ThinkTimeUs == nil || *r.ThinkTimeUs != 2_500_000 {
			t.Errorf("%s round %d think = %v, want 2500000 (end .5 s into the call, next call 3 s in)", r.SessionID, r.RoundIndex, r.ThinkTimeUs)
		}
	}
	want := []string{"sess-c", "sess-a", "sess-b"}
	for i := range want {
		if i >= len(order) || order[i] != want[i] {
			t.Fatalf("sessions in order %v, want the start order %v", order, want)
		}
	}
	if group["sess-c"] == "" || group["sess-c"] != group["sess-a"] {
		t.Errorf("sessions c and a share a system message: groups %q and %q", group["sess-c"], group["sess-a"])
	}
	if group["sess-b"] != "" {
		t.Errorf("session b's system message is its own, yet it has prefix group %q", group["sess-b"])
	}
	for _, r := range tr.Records {
		if r.RoundIndex == 0 && r.SessionID == "sess-c" && (r.PrefixLength < 1 || r.InputTokens+r.PrefixLength != 400) {
			t.Errorf("round 0 of sess-c: prefix %d plus suffix %d tokens, want the call's 400", r.PrefixLength, r.InputTokens)
		}
	}
}
