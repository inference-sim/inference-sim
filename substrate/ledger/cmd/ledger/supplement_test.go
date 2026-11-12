package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/ledger"
	"github.com/inference-sim/blis-substrate/ledger/inferencex"
)

func syntheticRecord(id int, ttft string) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"id":"%d","model":"m","hardware":"h200","precision":"fp4","framework":"vllm","benchmark_type":"agentic_traces",`+
		`"conc":1,"date":"2026-11-09","run_url":"https://github.com/o/r/actions/runs/42/attempts/1","isl":null,"osl":null,`+
		`"metrics":{"mean_ttft":%s}}`, id, ttft))
}

// ledgerWithSupplement writes a ledger directory whose InferenceX source pins the records the
// fake API serves, and points the command's fetcher at the API.
func ledgerWithSupplement(t *testing.T, served []json.RawMessage) string {
	t.Helper()
	body, _ := json.Marshal(served)
	old := fetcher
	fetcher = func(string) ([]byte, error) { return body, nil }
	t.Cleanup(func() { fetcher = old })

	src, err := ledger.LoadSources("../../../catalog/ledger")
	if err != nil {
		t.Fatal(err)
	}
	s := src.Lookup(inferencex.SourceID)
	s.Supplements = nil
	sup := ledger.Supplement{ID: "t", Kind: ledger.SupplementKindAPI, URL: "https://api.test/benchmarks",
		Run: "https://github.com/o/r/actions/runs/42/attempts/1"}
	for _, r := range served {
		id, _ := inferencex.RecordSHA256(r)
		var h struct{ ID string }
		_ = json.Unmarshal(r, &h)
		var n int64
		_, _ = fmt.Sscan(h.ID, &n)
		sup.Records = append(sup.Records, ledger.PinnedRecord{ID: n, SHA256: id})
	}
	s.Supplements = []ledger.Supplement{sup}
	out, err := yaml.Marshal(src)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ledger.FileSources), out, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// The dashboard gets the pinned records through the ledger's own tool, checked against their
// pins, and commits none of them (X12).
func TestSupplementRecordsAreFetchedAndChecked(t *testing.T) {
	served := []json.RawMessage{syntheticRecord(901, "1.5"), syntheticRecord(902, "2.5")}
	dir := ledgerWithSupplement(t, served)
	out := filepath.Join(t.TempDir(), "records.json")
	var stdout, stderr bytes.Buffer
	if code := run([]string{"supplement-records", "-out", out, dir}, &stdout, &stderr); code != 0 {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var got []map[string]any
	if err := json.Unmarshal(raw, &got); err != nil || len(got) != 2 || got[0]["id"] != "901" || got[1]["id"] != "902" {
		t.Fatalf("records %s (%v)", raw, err)
	}

	// The API now serves record 902 with another value: the tool stops and writes nothing.
	fetcher = func(string) ([]byte, error) {
		b, _ := json.Marshal([]json.RawMessage{served[0], syntheticRecord(902, "2.6")})
		return b, nil
	}
	out2 := filepath.Join(t.TempDir(), "records.json")
	stderr.Reset()
	if code := run([]string{"supplement-records", "-out", out2, dir}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "902") {
		t.Errorf("a changed record: exit %d: %s", code, stderr.String())
	}
	if _, err := os.Stat(out2); err == nil {
		t.Error("records were written although one differs from its pin")
	}
}

// The pins a ledger owner pastes are the ones the check later compares against.
func TestPinSupplementPrintsThePinsTheCheckUses(t *testing.T) {
	served := []json.RawMessage{syntheticRecord(901, "1.5"), syntheticRecord(902, "2.5"), syntheticRecord(903, "3.5")}
	ledgerWithSupplement(t, served)
	var stdout, stderr bytes.Buffer
	if code := run([]string{"pin-supplement", "-url", "https://api.test/benchmarks", "-ids", "901-902"}, &stdout, &stderr); code != 0 {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	h, _ := inferencex.RecordSHA256(served[1])
	if lines := strings.Split(strings.TrimSpace(stdout.String()), "\n"); len(lines) != 2 || !strings.Contains(lines[1], "id: 902") || !strings.Contains(lines[1], h) {
		t.Errorf("pins %q", stdout.String())
	}
	if code := run([]string{"pin-supplement", "-url", "https://api.test/benchmarks", "-ids", "999"}, &stdout, &stderr); code != 1 {
		t.Errorf("a record the API does not serve: exit %d", code)
	}
}
