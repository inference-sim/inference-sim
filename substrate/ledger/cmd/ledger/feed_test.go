package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/substrate/ledger"
)

// dumpLine is one exported row of the pinned dump; its numbers are invented (X12).
func dumpLine(id int, conc int, date string) string {
	return `{"id":` + itoa(id) + `,"run_url":"https://github.com/o/r/actions/runs/42/attempts/1","run_attempt":1,` +
		`"config_id":7,"model":"m","hardware":"h200","precision":"fp4","framework":"vllm","benchmark_type":"agentic_traces",` +
		`"conc":` + itoa(conc) + `,"date":"` + date + `","metrics":{"mean_ttft":0.5}}`
}

func itoa(n int) string { b, _ := json.Marshal(n); return string(b) }

// The scheduled job's tool reads the feed with the dump, writes a report and the pins to
// propose, and changes nothing in the ledger. A run whose records are on both sides of the
// cutoff is reported, and the records a supplement pins are not proposed again.
func TestFeedWritesReportAndProposalsOnly(t *testing.T) {
	pinned := []json.RawMessage{syntheticRecord(901, "1.5"), syntheticRecord(902, "2.5")}
	dir := ledgerWithSupplement(t, pinned)
	before, err := os.ReadFile(filepath.Join(dir, ledger.FileSources))
	if err != nil {
		t.Fatal(err)
	}
	rows := filepath.Join(t.TempDir(), "rows.jsonl")
	if err := os.WriteFile(rows, []byte(dumpLine(5, 1, "2026-09-20")+"\n"+dumpLine(6, 8, "2026-09-20")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	served := append([]json.RawMessage{syntheticRecord(6, "7.7"), syntheticRecord(903, "3.5")}, pinned...)
	body, _ := json.Marshal(served)
	fetcher = func(url string) ([]byte, error) {
		if url == "https://api.test/benchmarks" { // the pinned supplement's URL and the feed's
			return body, nil
		}
		t.Errorf("fetched %s", url)
		return nil, os.ErrNotExist
	}

	out := filepath.Join(t.TempDir(), "feed")
	var stdout, stderr bytes.Buffer
	if code := run([]string{"feed", "-rows", rows, "-url", "https://api.test/benchmarks", "-out", out, dir}, &stdout, &stderr); code != 0 {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	after, _ := os.ReadFile(filepath.Join(dir, ledger.FileSources))
	if !bytes.Equal(before, after) {
		t.Error("the feed changed the ledger's sources.yaml")
	}

	rep, err := os.ReadFile(filepath.Join(out, "feed-report.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var r struct {
		Feed struct {
			Cutoff         int64              `yaml:"cutoff"`
			APIRecords     int                `yaml:"api_records"`
			OwnedByDump    int                `yaml:"owned_by_dump"`
			RowDiffers     []int64            `yaml:"row_differs_from_dump"`
			StraddlingRuns []int64            `yaml:"straddling_runs"`
			ByRevision     map[string]int     `yaml:"units_by_source_revision"`
			Unreadable     map[string][]int64 `yaml:"unreadable"`
		} `yaml:"feed"`
	}
	if err := yaml.Unmarshal(rep, &r); err != nil {
		t.Fatal(err)
	}
	f := r.Feed
	if f.Cutoff != 6 || f.APIRecords != 3 || f.OwnedByDump != 1 || len(f.RowDiffers) != 1 || f.RowDiffers[0] != 6 ||
		len(f.StraddlingRuns) != 1 || f.StraddlingRuns[0] != 42 {
		t.Errorf("report %s", rep)
	}

	prop, err := os.ReadFile(filepath.Join(out, "proposed-supplements.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var p struct {
		Supplements []ledger.Supplement `yaml:"supplements"`
	}
	if err := yaml.Unmarshal(prop, &p); err != nil {
		t.Fatal(err)
	}
	if len(p.Supplements) != 1 || p.Supplements[0].ID != "run-42-attempt-1" || len(p.Supplements[0].Records) != 1 || p.Supplements[0].Records[0].ID != 903 {
		t.Errorf("proposed %s", prop)
	}
	if strings.Contains(string(prop)+string(rep), "mean_ttft") {
		t.Error("a measured value is in the output (X12)")
	}

	// The proposal is a valid supplement: appended to sources.yaml it passes the ledger's check.
	src, err := ledger.LoadSources(dir)
	if err != nil {
		t.Fatal(err)
	}
	s := src.Lookup("inferencex")
	s.Supplements = append(s.Supplements, p.Supplements...)
	b, _ := yaml.Marshal(src)
	next := t.TempDir()
	if err := os.WriteFile(filepath.Join(next, ledger.FileSources), b, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ledger.LoadSources(next); err != nil {
		t.Errorf("sources.yaml with the proposed supplement: %v", err)
	}
}

// A pinned record the feed serves with other content stops the run and writes nothing.
func TestFeedRefusesAChangedPinnedRecord(t *testing.T) {
	pinned := []json.RawMessage{syntheticRecord(901, "1.5")}
	dir := ledgerWithSupplement(t, pinned)
	rows := filepath.Join(t.TempDir(), "rows.jsonl")
	if err := os.WriteFile(rows, []byte(dumpLine(5, 1, "2026-09-20")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// The pin fetch sees the pinned content; the feed URL serves record 901 changed.
	changed, _ := json.Marshal([]json.RawMessage{syntheticRecord(901, "1.6")})
	good, _ := json.Marshal(pinned)
	fetcher = func(url string) ([]byte, error) {
		if strings.Contains(url, "feed") {
			return changed, nil
		}
		return good, nil
	}
	out := filepath.Join(t.TempDir(), "feed")
	var stdout, stderr bytes.Buffer
	if code := run([]string{"feed", "-rows", rows, "-url", "https://api.test/feed", "-out", out, dir}, &stdout, &stderr); code != 1 || !strings.Contains(stderr.String(), "901") {
		t.Errorf("exit %d: %s", code, stderr.String())
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("output was written although a pinned record changed")
	}
	if code := run([]string{"feed", "-rows", rows, "-out", out, dir}, &stdout, &stderr); code != 2 {
		t.Errorf("no -url: exit %d, want 2", code)
	}
}
