package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/ledger"
	"github.com/inference-sim/inference-sim/substrate/load"
)

const committedTransferValidation = "../../../registry/docs/transfer-rule-validation.json"

type transferDoc struct {
	Kind      string               `json:"kind"`
	MinPoints int                  `json:"min_points"`
	Rows      []ledger.TransferRow `json:"rows"`
}

func runTransferError(t *testing.T, args ...string) (int, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := run(append([]string{"transfer-error"}, args...), &out, &errb)
	return code, errb.String()
}

// The validation is computed from the registry's measured coefficients and the catalog's
// datasheet figures alone. Every published row covers at least MinAggregateUnits points, and its
// use_rule is exactly "the rule's median ratio is below roofline's".
func TestTransferErrorPublishesOnlyAggregatesAndAHonestGate(t *testing.T) {
	out := filepath.Join(t.TempDir(), "t.json")
	if code, msg := runTransferError(t, "-out", out); code != 0 {
		t.Fatalf("exit %d: %s", code, msg)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var doc transferDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.Kind != "TransferRuleValidation" || doc.MinPoints != ledger.MinAggregateUnits {
		t.Errorf("kind %q, min_points %d", doc.Kind, doc.MinPoints)
	}
	if len(doc.Rows) == 0 {
		t.Fatal("no row published: the registry's Hopper and Blackwell chips have points in common")
	}
	for _, r := range doc.Rows {
		if r.Points < ledger.MinAggregateUnits {
			t.Errorf("%+v covers fewer than %d points", r, ledger.MinAggregateUnits)
		}
		if r.UseRule != (r.RuleMedian < r.RooflineMedian) {
			t.Errorf("%+v: use_rule disagrees with the medians", r)
		}
		if r.From == r.To {
			t.Errorf("%+v is not a cross-generation transfer", r)
		}
	}
}

// The GB200 (Blackwell) and Hopper directions are each published (VAL-5).
func TestTransferErrorCoversBothHopperAndBlackwellDirections(t *testing.T) {
	out := filepath.Join(t.TempDir(), "t.json")
	if code, msg := runTransferError(t, "-out", out); code != 0 {
		t.Fatalf("exit %d: %s", code, msg)
	}
	raw, _ := os.ReadFile(out)
	var doc transferDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, r := range doc.Rows {
		seen[r.From+">"+r.To] = true
	}
	for _, dir := range []string{"Blackwell>Hopper", "Hopper>Blackwell"} {
		if !seen[dir] {
			t.Errorf("no published row for %s; got %v", dir, seen)
		}
	}
}

// Freshness: the committed validation equals what the registry and catalog in the tree compute.
// A coefficient or datasheet change that moves a figure fails here until the file is regenerated
// with `ledger transfer-error -out registry/method-error/transfer-rule.json`.
func TestCommittedTransferValidationIsCurrent(t *testing.T) {
	if code, msg := runTransferError(t, "-check", committedTransferValidation); code != 0 {
		t.Fatalf("exit %d: %s", code, msg)
	}
}

func TestTransferErrorCheckRefusesAStaleFile(t *testing.T) {
	raw, err := os.ReadFile(committedTransferValidation)
	if err != nil {
		t.Fatal(err)
	}
	stale := filepath.Join(t.TempDir(), "stale.json")
	if err := os.WriteFile(stale, []byte(strings.Replace(string(raw), `"points": `, `"points": 1`, 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if code, _ := runTransferError(t, "-check", stale); code != 1 {
		t.Errorf("a stale file passed -check (exit %d)", code)
	}
	if code, _ := runTransferError(t, "-check", filepath.Join(t.TempDir(), "absent.json")); code != 1 {
		t.Errorf("a missing file passed -check (exit %d)", code)
	}
	for _, args := range [][]string{{}, {"-out", "a", "-check", "b"}} {
		if code, _ := runTransferError(t, args...); code != 2 {
			t.Errorf("%v: exit %d, want 2", args, code)
		}
	}
}

// Each datasheet axis the families are mapped to is a figure every chip in the catalog states,
// and the mapped families are the rate families the cost model reads, not floors or counts.
func TestTransferAxesAreDatasheetFiguresAndNameRateFamilies(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	sheet, err := datasheet(d)
	if err != nil {
		t.Fatal(err)
	}
	for axis, byChip := range sheet {
		for _, c := range d.Chips() {
			if !(byChip[c] > 0) {
				t.Errorf("%s of %s: %v", axis, c, byChip[c])
			}
		}
	}
	for family, want := range map[string]string{
		"attention_decode_rate":               "memory_bandwidth",
		"attention_decode_rate_swa":           "memory_bandwidth",
		"collective_peak_rate_all_reduce":     "intra_node_bandwidth",
		"collective_transition_rate_alltoall": "intra_node_bandwidth",
	} {
		if got, ok := transferAxis(family); !ok || got != want {
			t.Errorf("%s: axis %q (%v), want %q", family, got, ok, want)
		}
	}
	for _, family := range []string{
		"attention_decode_floor", "collective_floor_all_reduce", "gemm_eps_max", "gemm_m_half",
		"attention_prefill_work_scale", "recurrent_decode_rate_gdn", "attention_decode_serial_rate_sparse_mla",
		"host_link_bandwidth", "cudagraph_capture_floor_bytes_full_and_piecewise",
	} {
		if axis, ok := transferAxis(family); ok {
			t.Errorf("%s is mapped to %s; it is not limited by a datasheet quantity", family, axis)
		}
	}
}
