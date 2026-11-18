package coefficient

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

func entryNamed(name string, u vocab.Unit) *Set {
	return &Set{Kind: "CoefficientSet", Name: "s", Coefficients: []Entry{{
		Name: name, Value: 1, Units: u, Method: vocab.MethodAssumed, Fitted: false,
		Scope:   Scope{Hardware: []string{"h100"}},
		Sources: []Source{{Kind: "vendor_doc", Cite: "x", Role: "primary"}}, Rationale: "r",
	}}}
}

// SUB-16: an entry whose unit is not the one its formula needs is refused when the set loads.
func TestAUnitThatTheFormulaDoesNotNeedIsRefused(t *testing.T) {
	for _, c := range []struct {
		name string
		unit vocab.Unit
	}{
		{"collective_peak_rate_all_reduce_fp16_8rank_h100", vocab.UnitMicrosecondsPerXfer},
		{"collective_floor_all_reduce_fp16_8rank_h100", vocab.UnitBytesPerMicrosecond},
		{"attention_decode_rate_mla", vocab.UnitDimensionless},
		{"gemm_eps_max_bf16", vocab.UnitTokens},
		{"gemm_m_half_bf16", vocab.UnitDimensionless},
		{"host_output_token", vocab.UnitMicrosecondsPerRequest},
		{"cudagraph_capture_bytes_full", vocab.UnitBytesPerMicrosecond},
		// https://github.com/inference-sim/blis-schemas/issues/86: the per-token capture slopes multiply a token count, so a size per rank
		// is the wrong unit for them, and a per-token unit is the wrong unit for a size.
		{"cudagraph_capture_bytes_per_mixed_token_full_and_piecewise", vocab.UnitBytesPerRank},
		{"cudagraph_capture_bytes_per_decode_token_full_and_piecewise", vocab.UnitBytesPerRank},
		{"cudagraph_capture_floor_bytes_full_and_piecewise", vocab.UnitBytesPerToken},
		{"cudagraph_capture_bytes_full", vocab.UnitBytesPerToken},
		{"recurrent_decode_rate_kda", vocab.UnitMicrosecondsPerXfer},
	} {
		p := entryNamed(c.name, c.unit).Validate()
		if p.OK() || !strings.Contains(p.Error(), "the formula that reads it requires") {
			t.Errorf("%s as %s: %v", c.name, c.unit, p)
		}
	}
}

func TestTheRightUnitLoads(t *testing.T) {
	for name, u := range map[string]vocab.Unit{
		"collective_peak_rate_all_reduce_fp16_8rank_h100": vocab.UnitBytesPerMicrosecond,
		"gemm_m_half_bf16":                                            vocab.UnitTokens,
		"host_output_token":                                           vocab.UnitMicrosecondsPerToken,
		"a_name_no_family_covers":                                     vocab.UnitDimensionless,
		"cudagraph_capture_floor_bytes_full_and_piecewise":            vocab.UnitBytesPerRank,
		"cudagraph_capture_bytes_full_decode_only":                    vocab.UnitBytesPerRank,
		"cudagraph_capture_bytes_per_mixed_token_full_and_piecewise":  vocab.UnitBytesPerToken,
		"cudagraph_capture_bytes_per_decode_token_full_and_piecewise": vocab.UnitBytesPerToken,
	} {
		if p := entryNamed(name, u).Validate(); !p.OK() {
			t.Errorf("%s as %s was refused: %v", name, u, p)
		}
	}
}

// Every family's pattern matches a name of its own, and no name matches two families with
// different units, so the table has no dead row and no ambiguity.
func TestFormUnitsHaveNoDeadRowAndNoConflict(t *testing.T) {
	files, err := filepath.Glob("../../../registry/coefficients/*.yaml")
	if err != nil || len(files) == 0 {
		t.Fatalf("no registry files: %v", err)
	}
	hit := make([]int, len(FormUnits))
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var s Set
		if err := yaml.Unmarshal(raw, &s); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		for _, e := range s.Coefficients {
			var units []vocab.Unit
			for i, re := range formUnitRes {
				if re.MatchString(e.Name) {
					hit[i]++
					units = append(units, FormUnits[i].Unit)
				}
			}
			if len(units) == 0 {
				t.Errorf("%s: %s is covered by no family in FormUnits", filepath.Base(f), e.Name)
			}
			for _, u := range units {
				if u != units[0] {
					t.Errorf("%s matches families with units %v", e.Name, units)
				}
			}
			if len(units) > 0 && e.Units != units[0] {
				t.Errorf("%s: %s has units %q, its family requires %q", filepath.Base(f), e.Name, e.Units, units[0])
			}
		}
	}
	// A row for a coefficient the cost model reads and no committed set carries yet is allowed
	// by name; every other row must match an entry.
	notYetCarried := map[string]bool{`^host_step_per_request$`: true}
	for i, n := range hit {
		if n == 0 && !notYetCarried[FormUnits[i].Pattern] {
			t.Errorf("the family %q matches no committed entry; remove the row or fix its pattern", FormUnits[i].Pattern)
		}
	}
}

// RequiredUnit keeps its answer per name; the kept answer is the one the table gives, for a name a
// family covers, for a name none covers, and when asked again.
func TestRequiredUnitKeptAnswerIsTheTablesAnswer(t *testing.T) {
	uncached := func(name string) (FormUnit, bool) {
		for i, re := range formUnitRes {
			if re.MatchString(name) {
				return FormUnits[i], true
			}
		}
		return FormUnit{}, false
	}
	names := []string{
		"collective_peak_rate_all_reduce_fp16_8rank_h100", "attention_decode_rate_mla", "gemm_eps_max_bf16",
		"host_output_token", "cudagraph_capture_bytes_full", "no_such_family_at_all", "",
	}
	for round := 0; round < 2; round++ { // the second round reads the kept answers
		for _, n := range names {
			wf, wok := uncached(n)
			if gf, gok := RequiredUnit(n); gok != wok || gf != wf {
				t.Errorf("round %d, %q: got (%+v, %t), want (%+v, %t)", round, n, gf, gok, wf, wok)
			}
		}
	}
	if _, ok := RequiredUnit("no_such_family_at_all"); ok {
		t.Error("a name no family covers is covered")
	}
}

// BenchmarkValidateSet times the validation every kernel build runs over each coefficient set.
func BenchmarkValidateSet(b *testing.B) {
	s := &Set{Kind: "CoefficientSet", Name: "s"}
	for i := 0; i < 200; i++ {
		s.Coefficients = append(s.Coefficients, Entry{
			Name: "collective_peak_rate_all_reduce_fp16_" + strconv.Itoa(i+2) + "rank_h100", Value: 1,
			Units: vocab.UnitBytesPerMicrosecond, Method: vocab.MethodAssumed,
			Scope:     Scope{Hardware: []string{"h100"}},
			Sources:   []Source{{Kind: "vendor_doc", Cite: "x", Role: "primary"}, {Kind: "vendor_doc", Cite: "y", Role: "primary"}},
			Rationale: "r"})
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		s.Validate()
	}
}
