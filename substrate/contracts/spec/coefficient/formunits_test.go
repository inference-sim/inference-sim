package coefficient

import (
	"os"
	"path/filepath"
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
		// blis-schemas#86: the per-token capture slopes multiply a token count, so a size per rank
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
