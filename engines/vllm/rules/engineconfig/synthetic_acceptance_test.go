// Original to blis-vllm.

package engineconfig

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
)

// syntheticAcceptanceRecord is oracle/synthetic_acceptance's output: vLLM v0.32.0's own
// SpeculativeConfig._resolve_synthetic_acceptance_rates over a grid of (num_speculative_tokens,
// length), valid and not.
type syntheticAcceptanceRecord struct {
	Commit               string `json:"commit"`
	LengthNeedsSynthetic bool   `json:"length_needs_synthetic"`
	Cases                []struct {
		K      int       `json:"num_speculative_tokens"`
		Length *float64  `json:"length"`
		Rates  []float64 `json:"rates"`
		Error  *string   `json:"error"`
	} `json:"cases"`
}

func loadSyntheticAcceptance(t *testing.T) syntheticAcceptanceRecord {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "synthetic_acceptance.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r syntheticAcceptanceRecord
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	if r.Commit != "5df63a7203cdb9328d824a27b682e77c7b9928be" {
		t.Fatalf("the record is for commit %s, not the pinned release", r.Commit)
	}
	return r
}

// For every length vLLM resolves, the rates it returns are the minimum-variance schedule the
// acceptance resolver derives for a length alone (acceptance.MinimumVarianceRates), bit for bit, and
// the engine block accepts the length. For every length vLLM refuses, the block's validation refuses
// it and the resolver's schedule refuses it too. A length is valid only with rejection_sample_method
// synthetic, which vLLM's source says (the record's flag) and the block's validation enforces.
func TestSyntheticAcceptanceLengthIsVLLMsOwnResolution(t *testing.T) {
	r := loadSyntheticAcceptance(t)
	if !r.LengthNeedsSynthetic {
		t.Error("the record does not show vLLM requiring rejection_sample_method synthetic for a length")
	}
	checked, refused := 0, 0
	for _, c := range r.Cases {
		if c.Length == nil || math.IsNaN(*c.Length) || (c.Rates != nil && c.Error != nil) {
			continue // the exactly-one cases: no length to compare, or both stated
		}
		l := *c.Length
		e := Engine{Speculative: &Speculative{Method: "eagle3", NumSpecTokens: c.K,
			RejectionSampleMethod: "synthetic", SyntheticAcceptanceLength: &l}}
		p := engineOnly(e).Validate()
		rates, rerr := acceptance.MinimumVarianceRates(l, c.K)
		if c.Error != nil {
			refused++
			if p.OK() || !strings.Contains(p.Error(), ".engine.speculative.synthetic_acceptance_length") {
				t.Errorf("K %d length %v: vLLM refuses (%s) and the block does not:\n%s", c.K, l, *c.Error, p.Error())
			}
			if rerr == nil {
				t.Errorf("K %d length %v: vLLM refuses it and MinimumVarianceRates returns %v", c.K, l, rates)
			}
			continue
		}
		checked++
		if !p.OK() {
			t.Errorf("K %d length %v: vLLM resolves it and the block refuses:\n%s", c.K, l, p.Error())
		}
		if rerr != nil || !slices.Equal(rates, c.Rates) {
			t.Errorf("K %d length %v: vLLM's rates %v, MinimumVarianceRates %v (%v)", c.K, l, c.Rates, rates, rerr)
		}
	}
	if checked < 60 || refused < 30 {
		t.Errorf("the record held %d resolved and %d refused lengths, too few to hold the engine to", checked, refused)
	}

	// Stated without synthetic rejection sampling, a length is refused, whatever its value.
	l := 2.0
	for _, m := range []string{"", "standard", "block"} {
		e := Engine{Speculative: &Speculative{Method: "eagle3", NumSpecTokens: 3, RejectionSampleMethod: m, SyntheticAcceptanceLength: &l}}
		if p := engineOnly(e).Validate(); p.OK() || !strings.Contains(p.Error(), "only valid with rejection_sample_method synthetic") {
			t.Errorf("rejection_sample_method %q with a length: %v", m, p.Error())
		}
	}
	// NaN is outside every range.
	nan := math.NaN()
	e := Engine{Speculative: &Speculative{Method: "eagle3", NumSpecTokens: 3, RejectionSampleMethod: "synthetic", SyntheticAcceptanceLength: &nan}}
	if p := engineOnly(e).Validate(); p.OK() {
		t.Error("a NaN length is accepted")
	}
}
