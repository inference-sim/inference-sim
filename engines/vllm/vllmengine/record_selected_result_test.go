package vllmengine

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// A run that prices the trtllm-gen prefill lane on a b200 (the lane an unstated setting selects there;
// a stated use_trtllm_attention is refused until an engine reports the implementation of
// attention.prefill, see TestTheKernelDecidesUseTRTLLMAttention) must produce selected entries the Result validator accepts: the
// lane's registry entries carry a fit range, and the report holds the closeness data the engine's
// kernel recorded for them. This builds the engine through engine.New, prices a prefill step,
// registers the engine's kernel with the host's Data as the host does, and validates the report's
// selected_entries. Problems the Result has for other reasons (this test builds no run) are not
// looked at; only those at selected_entries are.
func TestARunPricingTheTRTLLMLaneOnB200ProducesValidSelectedEntries(t *testing.T) {
	Register()
	for name, c := range map[string]struct {
		extra     string
		needsLane bool
	}{
		"nothing stated on b200": {"", false},
	} {
		t.Run(name, func(t *testing.T) {
			e, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.32.0", Roots: withEngine(t, c.extra)})
			if err != nil {
				t.Fatal(err)
			}
			priceStep(t, e, "r", 512)
			host := embeddedData(t)
			register(t, e, host)
			sel := host.Report().Result()
			withFitRange := 0
			for _, c := range sel.Coefficients {
				if len(c.FitRange) > 0 && c.Method != "measured" && c.Method != "calibrated" {
					withFitRange++
				}
			}
			if c.needsLane && withFitRange == 0 {
				t.Fatalf("no transferred coefficient with a fit range is selected (%d coefficients): the lane is not priced", len(sel.Coefficients))
			}
			res := result.Result{SelectedEntries: sel}
			for _, p := range res.Validate().All() {
				if strings.Contains(p.String(), "selected_entries") {
					t.Errorf("the Result refuses the engine's selected entries: %s", p.String())
				}
			}
		})
	}
}
