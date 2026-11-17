package latencykernel

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/load"
)

// The per-launch term a step pays is the chip's own. The part-wide 3.2 us was backed out of two
// Hopper runs; on Blackwell the same value put the host term at about half of a single-request
// decode step and priced every concurrency-1 step 1-2 ms above its measurement (accuracy
// pre-registration K1). The price of one launch is read here from the step itself: removing one
// launch a layer (plan.PoolPlan.Launches) shortens a FULL-graph decode step by layers times the
// per-launch term, whatever the model's device time is.
func TestBlackwellIsChargedItsOwnPerLaunchTermAndHopperKeepsTheCommittedOne(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	perLaunch := func(chip string) time.Duration {
		open := func(launches []plan.LaunchChange) *Kernel {
			return mustKernel(t, planCase(t, d, "gpt-oss-120b", chip, 1, "tp: 4, pp: 1, dp: 1", func(l *plan.Lowered) {
				l.Plan.Launches = launches
				l.Plan.Batch.Requests = 16
				l.Plan.Graphs.Ceiling = 16
				l.Plan.Graphs.Buckets = []int{1, 2, 4, 8, 16}
				l.Plan.Graphs.Decode = plan.LaunchGraph
			}))
		}
		before := open(nil)
		after := open([]plan.LaunchChange{{FromTokens: 1, ToTokens: 16, PerLayer: -1}})
		b := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 1024, PromptLen: 1024}}}
		saved := before.StepTime(b).NoOverlap - after.StepTime(b).NoOverlap
		return saved / time.Duration(before.plan.TotalLayers)
	}
	hopper := perLaunch("h200")
	if hopper != 3200*time.Nanosecond {
		t.Errorf("an h200 launch costs %v, want the committed part-wide 3.2us", hopper)
	}
	for _, chip := range []string{"b200", "b300"} {
		got := perLaunch(chip)
		t.Logf("%s: %v per launch (h200 %v)", chip, got, hopper)
		if got <= 0 || got >= hopper {
			t.Errorf("a %s launch costs %v; it must be priced below the Hopper value %v it was backed out of", chip, got, hopper)
		}
	}
}
