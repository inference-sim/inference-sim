package cluster

import (
	"fmt"
	"testing"
	"time"

	"pgregory.net/rapid"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// A pull runs alone for the demand's AloneTime, rounded up to a whole tick and at least one: the
// host's rounding of the cost model's answer (AloneTicks of its Flow) is the substrate's
// definition, for every payload, floor and set of links, so the Environment's demand and the
// arbiter cannot disagree about how long a pull takes alone.
func TestAloneTicksOfAFlowIsTheDemandsAloneTimeRoundedUp(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		d := kernel.TransferDemand{
			Bytes: rapid.Int64Range(0, 1<<40).Draw(rt, "bytes"),
			Floor: time.Duration(rapid.Int64Range(0, 5_000_000).Draw(rt, "floorNs")),
		}
		for i := range rapid.IntRange(0, 3).Draw(rt, "links") {
			d.Links = append(d.Links, kernel.LinkUse{Link: kernel.LinkID(fmt.Sprintf("l%d", i)),
				BytesPerSecond: float64(rapid.Int64Range(1_000_000, 900_000_000_000).Draw(rt, "rate"))})
		}
		if d.Bytes > 0 && len(d.Links) == 0 {
			d.Bytes = 0 // a payload crosses a link (TransferDemand.Check)
		}
		want := max(1, (int64(d.AloneTime())+999)/1000)
		if got := AloneTicks(FlowOf(d)); got != want {
			rt.Fatalf("demand %+v: %d ticks, AloneTime %v rounds up to %d", d, got, d.AloneTime(), want)
		}
	})
}

// FlowOf keeps the demand's links in order, with their rates, and its floor in nanoseconds.
func TestFlowOfKeepsTheDemand(t *testing.T) {
	d := kernel.TransferDemand{Bytes: 7, Floor: 1500 * time.Nanosecond, Links: []kernel.LinkUse{
		{Link: "b", BytesPerSecond: 2e9}, {Link: "a", BytesPerSecond: 1e9}}}
	f := FlowOf(d)
	if f.Bytes != 7 || f.FloorNs != 1500 || len(f.Links) != 2 || f.Links[0] != (Link{"b", 2e9}) || f.Links[1] != (Link{"a", 1e9}) {
		t.Errorf("FlowOf(%+v) = %+v", d, f)
	}
	if e := FlowOf(kernel.TransferDemand{}); AloneTicks(e) != 1 {
		t.Errorf("an empty demand (a pull of no tokens) takes %d ticks, want 1", AloneTicks(e))
	}
}
