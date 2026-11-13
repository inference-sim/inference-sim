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
				BytesPerSecond: float64(rapid.Int64Range(1_000_000, 900_000_000_000).Draw(rt, "rate")),
				Streams:        rapid.IntRange(0, 8).Draw(rt, "streams")})
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
		{Link: "b", BytesPerSecond: 2e9, Streams: 3}, {Link: "a", BytesPerSecond: 1e9}}}
	f := FlowOf(d)
	if f.Bytes != 7 || f.FloorNs != 1500 || len(f.Links) != 2 || f.Links[0] != (Link{ID: "b", BytesPerSecond: 2e9, Streams: 3}) || f.Links[1] != (Link{ID: "a", BytesPerSecond: 1e9}) {
		t.Errorf("FlowOf(%+v) = %+v", d, f)
	}
	if e := FlowOf(kernel.TransferDemand{}); AloneTicks(e) != 1 {
		t.Errorf("an empty demand (a pull of no tokens) takes %d ticks, want 1", AloneTicks(e))
	}
}

// A transfer that crosses a link with n streams (n ranks' rails feeding one uplink) divides the
// link's rate among them: alone it drains its payload at rate/n, as the demand's AloneTime says,
// and beside a one-stream transfer on the same link it takes n shares of the rate to the other's
// one. An arbiter that ignores Streams lets the uplink look n times too fast.
func TestAFlowsStreamsDivideTheLinkItCrosses(t *testing.T) {
	const rate, bytes = 100e9, int64(8e9) // 8 GB over a 100 GB/s link
	uplink := func(streams int) kernel.TransferDemand {
		return kernel.TransferDemand{Bytes: bytes, Links: []kernel.LinkUse{{Link: "uplink", BytesPerSecond: rate, Streams: streams}}}
	}
	alone := func(streams int) int64 {
		a := NewLinkArbiter()
		a.Start(0, FlowOf(uplink(streams)))
		wake, _ := a.NextWake()
		return wake
	}
	// Alone: 8 GB at 100 GB/s is 80 ms; four streams take a quarter of the rate each: 320 ms.
	if got, want := alone(1), int64(80_000); got != want {
		t.Errorf("one stream alone drains at tick %d, want %d", got, want)
	}
	if got, want := alone(4), int64(320_000); got != want {
		t.Errorf("four streams alone drain at tick %d, want %d", got, want)
	}
	// Together: 4 + 1 streams share 100 GB/s, 20 GB/s a stream. The one-stream transfer drains
	// 8 GB at 20 GB/s in 400 ms; the four-stream one has moved 8 GB by then too (the same rate
	// per stream), so both drain together.
	a := NewLinkArbiter()
	a.Start(0, FlowOf(uplink(4)))
	a.Start(0, FlowOf(uplink(1)))
	if wake, _ := a.NextWake(); wake != 400_000 {
		t.Errorf("a 4-stream and a 1-stream transfer on one link first drain at tick %d, want 400000", wake)
	}
}
