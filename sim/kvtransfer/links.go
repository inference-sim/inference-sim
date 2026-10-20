package kvtransfer

import (
	"fmt"
	"math"
	"sort"
)

// LinkArbiter shares fabric links among concurrent transfers (P/D KV handoffs, #1922). It is
// the simulator's half of blis-schemas kernel.TransferDemand: the kernel states each
// transfer's payload, its floor and the links it crosses with their full rates, and says how
// one transfer alone is priced (max(floor, bytes / slowest link)); how links are divided among
// transfers in flight together is left to the simulator, and this is that division.
//
// The rule is max-min fairness (progressive filling): every link's bandwidth is divided among
// the transfers crossing it so that no transfer's rate can rise without lowering the rate of a
// transfer that is no faster. A transfer alone on its links gets the slowest one's full rate,
// so it costs exactly the kernel's price. Rates are recomputed whenever a transfer starts,
// drains its payload, or finishes. The floor is per transfer and never shared: a transfer
// completes at the later of start + floor and the tick its last byte drains, so a drained
// transfer that is still inside its floor holds no bandwidth.
//
// Time is in integer ticks (microseconds), as in the rest of BLIS; a completion is rounded up
// to the next whole tick, since a transfer is not done until its last byte lands. Like
// TransferStation it is pure and event-free: callers Start transfers, ask NextWake, and
// Advance to it. Deterministic: transfers are processed in Start order and links in ID order
// (INV-6).
type LinkArbiter struct {
	now   int64
	nextI FlowID
	flows []*flowState // Start order, which is FlowID order
	caps  map[string]float64

	inFlight map[string]int // transfers not yet complete on each link
	peak     map[string]int // the most ever in flight together on each link
}

// Link is one link a transfer crosses and the full rate it offers alone, in bytes per second.
type Link struct {
	ID             string
	BytesPerSecond float64
}

// Flow is one transfer's demand: payload bytes, a floor in nanoseconds that no bandwidth
// removes, and the links the payload crosses in series.
type Flow struct {
	Bytes   int64
	FloorNs int64
	Links   []Link
}

// FlowID names a transfer within one arbiter.
type FlowID int64

type flowState struct {
	id        FlowID
	links     []string
	start     int64
	floorTick int64

	rate      float64 // bytes per second since `since`
	remaining float64 // bytes left at `since`
	since     int64
	drained   bool
	drainTick int64 // the tick the payload drains at the current rate (or did drain)
}

// NewLinkArbiter returns an empty arbiter at tick 0.
func NewLinkArbiter() *LinkArbiter {
	return &LinkArbiter{caps: map[string]float64{}, inFlight: map[string]int{}, peak: map[string]int{}}
}

// AloneTicks is what f costs with every link to itself, in whole ticks rounded up and at least
// one: the kernel's AloneTime, max(floor, bytes / slowest link), converted as BLIS converts a
// handoff price (kernelmodel.PDTransferTicks).
func AloneTicks(f Flow) int64 {
	ns := f.FloorNs
	if len(f.Links) > 0 && f.Bytes > 0 {
		rate := math.Inf(1)
		for _, l := range f.Links {
			rate = math.Min(rate, l.BytesPerSecond)
		}
		ns = max(ns, drainNs(float64(f.Bytes), rate))
	}
	return max(1, ceilTicks(ns))
}

// drainNs is how long bytes take at rate, in nanoseconds truncated as time.Duration truncates,
// computed in the same order as blis-schemas TransferDemand.AloneTime so that a transfer alone
// reproduces the kernel's price exactly.
func drainNs(bytes, rate float64) int64 {
	if bytes <= 0 {
		return 0
	}
	if !(rate > 0) {
		return math.MaxInt64 / 4
	}
	ns := bytes / rate * 1e9
	if ns >= math.MaxInt64/4 {
		return math.MaxInt64 / 4
	}
	return int64(ns)
}

func ceilTicks(ns int64) int64 { return (ns + 999) / 1000 }

// Start begins transfer f at tick now and returns its ID. now must not precede an earlier
// Start or Advance. A link already in use must be stated at the rate it was stated at before:
// two rates for one link would be two links. Panics on a malformed flow (negative bytes or
// floor, bytes on no link, an empty or repeated link ID, a rate that is not positive and
// finite) -- the checks of blis-schemas TransferDemand.Check.
func (a *LinkArbiter) Start(now int64, f Flow) FlowID {
	a.moveTo(now)
	if err := checkFlow(f); err != nil {
		panic(fmt.Sprintf("LinkArbiter.Start: %v", err))
	}
	for _, l := range f.Links {
		if c, ok := a.caps[l.ID]; ok && c != l.BytesPerSecond {
			panic(fmt.Sprintf("LinkArbiter.Start: link %q stated at %v B/s, already in use at %v B/s",
				l.ID, l.BytesPerSecond, c))
		}
	}
	id := a.nextI
	a.nextI++
	fs := &flowState{id: id, start: now, since: now, remaining: float64(f.Bytes),
		floorTick: now + max(1, ceilTicks(f.FloorNs))}
	for _, l := range f.Links {
		a.caps[l.ID] = l.BytesPerSecond
		fs.links = append(fs.links, l.ID)
		a.inFlight[l.ID]++
		a.peak[l.ID] = max(a.peak[l.ID], a.inFlight[l.ID])
	}
	if f.Bytes <= 0 || len(f.Links) == 0 {
		fs.drained, fs.drainTick = true, now
	}
	a.flows = append(a.flows, fs)
	a.rerate()
	return id
}

func checkFlow(f Flow) error {
	if f.Bytes < 0 || f.FloorNs < 0 {
		return fmt.Errorf("negative bytes %d or floor %d", f.Bytes, f.FloorNs)
	}
	if f.Bytes > 0 && len(f.Links) == 0 {
		return fmt.Errorf("%d bytes cross no link", f.Bytes)
	}
	seen := map[string]bool{}
	for i, l := range f.Links {
		if l.ID == "" || seen[l.ID] {
			return fmt.Errorf("links[%d] %q is empty or repeated", i, l.ID)
		}
		seen[l.ID] = true
		if !(l.BytesPerSecond > 0) || math.IsInf(l.BytesPerSecond, 1) {
			return fmt.Errorf("links[%d] %q: rate %v must be positive and finite", i, l.ID, l.BytesPerSecond)
		}
	}
	return nil
}

func (f *flowState) completeTick() int64 { return max(f.floorTick, f.drainTick) }

// NextWake is the next tick at which a transfer drains or completes, when any is in flight.
// The caller Advances to it so that rates are recomputed when a drained transfer frees its
// share.
func (a *LinkArbiter) NextWake() (int64, bool) {
	next, ok := int64(math.MaxInt64), false
	for _, f := range a.flows {
		t := f.completeTick()
		if !f.drained {
			t = f.drainTick
		}
		if t < next {
			next, ok = t, true
		}
	}
	return next, ok
}

// Advance moves to tick now and returns, in Start order, the transfers that completed at or
// before it, which leave the arbiter.
func (a *LinkArbiter) Advance(now int64) []FlowID {
	a.moveTo(now)
	var done []FlowID
	kept := a.flows[:0]
	for _, f := range a.flows {
		if f.drained && f.completeTick() <= now {
			done = append(done, f.id)
			for _, l := range f.links {
				a.inFlight[l]--
			}
			continue
		}
		kept = append(kept, f)
	}
	a.flows = kept
	a.rerate()
	return done
}

// moveTo marks every transfer whose payload has drained by now.
func (a *LinkArbiter) moveTo(now int64) {
	if now < a.now {
		panic(fmt.Sprintf("LinkArbiter: time moved back from %d to %d", a.now, now))
	}
	a.now = now
	for _, f := range a.flows {
		if !f.drained && f.drainTick <= now {
			f.drained, f.remaining, f.since = true, 0, f.drainTick
		}
	}
}

// rerate divides the links among the undrained transfers, max-min fairly, at a.now. A
// transfer whose rate does not change keeps its drain tick untouched, so transfers on links
// no other transfer touches finish exactly when they would alone.
func (a *LinkArbiter) rerate() {
	var open []*flowState
	for _, f := range a.flows {
		if !f.drained {
			open = append(open, f)
		}
	}
	rates := maxMinRates(open, a.caps)
	for i, f := range open {
		r := rates[i]
		if r == f.rate {
			continue
		}
		f.remaining = max(0, f.remaining-f.rate*float64(a.now-f.since)/1e6)
		f.since = a.now
		f.rate = r
		f.drainTick = a.now + ceilTicks(drainNs(f.remaining, r))
	}
}

// maxMinRates is progressive filling: raise every unfrozen transfer's rate together until a
// link saturates, freeze the transfers crossing it at the rate reached, and repeat. The share
// a link offers each unfrozen transfer is its unallocated rate over their count.
func maxMinRates(flows []*flowState, caps map[string]float64) []float64 {
	rates := make([]float64, len(flows))
	left := map[string]float64{}
	count := map[string]int{}
	for _, f := range flows {
		for _, l := range f.links {
			left[l] = caps[l]
			count[l]++
		}
	}
	links := make([]string, 0, len(left))
	for l := range left {
		links = append(links, l)
	}
	sort.Strings(links)
	frozen := make([]bool, len(flows))
	for remaining := len(flows); remaining > 0; {
		best := math.Inf(1)
		for _, l := range links {
			if count[l] > 0 {
				best = math.Min(best, left[l]/float64(count[l]))
			}
		}
		saturated := map[string]bool{}
		for _, l := range links {
			if count[l] > 0 && left[l]/float64(count[l]) == best {
				saturated[l] = true
			}
		}
		for i, f := range flows {
			if frozen[i] {
				continue
			}
			hit := false
			for _, l := range f.links {
				hit = hit || saturated[l]
			}
			if !hit {
				continue
			}
			frozen[i], rates[i] = true, best
			remaining--
			for _, l := range f.links {
				left[l] = max(0, left[l]-best)
				count[l]--
			}
		}
	}
	return rates
}

// InFlight is the number of transfers not yet complete.
func (a *LinkArbiter) InFlight() int { return len(a.flows) }

// PeakConcurrent is, per link, the most transfers ever in flight on it together.
func (a *LinkArbiter) PeakConcurrent() map[string]int {
	out := make(map[string]int, len(a.peak))
	for l, n := range a.peak {
		out[l] = n
	}
	return out
}
