package kvtransfer

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"pgregory.net/rapid"
)

// The arbiter's laws (#1922, INV-P2-2). A transfer alone costs exactly the kernel's price; a
// shared link is divided max-min fairly and never carries more than its rate; links nobody
// else uses are unaffected; more bandwidth never delays anyone.

type timedFlow struct {
	at int64
	f  Flow
}

// simulate runs flows (sorted by start) through a fresh arbiter, driven the way the cluster
// drives it, and returns each flow's completion tick. check, when non-nil, runs after every
// Start and Advance.
func simulate(t interface{ Fatalf(string, ...any) }, flows []timedFlow, check func(*LinkArbiter)) []int64 {
	a := NewLinkArbiter()
	done := make([]int64, len(flows))
	ids := map[FlowID]int{}
	next := 0
	for next < len(flows) || a.InFlight() > 0 {
		wake, ok := a.NextWake()
		if next < len(flows) && (!ok || flows[next].at <= wake) {
			ids[a.Start(flows[next].at, flows[next].f)] = next
			next++
		} else {
			for _, id := range a.Advance(wake) {
				done[ids[id]] = wake
			}
		}
		if check != nil {
			check(a)
		}
		if len(ids) > len(flows)+1 {
			t.Fatalf("more starts than flows")
		}
	}
	return done
}

var linkNames = []string{"nic/egress/node/0", "nic/egress/node/1", "nic/ingress/node/2", "nic/ingress/node/3", "nvlink/node/0"}

func drawLinkRates(rt *rapid.T) map[string]float64 {
	rates := map[string]float64{}
	for _, l := range linkNames {
		rates[l] = rapid.Float64Range(1e6, 1e11).Draw(rt, "rate "+l)
	}
	return rates
}

func drawFlow(rt *rapid.T, rates map[string]float64, label string) Flow {
	f := Flow{Bytes: rapid.Int64Range(0, 5e9).Draw(rt, label+" bytes"),
		FloorNs: rapid.Int64Range(0, 5e8).Draw(rt, label+" floor")}
	n := rapid.IntRange(1, 3).Draw(rt, label+" links")
	perm := rapid.Permutation(linkNames).Draw(rt, label+" perm")
	for _, l := range perm[:n] {
		f.Links = append(f.Links, Link{ID: l, BytesPerSecond: rates[l]})
	}
	return f
}

func drawFlows(rt *rapid.T, rates map[string]float64) []timedFlow {
	n := rapid.IntRange(1, 8).Draw(rt, "n")
	out := make([]timedFlow, n)
	var at int64
	for i := range out {
		at += rapid.Int64Range(0, 2e6).Draw(rt, "gap")
		out[i] = timedFlow{at: at, f: drawFlow(rt, rates, fmt.Sprint("flow", i))}
	}
	return out
}

func toDemand(f Flow) kernel.TransferDemand {
	d := kernel.TransferDemand{Bytes: f.Bytes, Floor: time.Duration(f.FloorNs)}
	for _, l := range f.Links {
		d.Links = append(d.Links, kernel.LinkUse{Link: kernel.LinkID(l.ID), BytesPerSecond: l.BytesPerSecond})
	}
	return d
}

// A transfer alone completes at exactly the kernel's price for it, rounded up to a tick as
// BLIS rounds every handoff price (kernelmodel.PDTransferTicks): the blis-schemas law
// PDTransferTime == PDTransferDemand.AloneTime() carries through the arbiter unchanged.
func TestLinkArbiter_AloneIsTheKernelPrice(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		f := drawFlow(rt, drawLinkRates(rt), "f")
		at := rapid.Int64Range(0, 1e9).Draw(rt, "at")
		want := max(1, (toDemand(f).AloneTime().Nanoseconds()+999)/1000)
		if got := AloneTicks(f); got != want {
			rt.Fatalf("AloneTicks = %d, kernel AloneTime in ticks = %d", got, want)
		}
		if got := simulate(rt, []timedFlow{{at, f}}, nil)[0] - at; got != want {
			rt.Fatalf("alone in the arbiter the flow took %d ticks, the kernel prices %d", got, want)
		}
	})
}

// Transfers on links no other transfer touches finish exactly as they would alone, however
// busy the rest of the fabric is.
func TestLinkArbiter_DisjointLinksAreIndependent(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		rates := drawLinkRates(rt)
		a := drawFlows(rt, rates)
		// Rename every link of the second set so the two sets share nothing.
		b := drawFlows(rt, rates)
		for i := range b {
			links := make([]Link, len(b[i].f.Links))
			for j, l := range b[i].f.Links {
				links[j] = Link{ID: "other/" + l.ID, BytesPerSecond: l.BytesPerSecond}
			}
			b[i].f.Links = links
		}
		alone := simulate(rt, a, nil)
		merged, origin := mergeByStart(a, b)
		together := simulate(rt, merged, nil)
		for i, o := range origin {
			if o >= 0 && together[i] != alone[o] {
				rt.Fatalf("flow %d finishes at %d beside disjoint traffic, %d without it", o, together[i], alone[o])
			}
		}
	})
}

// mergeByStart interleaves a and b by start tick (a first on ties) and reports, for each
// merged flow, its index in a or -1.
func mergeByStart(a, b []timedFlow) ([]timedFlow, []int) {
	var out []timedFlow
	var origin []int
	i, j := 0, 0
	for i < len(a) || j < len(b) {
		if j >= len(b) || (i < len(a) && a[i].at <= b[j].at) {
			out, origin = append(out, a[i]), append(origin, i)
			i++
		} else {
			out, origin = append(out, b[j]), append(origin, -1)
			j++
		}
	}
	return out, origin
}

// At every instant the division is max-min fair: no link carries more than its rate, every
// undrained transfer moves, and every undrained transfer has a bottleneck -- a saturated link
// on which no transfer is faster than it. That is the definition of a max-min fair
// allocation, so it pins the rule rather than one implementation of it.
func TestLinkArbiter_DivisionIsMaxMinFairAndConservesLinks(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		rates := drawLinkRates(rt)
		simulate(rt, drawFlows(rt, rates), func(a *LinkArbiter) {
			load := map[string]float64{}
			fastest := map[string]float64{}
			for _, f := range a.flows {
				if f.drained {
					continue
				}
				if !(f.rate > 0) {
					rt.Fatalf("an undrained transfer has rate %v", f.rate)
				}
				for _, l := range f.links {
					load[l] += f.rate
					fastest[l] = math.Max(fastest[l], f.rate)
				}
			}
			const tol = 1e-9
			for l, x := range load {
				if x > a.caps[l]*(1+tol) {
					rt.Fatalf("link %s carries %v B/s, above its %v", l, x, a.caps[l])
				}
			}
			for _, f := range a.flows {
				if f.drained {
					continue
				}
				bottleneck := false
				for _, l := range f.links {
					if load[l] >= a.caps[l]*(1-tol) && f.rate >= fastest[l]*(1-tol) {
						bottleneck = true
					}
				}
				if !bottleneck {
					rt.Fatalf("transfer %d at %v B/s has no bottleneck link", f.id, f.rate)
				}
			}
		})
	})
}

// Doubling every link's rate never makes any transfer finish later. (Raising ONE link can
// delay another transfer: max-min fairness is not monotone in a single link's capacity. Doubling
// all of them doubles every instantaneous allocation; with fixed start times that is a checked
// property here, not a theorem quoted.)
func TestLinkArbiter_DoublingBandwidthNeverDelays(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		rates := drawLinkRates(rt)
		flows := drawFlows(rt, rates)
		fast := make([]timedFlow, len(flows))
		for i, tf := range flows {
			links := make([]Link, len(tf.f.Links))
			for j, l := range tf.f.Links {
				links[j] = Link{ID: l.ID, BytesPerSecond: 2 * l.BytesPerSecond}
			}
			fast[i] = timedFlow{tf.at, Flow{Bytes: tf.f.Bytes, FloorNs: tf.f.FloorNs, Links: links}}
		}
		slow, quick := simulate(rt, flows, nil), simulate(rt, fast, nil)
		for i := range flows {
			if quick[i] > slow[i] {
				rt.Fatalf("flow %d finishes at %d with doubled links, %d without", i, quick[i], slow[i])
			}
		}
	})
}

// On one shared set of links -- every handoff of a 1P1D deployment crosses the prefill node's
// NIC egress and the decode node's NIC ingress -- the arbiter is processor sharing: no
// transfer finishes earlier than alone, and one more transfer never makes another finish
// earlier.
func TestLinkArbiter_SharedLinksSlowButNeverSpeedUp(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		eg := rapid.Float64Range(1e6, 1e11).Draw(rt, "egress")
		in := rapid.Float64Range(1e6, 1e11).Draw(rt, "ingress")
		links := []Link{{"nic/egress/node/0", eg}, {"nic/ingress/node/1", in}}
		n := rapid.IntRange(1, 8).Draw(rt, "n")
		var flows []timedFlow
		var at int64
		for i := 0; i < n; i++ {
			at += rapid.Int64Range(0, 2e6).Draw(rt, "gap")
			flows = append(flows, timedFlow{at, Flow{Bytes: rapid.Int64Range(0, 5e9).Draw(rt, "bytes"),
				FloorNs: rapid.Int64Range(0, 5e8).Draw(rt, "floor"), Links: links}})
		}
		base := simulate(rt, flows, nil)
		for i, tf := range flows {
			if alone := AloneTicks(tf.f); base[i]-tf.at < alone {
				rt.Fatalf("flow %d took %d ticks shared, %d alone", i, base[i]-tf.at, alone)
			}
		}
		extraAt := rapid.Int64Range(0, at+2e6).Draw(rt, "extraAt")
		extra := timedFlow{extraAt, Flow{Bytes: rapid.Int64Range(0, 5e9).Draw(rt, "extraBytes"), Links: links}}
		merged, origin := mergeByStart(flows, []timedFlow{extra})
		more := simulate(rt, merged, nil)
		for i, o := range origin {
			if o >= 0 && more[i] < base[o] {
				rt.Fatalf("flow %d finishes at %d with one more transfer, %d without", o, more[i], base[o])
			}
		}
	})
}

// Two equal transfers that start together on one link each get half of it: each takes twice
// as long as alone, to the tick.
func TestLinkArbiter_TwoTransfersShareALinkEqually(t *testing.T) {
	l := []Link{{"nic/ingress/node/1", 1e9}} // 1 GB/s
	f := Flow{Bytes: 1e9, Links: l}          // 1 s alone
	done := simulate(t, []timedFlow{{0, f}, {0, f}}, nil)
	if done[0] != 2_000_000 || done[1] != 2_000_000 {
		t.Fatalf("two 1-second transfers sharing one link finished at %v, want both at 2 s", done)
	}
}

// The floor overlaps the payload and is never shared: transfers whose floor dominates finish
// at start + floor even when several run together.
func TestLinkArbiter_FloorIsNotShared(t *testing.T) {
	l := []Link{{"nic/ingress/node/1", 1e9}}
	f := Flow{Bytes: 1000, FloorNs: 5e6, Links: l} // 1 us of payload, 5 ms of floor
	done := simulate(t, []timedFlow{{0, f}, {0, f}, {0, f}}, nil)
	for i, d := range done {
		if d != 5000 {
			t.Errorf("transfer %d finished at %d, want its 5 ms floor (5000 ticks)", i, d)
		}
	}
}

// Max-min fairness is NOT monotone across links, and this pins why no law claims it is: A
// crosses L1, C crosses L1 and the half-rate L2. Alone together, A and C split L1. Adding N on
// L2 cuts C to a quarter of L1's rate, which frees L1 for A, so A finishes EARLIER with more
// traffic on the fabric. The shared-link law above is
// stated for one link set only for this reason.
func TestLinkArbiter_MaxMinIsNotMonotoneAcrossLinks(t *testing.T) {
	l1, l2 := Link{"L1", 1e9}, Link{"L2", 5e8}
	a := Flow{Bytes: 1e9, Links: []Link{l1}}
	c := Flow{Bytes: 1e9, Links: []Link{l1, l2}}
	n := Flow{Bytes: 1e9, Links: []Link{l2}}
	without := simulate(t, []timedFlow{{0, a}, {0, c}}, nil)
	with := simulate(t, []timedFlow{{0, a}, {0, c}, {0, n}}, nil)
	if with[0] >= without[0] {
		t.Fatalf("expected A to finish earlier with N present (%d) than without (%d)", with[0], without[0])
	}
}

// Same inputs, same completions (INV-6).
func TestLinkArbiter_Deterministic(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		flows := drawFlows(rt, drawLinkRates(rt))
		if a, b := simulate(rt, flows, nil), simulate(rt, flows, nil); fmt.Sprint(a) != fmt.Sprint(b) {
			rt.Fatalf("two runs differ: %v vs %v", a, b)
		}
	})
}

// A link stated at two rates, and a malformed demand, are refused rather than arbitrated.
func TestLinkArbiter_RefusesInconsistentOrMalformedDemand(t *testing.T) {
	for name, fn := range map[string]func(a *LinkArbiter){
		"two rates for one link": func(a *LinkArbiter) {
			a.Start(0, Flow{Bytes: 1, Links: []Link{{"L", 1e9}}})
			a.Start(0, Flow{Bytes: 1, Links: []Link{{"L", 2e9}}})
		},
		"bytes on no link": func(a *LinkArbiter) { a.Start(0, Flow{Bytes: 1}) },
		"zero rate":        func(a *LinkArbiter) { a.Start(0, Flow{Bytes: 1, Links: []Link{{"L", 0}}}) },
		"repeated link":    func(a *LinkArbiter) { a.Start(0, Flow{Bytes: 1, Links: []Link{{"L", 1}, {"L", 1}}}) },
		"time moves back": func(a *LinkArbiter) {
			a.Start(10, Flow{Bytes: 1, Links: []Link{{"L", 1e9}}})
			a.Advance(5)
		},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected a panic")
				}
			}()
			fn(NewLinkArbiter())
		})
	}
}

// Peak concurrency is per link: transfers on different links do not add up.
func TestLinkArbiter_PeakConcurrentIsPerLink(t *testing.T) {
	a := NewLinkArbiter()
	a.Start(0, Flow{Bytes: 1e9, Links: []Link{{"L1", 1e9}}})
	a.Start(0, Flow{Bytes: 1e9, Links: []Link{{"L1", 1e9}}})
	a.Start(0, Flow{Bytes: 1e9, Links: []Link{{"L2", 1e9}}})
	if p := a.PeakConcurrent(); p["L1"] != 2 || p["L2"] != 1 {
		t.Fatalf("peak = %v, want L1:2 L2:1", p)
	}
}

// A re-rate that leaves a transfer's rate unchanged leaves its accounting untouched: its
// remaining bytes, the tick they are counted from and its drain tick. Re-deriving them from the
// float remainder would agree only up to rounding, and "a transfer on links nobody else uses
// finishes exactly as alone" is meant by construction, not by luck of the arithmetic. A
// 50-million-case search found no rounding flip, so this pins the guard directly.
func TestLinkArbiter_UnchangedRateKeepsItsAccounting(t *testing.T) {
	a := NewLinkArbiter()
	a.Start(0, Flow{Bytes: 1e9, Links: []Link{{"L1", 3e8}}})
	f := a.flows[0]
	before := *f
	for _, at := range []int64{1, 333_333, 2_000_001} { // starts on a disjoint link force re-rates
		a.Start(at, Flow{Bytes: 1, Links: []Link{{ID: "L2", BytesPerSecond: 1e9}}})
		if f.remaining != before.remaining || f.since != before.since || f.drainTick != before.drainTick {
			t.Fatalf("re-rate at %d with an unchanged rate rewrote the accounting: %+v -> %+v", at, before, *f)
		}
	}
}
