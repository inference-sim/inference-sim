package enginetest

import (
	"errors"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// loader is a request that loads 4 tokens from FIFO's tier before it runs.
func loader() Arrival {
	r := req("loads", 4, 1)
	r.Tags = map[string]string{"fifo.load_tokens": "4"}
	return at(0, r)
}

// rewrites is a FIFO whose started transfers are changed after FIFO states them: a known-bad
// engine for each way a Transfer and its Move can fail to describe one move (X18).
type rewrites struct {
	*FIFO
	change func(*engine.Transfer)
}

func (e rewrites) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := e.FIFO.Schedule(now)
	for i := range b.TransfersStarted {
		e.change(&b.TransfersStarted[i])
	}
	return b, err
}

// unplanned hides FIFO's Plan: an engine that starts transfers and states no plan to price them
// against.
type unplanned struct{ engine.Engine }

// replans is a FIFO whose Plan is not FIFO's.
type replans struct {
	*FIFO
	plan func(call int) plan.PoolPlan
	n    *int
}

func (e replans) Plan() plan.PoolPlan {
	*e.n++
	return e.plan(*e.n)
}

// The transfer FIFO returns in Batch.TransfersStarted reaches the boundary with FIFO's own tier
// name resolved to a catalog device by the engine. Only the checks that need no Environment run
// here; envtest takes the transfer on to env.DemandFor.
func TestAStartedTransferStatesItsMove(t *testing.T) {
	var seen []engine.Transfer
	_, err := Drive(newFIFO(fifoCfg()), []Arrival{loader()}, Options{
		Boundary: func(p plan.PoolPlan, x engine.Transfer) error {
			seen = append(seen, x)
			if x.Tier != FIFOTier || x.Move.Tier != FIFOTierDevice || x.Move.Kind != plan.TransferFromTier {
				t.Errorf("transfer %+v: want the engine's tier %q resolved to %q", x, FIFOTier, FIFOTierDevice)
			}
			if len(p.Tiers) != 1 || p.Tiers[0].Device != FIFOTierDevice {
				t.Errorf("plan tiers %+v do not list the device the move names", p.Tiers)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(seen) == 0 {
		t.Fatal("no transfer reached the boundary; the check processes at least one (X22)")
	}
}

func TestTheDriverCatchesTransfersThatCannotBeDemanded(t *testing.T) {
	peer := plan.Endpoint{Node: 4, Rack: 2}
	pullReq := req("pulls", 4, 1)
	pullReq.RemotePrefill, pullReq.Peer = true, &peer
	pullReq.Tags = map[string]string{"fifo.load_tokens": "4"}
	asPeer := func(from plan.Endpoint) func(*engine.Transfer) {
		return func(x *engine.Transfer) {
			x.Tier = ""
			x.Move = plan.Transfer{Tokens: x.Move.Tokens, Kind: plan.TransferPeer, From: from}
		}
	}
	for _, c := range []struct {
		name     string
		build    func(*FIFO) engine.Engine
		arrivals []Arrival
		want     string
	}{
		{"a transfer that states no Move", func(f *FIFO) engine.Engine {
			return rewrites{f, func(x *engine.Transfer) { x.Move = plan.Transfer{} }}
		}, []Arrival{loader()}, "tokens"},
		{"a Move that sizes the move differently", func(f *FIFO) engine.Engine {
			return rewrites{f, func(x *engine.Transfer) { x.Move.Tokens++ }}
		}, []Arrival{loader()}, "reports 4 tokens and Move 5"},
		{"a load whose Move stores", func(f *FIFO) engine.Engine {
			return rewrites{f, func(x *engine.Transfer) { x.Move.Kind = plan.TransferToTier }}
		}, []Arrival{loader()}, "a load whose Move goes to a tier"},
		{"the engine's tier name left in the Move", func(f *FIFO) engine.Engine {
			return rewrites{f, func(x *engine.Transfer) { x.Move.Tier = x.Tier }}
		}, []Arrival{loader()}, "not a device of this pool's plan"},
		{"a tier move on a node the pool lacks", func(f *FIFO) engine.Engine {
			return rewrites{f, func(x *engine.Transfer) { x.Move.To.Node = 3 }}
		}, []Arrival{loader()}, "node 3 of a pool with 1"},
		{"a pull from a peer the request did not name", func(f *FIFO) engine.Engine {
			return rewrites{f, asPeer(plan.Endpoint{Node: 9})}
		}, []Arrival{at(0, pullReq)}, "names peer"},
		{"transfers with no plan to price them", func(f *FIFO) engine.Engine { return unplanned{f} }, []Arrival{loader()}, "does not implement engine.Planned"},
		{"a plan the contract refuses", func(f *FIFO) engine.Engine {
			n := 0
			return replans{f, func(int) plan.PoolPlan { return plan.PoolPlan{} }, &n}
		}, []Arrival{loader()}, "not a valid PoolPlan"},
		{"a plan that changes between calls", func(f *FIFO) engine.Engine {
			n := 0
			return replans{f, func(call int) plan.PoolPlan {
				p := f.Plan()
				p.Tiers = append(p.Tiers, plan.KVTier{Device: "nvme_gen4"})
				if call == 1 {
					p.Tiers = p.Tiers[:1]
				}
				return p
			}, &n}
		}, []Arrival{loader(), at(1000, func() engine.Request {
			r := req("loads2", 4, 1)
			r.Tags = map[string]string{"fifo.load_tokens": "4"}
			return r
		}())}, "different plan from an earlier call"},
	} {
		t.Run(c.name, func(t *testing.T) {
			build := func() (engine.Engine, error) {
				f, err := NewFIFO(fifoCfg())
				if err != nil {
					return nil, err
				}
				return c.build(f), nil
			}
			_, err := Drive(build, c.arrivals, Options{MaxSteps: 1000})
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("Drive returned %v, want an error containing %q", err, c.want)
			}
		})
	}
}

// A pull that reads from the peer its request named, and states that peer as the Move's far end,
// is accepted: the refusal above is for the peer that differs.
func TestAPullFromTheNamedPeerPasses(t *testing.T) {
	peer := plan.Endpoint{Node: 4, Rack: 2}
	r := req("pulls", 4, 1)
	r.RemotePrefill, r.Peer = true, &peer
	r.Tags = map[string]string{"fifo.load_tokens": "4"}
	build := func() (engine.Engine, error) {
		f, err := NewFIFO(fifoCfg())
		if err != nil {
			return nil, err
		}
		return rewrites{f, func(x *engine.Transfer) {
			x.Tier = ""
			x.Move = plan.Transfer{Tokens: x.Move.Tokens, Kind: plan.TransferPeer, From: peer}
		}}, nil
	}
	if _, err := Drive(build, []Arrival{at(0, r)}, Options{}); err != nil {
		t.Fatal(err)
	}
}

// Boundary is the simulator's side: what it refuses fails the run, and says which transfer.
func TestTheBoundaryFailsTheRun(t *testing.T) {
	boom := errors.New("no demand for this transfer")
	_, err := Drive(newFIFO(fifoCfg()), []Arrival{loader()}, Options{
		Boundary: func(plan.PoolPlan, engine.Transfer) error { return boom },
	})
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), "across the boundary") {
		t.Fatalf("Drive returned %v, want the boundary's error", err)
	}
}

// A remote prefill whose KV is already here starts a pull of no tokens: the driver accepts it, and
// refuses the same emptiness on a tier load, where it is a bug.
func TestAPullOfNothingPassesAndATierLoadOfNothingFails(t *testing.T) {
	peer := plan.Endpoint{Node: 1}
	r := req("pulls", 4, 1)
	r.RemotePrefill, r.Peer = true, &peer
	r.Tags = map[string]string{"fifo.load_tokens": "4"}
	empty := func(f *FIFO, asPeer bool) engine.Engine {
		return rewrites{f, func(x *engine.Transfer) {
			x.Tokens, x.Blocks = 0, 0
			x.Move.Tokens = 0
			if asPeer {
				x.Tier = ""
				x.Move = plan.Transfer{Kind: plan.TransferPeer, From: peer}
			}
		}}
	}
	build := func(asPeer bool) func() (engine.Engine, error) {
		return func() (engine.Engine, error) {
			f, err := NewFIFO(fifoCfg())
			if err != nil {
				return nil, err
			}
			return empty(f, asPeer), nil
		}
	}
	if _, err := Drive(build(true), []Arrival{at(0, r)}, Options{}); err != nil {
		t.Errorf("a pull of no tokens: %v", err)
	}
	if _, err := Drive(build(false), []Arrival{loader()}, Options{}); err == nil || !strings.Contains(err.Error(), "0 tokens") {
		t.Errorf("a tier load of no tokens: Drive returned %v, want a refusal", err)
	}
}
