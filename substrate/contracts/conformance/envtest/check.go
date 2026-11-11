package envtest

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/env"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"
)

// Case is one plan, with the steps and transfer to run against it.
type Case struct {
	Name     string
	Plan     plan.PoolPlan
	Steps    []plan.StepShape
	Transfer plan.Transfer
}

// Check runs the laws of the Environment contract against e. Every implementation, and every
// fake a simulator's tests build, passes them.
//
//   - Pricing is pure: the same plan and step price the same, in any order.
//   - A step with more launched tokens never prices lower than one with fewer, other things
//     equal: the cost model prices the padded shape, so padding costs.
//   - A plan or step the contract refuses (plan.Validate, StepShape.Validate) is refused by
//     the environment, not priced: nothing is priced at zero because its input was absent.
//   - Memory and TransferDemand are pure in the same way, and a transfer's demand carries its
//     payload and at least one link.
//   - A pull from a peer of no tokens demands zero bytes; a tier move of no tokens is refused.
//   - A transfer the plan cannot make is refused, not priced: a kind outside the list, and a
//     move to or from a tier that is not a device in the plan's Tiers. How an engine's
//     transfer reaches TransferDemand is checked by Boundary, with enginetest.
func Check(t testing.TB, e env.Environment, cases []Case) {
	t.Helper()
	if len(cases) == 0 {
		t.Fatal("envtest: no cases; the check processes at least one (X22)")
	}
	for _, c := range cases {
		if err := c.Plan.Validate(); err != nil {
			t.Fatalf("%s: the case's plan is invalid: %v", c.Name, err)
		}
		first := make([]string, len(c.Steps))
		for i, s := range c.Steps {
			est, err := e.Price(c.Plan, s)
			if err != nil {
				t.Errorf("%s step %d: %v", c.Name, i, err)
				continue
			}
			first[i] = fmt.Sprintf("%+v", est)
		}
		for i := len(c.Steps) - 1; i >= 0; i-- {
			est, err := e.Price(c.Plan, c.Steps[i])
			if err == nil && fmt.Sprintf("%+v", est) != first[i] {
				t.Errorf("%s step %d: priced twice, two answers:\n%s\n%+v", c.Name, i, first[i], est)
			}
		}
		m1, err := e.Memory(c.Plan)
		if err != nil {
			t.Errorf("%s: Memory: %v", c.Name, err)
		} else if m2, _ := e.Memory(c.Plan); m1 != m2 {
			t.Errorf("%s: Memory differs between calls: %+v %+v", c.Name, m1, m2)
		}
		if c.Transfer.Tokens > 0 {
			d, err := e.TransferDemand(c.Plan, c.Transfer)
			switch {
			case err != nil:
				t.Errorf("%s: TransferDemand: %v", c.Name, err)
			case d.Bytes <= 0 || len(d.Links) == 0:
				t.Errorf("%s: a transfer of %d tokens has demand %+v; it carries a payload over at least one link", c.Name, c.Transfer.Tokens, d)
			}
		}
		// A pull that moves nothing demands zero bytes, and a tier move of nothing is refused.
		if c.Transfer.Kind == plan.TransferPeer {
			empty := c.Transfer
			empty.Tokens = 0
			d, err := e.TransferDemand(c.Plan, empty)
			switch {
			case err != nil:
				t.Errorf("%s: a pull of no tokens was refused: %v", c.Name, err)
			case d.Bytes != 0:
				t.Errorf("%s: a pull of no tokens has demand %+v; it demands zero bytes", c.Name, d)
			}
		} else if c.Transfer.Tokens > 0 {
			empty := c.Transfer
			empty.Tokens = 0
			if _, err := e.TransferDemand(c.Plan, empty); err == nil {
				t.Errorf("%s: a tier move of no tokens was priced", c.Name)
			}
		}
		// A transfer the plan cannot make is refused: a tier that is not a device of the plan, and
		// a kind outside the list (plan.PoolPlan.CheckTransfer).
		if c.Transfer.Tokens > 0 {
			badKind := c.Transfer
			badKind.Kind = "no-such-kind"
			if _, err := e.TransferDemand(c.Plan, badKind); err == nil {
				t.Errorf("%s: a transfer of an unknown kind was priced", c.Name)
			}
			if c.Transfer.Kind != plan.TransferPeer {
				elsewhere := c.Transfer
				elsewhere.Tier = "no-such-device"
				if _, err := e.TransferDemand(c.Plan, elsewhere); err == nil {
					t.Errorf("%s: a transfer to a tier the plan does not list was priced", c.Name)
				}
			}
		}
		// Padding costs: grow the launched tokens of a priced step and the price does not fall.
		for i, s := range c.Steps {
			if len(s.Padded) == 0 || s.Mode == plan.StepIdle || s.Mode == plan.StepPrebuilt {
				continue
			}
			bigger := s
			bigger.Padded = append([]plan.PaddedShape(nil), s.Padded...)
			for r := range bigger.Padded {
				bigger.Padded[r].Tokens += units.Tokens(64)
			}
			a, errA := e.Price(c.Plan, s)
			b, errB := e.Price(c.Plan, bigger)
			if errA == nil && errB == nil && b.NoOverlap < a.NoOverlap {
				t.Errorf("%s step %d: 64 more launched tokens priced lower (%v then %v)", c.Name, i, a.NoOverlap, b.NoOverlap)
			}
		}
		// A refused plan is refused.
		bad := c.Plan
		bad.Hardware = ""
		if len(c.Steps) > 0 {
			if _, err := e.Price(bad, c.Steps[0]); err == nil {
				t.Errorf("%s: a plan naming no chip was priced", c.Name)
			}
		}
		if _, err := e.Memory(bad); err == nil {
			t.Errorf("%s: Memory sized a plan naming no chip", c.Name)
		}
		for i, s := range c.Steps {
			badStep := s
			badStep.Mode = "no-such-mode"
			if _, err := e.Price(c.Plan, badStep); err == nil || !strings.Contains(err.Error(), "mode") {
				t.Errorf("%s step %d: a step with an unknown mode was priced: %v", c.Name, i, err)
			}
		}
	}
}
