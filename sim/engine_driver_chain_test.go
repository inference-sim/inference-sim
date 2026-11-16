package sim

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
)

// chainDriver is a driver holding n requests, r0..r{n-1}, drawing acceptance from a chain over
// rates with spread sd.
func chainDriver(t *testing.T, rates []float64, sd float64, seed int64, n int) *EngineDriver {
	t.Helper()
	c, err := acceptance.NewChain(rates, sd)
	if err != nil {
		t.Fatal(err)
	}
	d := &EngineDriver{cfg: EngineDriverConfig{Acceptance: 0.71, AcceptanceChain: c, AcceptanceSeed: seed},
		reqs: map[string]*driverReq{}}
	for i := range n {
		d.reqs[fmt.Sprintf("r%d", i)] = &driverReq{}
	}
	return d
}

// accepted runs steps batches of every request verifying drafts drafts and returns, per
// request, the drafts each step accepted.
func accepted(d *EngineDriver, n, drafts, steps int) [][]int {
	out := make([][]int, n)
	for range steps {
		var b engine.Batch
		for i := range n {
			b.Reqs = append(b.Reqs, engine.Scheduled{ID: fmt.Sprintf("r%d", i), Drafts: drafts})
		}
		o := d.outcome(b)
		for i := range n {
			out[i] = append(out[i], o.Accepted[fmt.Sprintf("r%d", i)])
		}
	}
	return out
}

// Under a chain, a step accepts any prefix of its drafts, from none to all, and over many
// requests the first i drafts are accepted at the stated rate p_i.
func TestTheDriverDrawsAcceptanceFromTheChain(t *testing.T) {
	rates := []float64{0.84, 0.70, 0.59}
	const n, steps = 300, 400
	got := accepted(chainDriver(t, rates, 0.4, 42, n), n, 3, steps)
	atLeast := make([]float64, 3)
	seen := map[int]bool{}
	for _, req := range got {
		for _, a := range req {
			seen[a] = true
			for i := range a {
				atLeast[i] += 1.0 / (n * steps)
			}
		}
	}
	for a := 0; a <= 3; a++ {
		if !seen[a] {
			t.Errorf("no step accepted %d of 3 drafts", a)
		}
	}
	for i, p := range rates {
		if math.Abs(atLeast[i]-p) > 0.01 {
			t.Errorf("P(first %d accepted) = %.4f over requests, want %.2f", i+1, atLeast[i], p)
		}
	}
}

// A step that verifies fewer drafts than the chain covers accepts at most those.
func TestTheDriverNeverAcceptsMoreThanWereVerified(t *testing.T) {
	for _, req := range accepted(chainDriver(t, []float64{0.99, 0.98, 0.97}, 0, 1, 20), 20, 1, 50) {
		for _, a := range req {
			if a > 1 {
				t.Fatalf("a step verifying one draft accepted %d", a)
			}
		}
	}
}

// The same seed draws the same acceptance for a request whatever other requests run beside it;
// another seed draws another.
func TestTheDriversDrawsAreSeededPerRequest(t *testing.T) {
	rates := []float64{0.84, 0.70, 0.59}
	alone := accepted(chainDriver(t, rates, 0.4, 42, 1), 1, 3, 300)[0]
	crowded := accepted(chainDriver(t, rates, 0.4, 42, 8), 8, 3, 300)[0]
	other := accepted(chainDriver(t, rates, 0.4, 43, 1), 1, 3, 300)[0]
	same := func(a, b []int) bool {
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}
	if !same(alone, crowded) {
		t.Error("request r0's draws changed when seven other requests ran beside it")
	}
	if same(alone, other) {
		t.Error("seeds 42 and 43 drew the same acceptance for r0")
	}
}
