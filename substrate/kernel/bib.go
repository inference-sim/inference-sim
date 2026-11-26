
package latencykernel

import (
	"math"
	"os"
	"sync"
)

// routingImbalanceEnv switches the balls-in-bins expectation on: BLIS_ROUTING_IMBALANCE=balls-in-bins.
// Unset (the default) leaves the kernel byte-identical to the release: an expert-parallel group with no
// routing resolution is priced as evenly loaded.
const routingImbalanceEnv = "BLIS_ROUTING_IMBALANCE"

func ballsInBinsOn() bool { return os.Getenv(routingImbalanceEnv) == "balls-in-bins" }

// ballsInBins is the expected busiest-rank load over the mean rank's, for n routed assignments (a step's
// tokens x top-k) thrown uniformly and independently over r ranks that hold whole experts: E[max_i N_i] /
// (n / r) where (N_i) is multinomial(n; 1/r, ..., 1/r). It is at least 1 and at most r.
//
// Assumed, not measured: uniform independent routing. A token's k experts are distinct, which lowers the
// variance by (E-k)/(E-1) (under 4% for E >= 128, k <= 8); this form ignores that, so it is a slight
// overestimate. Real routing is skewed and correlated across tokens, which this does not carry.
//
// n <= ballsExactMax is exact: P(max <= t) by dynamic programming over ranks with the conditional
// binomial of each rank's count, E[max] = sum_t P(max > t). Above it the count of the busiest of r ranks is
// the mean plus sigma times the expected maximum of r standard normals (exact quadrature), sigma^2 =
// n (1/r)(1 - 1/r); at n > 128 balls the skew left out is under 3% of the busiest rank's excess.
func ballsInBins(n, r int) float64 {
	if r <= 1 || n <= 0 {
		return 1
	}
	type key struct{ n, r int }
	ballsMu.Lock()
	if v, ok := ballsMemo[key{n, r}]; ok {
		ballsMu.Unlock()
		return v
	}
	ballsMu.Unlock()
	mean := float64(n) / float64(r)
	var emax float64
	if n <= ballsExactMax {
		emax = exactExpectedMax(n, r)
	} else {
		p := 1 / float64(r)
		emax = mean + math.Sqrt(float64(n)*p*(1-p))*expectedMaxNormal(r)
	}
	f := math.Min(math.Max(emax/mean, 1), float64(r))
	ballsMu.Lock()
	ballsMemo[key{n, r}] = f
	ballsMu.Unlock()
	return f
}

const ballsExactMax = 128

var (
	ballsMu   sync.Mutex
	ballsMemo = map[struct{ n, r int }]float64{}
	normMemo  = map[int]float64{}
)

// exactExpectedMax is E[max_i N_i] of a multinomial(n; uniform over r).
func exactExpectedMax(n, r int) float64 {
	// binom[m][c] = C(m, c) q^c (1-q)^(m-c) for the q each rank draws with: 1/(r-i) for rank i.
	// P(max <= t) by DP: a[i][m] is the probability that ranks i.. take the m balls left with every
	// count at most t.
	pmax := func(t int) float64 {
		next := make([]float64, n+1)
		for m := 0; m <= n; m++ {
			if m <= t {
				next[m] = 1
			}
		}
		for i := r - 2; i >= 0; i-- {
			q := 1 / float64(r-i)
			cur := make([]float64, n+1)
			for m := 0; m <= n; m++ {
				s := 0.0
				lim := min(m, t)
				// binomial pmf by recurrence: pmf(0) = (1-q)^m, pmf(c+1) = pmf(c) (m-c)/(c+1) q/(1-q).
				pmf := math.Pow(1-q, float64(m))
				for c := 0; c <= lim; c++ {
					s += pmf * next[m-c]
					pmf *= float64(m-c) / float64(c+1) * q / (1 - q)
				}
				cur[m] = s
			}
			next = cur
		}
		return next[n]
	}
	lo := (n + r - 1) / r // the max is at least ceil(n/r)
	e := float64(lo)      // P(max > t) = 1 for t < lo
	for t := lo; t < n; t++ {
		p := 1 - pmax(t)
		if p < 1e-12 {
			break
		}
		e += p
	}
	return e
}

// expectedMaxNormal is E[max of r iid N(0,1)] = integral of x r phi(x) Phi(x)^(r-1) dx.
func expectedMaxNormal(r int) float64 {
	ballsMu.Lock()
	if v, ok := normMemo[r]; ok {
		ballsMu.Unlock()
		return v
	}
	ballsMu.Unlock()
	const lo, hi, steps = -9.0, 9.0, 18000
	h := (hi - lo) / steps
	sum := 0.0
	for i := 0; i <= steps; i++ {
		x := lo + float64(i)*h
		phi := math.Exp(-x*x/2) / math.Sqrt(2*math.Pi)
		cdf := 0.5 * math.Erfc(-x/math.Sqrt2)
		w := 1.0
		if i == 0 || i == steps {
			w = 0.5
		}
		sum += w * x * float64(r) * phi * math.Pow(cdf, float64(r-1))
	}
	v := sum * h
	ballsMu.Lock()
	normMemo[r] = v
	ballsMu.Unlock()
	return v
}
