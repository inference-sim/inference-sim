package saturation

import (
	"math"
	"math/rand"
	"testing"
)

// service draws a service time with mean 1 second.
type service func(r *rand.Rand) float64

func expService(r *rand.Rand) float64 { return r.ExpFloat64() }

// paretoService is a Pareto with tail index a and mean 1.
func paretoService(a float64) service {
	xm := (a - 1) / a
	return func(r *rand.Rand) float64 { return xm / math.Pow(1-r.Float64(), 1/a) }
}

// openQueue simulates a FIFO single server fed by Poisson arrivals at rate rho (service mean
// 1 s) and returns the jobs of the arrivals in [0, n), and the window [warm, last arrival)
// holding the last (1-warmFrac) of them. A job still in the system at the window's end is open.
func openQueue(seed int64, rho float64, n int, warmFrac float64, s service) ([]Job, int64, int64) {
	r := rand.New(rand.NewSource(seed))
	jobs := make([]Job, n)
	var t, free float64
	for i := range jobs {
		t += r.ExpFloat64() / rho
		start := math.Max(t, free)
		free = start + s(r)
		jobs[i] = Job{ArrivalUs: int64(t * 1e6), DepartUs: int64(free * 1e6), LatencyMs: (free - t) * 1e3}
	}
	end := jobs[n-1].ArrivalUs
	for i := range jobs {
		if jobs[i].DepartUs > end {
			jobs[i].DepartUs, jobs[i].Open, jobs[i].LatencyMs = end, true, 0
		}
	}
	return jobs, jobs[int(float64(n)*warmFrac)].ArrivalUs, end
}

// closedQueue simulates c clients, each sending its next request an exponential think time of
// mean 0 after the last returns (a fixed concurrency c), served FIFO by one server.
func closedQueue(seed int64, c, n int) ([]Job, int64, int64) {
	r := rand.New(rand.NewSource(seed))
	free := 0.0
	ready := make([]float64, c)
	var jobs []Job
	for len(jobs) < n {
		k := 0
		for i := range ready {
			if ready[i] < ready[k] {
				k = i
			}
		}
		arr := ready[k]
		start := math.Max(arr, free)
		free = start + r.ExpFloat64()
		ready[k] = free
		jobs = append(jobs, Job{ArrivalUs: int64(arr * 1e6), DepartUs: int64(free * 1e6), LatencyMs: (free - arr) * 1e3})
	}
	end := jobs[len(jobs)-1].ArrivalUs
	for i := range jobs {
		if jobs[i].DepartUs > end {
			jobs[i].DepartUs, jobs[i].Open, jobs[i].LatencyMs = end, true, 0
		}
	}
	return jobs, jobs[n/2].ArrivalUs, end
}

// rate runs a case over many seeds and returns the fraction classified as want.
func rate(seeds int, want State, run func(seed int64) Verdict) float64 {
	hit := 0
	for s := int64(1); s <= int64(seeds); s++ {
		if run(s).State == want {
			hit++
		}
	}
	return float64(hit) / float64(seeds)
}

func TestSettleOperatingCharacteristics(t *testing.T) {
	type cell struct {
		name  string
		rho   float64
		n     int
		s     service
		want  State
		bound float64 // least fraction classified as want
	}
	for _, c := range []cell{
		{"M/M/1 rho 0.5, 2000 arrivals, settled", 0.5, 2000, expService, Settled, 0.97},
		{"M/M/1 rho 0.7, 4000 arrivals, settled", 0.7, 4000, expService, Settled, 0.97},
		{"M/M/1 rho 0.9, 8000 arrivals, settled", 0.9, 8000, expService, Settled, 0.95},
		{"M/M/1 rho 0.97, 8000 arrivals, settled (near critical; reported)", 0.97, 8000, expService, Settled, 0},
		{"M/M/1 rho 1.0, 8000 arrivals, overloaded (critical; reported)", 1.0, 8000, expService, Overloaded, 0},
		{"M/M/1 rho 1.5, 1000 arrivals, overloaded", 1.5, 1000, expService, Overloaded, 0.95},
		{"M/M/1 rho 1.5, 4000 arrivals, overloaded", 1.5, 4000, expService, Overloaded, 0.99},
		{"M/M/1 rho 1.2, 4000 arrivals, overloaded", 1.2, 4000, expService, Overloaded, 0.9},
		{"M/M/1 rho 1.1, 8000 arrivals, overloaded", 1.1, 8000, expService, Overloaded, 0.8},
		{"M/G/1 Pareto 1.5 service rho 0.6, 4000 arrivals, incomplete", 0.6, 4000, paretoService(1.5), Incomplete, 0.7},
		{"M/G/1 Pareto 1.5 service rho 0.6, 16000 arrivals, incomplete", 0.6, 16000, paretoService(1.5), Incomplete, 0.8},
		{"M/G/1 Pareto 3.5 service rho 0.6, 4000 arrivals, settled", 0.6, 4000, paretoService(3.5), Settled, 0.85},
	} {
		got := rate(200, c.want, func(seed int64) Verdict {
			jobs, a, b := openQueue(seed, c.rho, c.n, 0.4, c.s)
			return Settle(jobs, a, b, Options{})
		})
		t.Logf("%-62s %.3f", c.name, got)
		if got < c.bound {
			t.Errorf("%s: %.3f, want at least %.2f", c.name, got, c.bound)
		}
	}
	got := rate(200, Settled, func(seed int64) Verdict {
		jobs, a, b := closedQueue(seed, 8, 4000)
		return Settle(jobs, a, b, Options{})
	})
	t.Logf("closed queue, 8 clients, 4000 requests, settled %.3f", got)
	if got < 0.97 {
		t.Errorf("a closed queue at fixed concurrency reads settled in %.3f of runs, want at least 0.97", got)
	}
}
