// Vendored from tantawi/lora-control epp-scorer/pkg/lorascore/score.go at fdeb55c,
// with one change: Demand.Share sums rates in sorted adapter order, so the
// result does not depend on map iteration order (INV-6). Keep in step with
// the source rather than editing here.

// Package lorascore scores candidate pods for a LoRA request by the estimated
// cost of serving it there: the adapter miss itself, the eviction it causes,
// and whether every GPU slot is busy. It has no llm-d or BLIS dependency.
package lorascore

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/inference-sim/inference-sim/sim/lorascore/residency"
)

// Weights are relative costs, in units of one load from storage. The defaults
// are placeholders until measured on the target GPUs.
type Weights struct {
	// CopyCost is a CPU-to-GPU copy (adapter in the CPU cache only).
	CopyCost float64
	// LoadCost is a load from storage (adapter in neither tier).
	LoadCost float64
	// BlockedCost is charged when every GPU slot is running or reported
	// active, so vLLM would skip the request until a slot drains.
	BlockedCost float64
	// PendingCost replaces the tier cost when an earlier request for the
	// adapter is already loading on the pod: the request waits for that load
	// rather than paying for its own.
	PendingCost float64
	// EvictionWeight scales the cost of the evictions an admission causes:
	// each victim's demand share times the cost of bringing it back, a copy
	// for a GPU adapter demoted to CPU and a load for one dropped to storage.
	EvictionWeight float64
}

// DefaultWeights returns the starting weights.
func DefaultWeights() Weights {
	return Weights{CopyCost: 0.2, LoadCost: 1, BlockedCost: 1, PendingCost: 0.1, EvictionWeight: 1}
}

// Validate reports an error if any weight is negative or not finite, which
// would turn costs, and so scores, into NaN or infinities.
func (w Weights) Validate() error {
	for name, v := range map[string]float64{
		"CopyCost": w.CopyCost, "LoadCost": w.LoadCost, "BlockedCost": w.BlockedCost,
		"PendingCost": w.PendingCost, "EvictionWeight": w.EvictionWeight,
	} {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("lorascore: weight %s = %v, want finite and non-negative", name, v)
		}
	}
	return nil
}

// Score returns a score in [0,1] per pod for a request for adapter, min-max
// normalized over the candidates' costs: the cheapest pod scores 1, the
// costliest 0, and equal costs score every pod 1 so they do not discriminate.
// A base-model request scores every pod 1, leaving the choice to other scorers.
//
// isBase marks a base-model request. It must come from configuration (the
// pool's base model names), not from the adapter being absent from metrics,
// which is indistinguishable from a cold adapter.
func Score(m *residency.Model, d *Demand, adapter string, isBase bool, pods []string, w Weights, now time.Time) map[string]float64 {
	scores := make(map[string]float64, len(pods))
	if isBase {
		for _, p := range pods {
			scores[p] = 1
		}
		return scores
	}
	costs := make(map[string]float64, len(pods))
	minCost, maxCost := math.Inf(1), 0.0
	for _, p := range pods {
		c := cost(m.Admit(p, adapter), d, w, now)
		costs[p] = c
		minCost = math.Min(minCost, c)
		maxCost = math.Max(maxCost, c)
	}
	for _, p := range pods {
		if maxCost == minCost {
			scores[p] = 1
		} else {
			scores[p] = 1 - (costs[p]-minCost)/(maxCost-minCost)
		}
	}
	return scores
}

func cost(a residency.Admission, d *Demand, w Weights, now time.Time) float64 {
	c := 0.0
	switch {
	case a.Pending:
		c += w.PendingCost
	case a.Tier == residency.CPU:
		c += w.CopyCost
	case a.Tier == residency.Absent:
		c += w.LoadCost
	}
	if a.Blocked {
		c += w.BlockedCost
	}
	if a.GPUVictim != "" {
		c += w.EvictionWeight * d.Share(a.GPUVictim, now) * w.CopyCost
	}
	if a.CPUVictim != "" {
		c += w.EvictionWeight * d.Share(a.CPUVictim, now) * w.LoadCost
	}
	return c
}

// Demand estimates each adapter's request rate as an exponentially decaying
// count, in requests per second. It is safe for concurrent use.
type Demand struct {
	mu   sync.Mutex
	tau  float64 // decay time constant in seconds
	rate map[string]float64
	last map[string]time.Time
}

// NewDemand returns an estimator whose estimate halves every halfLife, which
// must be positive.
func NewDemand(halfLife time.Duration) (*Demand, error) {
	if halfLife <= 0 {
		return nil, errors.New("lorascore: demand half-life must be positive")
	}
	return &Demand{
		tau:  halfLife.Seconds() / math.Ln2,
		rate: map[string]float64{},
		last: map[string]time.Time{},
	}, nil
}

// Observe records one request for adapter at time t.
func (d *Demand) Observe(adapter string, t time.Time) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.rate[adapter] = d.decayed(adapter, t) + 1/d.tau
	if t.After(d.last[adapter]) {
		d.last[adapter] = t
	}
}

// Rate returns adapter's estimated request rate at time t.
func (d *Demand) Rate(adapter string, t time.Time) float64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.decayed(adapter, t)
}

// Share returns adapter's fraction of the total estimated rate at time t, or 0
// when nothing has been observed.
func (d *Demand) Share(adapter string, t time.Time) float64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	names := make([]string, 0, len(d.rate))
	for a := range d.rate {
		names = append(names, a)
	}
	sort.Strings(names) // a fixed summation order, so the share is deterministic
	total := 0.0
	for _, a := range names {
		total += d.decayed(a, t)
	}
	if total == 0 {
		return 0
	}
	return d.decayed(adapter, t) / total
}

func (d *Demand) decayed(adapter string, t time.Time) float64 {
	dt := t.Sub(d.last[adapter]).Seconds()
	if dt <= 0 {
		return d.rate[adapter]
	}
	return d.rate[adapter] * math.Exp(-dt/d.tau)
}
