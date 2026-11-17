// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/backpressure.py and tiering/factory.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// BackpressureIssue tracks the tiered-offload decision clocks on virtual time.
const BackpressureIssue = "https://github.com/inference-sim/blis-vllm/issues/13"

// Policy of a detector: BackpressurePolicy (backpressure.py:9-24).
type bpPolicy interface {
	shouldStore(d *EMADetector) bool
	onStoreSkipped(numBlocks int)
	popStoresDropped() (stores, blocks int)
	reset()
}

// dropAccounting is DropAccountingPolicy's drop counters (:27-52).
type dropAccounting struct{ stores, blocks int }

func (p *dropAccounting) onStoreSkipped(n int) { p.stores++; p.blocks += n }
func (p *dropAccounting) popStoresDropped() (int, int) {
	s, b := p.stores, p.blocks
	p.stores, p.blocks = 0, 0
	return s, b
}
func (p *dropAccounting) reset() { p.stores, p.blocks = 0, 0 }

// dropStore is DropStorePolicy (:55-59): every store is dropped under pressure.
type dropStore struct{ dropAccounting }

func (p *dropStore) shouldStore(d *EMADetector) bool { return !d.IsUnderPressure() }

// throttledDrop is ThrottledDropPolicy (:62-116) with its default ramp_factor 2.0: under
// pressure the drop rate scales with the EMA's overshoot of the high water mark.
type throttledDrop struct {
	dropAccounting
	acc, ramp float64
}

func (p *throttledDrop) shouldStore(d *EMADetector) bool {
	if !d.IsUnderPressure() {
		return true
	}
	high := d.high
	if high <= 0 {
		return false
	}
	over := math.Max(0, d.ema-high) / high
	rate := math.Min(1, over/p.ramp)
	if rate <= 0 {
		return true
	}
	if rate >= 1 {
		return false
	}
	p.acc += rate
	if p.acc >= 1 {
		p.acc--
		return false
	}
	return true
}

func (p *throttledDrop) reset() { p.dropAccounting.reset(); p.acc = 0 }

// EMADetector is EMABackpressureDetector (backpressure.py:189-451): an EMA of store latency
// in seconds per MiB, with warmup, idle decay, a cooldown after pressure clears, and a
// failure streak that puts a storage tier under pressure. Its clock is the virtual time
// (time.monotonic() on the simulator's clock), the same clock that stamps a job's submit time.
type EMADetector struct {
	clock  func() float64
	policy bpPolicy

	alpha, high, low, halfLife, cooldown float64
	warmup                               int
	failureThreshold                     int // 0: failures ignored (None)

	ema               float64
	underPressure     bool
	completions       int
	warmupSamples     []float64
	lastUpdate        float64
	healthyStreak     int
	pressureClearedAt float64
	cooldownCount     int
	consecFailures    int
	failureTripped    bool
}

const (
	mib              = 1 << 20
	healthyThreshold = 10
	failureEMAFactor = 4.0
)

func (d *EMADetector) applyIdleDecay() {
	now := d.clock()
	dt := now - d.lastUpdate
	if dt <= 0 || d.halfLife <= 0 {
		return
	}
	d.ema *= math.Pow(0.5, dt/d.halfLife)
	d.lastUpdate = now
}

// Update is update (:148-159): a completed store of numBytes submitted at submitTime.
func (d *EMADetector) Update(submitTime float64, numBytes int64) {
	if numBytes <= 0 {
		return
	}
	d.onStoreCompleted(d.clock()-submitTime, numBytes)
}

func (d *EMADetector) onStoreCompleted(elapsed float64, numBytes int64) {
	now := d.clock()
	sPerMiB := elapsed / (float64(numBytes) / mib)
	d.consecFailures = 0
	d.completions++
	if d.completions <= d.warmup {
		d.warmupSamples = append(d.warmupSamples, sPerMiB)
		if d.completions == d.warmup {
			sum := 0.0
			for _, v := range d.warmupSamples {
				sum += v
			}
			d.ema = sum / float64(len(d.warmupSamples))
			d.lastUpdate = now
		}
		return
	}
	d.ema = d.alpha*sPerMiB + (1-d.alpha)*d.ema
	d.lastUpdate = now
	switch {
	case d.ema > d.high:
		d.underPressure = true
		d.pressureClearedAt = 0
		d.healthyStreak = 0
	case d.ema < d.low:
		if d.underPressure {
			d.pressureClearedAt = now
			d.cooldownCount = 0
		}
		d.underPressure = false
		d.failureTripped = false
		d.healthyStreak++
	default:
		d.healthyStreak = 0
	}
}

// OnStoreFailed is on_store_failed (:368-384).
func (d *EMADetector) OnStoreFailed() {
	if d.failureThreshold == 0 {
		return
	}
	d.consecFailures++
	d.healthyStreak = 0
	if d.consecFailures < d.failureThreshold {
		return
	}
	d.ema = math.Max(d.ema, d.high*failureEMAFactor)
	d.lastUpdate = d.clock()
	d.underPressure = true
	d.failureTripped = true
	d.pressureClearedAt = 0
}

// IsUnderPressure is is_under_pressure (:386-396). It is not a pure query: it applies the idle
// decay and may clear the pressure.
func (d *EMADetector) IsUnderPressure() bool {
	if d.underPressure && (d.completions >= d.warmup || d.failureTripped) {
		d.applyIdleDecay()
		if d.ema < d.low {
			d.underPressure = false
			d.failureTripped = false
			d.pressureClearedAt = d.clock()
			d.cooldownCount = 0
		}
	}
	return d.underPressure
}

func (d *EMADetector) isHealthy() bool {
	return !d.underPressure && d.healthyStreak >= healthyThreshold
}

// ShouldStore is should_store (:410-423, :161-166): the healthy fast path, every second store
// dropped during the cooldown, then the policy.
func (d *EMADetector) ShouldStore(numBlocks int) bool {
	under := d.IsUnderPressure()
	if d.isHealthy() {
		return true
	}
	if !under && d.pressureClearedAt > 0 && d.clock()-d.pressureClearedAt < d.cooldown {
		d.cooldownCount++
		if d.cooldownCount%2 == 0 {
			d.policy.onStoreSkipped(numBlocks)
			return false
		}
	}
	if !d.policy.shouldStore(d) {
		d.policy.onStoreSkipped(numBlocks)
		return false
	}
	return true
}

// Reset is reset (:425-436).
func (d *EMADetector) Reset() {
	d.ema, d.underPressure, d.completions = 0, false, 0
	d.warmupSamples = d.warmupSamples[:0]
	d.lastUpdate, d.healthyStreak, d.pressureClearedAt, d.cooldownCount = 0, 0, 0, 0
	d.consecFailures, d.failureTripped = 0, false
	d.policy.reset()
}

// StoreLatencyEMA is store_latency_ema.
func (d *EMADetector) StoreLatencyEMA() float64 { return d.ema }

// ConsecutiveFailures is stats' consecutive_store_failures.
func (d *EMADetector) ConsecutiveFailures() int { return d.consecFailures }

// PopStoresDropped is the policy's pop_stores_dropped: the stores and blocks dropped since the
// last call.
func (d *EMADetector) PopStoresDropped() (stores, blocks int) { return d.policy.popStoresDropped() }

// NewBackpressure is the factory's detector for one tier (factory.py:73-90, with the merge of
// tiering/spec.py:329-351 already applied to cfg): backpressure_cls and policy_cls by short
// name, EMABackpressureDetector.default_config's water marks for the tier type and locality
// (backpressure.py:259-273) under the stated keys, and the constructor's checks (:275-323).
// clock is the virtual time. A dotted class name loads code vLLM does not ship and is refused
// before this, as code_not_visible.
func NewBackpressure(cfg map[string]any, tierType, locality string, clock func() float64) (*EMADetector, error) {
	c := map[string]any{}
	for k, v := range cfg {
		c[k] = v
	}
	cls, _ := c["backpressure_cls"].(string)
	delete(c, "backpressure_cls")
	if cls == "" {
		return nil, fmt.Errorf("tiering: backpressure config must include 'backpressure_cls' (e.g. 'EMABackpressureDetector')")
	}
	if cls != "EMABackpressureDetector" {
		// BackpressureDetector is abstract; any other short name is not a detector class.
		return nil, fmt.Errorf("tiering: backpressure_cls %q is not a detector vLLM v0.33.0 ships; EMABackpressureDetector is (%s)", cls, BackpressureIssue)
	}
	var policy bpPolicy = &throttledDrop{ramp: 2}
	if p, ok := c["policy_cls"]; ok {
		delete(c, "policy_cls")
		switch p {
		case nil, "":
		case "ThrottledDropPolicy":
		case "DropStorePolicy":
			policy = &dropStore{}
		default:
			// BackpressurePolicy and DropAccountingPolicy are abstract and cannot be built.
			return nil, fmt.Errorf("tiering: policy_cls %v is not a policy vLLM v0.33.0 can build; DropStorePolicy and ThrottledDropPolicy are (%s)", p, BackpressureIssue)
		}
	}
	remote := strings.EqualFold(locality, "REMOTE")
	hi, lo := 0.010, 0.002 // LOCAL_HIGH_WATER_S, LOCAL_LOW_WATER_S
	if remote || tierType == "obj" {
		hi, lo = 0.050, 0.010 // NETWORK_HIGH_WATER_S, NETWORK_LOW_WATER_S
	}
	setdefault := func(k string, v any) {
		if _, ok := c[k]; !ok {
			c[k] = v
		}
	}
	setdefault("high_water_s", hi)
	setdefault("low_water_s", lo)
	if tierType == "fs" || tierType == "obj" {
		setdefault("failure_threshold", 3)
	}
	d := &EMADetector{clock: clock, policy: policy, alpha: 0.15, warmup: 10, halfLife: 2.0, cooldown: 1.0}
	num := func(k string) (float64, bool, error) {
		v, ok := c[k]
		if !ok {
			return 0, false, nil
		}
		delete(c, k)
		switch x := v.(type) {
		case float64:
			return x, true, nil
		case int:
			return float64(x), true, nil
		case int64:
			return float64(x), true, nil
		case nil:
			return 0, false, nil
		}
		return 0, false, fmt.Errorf("tiering: backpressure %s is %v, want a number", k, v)
	}
	for _, f := range []struct {
		k   string
		dst *float64
	}{{"high_water_s", &d.high}, {"low_water_s", &d.low}, {"alpha", &d.alpha},
		{"decay_half_life_s", &d.halfLife}, {"cooldown_s", &d.cooldown}} {
		if v, ok, err := num(f.k); err != nil {
			return nil, err
		} else if ok {
			*f.dst = v
		}
	}
	if v, ok, err := num("warmup_completions"); err != nil {
		return nil, err
	} else if ok {
		d.warmup = int(v)
	}
	if _, stated := c["failure_threshold"]; stated && c["failure_threshold"] == nil {
		delete(c, "failure_threshold")
	} else if v, ok, err := num("failure_threshold"); err != nil {
		return nil, err
	} else if ok {
		if v < 1 {
			return nil, fmt.Errorf("tiering: failure_threshold must be >= 1, got %v", v)
		}
		d.failureThreshold = int(v)
	}
	if len(c) > 0 {
		var ks []string
		for k := range c {
			ks = append(ks, k)
		}
		sort.Strings(ks)
		return nil, fmt.Errorf("tiering: EMABackpressureDetector takes no argument %q (vLLM raises TypeError)", ks[0])
	}
	if !(d.alpha > 0 && d.alpha <= 1) {
		return nil, fmt.Errorf("tiering: alpha must be in (0, 1], got %v", d.alpha)
	}
	if d.low > d.high {
		return nil, fmt.Errorf("tiering: low_water_s (%v) must be <= high_water_s (%v)", d.low, d.high)
	}
	return d, nil
}
