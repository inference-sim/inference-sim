package workload

import (
	"fmt"

	tracespec "github.com/inference-sim/inference-sim/substrate/contracts/spec/trace"

	"github.com/inference-sim/inference-sim/sim"
)

// TracePhases returns the spec's phases as the trace header lists them, which is what the run
// writes to the exported trace and the replay's PhaseGate reads. Nil when the spec has none.
func (s *WorkloadSpec) TracePhases() []tracespec.Phase {
	var out []tracespec.Phase
	for _, p := range s.Phases {
		out = append(out, tracespec.Phase{Name: p.Name, Start: p.Start, OffsetUs: p.OffsetUs, Concurrency: p.Concurrency})
	}
	return out
}

// validatePhases checks a phased workload: the phase list as the trace header's validator checks
// it, every client in exactly one phase, every phase with clients and a request count, and the
// client kinds the drain gate can run. A workload with no phases has no client naming one.
func (s *WorkloadSpec) validatePhases() error {
	clients := append([]ClientSpec{}, s.Clients...)
	for _, c := range s.Cohorts {
		clients = append(clients, ClientSpec{ID: c.ID, Phase: c.Phase, Reasoning: c.Reasoning, ClosedLoop: c.ClosedLoop,
			Lifecycle: cohortLifecycleMarker(c)})
	}
	if len(s.Phases) == 0 {
		for _, c := range clients {
			if c.Phase != "" {
				return fmt.Errorf("client %q names phase %q, but the workload lists no phases", c.ID, c.Phase)
			}
		}
		return nil
	}
	seed := s.Seed
	header := tracespec.Header{Version: TraceVersion, TimeUnit: "microseconds", Mode: tracespec.ModeGenerated,
		WorkloadSeed: &seed, Phases: s.TracePhases()}
	if p := header.Validate(); !p.OK() {
		return fmt.Errorf("phases: %w", p)
	}
	used := map[string]int{}
	for i, ph := range s.Phases {
		if ph.NumRequests <= 0 {
			return fmt.Errorf("phases[%d] %q: num_requests must be positive, got %d: a phase is a count of requests", i, ph.Name, ph.NumRequests)
		}
		if ph.AggregateRate < 0 {
			return fmt.Errorf("phases[%d] %q: aggregate_rate must not be negative, got %v", i, ph.Name, ph.AggregateRate)
		}
		if ph.AggregateRate > 0 && ph.Concurrency > 0 {
			return fmt.Errorf("phases[%d] %q: aggregate_rate and concurrency are exclusive: a concurrency-limited phase sends a request when a slot frees", i, ph.Name)
		}
		used[ph.Name] = 0
	}
	if s.NumRequests != 0 {
		var total int64
		for _, ph := range s.Phases {
			total += ph.NumRequests
		}
		if s.NumRequests != total {
			return fmt.Errorf("num_requests %d is not the sum %d of the phases' num_requests; omit it, a phased workload's size is its phases'", s.NumRequests, total)
		}
	}
	for _, c := range clients {
		if _, ok := used[c.Phase]; !ok {
			return fmt.Errorf("client %q is in phase %q, which the workload does not list (phases: %v)", c.ID, c.Phase, s.phaseNames())
		}
		used[c.Phase]++
		switch {
		case c.Concurrency > 0:
			return fmt.Errorf("client %q: a phased workload limits concurrency per phase (phases[].concurrency), not per client", c.ID)
		case c.Lifecycle != nil:
			return fmt.Errorf("client %q: lifecycle windows and phases both stage a workload; a phased workload stages by phases", c.ID)
		case c.Reasoning != nil && c.Reasoning.MultiTurn != nil, c.ClosedLoop != nil && *c.ClosedLoop:
			return fmt.Errorf("client %q: a phased workload runs single-request clients; closed-loop sessions in a phase are not supported yet (inference-sim#1926 follow-up)", c.ID)
		}
	}
	for _, ph := range s.Phases {
		if used[ph.Name] == 0 {
			return fmt.Errorf("phase %q has no client", ph.Name)
		}
	}
	return nil
}

func (s *WorkloadSpec) phaseNames() []string {
	var names []string
	for _, p := range s.Phases {
		names = append(names, p.Name)
	}
	return names
}

// cohortLifecycleMarker is a non-nil lifecycle when the cohort synthesizes lifecycle windows
// (diurnal, spike, drain), which a phased workload refuses as it refuses a client's.
func cohortLifecycleMarker(c CohortSpec) *LifecycleSpec {
	if c.Diurnal != nil || c.Spike != nil || c.Drain != nil {
		return &LifecycleSpec{}
	}
	return nil
}

// GeneratePhased builds a phased workload's requests: each phase's from its own clients, its
// NumRequests of them, in the phase's own time (arrivals from 0). The clients keep their place
// in the spec, so the phases' draws are distinct streams, and a prefix group is one token
// sequence in every phase that names it, which is what lets a measured phase hit the cache the
// warm-up filled. Request identifiers run on across the phases. Each request carries its phase.
// The drain gate (NewPhaseGate) places the phases on the clock.
func GeneratePhased(spec *WorkloadSpec) ([]*sim.Request, error) {
	if err := validateAndExpandSpec(spec); err != nil {
		return nil, err
	}
	if len(spec.Phases) == 0 {
		return nil, fmt.Errorf("GeneratePhased: the workload has no phases")
	}
	allClients := append([]ClientSpec{}, spec.Clients...)
	if len(spec.Cohorts) > 0 {
		allClients = append(allClients, ExpandCohorts(spec.Cohorts)...)
	}
	prefixes := generatePrefixTokens(allClients, spec.Seed)
	var out []*sim.Request
	for _, ph := range spec.Phases {
		var members []ClientSpec
		var idx []int
		for i, c := range allClients {
			if c.Phase == ph.Name {
				members = append(members, c)
				idx = append(idx, i)
			}
		}
		rate := ph.AggregateRate
		if rate == 0 {
			rate = spec.AggregateRate
		}
		rates := normalizeRateFractions(members, rate)
		var preps []clientPrep
		for k, i := range idx {
			if rates[k] <= 0 && ph.Concurrency == 0 {
				continue
			}
			preps = append(preps, clientPrep{idx: i, client: &allClients[i], specSeed: spec.Seed,
				rate: rates[k], prefix: prefixes[allClients[i].PrefixGroup]})
		}
		src, _, _, err := assembleLazySourceFromPreps(preps, prefixes, allClients, spec.Seed, 1<<62, ph.NumRequests)
		if err != nil {
			return nil, fmt.Errorf("phase %q: %w", ph.Name, err)
		}
		n := 0
		for {
			r, ok := src.Next()
			if !ok {
				break
			}
			r.ID = fmt.Sprintf("request_%d", len(out))
			r.Phase = sim.Phase(ph.Name)
			out = append(out, r)
			n++
		}
		if err := src.Err(); err != nil {
			return nil, fmt.Errorf("phase %q: %w", ph.Name, err)
		}
		if int64(n) != ph.NumRequests {
			return nil, fmt.Errorf("phase %q generated %d requests, not its num_requests %d", ph.Name, n, ph.NumRequests)
		}
	}
	return out, nil
}

// OpenLoopAccumulate reports whether the workload's session clients all grow their context by
// accumulation with every round pre-generated (closed_loop false), the case whose trace export
// must carry deltas (#1914). A workload with a closed-loop or independent-round session client,
// or none, is not.
func OpenLoopAccumulate(spec *WorkloadSpec) bool {
	clients := append([]ClientSpec{}, spec.Clients...)
	clients = append(clients, ExpandCohorts(spec.Cohorts)...)
	sessions := 0
	for i := range clients {
		c := &clients[i]
		if c.Reasoning == nil || c.Reasoning.MultiTurn == nil {
			continue
		}
		if isClosedLoop(c) || c.Reasoning.MultiTurn.ContextGrowth != "accumulate" {
			return false
		}
		sessions++
	}
	return sessions > 0
}
