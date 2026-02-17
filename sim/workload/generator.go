package workload

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// GenerateRequests creates a request sequence from a WorkloadSpec.
// Deterministic given the same spec and seed.
// Returns requests sorted by ArrivalTime with sequential IDs.
func GenerateRequests(spec *WorkloadSpec, horizon int64) ([]*sim.Request, error) {
	if horizon <= 0 {
		return nil, nil // EC-5: zero/negative horizon returns empty
	}
	if err := spec.Validate(); err != nil {
		return nil, fmt.Errorf("invalid workload spec: %w", err)
	}

	// Create partitioned RNG for deterministic generation
	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(spec.Seed))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkloadGen)

	// Normalize rate fractions
	clientRates := normalizeRateFractions(spec.Clients, spec.AggregateRate)

	// Generate shared prefix tokens per prefix group
	prefixes := generatePrefixTokens(spec.Clients, workloadRNG)

	// Per-client generation
	var allRequests []*sim.Request
	for i := range spec.Clients {
		client := &spec.Clients[i]
		clientRate := clientRates[i]
		if clientRate <= 0 {
			continue // EC-4: skip zero-rate clients
		}

		// Create per-client RNG (derived from main RNG for isolation)
		clientSeed := workloadRNG.Int63()
		clientRNG := newRandFromSeed(clientSeed)

		// Create samplers
		arrivalSampler := NewArrivalSampler(client.Arrival, clientRate)
		inputSampler, err := NewLengthSampler(client.InputDist)
		if err != nil {
			return nil, fmt.Errorf("client %q input distribution: %w", client.ID, err)
		}
		outputSampler, err := NewLengthSampler(client.OutputDist)
		if err != nil {
			return nil, fmt.Errorf("client %q output distribution: %w", client.ID, err)
		}

		// Get prefix for this client's group
		var prefix []int
		if client.PrefixGroup != "" {
			prefix = prefixes[client.PrefixGroup]
		}

		// Generate requests for this client
		currentTime := int64(0)
		for currentTime < horizon {
			iat := arrivalSampler.SampleIAT(clientRNG)
			currentTime += iat
			if currentTime >= horizon {
				break
			}

			// Check lifecycle windows
			if client.Lifecycle != nil && !isInActiveWindow(currentTime, client.Lifecycle) {
				continue
			}

			// Sample token lengths
			inputLen := inputSampler.Sample(clientRNG)
			outputLen := outputSampler.Sample(clientRNG)

			// Generate token IDs
			inputTokens := sim.GenerateRandomTokenIDs(clientRNG, inputLen)
			if len(prefix) > 0 {
				inputTokens = append(append([]int{}, prefix...), inputTokens...)
			}
			outputTokens := sim.GenerateRandomTokenIDs(clientRNG, outputLen)

			req := &sim.Request{
				ID:               "", // assigned after merge+sort
				ArrivalTime:      currentTime,
				InputTokens:      inputTokens,
				OutputTokens:     outputTokens,
				State:            "queued",
				ScheduledStepIdx: 0,
				FinishedStepIdx:  0,
				TenantID:         client.TenantID,
				SLOClass:         client.SLOClass,
				Streaming:        client.Streaming,
			}
			allRequests = append(allRequests, req)
		}
	}

	// Sort by arrival time (stable sort preserves client order for ties)
	sort.SliceStable(allRequests, func(i, j int) bool {
		return allRequests[i].ArrivalTime < allRequests[j].ArrivalTime
	})

	// Assign sequential IDs
	for i, req := range allRequests {
		req.ID = fmt.Sprintf("request_%d", i)
	}

	return allRequests, nil
}

// isInActiveWindow checks if a timestamp falls within any active window.
func isInActiveWindow(timeUs int64, lifecycle *LifecycleSpec) bool {
	for _, w := range lifecycle.Windows {
		if timeUs >= w.StartUs && timeUs < w.EndUs {
			return true
		}
	}
	return false
}

// newRandFromSeed creates a new *rand.Rand from a seed (avoids importing math/rand in callers).
func newRandFromSeed(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}
