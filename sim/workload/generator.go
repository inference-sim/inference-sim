package workload

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// GenerateRequests creates a request sequence from a WorkloadSpec.
// Deterministic given the same spec, seed, and maxRequests.
// maxRequests caps the total number of requests (0 = unlimited, use horizon only).
// Returns requests sorted by ArrivalTime with sequential IDs.
func GenerateRequests(spec *WorkloadSpec, horizon int64, maxRequests int64) ([]*sim.Request, error) {
	if horizon <= 0 {
		return nil, nil // EC-5: zero/negative horizon returns empty
	}
	if maxRequests < 0 {
		return nil, fmt.Errorf("maxRequests must be non-negative, got %d", maxRequests)
	}
	// Expand inference-perf spec if specified (populates spec.Clients)
	if spec.InferencePerf != nil && len(spec.Clients) == 0 {
		expanded, err := ExpandInferencePerfSpec(spec.InferencePerf, spec.Seed)
		if err != nil {
			return nil, fmt.Errorf("expanding inference-perf spec: %w", err)
		}
		spec.Clients = expanded.Clients
		if spec.Category == "" {
			spec.Category = expanded.Category
		}
		if spec.AggregateRate <= 0 {
			spec.AggregateRate = expanded.AggregateRate
		}
	}

	// Load ServeGen data if specified (populates spec.Clients)
	if spec.ServeGenData != nil && len(spec.Clients) == 0 {
		if err := loadServeGenData(spec); err != nil {
			return nil, fmt.Errorf("loading ServeGen data: %w", err)
		}
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

	// Per-client generation cap: prevent OOM when horizon >> maxRequests.
	// Each client generates at most 2x maxRequests, then post-merge truncation finalizes.
	perClientCap := int64(0)
	if maxRequests > 0 {
		perClientCap = 2 * maxRequests
		if perClientCap < maxRequests { // int64 overflow guard
			perClientCap = math.MaxInt64
		}
	}

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

		// Handle reasoning/multi-turn clients: generate multiple sessions
		// based on the arrival process, each session producing MaxRounds requests.
		if client.Reasoning != nil && client.Reasoning.MultiTurn != nil {
			var clientReqCount int64
			currentTime := int64(0)
			for currentTime < horizon {
				if perClientCap > 0 && clientReqCount >= perClientCap {
					break
				}
				iat := arrivalSampler.SampleIAT(clientRNG)
				currentTime += iat
				if currentTime >= horizon {
					break
				}
				reasoningReqs, err := GenerateReasoningRequests(
					clientRNG, client.Reasoning,
					inputSampler, outputSampler,
					currentTime,
					client.ID, client.TenantID, client.SLOClass,
				)
				if err != nil {
					return nil, fmt.Errorf("client %q reasoning: %w", client.ID, err)
				}
				allRequests = append(allRequests, reasoningReqs...)
				clientReqCount += int64(len(reasoningReqs))
				// Note: we do NOT skip ahead by session duration. Sessions overlap
				// in time — the arrival process controls inter-session spacing.
				// This models concurrent chat users starting sessions independently.
			}
			continue
		}

		// Generate requests for this client
		var clientReqCount int64
		currentTime := int64(0)
		for currentTime < horizon {
			if perClientCap > 0 && clientReqCount >= perClientCap {
				break
			}

			iat := arrivalSampler.SampleIAT(clientRNG)
			currentTime += iat
			if currentTime >= horizon {
				break
			}

			// Check lifecycle windows
			if client.Lifecycle != nil && !isInActiveWindow(currentTime, client.Lifecycle) {
				continue
			}

			var inputTokens []int
			var outputTokens []int
			var textCount, imageCount, audioCount, videoCount int

			if client.Multimodal != nil {
				// Multimodal generation (BC-8)
				var err error
				inputTokens, textCount, imageCount, audioCount, videoCount, err = GenerateMultimodalTokens(clientRNG, client.Multimodal)
				if err != nil {
					return nil, fmt.Errorf("client %q multimodal: %w", client.ID, err)
				}
				outputLen := outputSampler.Sample(clientRNG)
				outputTokens = sim.GenerateRandomTokenIDs(clientRNG, outputLen)
			} else {
				// Standard language generation
				inputLen := inputSampler.Sample(clientRNG)
				outputLen := outputSampler.Sample(clientRNG)
				inputTokens = sim.GenerateRandomTokenIDs(clientRNG, inputLen)
				outputTokens = sim.GenerateRandomTokenIDs(clientRNG, outputLen)
			}

			if len(prefix) > 0 {
				inputTokens = append(append([]int{}, prefix...), inputTokens...)
			}

			req := &sim.Request{
				ID:               "", // assigned after merge+sort
				ArrivalTime:      currentTime,
				InputTokens:      inputTokens,
				OutputTokens:     outputTokens,
				State:            sim.StateQueued,
				ScheduledStepIdx: 0,
				FinishedStepIdx:  0,
				TenantID:         client.TenantID,
				SLOClass:         client.SLOClass,
				Streaming:        client.Streaming,
				TextTokenCount:   textCount,
				ImageTokenCount:  imageCount,
				AudioTokenCount:  audioCount,
				VideoTokenCount:  videoCount,
			}
			allRequests = append(allRequests, req)
			clientReqCount++
		}
	}

	// Sort by arrival time (stable sort preserves client order for ties)
	sort.SliceStable(allRequests, func(i, j int) bool {
		return allRequests[i].ArrivalTime < allRequests[j].ArrivalTime
	})

	// Truncate to maxRequests after merge-sort (preserves client proportionality)
	if maxRequests > 0 && int64(len(allRequests)) > maxRequests {
		allRequests = allRequests[:maxRequests]
	}

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
