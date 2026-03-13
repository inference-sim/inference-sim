package workload

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
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
		// Always use the expanded aggregate rate — per-stage rates define the
		// ground truth. A user-specified aggregate_rate would silently scale
		// all per-stage rates by the wrong factor.
		if spec.AggregateRate > 0 && spec.AggregateRate != expanded.AggregateRate {
			logrus.Warnf("overriding aggregate_rate %.2f with sum of stage rates %.2f",
				spec.AggregateRate, expanded.AggregateRate)
		}
		spec.AggregateRate = expanded.AggregateRate
	}

	// Load ServeGen data if specified (populates spec.Clients)
	if spec.ServeGenData != nil && len(spec.Clients) == 0 {
		if err := loadServeGenData(spec); err != nil {
			return nil, fmt.Errorf("loading ServeGen data: %w", err)
		}
	}

	UpgradeV1ToV2(spec)

	if err := spec.Validate(); err != nil {
		return nil, fmt.Errorf("invalid workload spec: %w", err)
	}

	// Build working client list without mutating spec.Clients (idempotency, INV-6).
	allClients := append([]ClientSpec{}, spec.Clients...)
	if len(spec.Cohorts) > 0 {
		expanded := ExpandCohorts(spec.Cohorts, spec.Seed)
		allClients = append(allClients, expanded...)
	}

	// Create partitioned RNG for deterministic generation
	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(spec.Seed))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkloadGen)

	// Normalize rate fractions
	clientRates := normalizeRateFractions(allClients, spec.AggregateRate)

	// Generate shared prefix tokens per prefix group
	prefixes := generatePrefixTokens(allClients, workloadRNG)

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
	for i := range allClients {
		client := &allClients[i]
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

		// Handle reasoning/multi-turn clients.
		if client.Reasoning != nil && client.Reasoning.MultiTurn != nil {
			mt := client.Reasoning.MultiTurn

			if mt.SingleSession {
				// Single session: sample one start time, generate one session,
				// filter rounds against horizon. Models inference-perf's behavior
				// where each client is one persistent session cycling through rounds.
				iat := arrivalSampler.SampleIAT(clientRNG)
				startTime := iat
				// For clients with lifecycle windows, offset into the first window.
				// The IAT sample provides staggering within the window.
				if client.Lifecycle != nil && len(client.Lifecycle.Windows) > 0 {
					startTime = client.Lifecycle.Windows[0].StartUs + iat
				}
				if startTime >= horizon {
					continue
				}
				if client.Lifecycle != nil && !isInActiveWindow(startTime, client.Lifecycle) {
					continue
				}
				reasoningReqs, err := GenerateReasoningRequests(
					clientRNG, client.Reasoning,
					inputSampler, outputSampler,
					startTime,
					client.ID, client.TenantID, client.SLOClass, client.Model,
				)
				if err != nil {
					return nil, fmt.Errorf("client %q reasoning: %w", client.ID, err)
				}
				// Prepend shared prefix to each round's input (BC-1, #516).
				// NOTE: reasoning.go builds contextPrefix from raw newInputTokens,
				// NOT from req.InputTokens. The prefix must be prepended here in
				// the caller, not passed into GenerateReasoningRequests, to avoid
				// double-prepend with context accumulation.
				if len(prefix) > 0 {
					for _, req := range reasoningReqs {
						req.InputTokens = append(append([]int{}, prefix...), req.InputTokens...)
					}
				}
				// Set Deadline on all reasoning requests (not set in reasoning.go)
				for _, req := range reasoningReqs {
					req.Deadline = computeDeadline(req.ArrivalTime, client.Timeout)
				}
				for _, req := range reasoningReqs {
					if req.ArrivalTime >= horizon {
						break // rounds are in chronological order
					}
					if client.Lifecycle != nil && !isInActiveWindow(req.ArrivalTime, client.Lifecycle) {
						continue // suppress rounds outside lifecycle windows (BC-6)
					}
					allRequests = append(allRequests, req)
				}
				continue
			}

			// Multi-session: generate multiple sessions based on the arrival process,
			// each session producing MaxRounds requests.
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
				// Check lifecycle windows (bug fix: reasoning path was missing this)
				if client.Lifecycle != nil && !isInActiveWindow(currentTime, client.Lifecycle) {
					continue
				}
				reasoningReqs, err := GenerateReasoningRequests(
					clientRNG, client.Reasoning,
					inputSampler, outputSampler,
					currentTime,
					client.ID, client.TenantID, client.SLOClass, client.Model,
				)
				if err != nil {
					return nil, fmt.Errorf("client %q reasoning: %w", client.ID, err)
				}
				// Prepend shared prefix to each round's input (BC-2, #516)
				if len(prefix) > 0 {
					for _, req := range reasoningReqs {
						req.InputTokens = append(append([]int{}, prefix...), req.InputTokens...)
					}
				}
				// Set Deadline on all reasoning requests (not set in reasoning.go)
				for _, req := range reasoningReqs {
					req.Deadline = computeDeadline(req.ArrivalTime, client.Timeout)
				}
				// Count all generated rounds for perClientCap safety (R19)
				clientReqCount += int64(len(reasoningReqs))
				// Filter individual rounds against horizon and lifecycle windows (BC-3, BC-4, #515)
				for _, req := range reasoningReqs {
					if req.ArrivalTime >= horizon {
						break // rounds are in chronological order (BC-4)
					}
					if client.Lifecycle != nil && !isInActiveWindow(req.ArrivalTime, client.Lifecycle) {
						continue // suppress rounds outside lifecycle windows (BC-3)
					}
					allRequests = append(allRequests, req)
				}
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
				MaxOutputLen:     len(outputTokens),
				State:            sim.StateQueued,
				ScheduledStepIdx: 0,
				FinishedStepIdx:  0,
				TenantID:         client.TenantID,
				SLOClass:         client.SLOClass,
				Model:            client.Model,
				TextTokenCount:   textCount,
				ImageTokenCount:  imageCount,
				AudioTokenCount:  audioCount,
				VideoTokenCount:  videoCount,
				Deadline:         computeDeadline(currentTime, client.Timeout),
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

// GeneratedWorkload holds the output of GenerateWorkload: requests plus session blueprints.
type GeneratedWorkload struct {
	Requests []*sim.Request
	Sessions []SessionBlueprint // nil for non-session workloads
}

// GenerateWorkload creates requests and session blueprints from a WorkloadSpec.
// For closed-loop reasoning/multi-turn clients, only round-0 requests are generated
// and SessionBlueprints are created for the SessionManager to generate follow-up rounds.
// For all other clients (including open-loop reasoning), identical to GenerateRequests.
func GenerateWorkload(spec *WorkloadSpec, horizon int64, maxRequests int64) (*GeneratedWorkload, error) {
	// Generate all requests using existing logic.
	// For closed-loop clients, this currently generates ALL rounds (open-loop style).
	// We'll filter to round-0 only below and create blueprints for the rest.
	reqs, err := GenerateRequests(spec, horizon, maxRequests)
	if err != nil {
		return nil, err
	}

	// Check if any client is closed-loop — if not, return early (no sessions)
	hasClosedLoop := false
	allClients := append([]ClientSpec{}, spec.Clients...)
	if len(spec.Cohorts) > 0 {
		allClients = append(allClients, ExpandCohorts(spec.Cohorts, spec.Seed)...)
	}
	for i := range allClients {
		if isClosedLoop(&allClients[i]) {
			hasClosedLoop = true
			break
		}
	}
	if !hasClosedLoop {
		return &GeneratedWorkload{Requests: reqs}, nil
	}

	// For closed-loop clients: filter requests to round-0 only, create blueprints.
	// Blueprint RNG uses a fixed offset from spec seed to avoid colliding with
	// GenerateRequests' internal RNG draws. The offset (spec.Seed + 7919) is a
	// prime shift that produces an independent stream.
	blueprintRNG := rand.New(rand.NewSource(spec.Seed + 7919))

	var sessions []SessionBlueprint
	round0Only := make([]*sim.Request, 0, len(reqs))
	closedLoopSessionIDs := make(map[string]bool)

	// Build session blueprints for closed-loop clients
	for i := range allClients {
		client := &allClients[i]
		if !isClosedLoop(client) {
			continue
		}
		if client.Reasoning == nil || client.Reasoning.MultiTurn == nil {
			continue
		}
		mt := client.Reasoning.MultiTurn

		// Create samplers for the blueprint
		inputSampler, err := NewLengthSampler(client.InputDist)
		if err != nil {
			return nil, fmt.Errorf("client %q input distribution for blueprint: %w", client.ID, err)
		}
		outputSampler, err := NewLengthSampler(client.OutputDist)
		if err != nil {
			return nil, fmt.Errorf("client %q output distribution for blueprint: %w", client.ID, err)
		}

		// Get prefix tokens by extracting from the first round-0 request for this client.
		// GenerateRequests already prepended the correct prefix — we extract it here
		// to pass to the SessionBlueprint for follow-up round generation.
		var prefixTokens []int
		if client.PrefixGroup != "" && client.PrefixLength > 0 {
			for _, req := range reqs {
				if req.SessionID != "" && req.RoundIndex == 0 &&
					req.TenantID == client.TenantID && req.SLOClass == client.SLOClass {
					// The first PrefixLength tokens of InputTokens are the prefix
					if len(req.InputTokens) >= client.PrefixLength {
						prefixTokens = make([]int, client.PrefixLength)
						copy(prefixTokens, req.InputTokens[:client.PrefixLength])
					}
					break
				}
			}
		}

		// Find all session IDs for this client in the generated requests
		sessionIDsForClient := make(map[string]bool)
		for _, req := range reqs {
			if req.SessionID != "" && req.RoundIndex == 0 {
				// Check if this request belongs to this client by matching metadata
				if req.TenantID == client.TenantID && req.SLOClass == client.SLOClass && req.Model == client.Model {
					sessionIDsForClient[req.SessionID] = true
					closedLoopSessionIDs[req.SessionID] = true
				}
			}
		}

		// Create a blueprint per session (R2: sort map keys for deterministic RNG draws)
		sortedSessionIDs := make([]string, 0, len(sessionIDsForClient))
		for sessID := range sessionIDsForClient {
			sortedSessionIDs = append(sortedSessionIDs, sessID)
		}
		sort.Strings(sortedSessionIDs)
		for _, sessID := range sortedSessionIDs {
			sessSeed := blueprintRNG.Int63()
			sessions = append(sessions, SessionBlueprint{
				SessionID:     sessID,
				ClientID:      client.ID,
				MaxRounds:     mt.MaxRounds,
				ContextGrowth: mt.ContextGrowth,
				ThinkTimeUs:   mt.ThinkTimeUs,
				Timeout:       client.Timeout,
				Horizon:       horizon,
				InputSampler:  inputSampler,
				OutputSampler: outputSampler,
				RNG:           rand.New(rand.NewSource(sessSeed)),
				Prefix:        prefixTokens,
				TenantID:      client.TenantID,
				SLOClass:      client.SLOClass,
				Model:         client.Model,
			})
		}
	}

	// Filter: keep round-0 only for closed-loop sessions, keep all for non-session requests
	for _, req := range reqs {
		if req.SessionID != "" && closedLoopSessionIDs[req.SessionID] {
			// Closed-loop session: keep only round 0
			if req.RoundIndex == 0 {
				round0Only = append(round0Only, req)
			}
		} else {
			// Non-session request or open-loop session: keep all
			round0Only = append(round0Only, req)
		}
	}

	// Re-assign sequential IDs (round-0 filter changed the count)
	for i, req := range round0Only {
		req.ID = fmt.Sprintf("request_%d", i)
	}

	return &GeneratedWorkload{Requests: round0Only, Sessions: sessions}, nil
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

// DefaultTimeoutUs is the default per-request timeout (300s = 5 minutes).
// Matches cmd/observe.go HTTP client timeout for consistency between
// simulated and real-backend modes.
const DefaultTimeoutUs = 300_000_000

// computeDeadline derives the absolute deadline tick for a request.
// nil timeout → default (300s). 0 timeout → no deadline (0). Positive → arrival + timeout.
func computeDeadline(arrivalTime int64, clientTimeout *int64) int64 {
	if clientTimeout == nil {
		return arrivalTime + DefaultTimeoutUs
	}
	if *clientTimeout == 0 {
		return 0 // no timeout
	}
	return arrivalTime + *clientTimeout
}

// isClosedLoop returns whether a client should use closed-loop session generation.
// Default: true for reasoning/multi-turn clients. Overridden by explicit ClosedLoop field.
func isClosedLoop(client *ClientSpec) bool {
	if client.ClosedLoop != nil {
		return *client.ClosedLoop
	}
	// Default: true for reasoning/multi-turn clients
	return client.Reasoning != nil && client.Reasoning.MultiTurn != nil
}
