package workload

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

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
	// Mutual exclusion: at most one primary workload source allowed (R1).
	// Clients+Cohorts compose (cohorts expand into clients), but
	// InferencePerf and ServeGenData are exclusive alternatives.
	var sourceNames []string
	if len(spec.Clients) > 0 {
		sourceNames = append(sourceNames, "clients")
	}
	if spec.ServeGenData != nil {
		sourceNames = append(sourceNames, "servegen_data")
	}
	if spec.InferencePerf != nil {
		sourceNames = append(sourceNames, "inference_perf")
	}
	if len(sourceNames) > 1 {
		return nil, fmt.Errorf("workload sources {%s} are mutually exclusive; specify exactly one of: clients, servegen_data, inference_perf", strings.Join(sourceNames, ", "))
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

	// Route to time-varying generator when any client has per-window parameter
	// overrides (TraceRate, Arrival, InputDist, OutputDist on ActiveWindow).
	// This path uses per-window proportional allocation and IAT rescaling.
	if hasPerWindowParameters(allClients) {
		return generateTimeVaryingRequests(spec, horizon, maxRequests, allClients, workloadRNG)
	}

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

		// Create samplers.
		// When CustomSamplerFactory is set, clientRate is only used for the
		// skip guard above (line 106); the factory overrides the actual arrival rate.
		var arrivalSampler ArrivalSampler
		if client.CustomSamplerFactory != nil {
			// Derive sub-RNG for factory with single entropy draw from clientRNG.
			// This isolates the sampler's N-draw RNG consumption (for N pre-generated intervals)
			// from downstream content sampling, keeping input/output distributions stable.
			subSeed := clientRNG.Int63()
			subRNG := newRandFromSeed(subSeed)
			arrivalSampler = client.CustomSamplerFactory(subRNG)
		} else {
			arrivalSampler = NewArrivalSampler(client.Arrival, clientRate)
		}
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
				if iat == 0 {
					// Stateful sampler exhausted (e.g., NormalizedExponentialSampler).
					// Stateless samplers (Poisson, Gamma, etc.) never return 0.
					continue
				}
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
						req.PrefixLength = len(prefix)
					}
				}
				// Set Deadline on all reasoning requests (not set in reasoning.go)
				for _, req := range reasoningReqs {
					req.Deadline = computeDeadline(req.ArrivalTime, client.Timeout, true)
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
				if iat == 0 {
					// Stateful sampler exhausted (e.g., NormalizedExponentialSampler).
					// Stateless samplers (Poisson, Gamma, etc.) never return 0.
					break
				}
				currentTime += iat
				if currentTime >= horizon {
					break
				}
				// Check lifecycle windows
				if client.Lifecycle != nil && !isInActiveWindow(currentTime, client.Lifecycle) {
					if currentTime >= lastWindowEndUs(client.Lifecycle) {
						break
					}
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
						req.PrefixLength = len(prefix)
					}
				}
				// Set Deadline on all reasoning requests (not set in reasoning.go)
				for _, req := range reasoningReqs {
					req.Deadline = computeDeadline(req.ArrivalTime, client.Timeout, true)
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
			if iat == 0 {
				// Stateful sampler exhausted (e.g., NormalizedExponentialSampler).
				// Stateless samplers (Poisson, Gamma, etc.) never return 0.
				break
			}
			currentTime += iat
			if currentTime >= horizon {
				break
			}

			// Check lifecycle windows
			if client.Lifecycle != nil && !isInActiveWindow(currentTime, client.Lifecycle) {
				if currentTime >= lastWindowEndUs(client.Lifecycle) {
					break
				}
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

			var prefixLength int
			if len(prefix) > 0 {
				inputTokens = append(append([]int{}, prefix...), inputTokens...)
				prefixLength = len(prefix)
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
				Deadline:         computeDeadline(currentTime, client.Timeout, isClosedLoop(client)),
				ClientID:         client.ID,
				PrefixGroup:      client.PrefixGroup,
				PrefixLength:     prefixLength,
				Streaming:        client.Streaming,
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
	Requests       []*sim.Request
	Sessions       []SessionBlueprint // nil for non-session workloads
	// FollowUpBudget is the cap on follow-up requests for concurrency sessions.
	// -1 = no cap (maxRequests was 0/unlimited, or only closed-loop multi-turn clients present).
	//  0 = no follow-ups allowed (seeds consumed the entire budget, or no sessions at all).
	// >0 = exactly that many follow-ups allowed.
	FollowUpBudget int64
}

// GenerateWorkload creates requests and session blueprints from a WorkloadSpec.
// For closed-loop reasoning/multi-turn clients, only round-0 requests are generated
// and SessionBlueprints are created for the SessionManager to generate follow-up rounds.
// For concurrency clients (Concurrency > 0), seed requests and unlimited-round
// SessionBlueprints are generated directly (concurrency clients have RateFraction=0,
// so GenerateRequests naturally skips them).
// For all other clients (including open-loop reasoning), identical to GenerateRequests.
func GenerateWorkload(spec *WorkloadSpec, horizon int64, maxRequests int64) (*GeneratedWorkload, error) {
	// Generate all requests using existing logic.
	// For closed-loop clients, this currently generates ALL rounds (open-loop style).
	// We'll filter to round-0 only below and create blueprints for the rest.
	// Concurrency clients (RateFraction=0) are skipped by GenerateRequests.
	reqs, err := GenerateRequests(spec, horizon, maxRequests)
	if err != nil {
		return nil, err
	}

	// Check if any client is closed-loop or concurrency — if neither, return early (no sessions)
	hasClosedLoop := false
	hasConcurrency := false
	allClients := append([]ClientSpec{}, spec.Clients...)
	if len(spec.Cohorts) > 0 {
		allClients = append(allClients, ExpandCohorts(spec.Cohorts, spec.Seed)...)
	}
	for i := range allClients {
		if isClosedLoop(&allClients[i]) {
			hasClosedLoop = true
		}
		if allClients[i].Concurrency > 0 {
			hasConcurrency = true
		}
	}
	if !hasClosedLoop && !hasConcurrency {
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

	// Build session blueprints for closed-loop multi-turn clients
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
		// Match by ClientID to avoid conflating clients that share TenantID/SLOClass
		// (e.g. all stages in a multi-stage workload share the same prefixGroup TenantID).
		var prefixTokens []int
		if client.PrefixGroup != "" && client.PrefixLength > 0 {
			for _, req := range reqs {
				if req.SessionID != "" && req.RoundIndex == 0 && req.ClientID == client.ID {
					// The first PrefixLength tokens of InputTokens are the prefix
					if len(req.InputTokens) >= client.PrefixLength {
						prefixTokens = make([]int, client.PrefixLength)
						copy(prefixTokens, req.InputTokens[:client.PrefixLength])
					}
					break
				}
			}
		}

		// Find all session IDs for this client in the generated requests.
		// Match by ClientID: GenerateReasoningRequests sets req.ClientID = client.ID,
		// so this is an exact 1:1 mapping. Matching by (TenantID, SLOClass, Model) was
		// incorrect — in multi-stage workloads, all stages share the same TenantID
		// (prefixGroup), causing the first client to claim all sessions (#974).
		sessionIDsForClient := make(map[string]bool)
		for _, req := range reqs {
			if req.SessionID != "" && req.RoundIndex == 0 && req.ClientID == client.ID {
				sessionIDsForClient[req.SessionID] = true
				closedLoopSessionIDs[req.SessionID] = true
			}
		}
		// Warn if a closed-loop client produced no sessions. This indicates that
		// round-0 requests for this client have ClientID unset or mismatched
		// (e.g. a future code path that bypasses GenerateReasoningRequests).
		// With the current implementation this should never fire.
		//
		// R1 note: warn-only is intentional. Returning an error here would abort the
		// entire workload generation for a condition that is only possible through a
		// future implementation bug (unreachable via current public API). The warning
		// makes the condition observable; the subsequent blueprint loop is a no-op
		// on an empty map, so execution continues safely with zero blueprints for
		// this client.
		if len(sessionIDsForClient) == 0 {
			logrus.Warnf("GenerateWorkload: closed-loop client %q produced no sessions — ClientID may not be set on round-0 requests", client.ID)
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

	// --- Handle concurrency clients ---
	// Concurrency clients have RateFraction=0, so GenerateRequests skips them.
	// We generate seed requests and SessionBlueprints here.
	//
	// concurrencyRNG drives per-user seed selection and blueprint RNG seeding.
	// Uses spec.Seed + 10007, distinct from blueprintRNG's spec.Seed + 7919 above,
	// so the two streams do not produce identical sequences for the same spec seed.
	// If new per-client RNG streams are added here, choose an offset not already
	// in use in this function and document it with the same pattern.
	concurrencyRNG := rand.New(rand.NewSource(spec.Seed + 10007))

	// Re-derive prefix tokens by initializing a fresh RNG from spec.Seed —
	// same seed produces same prefix tokens as GenerateRequests produced.
	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(spec.Seed))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkloadGen)
	prefixes := generatePrefixTokens(allClients, workloadRNG)

	var concurrencySeeds []*sim.Request
	totalConcurrencyUsers := 0

	for i := range allClients {
		client := &allClients[i]
		if client.Concurrency <= 0 {
			continue
		}

		inputSampler, err := NewLengthSampler(client.InputDist)
		if err != nil {
			return nil, fmt.Errorf("client %q input distribution: %w", client.ID, err)
		}
		outputSampler, err := NewLengthSampler(client.OutputDist)
		if err != nil {
			return nil, fmt.Errorf("client %q output distribution: %w", client.ID, err)
		}

		var prefix []int
		if client.PrefixGroup != "" {
			prefix = prefixes[client.PrefixGroup]
		}

		for u := 0; u < client.Concurrency; u++ {
			// Never generate more seeds than the global request budget allows.
			if maxRequests > 0 && int64(len(round0Only)+len(concurrencySeeds)) >= maxRequests {
				break
			}
			userSeed := concurrencyRNG.Int63()
			userRNG := rand.New(rand.NewSource(userSeed))

			sessionID := fmt.Sprintf("concurrency_%s_user_%d", client.ID, u)

			// BC-3: Stagger seed arrivals within [0, think_time)
			var arrivalTime int64
			if client.ThinkTimeUs > 0 && client.Concurrency > 1 {
				arrivalTime = int64(u) * client.ThinkTimeUs / int64(client.Concurrency)
			}

			// Sample token lengths
			inputLen := inputSampler.Sample(userRNG)
			outputLen := outputSampler.Sample(userRNG)
			inputTokens := sim.GenerateRandomTokenIDs(userRNG, inputLen)
			outputTokens := sim.GenerateRandomTokenIDs(userRNG, outputLen)

			var prefixLength int
			if len(prefix) > 0 {
				inputTokens = append(append([]int{}, prefix...), inputTokens...)
				prefixLength = len(prefix)
			}

			seed := &sim.Request{
				ID:           "", // assigned after merge+sort
				ArrivalTime:  arrivalTime,
				InputTokens:  inputTokens,
				OutputTokens: outputTokens,
				MaxOutputLen: len(outputTokens),
				State:        sim.StateQueued,
				Deadline:     computeDeadline(arrivalTime, client.Timeout, true),
				TenantID:     client.TenantID,
				SLOClass:     client.SLOClass,
				Model:        client.Model,
				ClientID:     client.ID,
				PrefixGroup:  client.PrefixGroup,
				PrefixLength: prefixLength,
				Streaming:    client.Streaming,
				SessionID:    sessionID,
				RoundIndex:   0,
			}
			concurrencySeeds = append(concurrencySeeds, seed)

			// Create blueprint for this virtual user's session
			bpSeed := concurrencyRNG.Int63()
			sessions = append(sessions, SessionBlueprint{
				SessionID:       sessionID,
				ClientID:        client.ID,
				UnlimitedRounds: true,
				ContextGrowth:   "", // no accumulation for concurrency clients
				ThinkTimeUs:     client.ThinkTimeUs,
				Timeout:         client.Timeout,
				Horizon:         horizon,
				InputSampler:    inputSampler,
				OutputSampler:   outputSampler,
				RNG:             rand.New(rand.NewSource(bpSeed)),
				Prefix:          prefix,
				TenantID:        client.TenantID,
				SLOClass:        client.SLOClass,
				Model:           client.Model,
			})
		}
		totalConcurrencyUsers += client.Concurrency
	}

	// Merge closed-loop round-0 requests with concurrency seeds
	allReqs := append(round0Only, concurrencySeeds...)

	// Sort by arrival time (stable sort preserves order for ties)
	sort.SliceStable(allReqs, func(i, j int) bool {
		return allReqs[i].ArrivalTime < allReqs[j].ArrivalTime
	})

	// Re-assign sequential IDs
	for i, req := range allReqs {
		req.ID = fmt.Sprintf("request_%d", i)
	}

	// Compute follow-up budget for concurrency sessions.
	// -1 = no cap (default); >= 0 = exact cap on follow-ups.
	followUpBudget := int64(-1)
	if maxRequests > 0 && totalConcurrencyUsers > 0 {
		budget := maxRequests - int64(len(allReqs))
		if budget < 0 {
			budget = 0
		}
		followUpBudget = budget
	}

	return &GeneratedWorkload{Requests: allReqs, Sessions: sessions, FollowUpBudget: followUpBudget}, nil
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

// lastWindowEndUs returns the maximum EndUs across all lifecycle windows.
// Returns 0 if Windows is empty; callers must ensure the lifecycle is validated.
func lastWindowEndUs(lifecycle *LifecycleSpec) int64 {
	var maxEnd int64
	for _, w := range lifecycle.Windows {
		if w.EndUs > maxEnd {
			maxEnd = w.EndUs
		}
	}
	return maxEnd
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
// nil timeout + session client → default (300s). nil timeout + non-session → no deadline (0).
// Explicit 0 → no deadline (0). Positive → arrival + timeout.
// The isSessionClient flag determines whether the 300s default applies.
// Non-session clients do NOT get a default timeout to preserve backward compatibility.
func computeDeadline(arrivalTime int64, clientTimeout *int64, isSessionClient bool) int64 {
	if clientTimeout == nil {
		if isSessionClient {
			return arrivalTime + DefaultTimeoutUs
		}
		return 0 // no timeout for non-session clients (backward compatible)
	}
	if *clientTimeout == 0 {
		return 0 // explicit no timeout
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

// hasPerWindowParameters checks if any client has per-window parameter overrides
// (TraceRate, Arrival, InputDist, or OutputDist set on any ActiveWindow).
// Returns true when time-varying generation should be used instead of static generation.
func hasPerWindowParameters(clients []ClientSpec) bool {
	for _, client := range clients {
		if client.Lifecycle == nil {
			continue
		}
		for _, window := range client.Lifecycle.Windows {
			if window.TraceRate != nil || window.Arrival != nil ||
				window.InputDist != nil || window.OutputDist != nil {
				return true
			}
		}
	}
	return false
}

// generateTimeVaryingRequests generates requests for workloads with per-window
// parameters, using proportional rate allocation and IAT rescaling to match
// window durations. Each client's lifecycle windows are iterated, generating
// requests per window using window-specific distributions and rates.
//
// Windows beyond the horizon are skipped. Requests from all clients and windows
// are merged, sorted by arrival time, truncated to maxRequests, and assigned
// sequential IDs.
func generateTimeVaryingRequests(
	spec *WorkloadSpec,
	horizon int64,
	maxRequests int64,
	allClients []ClientSpec,
	rng *rand.Rand,
) ([]*sim.Request, error) {
	var allRequests []*sim.Request

	// Generate requests for each client's windows.
	for i := range allClients {
		client := &allClients[i]

		if client.Lifecycle == nil || len(client.Lifecycle.Windows) == 0 {
			// Client has no lifecycle windows - skip.
			// Always-on clients mixed with windowed clients are handled
			// by computeProportionalRate (contributes RateFraction to denominator).
			logrus.Warnf("generateTimeVaryingRequests: client %q has no lifecycle windows and will generate no requests (mixed always-on + windowed clients are not supported)", client.ID)
			continue
		}

		// Create per-client RNG for determinism (isolates client entropy).
		clientSeed := rng.Int63()
		clientRNG := newRandFromSeed(clientSeed)

		// Generate requests for each window.
		for _, window := range client.Lifecycle.Windows {
			// Skip windows that start beyond the simulation horizon.
			if window.StartUs >= horizon {
				continue
			}

			// Clamp window end to horizon to avoid generating requests beyond it.
			effectiveWindow := window
			if effectiveWindow.EndUs > horizon {
				effectiveWindow.EndUs = horizon
			}

			windowRequests, err := generateRequestsForWindow(
				*client, effectiveWindow, allClients, spec.AggregateRate, clientRNG,
			)
			if err != nil {
				return nil, fmt.Errorf("generating window [%d-%d] for client %q: %w",
					effectiveWindow.StartUs, effectiveWindow.EndUs, client.ID, err)
			}
			allRequests = append(allRequests, windowRequests...)
		}
	}

	// Sort all requests by arrival time (stable sort preserves client order for ties).
	sort.SliceStable(allRequests, func(i, j int) bool {
		return allRequests[i].ArrivalTime < allRequests[j].ArrivalTime
	})

	// Apply maxRequests cap if specified.
	if maxRequests > 0 && int64(len(allRequests)) > maxRequests {
		allRequests = allRequests[:maxRequests]
	}

	// Assign sequential IDs.
	for i, req := range allRequests {
		req.ID = fmt.Sprintf("request_%d", i)
	}

	return allRequests, nil
}

// generateRequestsForWindow generates requests for a single lifecycle window
// with proportional rate allocation and IAT rescaling to match the window duration.
//
// Steps:
//  1. Resolve parameters with fallback to client-level defaults.
//  2. Compute allocated rate via proportional allocation across co-active windows.
//  3. Sample IATs from the resolved arrival process.
//  4. Rescale IATs to match the target window duration (exact rate matching).
//  5. Generate requests with window-specific token distributions.
//
// Requests are returned in arrival-time order. IDs are left empty (assigned by caller
// after merging requests from all windows).
func generateRequestsForWindow(
	client ClientSpec,
	window ActiveWindow,
	allClients []ClientSpec,
	aggregateRate float64,
	rng *rand.Rand,
) ([]*sim.Request, error) {
	// Step 1: Resolve parameters with fallback to client-level defaults.
	arrival, inputDist, outputDist, _ := resolveWindowParameters(client, window)

	// Step 2: Compute allocated rate for this window via proportional allocation.
	windowTargetRate := computeProportionalRate(client, window, allClients, aggregateRate)
	if windowTargetRate <= 0 {
		return nil, nil
	}

	windowDurationUs := window.EndUs - window.StartUs
	windowDurationSec := float64(windowDurationUs) / 1e6

	// Step 3: Determine number of requests from allocated rate and window duration.
	expectedRequests := windowTargetRate * windowDurationSec
	numRequests := int(math.Ceil(expectedRequests))
	if numRequests == 0 {
		return nil, nil
	}

	// Step 4: Create samplers from resolved parameters.
	// The rate passed to NewArrivalSampler is in requests/microsecond; it
	// affects the mean IAT of the underlying distribution. Post-hoc rescaling
	// (step 6) ensures the sum of IATs matches the window duration exactly.
	arrivalSampler := NewArrivalSampler(arrival, windowTargetRate/1e6)
	inputSampler, err := NewLengthSampler(inputDist)
	if err != nil {
		return nil, fmt.Errorf("client %q input dist: %w", client.ID, err)
	}
	outputSampler, err := NewLengthSampler(outputDist)
	if err != nil {
		return nil, fmt.Errorf("client %q output dist: %w", client.ID, err)
	}

	// Step 5: Sample IATs using the resolved arrival process (shape/scale for CV).
	iats := make([]int64, numRequests)
	for i := 0; i < numRequests; i++ {
		iats[i] = arrivalSampler.SampleIAT(rng)
	}

	// Step 6: Rescale IATs to match target window duration.
	// This preserves relative ratios (CV) while ensuring total span equals windowDurationUs.
	iats = rescaleIATsToMatchDuration(iats, windowDurationUs)

	// Step 7: Generate requests with window-specific distributions.
	requests := make([]*sim.Request, 0, numRequests)
	currentTime := window.StartUs

	for i := 0; i < numRequests; i++ {
		currentTime += iats[i]

		// Stop if we exceed window boundary.
		if currentTime >= window.EndUs {
			break
		}

		// Sample token lengths from resolved distributions.
		inputLen := inputSampler.Sample(rng)
		outputLen := outputSampler.Sample(rng)
		inputTokens := sim.GenerateRandomTokenIDs(rng, inputLen)
		outputTokens := sim.GenerateRandomTokenIDs(rng, outputLen)

		req := &sim.Request{
			ID:           "", // Assigned later after merge+sort across all windows.
			ArrivalTime:  currentTime,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			MaxOutputLen: outputLen,
			State:        sim.StateQueued,
			TenantID:     client.TenantID,
			SLOClass:     client.SLOClass,
			Model:        client.Model,
			ClientID:     client.ID,
			Streaming:    client.Streaming,
			Deadline:     0, // Set by caller if needed.
		}
		requests = append(requests, req)
	}

	return requests, nil
}

// computeProportionalRate computes the allocated rate for a window using
// proportional allocation across co-active clients. When aggregate_rate > 0,
// returns: target_aggregate_rate * (window_trace_rate / sum_of_co_active_trace_rates).
// When aggregate_rate is 0 (absolute rate mode), returns window.TraceRate directly.
//
// For each co-active client (any client whose window overlaps the queried window),
// the first overlapping window's trace rate is summed into totalTraceRate. Always-on
// clients (no lifecycle) contribute their RateFraction. If totalTraceRate is zero,
// returns zero to avoid division by zero (R11).
func computeProportionalRate(
	client ClientSpec,
	window ActiveWindow,
	allClients []ClientSpec,
	aggregateRate float64,
) float64 {
	// Get this window's trace rate (resolveWindowParameters falls back to client.RateFraction)
	_, _, _, traceRate := resolveWindowParameters(client, window)

	// Absolute rate mode: when aggregate_rate is 0, use trace_rate directly.
	// This signals "use per-window rates verbatim, don't scale". Useful for
	// workloads with time-varying aggregate load that cannot be represented
	// by a single scalar aggregate_rate.
	if aggregateRate == 0 && window.TraceRate != nil {
		return traceRate
	}
	// Sum trace rates of all co-active windows
	totalTraceRate := 0.0

	for _, otherClient := range allClients {
		// Always-on clients (no lifecycle) contribute their RateFraction
		if otherClient.Lifecycle == nil || len(otherClient.Lifecycle.Windows) == 0 {
			totalTraceRate += otherClient.RateFraction
			continue
		}

		// Check if any of otherClient's windows overlap with the current window.
		// Only count each client once (first overlapping window) to avoid
		// double-counting a client that has multiple overlapping windows.
		for _, otherWindow := range otherClient.Lifecycle.Windows {
			// Time-based overlap: windows overlap iff start_a < end_b AND start_b < end_a
			if otherWindow.StartUs < window.EndUs && window.StartUs < otherWindow.EndUs {
				_, _, _, otherRate := resolveWindowParameters(otherClient, otherWindow)
				totalTraceRate += otherRate
				break // Only count each client once per query window
			}
		}
	}

	if totalTraceRate == 0 {
		logrus.Warnf("computeProportionalRate: totalTraceRate is zero for client %q window [%d-%d] (no co-active windows with non-zero rates)",
			client.ID, window.StartUs, window.EndUs)
		return 0
	}

	// Proportional allocation
	return aggregateRate * (traceRate / totalTraceRate)
}

// rescaleIATsToMatchDuration rescales inter-arrival times to sum exactly to
// targetDuration, preserving relative ratios (CV). This ensures that N requests
// generated with a given rate will fill exactly the specified window duration.
//
// Each IAT is multiplied by (targetDuration / sumIATs). A rounding residual
// from int64 truncation is applied to the last element so the sum is exact.
// Returns nil for empty input; returns the original slice unchanged if all
// IATs are zero (no scaling possible).
func rescaleIATsToMatchDuration(iats []int64, targetDuration int64) []int64 {
	if len(iats) == 0 {
		return nil
	}

	// Compute sum of original IATs.
	sumIATs := int64(0)
	for _, iat := range iats {
		sumIATs += iat
	}

	// Guard against zero sum: scaling undefined, return original values.
	if sumIATs == 0 {
		return iats
	}

	// Scale factor preserves relative ratios (CV).
	scaleFactor := float64(targetDuration) / float64(sumIATs)

	// Rescale all IATs, tracking the accumulated sum for residual correction.
	rescaled := make([]int64, len(iats))
	rescaledSum := int64(0)
	for i, iat := range iats {
		rescaled[i] = int64(float64(iat) * scaleFactor)
		rescaledSum += rescaled[i]
	}

	// Distribute rounding residual to the last element so sum is exact.
	rescaled[len(rescaled)-1] += targetDuration - rescaledSum

	return rescaled
}

// resolveWindowParameters resolves per-window parameters with fallback to client-level defaults.
// Returns the effective arrival spec, input distribution, output distribution, and trace rate.
// When a window field is nil, the corresponding client-level value is used.
func resolveWindowParameters(client ClientSpec, window ActiveWindow) (ArrivalSpec, DistSpec, DistSpec, float64) {
	// Resolve arrival spec
	arrival := client.Arrival
	if window.Arrival != nil {
		arrival = *window.Arrival
	}

	// Resolve input distribution
	inputDist := client.InputDist
	if window.InputDist != nil {
		inputDist = *window.InputDist
	}

	// Resolve output distribution
	outputDist := client.OutputDist
	if window.OutputDist != nil {
		outputDist = *window.OutputDist
	}

	// Resolve trace rate (falls back to client RateFraction)
	traceRate := client.RateFraction
	if window.TraceRate != nil {
		traceRate = *window.TraceRate
	}

	return arrival, inputDist, outputDist, traceRate
}
