package cmd

import (
	"context"
	"encoding/binary"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var (
	observeServerURL    string
	observeAPIKey       string
	observeServerType   string
	observeMaxConcur    int
	observeWarmup       int
	observeNoStreaming  bool
	observeTraceHeader  string
	observeTraceData    string
	observeModel        string
	observeWorkloadSpec string
	observeRate         float64
	observeSeed         int64
	observeHorizon      int64
	observeNumRequests  int
	// Distribution synthesis flags (same as blis run)
	observePromptTokens  int
	observePromptStdDev  int
	observePromptMin     int
	observePromptMax     int
	observeOutputTokens  int
	observeOutputStdDev  int
	observeOutputMin     int
	observeOutputMax     int
	observePrefixTokens  int
	observeAPIFormat           string
	observeUnconstrainedOutput bool
	observeRttMs               float64
)

var observeCmd = &cobra.Command{
	Use:   "observe",
	Short: "Dispatch workload requests to a real inference server and record timing",
	Long: `Observe sends requests from a WorkloadSpec to a real OpenAI-compatible inference
server at precise arrival times, recording per-request timing into TraceV2 files.

This is the data collection step of the observe/replay/calibrate pipeline.
The output TraceV2 files can be fed to 'blis replay' for simulation comparison
and 'blis calibrate' for accuracy measurement.

Supports both --workload-spec (YAML) and --rate (distribution synthesis) input paths.
Closed-loop sessions with multi-turn follow-ups are supported when the WorkloadSpec
contains session clients.

API format: Use --api-format=chat for servers that expose /v1/chat/completions
(most production vLLM/SGLang deployments). Default is --api-format=completions
which uses /v1/completions with a "prompt" field.

Output control: Use --unconstrained-output to let the server decide output length
(omits max_tokens for chat, sends large value for completions). Default constrains
output to the workload spec's sampled MaxOutputTokens.

Network calibration: Use --rtt-ms to record measured network round-trip time
in the trace header for calibration normalization.

Example:
  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --workload-spec workload.yaml --trace-header trace.yaml --trace-data trace.csv

  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --api-format chat --rate 10 --num-requests 100 --trace-header trace.yaml --trace-data trace.csv`,
	Run: runObserve,
}

func init() {
	// Required flags
	observeCmd.Flags().StringVar(&observeServerURL, "server-url", "", "Inference server URL (required)")
	observeCmd.Flags().StringVar(&observeModel, "model", "", "Model name for API requests (required)")
	observeCmd.Flags().StringVar(&observeTraceHeader, "trace-header", "", "Output path for TraceV2 header YAML (required)")
	observeCmd.Flags().StringVar(&observeTraceData, "trace-data", "", "Output path for TraceV2 data CSV (required)")

	// Workload input
	observeCmd.Flags().StringVar(&observeWorkloadSpec, "workload-spec", "", "Path to WorkloadSpec YAML (alternative to --rate)")
	observeCmd.Flags().Float64Var(&observeRate, "rate", 0, "Requests per second for distribution synthesis")

	// Optional
	observeCmd.Flags().StringVar(&observeAPIKey, "api-key", "", "Bearer token for server authentication")
	observeCmd.Flags().StringVar(&observeServerType, "server-type", "vllm", "Server type (vllm, tgi, etc.)")
	observeCmd.Flags().IntVar(&observeMaxConcur, "max-concurrency", 256, "Maximum simultaneous in-flight requests")
	observeCmd.Flags().IntVar(&observeWarmup, "warmup-requests", 0, "Number of initial requests to exclude from trace")
	observeCmd.Flags().BoolVar(&observeNoStreaming, "no-streaming", false, "Disable streaming (use non-streaming HTTP)")
	observeCmd.Flags().Int64Var(&observeSeed, "seed", 42, "RNG seed for workload generation")
	observeCmd.Flags().Int64Var(&observeHorizon, "horizon", 0, "Observation horizon in microseconds (0 = from spec or unlimited)")
	observeCmd.Flags().IntVar(&observeNumRequests, "num-requests", 0, "Maximum requests to generate (0 = from spec or unlimited)")

	// Distribution synthesis flags (same names as blis run)
	observeCmd.Flags().IntVar(&observePromptTokens, "prompt-tokens", 512, "Average prompt token count (distribution mode)")
	observeCmd.Flags().IntVar(&observePromptStdDev, "prompt-tokens-stdev", 50, "Prompt token std dev (distribution mode)")
	observeCmd.Flags().IntVar(&observePromptMin, "prompt-tokens-min", 1, "Minimum prompt tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observePromptMax, "prompt-tokens-max", 2048, "Maximum prompt tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputTokens, "output-tokens", 512, "Average output token count (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputStdDev, "output-tokens-stdev", 50, "Output token std dev (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputMin, "output-tokens-min", 1, "Minimum output tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputMax, "output-tokens-max", 2048, "Maximum output tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observePrefixTokens, "prefix-tokens", 0, "Shared prefix token count (distribution mode)")
	observeCmd.Flags().StringVar(&observeAPIFormat, "api-format", "completions", "API format: 'completions' (/v1/completions) or 'chat' (/v1/chat/completions)")
	observeCmd.Flags().BoolVar(&observeUnconstrainedOutput, "unconstrained-output", false, "Do not set max_tokens (let server decide output length)")
	observeCmd.Flags().Float64Var(&observeRttMs, "rtt-ms", 0, "Measured network round-trip time in milliseconds (recorded in trace header)")

	rootCmd.AddCommand(observeCmd)
}

func runObserve(cmd *cobra.Command, _ []string) {
	// BC-13: Required flag validation
	if observeServerURL == "" {
		logrus.Fatalf("--server-url is required")
	}
	if observeModel == "" {
		logrus.Fatalf("--model is required")
	}
	if observeTraceHeader == "" {
		logrus.Fatalf("--trace-header is required")
	}
	if observeTraceData == "" {
		logrus.Fatalf("--trace-data is required")
	}
	if observeWorkloadSpec == "" && !cmd.Flags().Changed("rate") {
		logrus.Fatalf("Either --workload-spec or --rate is required")
	}

	// BC-14: Numeric flag validation (R3)
	if observeMaxConcur <= 0 {
		logrus.Fatalf("--max-concurrency must be > 0, got %d", observeMaxConcur)
	}
	if observeWarmup < 0 {
		logrus.Fatalf("--warmup-requests must be >= 0, got %d", observeWarmup)
	}
	if cmd.Flags().Changed("rate") && (observeRate <= 0 || math.IsNaN(observeRate) || math.IsInf(observeRate, 0)) {
		logrus.Fatalf("--rate must be a finite value > 0, got %v", observeRate)
	}
	if observeAPIFormat != "completions" && observeAPIFormat != "chat" {
		logrus.Fatalf("--api-format must be 'completions' or 'chat', got %q", observeAPIFormat)
	}
	if observeRttMs < 0 || math.IsNaN(observeRttMs) || math.IsInf(observeRttMs, 0) {
		logrus.Fatalf("--rtt-ms must be a finite value >= 0, got %v", observeRttMs)
	}

	// Generate workload
	var spec *workload.WorkloadSpec
	if observeWorkloadSpec != "" {
		var err error
		spec, err = workload.LoadWorkloadSpec(observeWorkloadSpec)
		if err != nil {
			logrus.Fatalf("Failed to load workload spec: %v", err)
		}
		if cmd.Flags().Changed("seed") {
			spec.Seed = observeSeed
		}
	} else {
		// Distribution synthesis (BC-2)
		spec = workload.SynthesizeFromDistribution(workload.DistributionParams{
			Rate:               observeRate,
			NumRequests:        observeNumRequests,
			PrefixTokens:       observePrefixTokens,
			PromptTokensMean:   observePromptTokens,
			PromptTokensStdDev: observePromptStdDev,
			PromptTokensMin:    observePromptMin,
			PromptTokensMax:    observePromptMax,
			OutputTokensMean:   observeOutputTokens,
			OutputTokensStdDev: observeOutputStdDev,
			OutputTokensMin:    observeOutputMin,
			OutputTokensMax:    observeOutputMax,
		})
		spec.Seed = observeSeed
	}

	// Resolve horizon
	horizon := int64(math.MaxInt64)
	if cmd.Flags().Changed("horizon") && observeHorizon > 0 {
		horizon = observeHorizon
	} else if spec.Horizon > 0 {
		horizon = spec.Horizon
	}

	// Resolve max requests
	maxRequests := spec.NumRequests
	if cmd.Flags().Changed("num-requests") && observeNumRequests > 0 {
		maxRequests = int64(observeNumRequests)
	}

	// Guard unbounded generation
	if maxRequests <= 0 && horizon == math.MaxInt64 {
		logrus.Fatalf("Workload requires either num_requests, --num-requests, or --horizon to bound generation")
	}

	// Generate requests and session blueprints (BC-1, BC-2, D1)
	wl, err := workload.GenerateWorkload(spec, horizon, maxRequests)
	if err != nil {
		logrus.Fatalf("Failed to generate workload: %v", err)
	}

	logrus.Infof("Generated %d requests", len(wl.Requests))
	if len(wl.Sessions) > 0 {
		logrus.Infof("Generated %d session blueprints (closed-loop)", len(wl.Sessions))
	}

	if len(wl.Requests) == 0 {
		logrus.Warn("No requests generated — writing empty trace")
	}

	// Setup
	streaming := !observeNoStreaming
	client := NewRealClient(observeServerURL, observeAPIKey, observeModel, observeServerType, WithAPIFormat(observeAPIFormat))
	recorder := &Recorder{}

	// Build prefix strings for prefix-group clients (BC-5)
	var prefixes map[string]string
	var prefixLengths map[string]int
	if spec != nil {
		groups := make(map[string]int)
		for _, c := range spec.Clients {
			if c.PrefixGroup != "" {
				prefixLen := c.PrefixLength
				if prefixLen <= 0 {
					prefixLen = 50
				}
				groups[c.PrefixGroup] = prefixLen
			}
		}
		if len(groups) > 0 {
			calibCtx, calibCancel := context.WithTimeout(context.Background(), 30*time.Second)
			tokensPerWord := calibratePrefixTokenRatio(calibCtx, client)
			calibCancel()
			prefixes, prefixLengths = buildPrefixStrings(groups, spec.Seed, tokensPerWord)
			logrus.Infof("Built prefix strings for %d prefix groups (%.3f tokens/word)", len(groups), tokensPerWord)
		}
	}

	var sessionMgr *workload.SessionManager
	if len(wl.Sessions) > 0 {
		sessionMgr = workload.NewSessionManager(wl.Sessions)
	}

	// Context for graceful shutdown (BC-12)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		logrus.Warn("Received interrupt signal, cancelling observation...")
		cancel()
	}()

	// Run orchestrator
	startTime := time.Now()
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, streaming, observeMaxConcur, observeWarmup, prefixes, prefixLengths, observeUnconstrainedOutput)
	logrus.Infof("Observation wall-clock time: %.3fs", time.Since(startTime).Seconds())

	// Export trace (BC-4)
	header := &workload.TraceHeader{
		Version:        2,
		TimeUnit:       "us",
		CreatedAt:      time.Now().UTC().Format(time.RFC3339),
		Mode:           "real",
		WarmUpRequests: observeWarmup,
		Server: &workload.TraceServerConfig{
			Type:  observeServerType,
			Model: observeModel,
		},
		Network: &workload.TraceNetworkConfig{
			MeasuredRTTMs: observeRttMs,
		},
	}
	if observeWorkloadSpec != "" {
		header.WorkloadSpec = observeWorkloadSpec
	}

	if err := recorder.Export(header, observeTraceHeader, observeTraceData); err != nil {
		logrus.Fatalf("Failed to export trace: %v", err)
	}

	records := recorder.Records()
	logrus.Infof("Trace exported: %d records to %s / %s", len(records), observeTraceHeader, observeTraceData)
}

// completionEvent carries HTTP completion info to the serializer goroutine.
type completionEvent struct {
	req       *sim.Request
	record    *RequestRecord
	wallClock int64 // wall-clock microseconds at completion
}

// runObserveOrchestrator implements the dispatch loop with session support.
// This is the core orchestration function, extracted for testability.
func runObserveOrchestrator(
	ctx context.Context,
	client *RealClient,
	recorder *Recorder,
	sessionMgr *workload.SessionManager,
	requests []*sim.Request,
	streaming bool,
	maxConcurrency int,
	warmupCount int,
	prefixes map[string]string,
	prefixLengths map[string]int,
	unconstrained bool,
) {
	if len(requests) == 0 {
		return
	}

	semaphore := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup
	startWall := time.Now()
	dispatchIndex := 0

	// Channel for session follow-ups (buffered to avoid blocking serializer)
	followUpCh := make(chan *sim.Request, maxConcurrency)

	// Completion channel for session serialization (BC-8, D7)
	completionCh := make(chan completionEvent, maxConcurrency)

	// Active session tracking for drain (count unique session IDs)
	activeSessionCount := int64(0)
	if sessionMgr != nil {
		sessionIDs := make(map[string]bool)
		for _, req := range requests {
			if req.SessionID != "" && !sessionIDs[req.SessionID] {
				sessionIDs[req.SessionID] = true
				activeSessionCount++
			}
		}
	}

	// Session serializer goroutine (BC-8: single-threaded OnComplete)
	var serializerDone chan struct{}
	if sessionMgr != nil {
		serializerDone = make(chan struct{})
		go func() {
			defer close(serializerDone)
			for ce := range completionCh {
				adapted := adaptForSessionManager(ce.req, ce.record)
				followUps := sessionMgr.OnComplete(adapted, ce.wallClock)
				for _, fu := range followUps {
					followUpCh <- fu
				}
				// If session terminated (no follow-up and session request), decrement
				// and send nil wakeup to unblock the main loop's select on followUpCh
				if ce.req.SessionID != "" && len(followUps) == 0 {
					atomic.AddInt64(&activeSessionCount, -1)
					followUpCh <- nil // wakeup sentinel
				}
			}
		}()
	}

	// Dispatch function (shared between pre-generated and follow-up requests)
	dispatch := func(req *sim.Request, idx int) {
		defer wg.Done()
		defer func() { <-semaphore }() // release concurrency slot

		pending := requestToPending(req, idx, streaming, unconstrained, prefixes, prefixLengths)
		record, sendErr := client.Send(ctx, pending)
		if sendErr != nil {
			logrus.Warnf("request %d: Send returned error: %v", idx, sendErr)
		}

		// Record trace (skip warmup by index)
		arrivalTimeUs := req.ArrivalTime
		if idx >= warmupCount {
			recorder.RecordRequest(pending, record, arrivalTimeUs, req.SessionID, req.RoundIndex)
		}

		// Session completion (BC-3)
		if sessionMgr != nil && req.SessionID != "" {
			completionCh <- completionEvent{
				req:       req,
				record:    record,
				wallClock: time.Since(startWall).Microseconds(),
			}
		}
	}

	// Merge pre-generated requests and follow-ups, dispatch in arrival order.
	// Follow-ups are buffered in a local slice and merged by arrival time
	// with pre-generated requests (deterministic, no select/default race).
	preGenIdx := 0
	var pendingFollowUps []*sim.Request

	drainFollowUps := func() {
		for {
			select {
			case fu := <-followUpCh:
				if fu != nil { // nil is a wakeup sentinel from the serializer
					pendingFollowUps = append(pendingFollowUps, fu)
				}
			default:
				return
			}
		}
	}

	for {
		// Drain any buffered follow-ups
		drainFollowUps()

		// Determine next request: pick earliest arrival time between
		// pre-generated and pending follow-ups
		var nextReq *sim.Request

		hasPreGen := preGenIdx < len(requests)
		hasFollowUp := len(pendingFollowUps) > 0

		if hasPreGen && hasFollowUp {
			if pendingFollowUps[0].ArrivalTime <= requests[preGenIdx].ArrivalTime {
				nextReq = pendingFollowUps[0]
				pendingFollowUps = pendingFollowUps[1:]
	
			} else {
				nextReq = requests[preGenIdx]
				preGenIdx++
			}
		} else if hasPreGen {
			nextReq = requests[preGenIdx]
			preGenIdx++
		} else if hasFollowUp {
			nextReq = pendingFollowUps[0]
			pendingFollowUps = pendingFollowUps[1:]

		} else if sessionMgr != nil && atomic.LoadInt64(&activeSessionCount) > 0 {
			// No pre-generated or buffered follow-ups — wait for new follow-up or drain
			select {
			case fu, ok := <-followUpCh:
				if !ok {
					goto drain
				}
				nextReq = fu
	
			case <-ctx.Done():
				goto drain
			}
		} else {
			break // no more requests and no sessions
		}

		if nextReq == nil {
			continue
		}

		// Rate-pace: sleep until target wall-clock time
		targetWall := startWall.Add(time.Duration(nextReq.ArrivalTime) * time.Microsecond)
		sleepDur := time.Until(targetWall)
		if sleepDur > 0 {
			select {
			case <-time.After(sleepDur):
			case <-ctx.Done():
				goto drain
			}
		}

		// Acquire concurrency slot (BC-7)
		select {
		case semaphore <- struct{}{}:
		case <-ctx.Done():
			goto drain
		}

		idx := dispatchIndex
		dispatchIndex++
		wg.Add(1)
		go dispatch(nextReq, idx)
	}

drain:
	// Wait for all in-flight requests
	wg.Wait()

	// Close session channels
	if sessionMgr != nil {
		close(completionCh)
		<-serializerDone
	}
}

// adaptForSessionManager converts an HTTP response into a sim.Request suitable
// for SessionManager.OnComplete. Only fields read by OnComplete are populated.
func adaptForSessionManager(original *sim.Request, record *RequestRecord) *sim.Request {
	adapted := &sim.Request{
		ID:          original.ID,
		SessionID:   original.SessionID,
		RoundIndex:  original.RoundIndex,
		InputTokens: original.InputTokens,
	}

	if record.Status == "ok" {
		adapted.State = sim.StateCompleted
	} else {
		adapted.State = sim.StateTimedOut
	}

	outputCount := record.OutputTokens
	adapted.ProgressIndex = int64(len(original.InputTokens) + outputCount)

	if outputCount > 0 {
		adapted.OutputTokens = make([]int, outputCount)
		for i := range adapted.OutputTokens {
			adapted.OutputTokens[i] = i + 1
		}
	}

	return adapted
}

// requestToPending converts a sim.Request to a PendingRequest for HTTP dispatch.
// prefixes maps prefix-group name to a pre-built prefix string; prefixLengths maps
// prefix-group name to the number of words in the prefix. Both may be nil if no
// prefix groups exist.
func requestToPending(req *sim.Request, reqIndex int, streaming, unconstrained bool, prefixes map[string]string, prefixLengths map[string]int) *PendingRequest {
	// Generate proportional prompt: ~N words for N InputTokens.
	// Actual token count varies by tokenizer; ServerInputTokens provides ground truth.
	wordCount := len(req.InputTokens)
	if wordCount <= 0 {
		wordCount = 1
	}

	var prompt string
	if req.PrefixGroup != "" && prefixes != nil {
		if prefix, ok := prefixes[req.PrefixGroup]; ok {
			prefixLen := prefixLengths[req.PrefixGroup]
			suffixWords := wordCount - prefixLen
			if suffixWords < 1 {
				suffixWords = 1
			}
			prompt = prefix + strings.Repeat("hello ", suffixWords)
		} else {
			prompt = strings.Repeat("hello ", wordCount)
		}
	} else {
		prompt = strings.Repeat("hello ", wordCount)
	}

	return &PendingRequest{
		RequestID:       reqIndex,
		InputTokens:     len(req.InputTokens),
		MaxOutputTokens: req.MaxOutputLen,
		Model:           req.Model,
		Streaming:       streaming,
		ClientID:        req.ClientID,
		TenantID:        req.TenantID,
		SLOClass:        req.SLOClass,
		Prompt:          prompt,
		Unconstrained:   unconstrained,
		DeadlineUs:      req.Deadline,
	}
}

// prefixVocabulary is a hardcoded 100-word vocabulary for generating deterministic
// prefix strings. Using distinct words (rather than repeating "hello") ensures
// that different prefix groups produce distinct token sequences, activating
// the server's prefix cache for within-group requests.
var prefixVocabulary = []string{
	"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
	"kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
	"uniform", "victor", "whiskey", "xray", "yankee", "zulu", "apple", "banana", "cherry", "date",
	"elder", "fig", "grape", "hazel", "iris", "jasmine", "kiwi", "lemon", "mango", "nutmeg",
	"olive", "peach", "quince", "rose", "sage", "thyme", "umber", "violet", "willow", "yarrow",
	"acorn", "birch", "cedar", "daisy", "elm", "fern", "ginger", "holly", "ivy", "juniper",
	"kelp", "laurel", "maple", "nettle", "oak", "pine", "quinoa", "reed", "spruce", "tulip",
	"umbra", "vine", "walnut", "xylem", "yew", "zinnia", "alder", "basil", "clover", "dill",
	"fennel", "garlic", "hemp", "indigo", "jade", "kumquat", "lily", "moss", "neem", "orchid",
	"poppy", "rye", "saffron", "tea", "urchin", "verbena", "wheat", "xeris", "yucca", "zest",
}

// calibrationWordCount is the number of vocabulary words used in the
// calibration request. Must equal len(prefixVocabulary) to avoid repetition.
var calibrationWordCount = len(prefixVocabulary)

// calibratePrefixTokenRatio sends a calibration request to measure how many
// tokens the server's tokenizer produces per vocabulary word. Returns the
// ratio (typically 1.5-1.7 for BPE tokenizers with multi-syllable words).
// The ratio includes a small chat template overhead (~10-20 tokens out of
// ~167 total, <10%) which is acceptable for prefix scaling purposes.
// On failure or out-of-bounds ratio, returns 1.0 (no scaling) with a warning.
func calibratePrefixTokenRatio(ctx context.Context, client *RealClient) float64 {
	prompt := strings.Join(prefixVocabulary[:calibrationWordCount], " ")

	pending := &PendingRequest{
		RequestID:       -1,
		Model:           client.modelName,
		Streaming:       false,
		Prompt:          prompt,
		MaxOutputTokens: 1,
	}

	record, err := client.Send(ctx, pending)
	if err != nil || record.Status != "ok" || record.ServerInputTokens <= 0 {
		msg := "unknown"
		if err != nil {
			msg = err.Error()
		} else if record != nil && record.ErrorMessage != "" {
			msg = record.ErrorMessage
		}
		logrus.Warnf("Prefix token calibration failed (%s); using 1:1 word-to-token ratio", msg)
		return 1.0
	}

	ratio := float64(record.ServerInputTokens) / float64(calibrationWordCount)
	if ratio < 1.0 || ratio > 3.0 {
		logrus.Warnf("Prefix token calibration ratio %.3f outside expected range [1.0, 3.0]; using 1:1 fallback", ratio)
		return 1.0
	}

	logrus.Infof("Prefix token calibration: %d words → %d server tokens (%.3f tokens/word)",
		calibrationWordCount, record.ServerInputTokens, ratio)
	return ratio
}

// buildPrefixStrings generates deterministic prefix strings for each prefix group.
// Each group gets a distinct sequence of words from the vocabulary, seeded by
// FNV hash of (seed, group name) for cross-run reproducibility.
func buildPrefixStrings(groups map[string]int, seed int64, tokensPerWord float64) (prefixes map[string]string, prefixLengths map[string]int) {
	prefixes = make(map[string]string, len(groups))
	prefixLengths = make(map[string]int, len(groups))
	for group, length := range groups {
		if length <= 0 {
			length = 50 // default prefix length
		}

		// Scale word count so the server's tokenizer produces ~length tokens.
		wordCount := int(math.Round(float64(length) / tokensPerWord))
		if wordCount <= 0 {
			wordCount = 1
		}

		// Derive per-group seed from FNV hash
		h := fnv.New64a()
		seedBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(seedBytes, uint64(seed))
		_, _ = h.Write(seedBytes)
		_, _ = h.Write([]byte(group))
		groupSeed := int64(h.Sum64())

		rng := rand.New(rand.NewSource(groupSeed)) //nolint:gosec // deterministic, not crypto
		var words []string
		for i := 0; i < wordCount; i++ {
			words = append(words, prefixVocabulary[rng.Intn(len(prefixVocabulary))])
		}
		prefixes[group] = strings.Join(words, " ") + " "
		// Store target token count (not word count) — downstream suffix
		// computation uses this against len(req.InputTokens) which is in tokens.
		prefixLengths[group] = length
	}
	return prefixes, prefixLengths
}
