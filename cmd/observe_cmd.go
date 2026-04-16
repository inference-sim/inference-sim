package cmd

import (
	"context"
	"encoding/binary"
	"fmt"
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
	// Distribution synthesis flags — same names and defaults as blis run.
	// Default values are defined in root.go (distDefaults const block).
	observePromptTokens  int
	observePromptStdDev  int
	observePromptMin     int
	observePromptMax     int
	observeOutputTokens  int
	observeOutputStdDev  int
	observeOutputMin     int
	observeOutputMax     int
	observePrefixTokens  int // hardcoded 0 — not in distDefaults (feature toggle, not distribution shape)
	observeAPIFormat           string
	observeUnconstrainedOutput bool
	observeRttMs               float64
	observeConcurrency         int
	observeThinkTimeMs         int
	observeWorkload            string
	observeDefaultsFilePath    string
	observeRecordITL           bool
	observeITLOutput           string
	observeMinTokens           int
)

var observeCmd = &cobra.Command{
	Use:   "observe",
	Short: "Dispatch workload requests to a real inference server and record timing",
	Long: `Observe sends requests from a WorkloadSpec to a real OpenAI-compatible inference
server at precise arrival times, recording per-request timing into TraceV2 files.

This is the data collection step of the observe/replay/calibrate pipeline.
The output TraceV2 files can be fed to 'blis replay' for simulation comparison
and 'blis calibrate' for accuracy measurement.

Supports --workload-spec (YAML), --workload <preset> (named preset; requires --rate),
--rate (distribution synthesis), or --concurrency (closed-loop virtual users) input paths.
Closed-loop sessions with multi-turn follow-ups are supported when the WorkloadSpec
contains session clients.

API format: Use --api-format=chat for servers that expose /v1/chat/completions
(most production vLLM/SGLang deployments). Default is --api-format=completions
which uses /v1/completions with a "prompt" field.

Output control: Use --unconstrained-output to let the server decide output length
(omits max_tokens for chat, sends large value for completions). Use --min-tokens N
to force the server to generate at least N tokens before EOS (set equal to
--output-tokens for exact token counts). Default constrains output to the workload
spec's sampled MaxOutputTokens.

Network calibration: Use --rtt-ms to record measured network round-trip time
in the trace header for calibration normalization.

Example:
  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --workload-spec workload.yaml --trace-header trace.yaml --trace-data trace.csv

  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --api-format chat --rate 10 --num-requests 100 --trace-header trace.yaml --trace-data trace.csv

  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --workload chatbot --rate 10 --num-requests 100 \
    --trace-header trace.yaml --trace-data trace.csv

  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --api-format chat --concurrency 50 --num-requests 500 --think-time-ms 200 \
    --trace-header trace.yaml --trace-data trace.csv`,
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
	observeCmd.Flags().StringVar(&observeWorkload, "workload", "", "Workload preset name (chatbot, summarization, contentgen, multidoc); requires --rate")
	observeCmd.Flags().StringVar(&observeDefaultsFilePath, "defaults-filepath", "defaults.yaml", "Path to defaults.yaml (for preset workload definitions)")
	observeCmd.Flags().Float64Var(&observeRate, "rate", 0, "Requests per second for distribution synthesis")

	// Optional
	observeCmd.Flags().StringVar(&observeAPIKey, "api-key", "", "Bearer token for server authentication")
	observeCmd.Flags().StringVar(&observeServerType, "server-type", "vllm", "Server type (vllm, tgi, etc.)")
	observeCmd.Flags().IntVar(&observeMaxConcur, "max-concurrency", 256, "Maximum simultaneous in-flight requests")
	observeCmd.Flags().IntVar(&observeWarmup, "warmup-requests", 0, "Number of initial requests to exclude from trace")
	observeCmd.Flags().BoolVar(&observeNoStreaming, "no-streaming", false, "Disable streaming (use non-streaming HTTP)")
	observeCmd.Flags().Int64Var(&observeSeed, "seed", 42, "RNG seed for workload generation")
	observeCmd.Flags().Int64Var(&observeHorizon, "horizon", 0, "Observation horizon in microseconds (0 = from spec or unlimited)")
	observeCmd.Flags().IntVar(&observeNumRequests, "num-requests", 0, "Maximum requests to generate (0 = from spec or unlimited; differs from blis run default of 100)")
	observeCmd.Flags().IntVar(&observeConcurrency, "concurrency", 0, "Number of concurrent virtual users (closed-loop, mutually exclusive with --rate)")
	observeCmd.Flags().IntVar(&observeThinkTimeMs, "think-time-ms", 0, "Think time in ms between response and next request (concurrency mode)")

	// Distribution synthesis flags — same names AND defaults as blis run.
	// Default values are defined in root.go (distDefaults const block).
	observeCmd.Flags().IntVar(&observePromptTokens, "prompt-tokens", defaultPromptMean, "Average prompt token count (distribution mode)")
	observeCmd.Flags().IntVar(&observePromptStdDev, "prompt-tokens-stdev", defaultPromptStdev, "Prompt token std dev (distribution mode)")
	observeCmd.Flags().IntVar(&observePromptMin, "prompt-tokens-min", defaultPromptMin, "Minimum prompt tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observePromptMax, "prompt-tokens-max", defaultPromptMax, "Maximum prompt tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputTokens, "output-tokens", defaultOutputMean, "Average output token count (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputStdDev, "output-tokens-stdev", defaultOutputStdev, "Output token std dev (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputMin, "output-tokens-min", defaultOutputMin, "Minimum output tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputMax, "output-tokens-max", defaultOutputMax, "Maximum output tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observePrefixTokens, "prefix-tokens", 0, "Shared prefix token count (distribution mode)")
	observeCmd.Flags().StringVar(&observeAPIFormat, "api-format", "completions", "API format: 'completions' (/v1/completions) or 'chat' (/v1/chat/completions)")
	observeCmd.Flags().BoolVar(&observeUnconstrainedOutput, "unconstrained-output", false, "Do not set max_tokens (let server decide output length)")
	observeCmd.Flags().IntVar(&observeMinTokens, "min-tokens", 0, "Set min_tokens in request body (forces server to generate at least N tokens before EOS)")
	observeCmd.Flags().Float64Var(&observeRttMs, "rtt-ms", 0, "Measured network round-trip time in milliseconds (recorded in trace header)")

	// ITL recording (optional, opt-in)
	observeCmd.Flags().BoolVar(&observeRecordITL, "record-itl", false, "Record per-chunk timestamps for ITL calibration (streaming only)")
	observeCmd.Flags().StringVar(&observeITLOutput, "itl-output", "", "Output path for ITL CSV file (default: <trace-data>.itl.csv if --record-itl is set)")

	rootCmd.AddCommand(observeCmd)
}

// validateObserveWorkloadFlags checks preset-mode flag constraints.
// Returns a non-empty error string if the combination is invalid, empty string if valid.
// Called from runObserve; extracted for unit testability (R14).
func validateObserveWorkloadFlags(preset, workloadSpec string, rateChanged bool, concurrency int) string {
	if preset == "" {
		return "" // no preset — nothing to validate
	}
	if workloadSpec != "" {
		return "--workload and --workload-spec are mutually exclusive"
	}
	if concurrency > 0 {
		return "--workload and --concurrency are mutually exclusive; use --workload-spec with clients[].concurrency for closed-loop preset workloads"
	}
	if !rateChanged {
		return fmt.Sprintf("--workload %q requires --rate (preset synthesis needs a request rate)", preset)
	}
	return ""
}

// buildPresetSpec loads the named preset from defaults.yaml and synthesizes a WorkloadSpec.
// Returns (nil, errMsg) if the preset is not defined or defaults.yaml cannot be accessed; (spec, "") on success.
// Extracted from runObserve for unit testability (R14). File read or YAML parse errors
// are CLI-fatal inside loadDefaultsConfig — consistent with all other defaults.yaml reads.
func buildPresetSpec(preset, defaultsPath string, rate float64, numRequests int) (*workload.WorkloadSpec, string) {
	if _, err := os.Stat(defaultsPath); err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Sprintf("--workload requires a defaults.yaml with preset definitions; "+
				"file not found at %q — use --defaults-filepath to specify its location", defaultsPath)
		}
		return nil, fmt.Sprintf("--workload requires a defaults.yaml with preset definitions; "+
			"cannot access %q: %v", defaultsPath, err)
	}
	wl := loadPresetWorkload(defaultsPath, preset)
	if wl == nil {
		return nil, fmt.Sprintf("Undefined workload %q. Use one among (chatbot, summarization, contentgen, multidoc) or --workload-spec", preset)
	}
	spec := workload.SynthesizeFromPreset(preset, workload.PresetConfig{
		PrefixTokens:      wl.PrefixTokens,
		PromptTokensMean:  wl.PromptTokensMean,
		PromptTokensStdev: wl.PromptTokensStdev,
		PromptTokensMin:   wl.PromptTokensMin,
		PromptTokensMax:   wl.PromptTokensMax,
		OutputTokensMean:  wl.OutputTokensMean,
		OutputTokensStdev: wl.OutputTokensStdev,
		OutputTokensMin:   wl.OutputTokensMin,
		OutputTokensMax:   wl.OutputTokensMax,
	}, rate, numRequests)
	return spec, ""
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
	// Warn if --itl-output is set without --record-itl (no ITL data will be written)
	if observeITLOutput != "" && !observeRecordITL {
		logrus.Warnf("--itl-output is set but --record-itl is not enabled; no ITL data will be written")
	}
	// BC-7: at least one workload input mode must be provided
	if observeWorkload == "" && observeWorkloadSpec == "" && !cmd.Flags().Changed("rate") && observeConcurrency <= 0 {
		logrus.Fatalf("Either --workload, --workload-spec, --rate, or --concurrency is required")
	}
	// BC-2/3/4: preset-mode constraint check (extracted for testability, R14).
	// Runs before the existing concurrency/rate exclusion so preset errors are shown first.
	if msg := validateObserveWorkloadFlags(observeWorkload, observeWorkloadSpec, cmd.Flags().Changed("rate"), observeConcurrency); msg != "" {
		logrus.Fatalf("%s", msg)
	}
	// BC-1: --concurrency and --rate are mutually exclusive
	if observeConcurrency > 0 && cmd.Flags().Changed("rate") {
		logrus.Fatalf("--concurrency and --rate are mutually exclusive; use one or the other")
	}
	if observeConcurrency < 0 {
		logrus.Fatalf("--concurrency must be >= 0, got %d", observeConcurrency)
	}
	if observeThinkTimeMs < 0 {
		logrus.Fatalf("--think-time-ms must be >= 0, got %d", observeThinkTimeMs)
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
	if observeMinTokens < 0 {
		logrus.Fatalf("--min-tokens must be >= 0, got %d", observeMinTokens)
	}

	// Generate workload
	var spec *workload.WorkloadSpec
	if observeWorkloadSpec != "" {
		if observeConcurrency > 0 {
			logrus.Fatalf("--concurrency cannot be used with --workload-spec; " +
				"define concurrency in the spec file using clients[].concurrency instead")
		}
		var err error
		spec, err = workload.LoadWorkloadSpec(observeWorkloadSpec)
		if err != nil {
			logrus.Fatalf("Failed to load workload spec: %v", err)
		}
		if cmd.Flags().Changed("seed") {
			spec.Seed = observeSeed
		}
	} else if observeWorkload != "" {
		// Preset synthesis — BC-1: same token distribution as blis run --workload <preset>
		// Rate was validated finite+positive by the earlier rate validation above (defense-in-depth:
		// also guarded by validateObserveWorkloadFlags above, which requires rateChanged to be true).
		// Use separate errMsg var + = (not :=) to avoid shadowing the outer spec variable.
		var errMsg string
		spec, errMsg = buildPresetSpec(observeWorkload, observeDefaultsFilePath, observeRate, observeNumRequests)
		if errMsg != "" {
			logrus.Fatalf("%s", errMsg)
		}
		spec.Seed = observeSeed
	} else {
		// Distribution or concurrency synthesis
		spec = workload.SynthesizeFromDistribution(workload.DistributionParams{
			Rate:               observeRate,
			Concurrency:        observeConcurrency,
			ThinkTimeMs:        observeThinkTimeMs,
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

	// Warn if --min-tokens exceeds max_tokens for any request: vLLM rejects such requests.
	if observeMinTokens > 0 && !observeUnconstrainedOutput {
		mismatch := 0
		for _, r := range wl.Requests {
			if observeMinTokens > r.MaxOutputLen {
				mismatch++
			}
		}
		if mismatch > 0 {
			logrus.Warnf("--min-tokens=%d exceeds max_tokens for %d/%d requests; those requests will likely be rejected by the server", observeMinTokens, mismatch, len(wl.Requests))
		}
	}

	// Enable streaming on all requests when --record-itl is set (BC-6)
	// ITL recording requires streaming responses to capture per-chunk timestamps.
	// The inference-perf format defaults to non-streaming for parity with the real tool,
	// so we override it here when ITL is explicitly requested.
	if observeRecordITL {
		streamingCount := 0
		for i := range wl.Requests {
			if !wl.Requests[i].Streaming {
				wl.Requests[i].Streaming = true
				streamingCount++
			}
		}
		if streamingCount > 0 {
			logrus.Infof("Enabled streaming on %d requests for ITL recording", streamingCount)
		}
	}

	// Setup
	client := NewRealClient(observeServerURL, observeAPIKey, observeModel, observeServerType, WithAPIFormat(observeAPIFormat))
	recorder := &Recorder{}

	// Calibrate tokens-per-word ratio for the server's tokenizer (BC-6).
	// Used for both prefix string building and non-prefix prompt scaling.
	calibCtx, calibCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer calibCancel()
	tokensPerWord := calibratePrefixTokenRatio(calibCtx, client)
	logrus.Infof("Calibrated tokens-per-word ratio: %.3f", tokensPerWord)

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
			prefixes, prefixLengths = buildPrefixStrings(groups, spec.Seed, tokensPerWord)
			logrus.Infof("Built prefix strings for %d prefix groups", len(groups))
		}
	}

	var sessionMgr *workload.SessionManager
	if len(wl.Sessions) > 0 {
		sessionMgr = workload.NewSessionManager(wl.Sessions)
		if wl.FollowUpBudget >= 0 {
			sessionMgr.SetFollowUpBudget(wl.FollowUpBudget)
		}
	}

	// Auto-set max-concurrency for concurrency mode
	if observeConcurrency > 0 && !cmd.Flags().Changed("max-concurrency") {
		observeMaxConcur = observeConcurrency
		logrus.Infof("Auto-setting --max-concurrency=%d to match --concurrency", observeConcurrency)
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
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, observeNoStreaming, observeMaxConcur, observeWarmup, prefixes, prefixLengths, observeUnconstrainedOutput, observeMinTokens, observeRecordITL, tokensPerWord)
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
	} else if observeWorkload != "" {
		header.WorkloadSpec = "preset:" + observeWorkload
	}
	if spec != nil {
		header.WorkloadSeed = &spec.Seed
	}

	if err := recorder.Export(header, observeTraceHeader, observeTraceData); err != nil {
		logrus.Fatalf("Failed to export trace: %v", err)
	}

	records := recorder.Records()
	logrus.Infof("Trace exported: %d records to %s / %s", len(records), observeTraceHeader, observeTraceData)

	// Export ITL if requested (BC-5: opt-in)
	if observeRecordITL {
		itlPath := observeITLOutput
		if itlPath == "" {
			// Default: <trace-data>.itl.csv (strip .csv extension to avoid trace.csv.itl.csv)
			itlPath = strings.TrimSuffix(observeTraceData, ".csv") + ".itl.csv"
		}

		itlRecords := recorder.ITLRecords()
		if len(itlRecords) == 0 {
			logrus.Warnf("--record-itl was set but no ITL data recorded (non-streaming requests?)")
		}

		if err := recorder.ExportITL(itlPath); err != nil {
			logrus.Fatalf("Failed to export ITL data: %v", err)
		}
		logrus.Infof("ITL data exported: %s (%d records)", itlPath, len(itlRecords))
	}
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
	noStreaming bool,
	maxConcurrency int,
	warmupCount int,
	prefixes map[string]string,
	prefixLengths map[string]int,
	unconstrained bool,
	minTokens int,
	recordITL bool,
	tokensPerWord float64,
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

		pending := requestToPending(req, idx, noStreaming, unconstrained, minTokens, prefixes, prefixLengths, tokensPerWord)
		record, sendErr := client.Send(ctx, pending)
		if sendErr != nil {
			logrus.Warnf("request %d: Send returned error: %v", idx, sendErr)
		}

		// Record trace (skip warmup by index)
		arrivalTimeUs := req.ArrivalTime
		if idx >= warmupCount {
			recorder.RecordRequest(pending, record, arrivalTimeUs, req.SessionID, req.RoundIndex)

			// Record ITL if requested (BC-1, BC-2, BC-7)
			if recordITL && record.Status == "ok" && len(record.ChunkTimestamps) > 0 {
				recorder.RecordITL(record.RequestID, record.ChunkTimestamps)
			} else if recordITL && !pending.Streaming {
				// BC-2: warn if ITL requested for non-streaming
				logrus.Warnf("request %d: --record-itl was set but request is non-streaming (NumChunks=1)", record.RequestID)
			}
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

// tokensToPrompt converts token IDs into a diverse prompt string using
// prefixVocabulary. Each token ID selects a vocabulary word via modular
// indexing, ensuring different token arrays produce different prompts.
func tokensToPrompt(tokens []int, wordCount int) string {
	vocabLen := len(prefixVocabulary)
	var b strings.Builder
	b.Grow(wordCount * 8) // average word ~7 chars + space
	for i := 0; i < wordCount; i++ {
		var idx int
		if i < len(tokens) {
			idx = tokens[i]
		} else {
			idx = i
		}
		b.WriteString(prefixVocabulary[((idx%vocabLen)+vocabLen)%vocabLen])
		b.WriteByte(' ')
	}
	return b.String()
}

// requestToPending converts a sim.Request to a PendingRequest for HTTP dispatch.
// prefixes maps prefix-group name to a pre-built prefix string; prefixLengths maps
// prefix-group name to the target token count for the prefix (not word count;
// see buildPrefixStrings). Both may be nil if no prefix groups exist.
// tokensPerWord is the calibrated ratio from calibratePrefixTokenRatio; it scales
// word count so the server tokenizes the prompt to approximately len(InputTokens) tokens.
func requestToPending(req *sim.Request, reqIndex int, noStreaming, unconstrained bool, minTokens int, prefixes map[string]string, prefixLengths map[string]int, tokensPerWord float64) *PendingRequest {
	// Scale token count to word count using calibrated ratio (BC-3, BC-6).
	if tokensPerWord <= 0 {
		tokensPerWord = 1.0
	}
	wordCount := int(math.Round(float64(len(req.InputTokens)) / tokensPerWord))
	if wordCount <= 0 {
		wordCount = 1
	}

	var prompt string
	if req.PrefixGroup != "" && prefixes != nil {
		if prefix, ok := prefixes[req.PrefixGroup]; ok {
			prefixLen := prefixLengths[req.PrefixGroup]
			suffixTokens := len(req.InputTokens) - prefixLen
			if suffixTokens < 1 {
				suffixTokens = 1
			}
			suffixWords := int(math.Round(float64(suffixTokens) / tokensPerWord))
			if suffixWords < 1 {
				suffixWords = 1
			}
			suffixStart := len(req.InputTokens) - suffixTokens
			if suffixStart < 0 {
				suffixStart = 0
			}
			if suffixStart > len(req.InputTokens) {
				suffixStart = len(req.InputTokens)
			}
			prompt = prefix + tokensToPrompt(req.InputTokens[suffixStart:], suffixWords)
		} else {
			prompt = tokensToPrompt(req.InputTokens, wordCount)
		}
	} else {
		prompt = tokensToPrompt(req.InputTokens, wordCount)
	}

	return &PendingRequest{
		RequestID:       reqIndex,
		InputTokens:     len(req.InputTokens),
		MaxOutputTokens: req.MaxOutputLen,
		Model:           req.Model,
		Streaming:       req.Streaming && !noStreaming,
		ClientID:        req.ClientID,
		TenantID:        req.TenantID,
		SLOClass:        req.SLOClass,
		PrefixGroup:     req.PrefixGroup,
		PrefixLength:    req.PrefixLength,
		Prompt:          prompt,
		Unconstrained:   unconstrained,
		MinTokens:       minTokens,
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
	if err != nil || record == nil {
		msg := "unknown"
		if err != nil {
			msg = err.Error()
		}
		logrus.Warnf("Prefix token calibration failed (%s); using 1:1 word-to-token ratio", msg)
		return 1.0
	}
	if record.Status != "ok" {
		msg := record.ErrorMessage
		if msg == "" {
			msg = "status=" + record.Status
		}
		logrus.Warnf("Prefix token calibration failed (%s); using 1:1 word-to-token ratio", msg)
		return 1.0
	}
	if record.ServerInputTokens <= 0 {
		logrus.Warnf("Prefix token calibration failed (server returned 0 prompt_tokens — check that usage reporting is enabled); using 1:1 word-to-token ratio")
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
		tpw := tokensPerWord
		if tpw <= 0 {
			tpw = 1.0
		}
		wordCount := int(math.Round(float64(length) / tpw))
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
