package cmd

import (
	"context"
	"math"
	"os"
	"os/signal"
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

Example:
  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --workload-spec workload.yaml --trace-header trace.yaml --trace-data trace.csv

  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --rate 10 --num-requests 100 --trace-header trace.yaml --trace-data trace.csv`,
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
	if observeWorkloadSpec == "" && observeRate <= 0 {
		if !cmd.Flags().Changed("rate") && observeWorkloadSpec == "" {
			logrus.Fatalf("Either --workload-spec or --rate is required")
		}
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
	client := NewRealClient(observeServerURL, observeAPIKey, observeModel, observeServerType)
	recorder := &Recorder{}

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
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, streaming, observeMaxConcur, observeWarmup)
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

		pending := requestToPending(req, idx, streaming)
		record, _ := client.Send(ctx, pending)

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
		var isFollowUp bool

		hasPreGen := preGenIdx < len(requests)
		hasFollowUp := len(pendingFollowUps) > 0

		if hasPreGen && hasFollowUp {
			if pendingFollowUps[0].ArrivalTime <= requests[preGenIdx].ArrivalTime {
				nextReq = pendingFollowUps[0]
				pendingFollowUps = pendingFollowUps[1:]
				isFollowUp = true
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
			isFollowUp = true
		} else if sessionMgr != nil && atomic.LoadInt64(&activeSessionCount) > 0 {
			// No pre-generated or buffered follow-ups — wait for new follow-up or drain
			select {
			case fu, ok := <-followUpCh:
				if !ok {
					goto drain
				}
				nextReq = fu
				isFollowUp = true
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
		_ = isFollowUp // used for logging if needed

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
func requestToPending(req *sim.Request, reqIndex int, streaming bool) *PendingRequest {
	return &PendingRequest{
		RequestID:       reqIndex,
		InputTokens:     len(req.InputTokens),
		MaxOutputTokens: req.MaxOutputLen,
		Model:           req.Model,
		Streaming:       streaming,
		ClientID:        req.ClientID,
		TenantID:        req.TenantID,
		SLOClass:        req.SLOClass,
	}
}
