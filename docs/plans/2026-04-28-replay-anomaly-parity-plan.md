# fix(replay): anomaly block missing TimedOutRequests, GatewayQueueDepth, GatewayQueueShed

**Goal:** Fix `cmd/replay.go` anomaly trigger + print block to match `cmd/root.go` parity for three missing counters.
**Source:** GitHub issue #1184
**Closes:** #1184
**Tier:** Small (1 file changed, bug fix, no new interfaces/flags)

No clarifications needed — issue provides exact line numbers and exact code to add.

---

## Behavioral Contracts

**BC-1:** GIVEN a `blis replay` run where `rawMetrics.TimedOutRequests > 0`,
WHEN simulation ends,
THEN stdout includes the "=== Anomaly Counters ===" header and a line "Timed Out Requests: N" between "Dropped Unservable" and "Length-Capped Requests".

**BC-2:** GIVEN a `blis replay` run where `rawMetrics.GatewayQueueDepth > 0`,
WHEN simulation ends,
THEN stdout includes "Gateway Queue Depth (horizon): N" after "Length-Capped Requests".

**BC-3:** GIVEN a `blis replay` run where `rawMetrics.GatewayQueueShed > 0`,
WHEN simulation ends,
THEN stdout includes "Gateway Queue Shed: N" after "Length-Capped Requests".

**BC-4 (parity / R23):** GIVEN the same `cluster.RawMetrics` values,
WHEN the anomaly block is printed by `blis replay`,
THEN the output section matches what `blis run` would produce for the same fields.

---

## Tasks

### Task 1 — Write failing test for TimedOutRequests anomaly (BC-1)

**Test:** `TestReplayCmd_AnomalyBlock_TimedOutRequests` in `cmd/replay_test.go`

**Approach:** Use `os.Pipe()` to capture stdout. Build a 1-request trace with
`deadline_us=1` (expires at t=1µs — before any real simulation step) and run
`replayCmd.Run(testCmd, nil)`. Assert output contains "Timed Out Requests: 1".

```go
func TestReplayCmd_AnomalyBlock_TimedOutRequests(t *testing.T) {
    // GIVEN a trace with deadline_us=1 that forces a timeout
    dir := t.TempDir()
    headerPath := filepath.Join(dir, "trace.yaml")
    dataPath := filepath.Join(dir, "trace.csv")

    headerContent := "trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"
    if err := os.WriteFile(headerPath, []byte(headerContent), 0644); err != nil {
        t.Fatal(err)
    }
    // deadline_us=1 forces timeout (expires before any execution step)
    csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
        "0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,1,0,0,0,0,0,0,ok,,\n"
    if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
        t.Fatal(err)
    }

    mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)

    // save/restore package-level vars (same pattern as other replay tests)
    // [full save/restore block — same as TestReplayCmd_EndToEnd_TrainedPhysicsMode]
    origModel := model
    /* ... (all package-level var saves) ... */
    defer func() {
        model = origModel
        /* ... (all restores) ... */
    }()

    model = "test-model"
    latencyModelBackend = "trained-physics"
    totalKVBlocks = 1000
    blockSizeTokens = 16
    maxRunningReqs = 64
    maxScheduledTokens = 2048
    numInstances = 1
    seed = 42
    resultsPath = ""
    longPrefillTokenThreshold = 0
    kvCPUBlocks = 0
    kvOffloadThreshold = 0.9
    kvTransferBandwidth = 100.0
    kvTransferBaseLatency = 0
    snapshotRefreshInterval = 0
    admissionPolicy = "always-admit"
    routingPolicy = "round-robin"
    priorityPolicy = "constant"
    scheduler = "fcfs"
    policyConfigPath = ""
    maxModelLen = 0
    traceLevel = "none"
    counterfactualK = 0
    traceHeaderPath = headerPath
    traceDataPath = dataPath
    simulationHorizon = math.MaxInt64
    replayTraceOutput = ""
    modelConfigFolder = mcFolder
    hwConfigPath = hwPath
    gpu = "H100"
    tensorParallelism = 1
    defaultsFilePath = "../defaults.yaml"
    cacheSignalDelay = 0
    flowControlEnabled = false
    flowControlDetector = "utilization"
    flowControlDispatchOrder = "fifo"
    flowControlMaxQueueDepth = 0
    flowControlQueueDepthThreshold = 5.0
    flowControlKVCacheUtilThreshold = 0.8
    flowControlMaxConcurrency = 0

    testCmd := &cobra.Command{}
    registerSimConfigFlags(testCmd)
    testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
    testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
    if err := testCmd.ParseFlags([]string{
        "--model", "test-model",
        "--latency-model", "trained-physics",
        "--total-kv-blocks", "1000",
        "--hardware", "H100",
        "--tp", "1",
        "--model-config-folder", mcFolder,
        "--hardware-config", hwPath,
        "--trace-header", headerPath,
        "--trace-data", dataPath,
        "--defaults-filepath", "../defaults.yaml",
    }); err != nil {
        t.Fatalf("ParseFlags failed: %v", err)
    }

    // Capture stdout
    r, w, err := os.Pipe()
    if err != nil {
        t.Fatal(err)
    }
    origStdout := os.Stdout
    os.Stdout = w
    replayCmd.Run(testCmd, nil)
    w.Close()
    os.Stdout = origStdout
    var buf strings.Builder
    if _, err := io.Copy(&buf, r); err != nil {
        t.Fatal(err)
    }
    out := buf.String()

    // THEN: anomaly block fires and "Timed Out Requests: 1" appears (BC-1)
    if !strings.Contains(out, "=== Anomaly Counters ===") {
        t.Errorf("BC-1: expected anomaly block header, got output:\n%s", out)
    }
    if !strings.Contains(out, "Timed Out Requests: 1") {
        t.Errorf("BC-1: expected 'Timed Out Requests: 1' in anomaly block, got:\n%s", out)
    }
}
```

**Run to fail:**
```bash
cd /Users/sri/Documents/Projects/inference-sim/.worktrees/pr1185-replay-anomaly-parity
go test ./cmd/... -run TestReplayCmd_AnomalyBlock_TimedOutRequests -v -count=1
```
Expected: FAIL — "Timed Out Requests: 1" not found (trigger condition doesn't include `TimedOutRequests > 0`)

**Commit:** `test(cmd): failing test for replay TimedOutRequests anomaly (BC-1)`

---

### Task 2 — Fix replay.go anomaly block and make test pass (BC-1, BC-2, BC-3, BC-4)

**File:** `cmd/replay.go`

**Change 1 — Populate GatewayQueue fields (after line 290):**
```go
rawMetrics.ShedByTier = cs.ShedByTier()                             // Phase 1B-1a: tier-shed per-tier breakdown (SC-004)
rawMetrics.GatewayQueueDepth = cs.GatewayQueueDepth()               // Issue #882: gateway queue depth at horizon
rawMetrics.GatewayQueueShed = cs.GatewayQueueShed()                 // Issue #882: gateway queue shed count
```

**Change 2 — Extend trigger condition (line 293):**
```go
if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.RoutingRejections > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 || rawMetrics.TimedOutRequests > 0 || rawMetrics.GatewayQueueDepth > 0 || rawMetrics.GatewayQueueShed > 0 {
```

**Change 3 — Add Timed Out Requests print (between "Dropped Unservable" and "Length-Capped Requests"):**
```go
fmt.Printf("Dropped Unservable: %d\n", rawMetrics.DroppedUnservable)
fmt.Printf("Timed Out Requests: %d\n", rawMetrics.TimedOutRequests)
fmt.Printf("Length-Capped Requests: %d\n", rawMetrics.LengthCappedRequests)
```

**Change 4 — Add conditional GatewayQueue prints (after "Length-Capped Requests"):**
```go
fmt.Printf("Length-Capped Requests: %d\n", rawMetrics.LengthCappedRequests)
if rawMetrics.GatewayQueueDepth > 0 {
    fmt.Printf("Gateway Queue Depth (horizon): %d\n", rawMetrics.GatewayQueueDepth)
}
if rawMetrics.GatewayQueueShed > 0 {
    fmt.Printf("Gateway Queue Shed: %d\n", rawMetrics.GatewayQueueShed)
}
```

**Run to pass:**
```bash
go test ./cmd/... -run TestReplayCmd_AnomalyBlock_TimedOutRequests -v -count=1
```
Expected: PASS

**Lint:**
```bash
golangci-lint run ./cmd/...
```
Expected: 0 issues

**Run all tests:**
```bash
go test ./... -count=1
```
Expected: all PASS

**Commit:**
```
fix(replay): add TimedOutRequests, GatewayQueueDepth, GatewayQueueShed to anomaly block (R23, #1184)

- BC-1: TimedOutRequests > 0 triggers anomaly block and prints count
- BC-2: GatewayQueueDepth > 0 triggers anomaly block and prints count
- BC-3: GatewayQueueShed > 0 triggers anomaly block and prints count
- BC-4: replay anomaly output now matches run.go parity (R23)

Fixes #1184
```

---

## Sanity Checklist

- [ ] `rawMetrics.GatewayQueueDepth = cs.GatewayQueueDepth()` and `rawMetrics.GatewayQueueShed = cs.GatewayQueueShed()` added (matching root.go:1603-1604)
- [ ] Trigger condition includes `|| rawMetrics.TimedOutRequests > 0 || rawMetrics.GatewayQueueDepth > 0 || rawMetrics.GatewayQueueShed > 0`
- [ ] Print order matches root.go: ... Dropped Unservable → **Timed Out** → Length-Capped → [conditional] GatewayQueueDepth → [conditional] GatewayQueueShed
- [ ] `go test ./... -count=1` passes
- [ ] `golangci-lint run ./...` clean
