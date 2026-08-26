package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/spf13/cobra"
)

// replayTraceWithNetwork replays a TraceV2 through replayCmd at the given TP and
// inter-node network config (#1530). When gpusPerNodeVal == 0 the network flags are
// omitted (inert). Modeled on replaySpecTrace; adds --tp / --gpus-per-node /
// --inter-node-bandwidth so INV-13 parity can be exercised for a cross-node config.
//
// NOTE: mutates package-level CLI vars; not for t.Parallel().
func replayTraceWithNetwork(t *testing.T, traceHeaderFile, traceDataFile string, tpVal, gpusPerNodeVal int, bwGBps float64) []workload.SimResult {
	t.Helper()
	tmpDir := t.TempDir()
	resultsFile := filepath.Join(tmpDir, "results.json")
	mcFolder, hwPath, defaultsPath := setupTrainedPhysicsTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()

	origTraceHeader := traceHeaderPath
	origTraceData := traceDataPath
	origSessionMode := replaySessionMode
	origGpusPerNode := gpusPerNode
	origInterNodeBw := interNodeBandwidth
	origInterNodeLat := interNodeLatency
	defer func() {
		traceHeaderPath = origTraceHeader
		traceDataPath = origTraceData
		replaySessionMode = origSessionMode
		gpusPerNode = origGpusPerNode
		interNodeBandwidth = origInterNodeBw
		interNodeLatency = origInterNodeLat
	}()

	model = "qwen/qwen3-14b"
	latencyModelBackend = "trained-physics"
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxNumSeqs = 64
	maxNumBatchedTokens = 2048
	numInstances = 1
	resultsPath = resultsFile
	longPrefillTokenThreshold = 0
	kvCPUBlocks = 0
	kvOffloadThreshold = 0.9
	kvTransferBandwidth = 100.0
	kvTransferBaseLatency = 0
	snapshotRefreshInterval = 0
	admissionPolicy = "always-admit"
	routingPolicy = "round-robin"
	scheduler = "fcfs"
	policyConfigPath = ""
	maxModelLen = 0
	traceLevel = "none"
	counterfactualK = 0
	traceHeaderPath = traceHeaderFile
	traceDataPath = traceDataFile
	modelConfigFolder = mcFolder
	hwConfigPath = hwPath
	gpu = "H100"
	tensorParallelism = tpVal
	defaultsFilePath = defaultsPath
	replaySessionMode = "fixed"

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd) // resets gpusPerNode/interNodeBandwidth/interNodeLatency to 0
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	testCmd.Flags().StringVar(&resultsPath, "results-path", "", "")
	args := []string{
		"--model", "qwen/qwen3-14b", "--latency-model", "trained-physics",
		"--total-kv-blocks", "1000", "--hardware", "H100",
		"--tp", strconv.Itoa(tpVal),
		"--model-config-folder", mcFolder, "--hardware-config", hwPath,
		"--trace-header", traceHeaderFile, "--trace-data", traceDataFile,
		"--results-path", resultsFile,
		"--num-instances", "1",
		"--horizon", "120000000",
		"--defaults-filepath", defaultsPath,
	}
	if gpusPerNodeVal > 0 {
		args = append(args,
			"--gpus-per-node", strconv.Itoa(gpusPerNodeVal),
			"--inter-node-bandwidth", strconv.FormatFloat(bwGBps, 'f', -1, 64),
		)
	}
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	replayCmd.Run(testCmd, nil)

	data, err := os.ReadFile(resultsFile)
	if err != nil {
		t.Fatalf("results file not written: %v", err)
	}
	var results []workload.SimResult
	if err := json.Unmarshal(data, &results); err != nil {
		t.Fatalf("parse SimResult JSON: %v", err)
	}
	return results
}

// TestResolveNetworkConfig_HappyPaths pins the CLI seam that both run and replay
// use (BC-6 non-fatal cases): an inert config passes on any backend, and an active,
// well-formed config on trained-physics returns the configured fabric. The fatal
// paths (invalid fabric config, active-on-roofline) delegate to NetworkConfig.Validate
// and the trained-physics backend check, covered by the sim/latency guard tests.
//
// NOTE: mutates package-level vars; do NOT use t.Parallel().
func TestResolveNetworkConfig_HappyPaths(t *testing.T) {
	o1, o2, o3 := gpusPerNode, interNodeBandwidth, interNodeLatency
	defer func() { gpusPerNode, interNodeBandwidth, interNodeLatency = o1, o2, o3 }()

	// Inert default: valid on any backend, contributes nothing.
	gpusPerNode, interNodeBandwidth, interNodeLatency = 0, 0, 0
	if nc := resolveNetworkConfig("roofline"); nc.IsActive() {
		t.Errorf("inert config must not be active, got %+v", nc)
	}

	// Active, well-formed config on trained-physics: returned verbatim.
	gpusPerNode, interNodeBandwidth, interNodeLatency = 4, 50.0, 0.001
	nc := resolveNetworkConfig("trained-physics")
	if !nc.IsActive() || nc.GPUsPerNode != 4 || nc.InterNodeBandwidthGBps != 50.0 || nc.InterNodeLatencyMs != 0.001 {
		t.Errorf("resolveNetworkConfig returned %+v, want {4, 50, 0.001} active", nc)
	}
}

// TestParity_InterNodeNetwork_RunReplay_INV13 is BC-5: replaying a workload trace
// with a cross-node network config is deterministic (INV-13/INV-6), and replay
// HONORS the network flags rather than silently dropping them — a cross-node
// replay differs from an inert one, and never completes faster (the cross-node
// term only adds cost). Uses TP=2 with --gpus-per-node 1 so the TP all-reduce
// group (2) spans 2 nodes.
//
// NOTE: mutates package-level vars; do NOT use t.Parallel().
func TestParity_InterNodeNetwork_RunReplay_INV13(t *testing.T) {
	const seed int64 = 20260826
	shape := paritySpecShapes()[0] // chatbot

	// A single workload trace (mode: generated); replay recomputes timing from the
	// latency config, so the same trace drives both the cross-node and inert legs.
	hdr, data := runSpecToTraceFiles(t, shape.yaml, seed, shape.horizon, false)

	withNet := replayTraceWithNetwork(t, hdr, data, 2, 1, 50.0)
	withNet2 := replayTraceWithNetwork(t, hdr, data, 2, 1, 50.0)
	inert := replayTraceWithNetwork(t, hdr, data, 2, 0, 0)

	if len(withNet) == 0 {
		t.Fatal("cross-node replay produced no completed requests")
	}

	// Determinism: identical cross-node config ⇒ byte-identical per-request metrics.
	wTTFT, wE2E := simResultMaps(withNet)
	w2TTFT, w2E2E := simResultMaps(withNet2)
	for id, v := range wTTFT {
		if w2TTFT[id] != v {
			t.Errorf("INV-6: request %d TTFT non-deterministic across identical cross-node replays: %f vs %f", id, v, w2TTFT[id])
		}
		if w2E2E[id] != wE2E[id] {
			t.Errorf("INV-6: request %d E2E non-deterministic: %f vs %f", id, wE2E[id], w2E2E[id])
		}
	}

	// Replay honors the network flags: a cross-node replay must differ from an inert
	// one (if replay silently dropped --gpus-per-node, they would be identical), and
	// the cross-node term only ADDS cost, so no request completes earlier.
	iTTFT, iE2E := simResultMaps(inert)
	if len(iTTFT) != len(wTTFT) {
		t.Fatalf("cross-node and inert replays completed different request counts: %d vs %d", len(wTTFT), len(iTTFT))
	}
	anyDifferent := false
	for id, wv := range wE2E {
		iv, ok := iE2E[id]
		if !ok {
			t.Errorf("request %d present in cross-node replay, missing from inert replay", id)
			continue
		}
		if wv != iv {
			anyDifferent = true
		}
		if wv < iv {
			t.Errorf("request %d: cross-node E2E (%f) < inert E2E (%f) — the network term must only add cost", id, wv, iv)
		}
	}
	if !anyDifferent {
		t.Error("cross-node replay produced identical metrics to inert replay — replay is not honoring --gpus-per-node/--inter-node-bandwidth (INV-13)")
	}
}
