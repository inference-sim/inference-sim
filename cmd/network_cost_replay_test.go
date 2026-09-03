// network_cost_replay_test.go — cross-path parity fences for the inter-node network
// cost (#1530).
//
// Cross-node collective traffic is charged to step time, and it can only arise from
// node-pool multi-node placement, which is `blis run`-only. Two fences keep that from
// degrading silently on the replay side, and each gets its own subprocess test because
// each closes a DIFFERENT hole:
//
//  1. a node_pools bundle handed to `blis replay` is rejected (the forward path);
//  2. a TRACE whose source run had a multi-node fleet is rejected (the backward path).
//
// Fence 2 exists because fence 1 does not cover it: dropping the node_pools section is
// the only way to replay such a trace at all, and replay would then reproduce the
// workload at single-node speed — measurably faster than the run that produced the
// trace, with nothing to indicate it. Verified before this feature: on main, that
// round-trip returned identical metrics, so the divergence is introduced here and is
// this feature's to fence (INV-13: a feature replay cannot reproduce must fail loudly,
// never degrade silently).
package cmd

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// minimalTraceHeaderYAML returns a valid one-record TraceV2 header/data pair, with an
// optional extra header line (used to inject max_nodes_spanned).
func writeMinimalTrace(t *testing.T, dir, extraHeaderLine string) (headerPath, dataPath string) {
	t.Helper()
	headerPath = filepath.Join(dir, "trace.yaml")
	dataPath = filepath.Join(dir, "trace.csv")
	header := "trace_version: 3\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n" + extraHeaderLine
	data := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message,finish_reason\n" +
		"0,c1,t1,standard,s1,0,,0,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,,\n"
	if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dataPath, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
	return headerPath, dataPath
}

// runReplayInSubprocess drives replayCmd.Run with a trace whose header carries
// extraHeaderLine, inside the subprocess half of a fatal-path test.
func runReplayInSubprocess(t *testing.T, extraHeaderLine string) {
	t.Helper()
	dir := t.TempDir()
	headerPath, dataPath := writeMinimalTrace(t, dir, extraHeaderLine)
	mcFolder, hwPath := setupTrainedPhysicsTestFixtures(t)

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", "test-model", "--latency-model", "trained-physics",
		"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
		"--model-config-folder", mcFolder, "--hardware-config", hwPath,
		"--trace-header", headerPath, "--trace-data", dataPath,
		"--defaults-filepath", "../defaults.yaml",
	}); err != nil {
		fmt.Fprintf(os.Stderr, "ParseFlags failed (test setup error): %v\n", err)
		os.Exit(2) // distinct from logrus.Fatalf exit code (1)
	}
	replaySessionMode = "fixed"
	resultsPath = ""
	replayTraceOutput = ""
	policyConfigPath = ""
	replayCmd.Run(testCmd, nil) // must Fatalf before here for the spanning case
	os.Exit(0)
}

// expectSubprocessFatal runs this test binary's named test as a subprocess and asserts it
// exited 1 (logrus.Fatalf) with a message containing want.
func expectSubprocessFatal(t *testing.T, testName, want string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run="+testName, "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("expected a non-zero exit, got 0; output:\n%s", out)
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("unexpected error type: %v", err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit code 1 (logrus.Fatalf), got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), want) {
		t.Errorf("fatal message should mention %q, got:\n%s", want, out)
	}
}

// TestReplayCmd_CrossNodeTraceFatal verifies fence 2: a trace whose header records that
// the source run placed an instance across multiple nodes is rejected, because replay
// cannot reconstruct a multi-node fleet and would otherwise silently model the same
// workload at single-node speed.
func TestReplayCmd_CrossNodeTraceFatal(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		runReplayInSubprocess(t, "max_nodes_spanned: 2\n")
		return
	}
	expectSubprocessFatal(t, "TestReplayCmd_CrossNodeTraceFatal", "max_nodes_spanned")
}

// TestReplayCmd_SingleNodeTraceAccepted is the negative control for fence 2: a header
// with no max_nodes_spanned — every trace a run without multi-node placement produces,
// including every trace written before this feature — replays normally. Without this,
// fence 2 could be over-broad and reject ordinary traces.
func TestReplayCmd_SingleNodeTraceAccepted(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		runReplayInSubprocess(t, "")
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestReplayCmd_SingleNodeTraceAccepted", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("a trace with no recorded multi-node placement must replay normally, got %v; output:\n%s", err, out)
	}
	if strings.Contains(string(out), "max_nodes_spanned") {
		t.Errorf("the cross-node fence must stay silent for a single-node trace; output:\n%s", out)
	}
}

// TestReplayCmd_SingleNodeSpanTraceAccepted verifies the boundary value: a header
// recording a span of exactly 1 (every instance on one node) is not a cross-node fleet
// and must replay normally.
func TestReplayCmd_SingleNodeSpanTraceAccepted(t *testing.T) {
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		runReplayInSubprocess(t, "max_nodes_spanned: 1\n")
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestReplayCmd_SingleNodeSpanTraceAccepted", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("a span of 1 is not a cross-node fleet and must replay normally, got %v; output:\n%s", err, out)
	}
	if strings.Contains(string(out), "max_nodes_spanned") {
		t.Errorf("the cross-node fence must stay silent for a span of 1; output:\n%s", out)
	}
}
