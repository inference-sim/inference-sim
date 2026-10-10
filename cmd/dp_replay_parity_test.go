package cmd

// CLI-level DP-as-placement parity for `blis replay` (issue #1556, follow-up to #1531).
// Both commands resolve the placement through the single shared resolveDPPlacement, so a
// scenario stating dp=N expands into --num-instances x N engine replicas on replay exactly as
// it does on run. These are the system-level laws (the pure-function contracts live in
// dp_placement_test.go), all on the vendored dp=2 expert-parallel MoE scenario (dpScenario):
//
//   - BC-1: replay expands the scenario's dp into --num-instances x dp engine replicas.
//   - BC-2 (INV-13): a trace exported by `blis run` replayed with the same flags produces
//     byte-identical stdout.
//   - BC-5 (INV-6): replay is byte-identical run to run.
//   - BC-6 (INV-1): request conservation holds across the expanded replicas.
//
// Every leg re-execs this test binary as a real `blis` command (runKernelCLI).

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// dpParityHorizon is passed explicitly to BOTH legs so the two commands agree on the
// simulation horizon: `blis run` defaults to math.MaxInt64 while `blis replay` derives one
// from the trace, a difference orthogonal to DP that would otherwise truncate the replay.
var dpParityHorizon = strconv.FormatInt(math.MaxInt64, 10)

// clusterMetricInt returns one integer field of the "cluster" aggregate metrics
// object in blis stdout. Used for non-vacuity checks (a parity assertion over two
// empty runs passes trivially). Reuses extractJSONObjects from dp_placement_test.go.
func clusterMetricInt(t *testing.T, stdout, field string) int {
	t.Helper()
	for _, raw := range extractJSONObjects(stdout) {
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(raw), &obj); err != nil {
			continue
		}
		if obj["instance_id"] != "cluster" {
			continue
		}
		v, ok := obj[field].(float64)
		if !ok {
			t.Fatalf("cluster metrics missing numeric field %q", field)
		}
		return int(v)
	}
	t.Fatalf("no cluster aggregate metrics object found in stdout:\n%s", stdout)
	return 0
}

// dpExportTrace runs the dp=2 scenario over numInstances logical instances and exports its
// trace, returning the run's stdout and the trace prefix.
func dpExportTrace(t *testing.T, numInstances int) (stdout, prefix string) {
	t.Helper()
	prefix = filepath.Join(t.TempDir(), "trace")
	out, stderr, err := runKernelCLI(t, dpRunArgs(numInstances, "--horizon", dpParityHorizon,
		"--trace-output", prefix)...)
	if err != nil {
		t.Fatalf("dp=2 run: %v\n%s", err, lastLines(stderr, 3))
	}
	return out, prefix
}

// dpReplay replays a trace on the dp=2 scenario over numInstances logical instances.
func dpReplay(t *testing.T, headerPath, dataPath string, numInstances int) string {
	t.Helper()
	out, stderr, err := runKernelCLI(t, "replay", "--scenario", dpScenario,
		"--trace-header", headerPath, "--trace-data", dataPath,
		"--num-instances", strconv.Itoa(numInstances), "--seed", "42", "--horizon", dpParityHorizon)
	if err != nil {
		t.Fatalf("dp=2 replay: %v\n%s", err, lastLines(stderr, 3))
	}
	return out
}

// TestINV13_RunReplayParity_MoEDPPlacement is BC-2: a trace exported by a dp=2 run, replayed
// with the same flags, reproduces the run byte for byte -- over one and two logical instances,
// so the replica expansion itself is part of what must agree.
func TestINV13_RunReplayParity_MoEDPPlacement(t *testing.T) {
	for _, n := range []int{1, 2} {
		t.Run(fmt.Sprintf("%d-logical", n), func(t *testing.T) {
			runOut, prefix := dpExportTrace(t, n)
			if completed := clusterMetricInt(t, runOut, "completed_requests"); completed <= 0 {
				t.Fatalf("parity check would be vacuous: the run completed %d requests", completed)
			}
			clusterConservationHolds(t, runOut, dpFixtureNumRequests)
			if ids := instanceIDs(t, runOut); len(ids) != 2*n {
				t.Fatalf("activation: %d logical x dp 2 must run %d replicas, got %v", n, 2*n, ids)
			}
			replayOut := dpReplay(t, prefix+".yaml", prefix+".csv", n)
			if replayOut != runOut {
				t.Errorf("INV-13: the replay of a dp=2 run's trace differs from the run\n--- run\n%s\n--- replay\n%s",
					runOut, replayOut)
			}
		})
	}
}

// TestReplayCmd_MoEDPPlacement_SpawnsReplicas is BC-1 + BC-5 + BC-6 on a trace this binary did
// not produce -- the realistic operator flow (an observe-origin or converted corpus): replay
// expands into numInstances x dp replicas, conserves the trace's requests, and is
// deterministic.
func TestReplayCmd_MoEDPPlacement_SpawnsReplicas(t *testing.T) {
	dir := t.TempDir()
	header := filepath.Join(dir, "trace.yaml")
	data := filepath.Join(dir, "trace.csv")
	if err := os.WriteFile(header, []byte("trace_version: 2\ntime_unit: microseconds\nmode: generated\nwarm_up_requests: 0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	csv := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length," +
		"streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio," +
		"model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us," +
		"last_chunk_time_us,num_chunks,status,error_message,finish_reason\n"
	const rows = 6
	for i := 0; i < rows; i++ {
		arrival := i * 50000
		csv += fmt.Sprintf("%d,c1,t1,standard,,0,,0,false,64,16,64,0,0,0,0.0,,0,0,%d,%d,0,0,0,ok,,\n", i, arrival, arrival)
	}
	if err := os.WriteFile(data, []byte(csv), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, n := range []int{1, 2} {
		out := dpReplay(t, header, data, n)
		if ids := instanceIDs(t, out); len(ids) != 2*n {
			t.Errorf("BC-1: replay over %d logical x dp 2 must run %d replicas, got %v", n, 2*n, ids)
		}
		clusterConservationHolds(t, out, rows)
		if again := dpReplay(t, header, data, n); again != out {
			t.Errorf("INV-6: two identical dp=2 replays over %d logical instances differ", n)
		}
	}
}
