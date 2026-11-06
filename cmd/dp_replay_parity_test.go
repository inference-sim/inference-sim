package cmd

// CLI-level run/replay parity for a data-parallel deployment, on the dp=2 expert-parallel MoE
// scenario (dpScenario). Each logical instance is one engine, the whole data-parallel group:
//
//   - BC-1: run and replay each build --num-instances instances, one per group (the Result
//     lists instances only when there are two or more).
//   - BC-2 (INV-13): a trace exported by `blis run` replayed with the same flags produces
//     byte-identical stdout.
//   - BC-5 (INV-6): replay is byte-identical run to run.
//   - BC-6 (INV-1): request conservation holds across the groups.
//
// Every leg re-execs this test binary as a real `blis` command (runKernelCLI).

import (
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

// clusterMetricInt is a whole-run cluster figure, by the name the MetricsOutput JSON gave it,
// read from the Result document (clusterFigures maps each name onto its home).
func clusterMetricInt(t *testing.T, stdout, field string) int {
	t.Helper()
	v, ok := clusterFigures(t, stdout)[field]
	if !ok {
		t.Fatalf("the Result has no cluster figure %q", field)
	}
	return int(v)
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

// TestINV13_RunReplayParity_MoEDP is BC-2: a trace exported by a dp=2 run, replayed
// with the same flags, reproduces the run byte for byte -- over one and two logical instances.
func TestINV13_RunReplayParity_MoEDP(t *testing.T) {
	for _, n := range []int{1, 2} {
		t.Run(fmt.Sprintf("%d-logical", n), func(t *testing.T) {
			runOut, prefix := dpExportTrace(t, n)
			if completed := clusterMetricInt(t, runOut, "completed_requests"); completed <= 0 {
				t.Fatalf("parity check would be vacuous: the run completed %d requests", completed)
			}
			clusterConservationHolds(t, runOut, dpFixtureNumRequests)
			if ids := instanceIDs(t, runOut); n > 1 && len(ids) != n {
				t.Fatalf("%d logical instances at dp 2 must run %d data-parallel groups, got %v", n, n, ids)
			}
			replayOut := dpReplay(t, prefix+".yaml", prefix+".csv", n)
			if a, b := inv13View(t, runOut), inv13View(t, replayOut); a != b {
				t.Errorf("INV-13: the replay of a dp=2 run's trace differs from the run\n--- run\n%s\n--- replay\n%s",
					a, b)
			}
		})
	}
}

// TestReplayCmd_MoEDP_OneGroupPerInstance is BC-1 + BC-5 + BC-6 on a trace this binary did not
// produce, as a converted corpus is: replay builds one data-parallel group per instance,
// conserves the trace's requests, and is deterministic.
func TestReplayCmd_MoEDP_OneGroupPerInstance(t *testing.T) {
	dir := t.TempDir()
	header := filepath.Join(dir, "trace.yaml")
	data := filepath.Join(dir, "trace.csv")
	if err := os.WriteFile(header, []byte("trace_version: 4\ntime_unit: microseconds\nmode: generated\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	csv := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length," +
		"streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio," +
		"model,deadline_us,arrival_time_us,first_chunk_time_us,last_chunk_time_us,status\n"
	const rows = 6
	for i := 0; i < rows; i++ {
		arrival := i * 50000
		csv += fmt.Sprintf("%d,c1,t1,standard,,0,,0,false,64,16,64,0,0,0,0.0,,0,%d,0,0,ok\n", i, arrival)
	}
	if err := os.WriteFile(data, []byte(csv), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, n := range []int{1, 2} {
		out := dpReplay(t, header, data, n)
		if ids := instanceIDs(t, out); n > 1 && len(ids) != n {
			t.Errorf("BC-1: replay over %d logical instances at dp 2 must run %d groups, got %v", n, n, ids)
		}
		clusterConservationHolds(t, out, rows)
		if again := dpReplay(t, header, data, n); withoutRuntime(t, again) != withoutRuntime(t, out) {
			t.Errorf("INV-6: two identical dp=2 replays over %d logical instances differ", n)
		}
	}
}
