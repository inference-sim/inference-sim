package cmd

// DEV-6 (design §8.4): every random draw comes from a substream named by component, purpose and
// entity, so a request gets the same random inputs whatever happened to earlier requests, in
// every configuration. This test changes one server setting under a closed-loop session workload,
// which moves every completion and so every follow-up's arrival, and checks that every round both
// runs produced has the same prompt and output lengths, byte for byte.

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// dev6SessionSpec is a closed-loop multi-turn workload: four users, each one session of six
// rounds, whose follow-ups the session manager generates when the previous round completes.
const dev6SessionSpec = `version: "2"
category: language
aggregate_rate: 8.0
num_requests: 24
cohorts:
  - id: user
    population: 4
    tenant_id: t
    slo_class: batch
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params: { mean: 160, std_dev: 60, min: 16, max: 600 }
    output_distribution:
      type: exponential
      params: { mean: 64 }
    reasoning:
      multi_turn:
        max_rounds: 6
        context_growth: accumulate
        think_time_us: 40000
        single_session: true
`

// roundKey names one round of one session.
type roundKey struct {
	session string
	round   int
}

// roundRecord is what the trace states of a round: its random inputs and its arrival.
type roundRecord struct {
	input, output int
	arrival       int64
}

// runRounds runs dev6SessionSpec with the given instance count and returns every round of the
// exported trace, by session and round.
func runRounds(t *testing.T, instances int) map[roundKey]roundRecord {
	t.Helper()
	tmpDir := t.TempDir()
	tracePrefix := filepath.Join(tmpDir, "trace")
	specPath := filepath.Join(tmpDir, "workload.yaml")
	if err := os.WriteFile(specPath, []byte(dev6SessionSpec), 0644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	catalogDir, defaultsPath := setupKernelTestFixturesWithDefaults(t)
	orig := captureCmdLevelVars()
	defer orig.restore()
	workloadType = ""
	requestTimeoutSecs = 300

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "")
	testCmd.Flags().StringVar(&traceOutput, "trace-output", "", "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	if err := testCmd.ParseFlags([]string{
		"--defaults-filepath", defaultsPath,
		"--catalog", catalogDir, "--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t), "--registry", kernelRegistryDir(t),
		"--seed", "7", "--horizon", "120000000",
		"--num-instances", fmt.Sprint(instances),
		"--workload-spec", specPath,
		"--trace-output", tracePrefix,
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	runCmd.Run(testCmd, nil)
	tr, err := workload.LoadTraceV2(tracePrefix+".yaml", tracePrefix+".csv")
	if err != nil {
		t.Fatalf("load trace: %v", err)
	}
	out := map[roundKey]roundRecord{}
	for _, r := range tr.Records {
		k := roundKey{r.SessionID, r.RoundIndex}
		if _, dup := out[k]; dup {
			t.Fatalf("session %s round %d appears twice in the trace", r.SessionID, r.RoundIndex)
		}
		out[k] = roundRecord{input: r.InputTokens, output: r.OutputTokens, arrival: r.ArrivalTimeUs}
	}
	return out
}

func TestDEV6_ServerSettingDoesNotMoveAnyRoundsRandomInputs(t *testing.T) {
	one, two := runRounds(t, 1), runRounds(t, 2)
	shared, movedArrivals, followUps := 0, 0, 0
	for k, a := range one {
		b, ok := two[k]
		if !ok {
			continue
		}
		shared++
		if k.round > 0 {
			followUps++
		}
		if a.arrival != b.arrival {
			movedArrivals++
		}
		if a.input != b.input || a.output != b.output {
			t.Errorf("session %s round %d: 1 instance drew input %d output %d, 2 instances drew input %d output %d",
				k.session, k.round, a.input, a.output, b.input, b.output)
		}
	}
	// Non-vacuity: the setting must have moved follow-up arrivals (else the test proves nothing
	// about closed-loop timing), and most rounds must be in both runs.
	if followUps < 12 {
		t.Fatalf("only %d follow-up rounds ran in both configurations; the workload no longer exercises sessions", followUps)
	}
	if movedArrivals == 0 {
		t.Fatalf("no round arrived at a different time with 1 and 2 instances; the setting did not change the timing")
	}
	t.Logf("%d rounds in both runs (%d follow-ups), %d arrived at different times", shared, followUps, movedArrivals)
}
