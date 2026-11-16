package cmd

import (
	"maps"
	"slices"
	"strings"
	"testing"
)

// The scenario is the only configuration input (#1602, #2037, #1451): every flag that restated a
// scenario field is gone from run and replay. Given through the real command line, in either
// spelling, each one is refused before anything runs, with a message that names the release
// that removed it and the scenario field that states its setting now. The closest accepted
// command, the same one without the flag, runs (SIM-9).
func TestRemovedFlags_AreRefusedNamingTheScenarioField(t *testing.T) {
	scenarios, _, _ := kernelRepos(t)
	names := map[string]string{
		"scheduler":            "engine.scheduling_policy",
		"preemption-policy":    "engine.scheduling_policy",
		"dp-load-balancing":    "parallel.data_parallel_load_balancing",
		"swa-bounded-replay":   "engine.swa_bounded_replay",
		"kv-offload-config":    "offload block",
		"kv-cpu-blocks":        "cpu_dram tier in the offload block",
		"kv-offload-threshold": "offload block",
		"lazy-generation":      "always streamed",
	}
	values := map[string]string{
		"scheduler": "fcfs", "preemption-policy": "fcfs", "dp-load-balancing": "internal",
		"swa-bounded-replay": "false", "kv-offload-config": "offload.yaml", "kv-cpu-blocks": "100",
		"kv-offload-threshold": "0.9", "lazy-generation": "false",
	}
	run := []string{"run", "--scenarios", scenarios, "--scenario", colocatedFixture,
		"--num-requests", "4", "--rate", "4", "--seed", "3"}
	replay := []string{"replay", "--scenarios", scenarios, "--scenario", colocatedFixture,
		"--trace-header", "absent.yaml", "--trace-data", "absent.csv", "--seed", "3"}
	for _, name := range slices.Sorted(maps.Keys(names)) {
		field := names[name]
		for _, cmdline := range [][]string{run, replay} {
			for _, spelling := range [][]string{{"--" + name, values[name]}, {"--" + name + "=" + values[name]}} {
				args := append(append([]string{}, cmdline...), spelling...)
				stdout, stderr, err := runKernelCLI(t, args...)
				if err == nil {
					t.Errorf("%s %v: accepted a removed flag\n%s", cmdline[0], spelling, lastLines(stdout, 3))
					continue
				}
				for _, want := range []string{"--" + name + " was removed in " + flagsRemovedIn, field} {
					if !strings.Contains(stderr, want) {
						t.Errorf("%s %v: the refusal lacks %q; stderr:\n%s", cmdline[0], spelling, want, lastLines(stderr, 4))
					}
				}
			}
		}
	}
	stdout, stderr, err := runKernelCLI(t, run...)
	if err != nil || !completedIs(stdout, 4) {
		t.Fatalf("the same run without a removed flag must run: %v\n%s", err, lastLines(stderr, 4))
	}
}
