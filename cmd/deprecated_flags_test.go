package cmd

import (
	"strings"
	"testing"
)

// Each deprecated flag is recorded as deprecated on every command that registers it, with a
// notice naming the deprecating release, the removing release and the replacement (LIF-7), and
// stays visible in --help. A flag that is not deprecated carries no notice.
func TestDeprecatedFlags_AreMarkedOnRunAndReplay(t *testing.T) {
	for _, c := range []struct {
		name  string
		flags []string
	}{
		{"run", []string{"scheduler", "preemption-policy", "dp-load-balancing", "swa-bounded-replay", "kv-offload-config", "lazy-generation"}},
		{"replay", []string{"scheduler", "preemption-policy", "dp-load-balancing", "swa-bounded-replay", "kv-offload-config", "lazy-generation"}},
	} {
		cmd := findCommandByPath(t, rootCmd, []string{c.name})
		for _, name := range c.flags {
			f := cmd.Flags().Lookup(name)
			if f == nil {
				t.Fatalf("%s: --%s is not registered", c.name, name)
			}
			for _, want := range []string{"since " + flagsDeprecatedIn, "removed in " + flagsRemovedIn, "Instead, ", "#"} {
				if !strings.Contains(f.Deprecated, want) {
					t.Errorf("%s --%s: deprecation notice %q lacks %q", c.name, name, f.Deprecated, want)
				}
			}
			if f.Hidden {
				t.Errorf("%s --%s: a deprecated flag must stay in --help until it is removed", c.name, name)
			}
		}
		if f := cmd.Flags().Lookup("seed"); f == nil || f.Deprecated != "" {
			t.Errorf("%s --seed: must exist and carry no deprecation notice", c.name)
		}
	}
}

// A deprecated flag still works: given through the real command line it prints its notice on
// standard error, and standard output, the Result document, is byte for byte the output of the
// same run stating the same setting in the scenario (INV-6). A run without the flag prints no
// notice.
func TestDeprecatedFlags_NoticeOnStderrResultUnchanged(t *testing.T) {
	scenarios, _, _ := kernelRepos(t)
	common := []string{"run", "--scenarios", scenarios, "--scenario", colocatedFixture,
		"--num-requests", "6", "--rate", "4", "--seed", "3"}
	plain, plainErr, err := runKernelCLI(t, common...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(plainErr, 3))
	}
	if strings.Contains(plainErr, "has been deprecated") {
		t.Errorf("a run without a deprecated flag printed a deprecation notice:\n%s", lastLines(plainErr, 5))
	}
	// --scheduler fcfs restates the fixture's scheduling_policy: fcfs, so the simulation is
	// the same and only the notice differs.
	flagged, flaggedErr, err := runKernelCLI(t, append(common, "--scheduler", "fcfs")...)
	if err != nil {
		t.Fatalf("run --scheduler fcfs: %v\n%s", err, lastLines(flaggedErr, 3))
	}
	want := "Flag --scheduler has been deprecated, " + deprecationNotice(deprecatedFlags["scheduler"])
	if !strings.Contains(flaggedErr, want) {
		t.Errorf("stderr lacks the notice %q:\n%s", want, lastLines(flaggedErr, 5))
	}
	if strings.Contains(flagged, "has been deprecated") {
		t.Error("the deprecation notice reached standard output")
	}
	if simulated(t, plain) != simulated(t, flagged) {
		t.Errorf("--scheduler fcfs on an fcfs scenario changed what was simulated\n%s", firstDiff(simulated(t, plain), simulated(t, flagged)))
	}
}

// Streaming generation is the default (#1449). A run that omits --lazy-generation streams, and
// its Result is byte for byte the Result of the same run with --lazy-generation=false, which
// selects the deprecated eager generator and prints its notice on standard error (#1450). The
// default run prints no notice. Replay's --lazy-generation, which has no effect, carries its
// own notice.
func TestLazyGeneration_StreamingIsTheDefaultAndEagerIsDeprecated(t *testing.T) {
	scenarios, _, _ := kernelRepos(t)
	common := []string{"run", "--scenarios", scenarios, "--scenario", colocatedFixture,
		"--num-requests", "8", "--rate", "4", "--seed", "11", "--log", "info"}
	streamed, streamedErr, err := runKernelCLI(t, common...)
	if err != nil {
		t.Fatalf("run: %v\n%s", err, lastLines(streamedErr, 3))
	}
	if !strings.Contains(streamedErr, "Generated streaming workload source") {
		t.Errorf("a run without --lazy-generation must stream its requests:\n%s", lastLines(streamedErr, 8))
	}
	if strings.Contains(streamedErr, "has been deprecated") {
		t.Errorf("the default run printed a deprecation notice:\n%s", lastLines(streamedErr, 5))
	}
	eager, eagerErr, err := runKernelCLI(t, append(common, "--lazy-generation=false")...)
	if err != nil {
		t.Fatalf("run --lazy-generation=false: %v\n%s", err, lastLines(eagerErr, 3))
	}
	if !strings.Contains(eagerErr, "Generated 8 requests via unified workload pipeline") {
		t.Errorf("--lazy-generation=false must run the eager generator:\n%s", lastLines(eagerErr, 8))
	}
	want := "Flag --lazy-generation has been deprecated, " + deprecationNotice(deprecatedFlags["lazy-generation"])
	if !strings.Contains(eagerErr, want) {
		t.Errorf("stderr lacks the notice %q:\n%s", want, lastLines(eagerErr, 5))
	}
	if simulated(t, streamed) != simulated(t, eager) {
		t.Errorf("streaming and eager generation simulated different runs (INV-6)\n%s", firstDiff(simulated(t, streamed), simulated(t, eager)))
	}
	f := replayCmd.Flags().Lookup("lazy-generation")
	if f == nil || f.Deprecated != deprecationNotice(replayDeprecatedFlags["lazy-generation"]) {
		t.Errorf("replay --lazy-generation must carry replay's notice, got %+v", f)
	}
}
