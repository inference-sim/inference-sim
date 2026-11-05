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
		{"run", []string{"scheduler", "preemption-policy", "dp-load-balancing", "swa-bounded-replay", "kv-offload-config"}},
		{"replay", []string{"scheduler", "preemption-policy", "dp-load-balancing", "swa-bounded-replay", "kv-offload-config"}},
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
