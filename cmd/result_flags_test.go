package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

// #1871 review B1: every flag that can change what a run simulates is recorded in the Result,
// either in input (and listed as exempt here with where) or as an overrides row. The guard walks
// both commands' real flag sets: setting any flag that is not exempt to a value other than its
// default produces an override row naming it, and an exempt flag names its home. A new flag
// fails this test until it is recorded or exempted with a reason.
func TestResultFlags_EveryFlagIsRecordedOrExempt(t *testing.T) {
	for _, c := range []struct {
		name string
		fs   *pflag.FlagSet
	}{{"run", runCmd.Flags()}, {"replay", replayCmd.Flags()}} {
		exempt := resultFlagsExempt[c.name]
		for name, why := range exempt {
			if c.fs.Lookup(name) == nil && name != "help" { // cobra adds --help at execution
				t.Errorf("%s: exempt flag --%s does not exist (stale entry)", c.name, name)
			}
			if strings.TrimSpace(why) == "" {
				t.Errorf("%s: exempt flag --%s gives no reason", c.name, name)
			}
		}
		n := 0
		c.fs.VisitAll(func(f *pflag.Flag) {
			if _, ok := exempt[f.Name]; ok {
				return
			}
			n++
			fs := pflag.NewFlagSet(c.name, pflag.ContinueOnError)
			fs.AddFlag(cloneFlag(t, f))
			v := otherValue(t, f)
			if err := fs.Set(f.Name, v); err != nil {
				t.Errorf("%s: --%s=%s: %v", c.name, f.Name, v, err)
				return
			}
			got, err := flagOverrides(c.name, fs)
			if err != nil {
				t.Errorf("%s: --%s: %v", c.name, f.Name, err)
				return
			}
			if len(got) != 1 || got[0].Setting != f.Name || got[0].Source != "--"+f.Name || got[0].Run == "" {
				t.Errorf("%s: setting --%s=%s records %+v, want one override row naming it", c.name, f.Name, v, got)
			}
		})
		if n < 20 {
			t.Fatalf("non-vacuity: %s has only %d recorded flags", c.name, n)
		}
	}
}

// A flag left at its default, or set to it, records nothing: the default is part of what the
// scenario and the build already state.
func TestResultFlags_ADefaultValueRecordsNothing(t *testing.T) {
	fs := pflag.NewFlagSet("run", pflag.ContinueOnError)
	fs.AddFlag(cloneFlag(t, runCmd.Flags().Lookup("timeout")))
	if err := fs.Set("timeout", runCmd.Flags().Lookup("timeout").DefValue); err != nil {
		t.Fatal(err)
	}
	if got, err := flagOverrides("run", fs); err != nil || len(got) != 0 {
		t.Errorf("--timeout at its default recorded %+v (%v)", got, err)
	}
}

// A file flag is recorded by the file's sha256, never its path, and a file that cannot be read
// is an error naming the flag.
func TestResultFlags_AFileFlagIsPinnedBySHA256(t *testing.T) {
	dir := t.TempDir()
	spec := filepath.Join(dir, "spec.yaml")
	if err := os.WriteFile(spec, []byte("version: \"2\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	fs := pflag.NewFlagSet("run", pflag.ContinueOnError)
	fs.AddFlag(cloneFlag(t, runCmd.Flags().Lookup("workload-spec")))
	if err := fs.Set("workload-spec", spec); err != nil {
		t.Fatal(err)
	}
	got, err := flagOverrides("run", fs)
	if err != nil || len(got) != 1 || got[0].Run != "sha256:"+bytesDigest([]byte("version: \"2\"\n")) {
		t.Errorf("--workload-spec recorded %+v (%v), want its sha256", got, err)
	}
	if strings.Contains(got[0].Run, dir) {
		t.Errorf("the recorded value carries the host path: %q", got[0].Run)
	}
	if err := fs.Set("workload-spec", filepath.Join(dir, "absent.yaml")); err != nil {
		t.Fatal(err)
	}
	if _, err := flagOverrides("run", fs); err == nil || !strings.Contains(err.Error(), "--workload-spec") {
		t.Errorf("an unreadable --workload-spec was not refused naming the flag: %v", err)
	}
}

// The reviewer's cases through the real CLI: each flag changes the metrics and changes the
// Result's overrides and name, so two documents with the same input, overrides and revisions
// describe the same simulation (INV-6).
func TestResultFlags_SimulationAffectingFlagsChangeTheDocument(t *testing.T) {
	base, stderr, err := runKernelCLI(t, resultRunArgs()...)
	if err != nil {
		t.Fatalf("baseline: %v\n%s", err, lastLines(stderr, 3))
	}
	b := parseResult(t, base)
	for _, extra := range [][]string{
		{"--timeout", "2"},
		{"--prefix-tokens", "512"},
		{"--gateway-latency", "5000"},
		{"--output-tokens", "64", "--output-tokens-max", "128", "--output-tokens-stdev", "16"},
		{"--num-instances", "2"},
	} {
		out, stderr, err := runKernelCLI(t, resultRunArgs(extra...)...)
		if err != nil {
			t.Fatalf("%v: %v\n%s", extra, err, lastLines(stderr, 3))
		}
		r := parseResult(t, out)
		if jsonEqual(t, r.Metrics, b.Metrics) {
			t.Fatalf("non-vacuity: %v left the metrics unchanged", extra)
		}
		if jsonEqual(t, r.Overrides, b.Overrides) || r.Name == b.Name {
			t.Errorf("%v changed the metrics but not the overrides (%+v) and name (%q)", extra, r.Overrides, r.Name)
		}
	}
}

// cloneFlag is a copy of f with a fresh value of the same type, so setting it leaves the
// command's own flag untouched.
func cloneFlag(t *testing.T, f *pflag.Flag) *pflag.Flag {
	t.Helper()
	fs := pflag.NewFlagSet("clone", pflag.ContinueOnError)
	switch f.Value.Type() {
	case "int":
		fs.Int(f.Name, 0, "")
	case "int64":
		fs.Int64(f.Name, 0, "")
	case "float64":
		fs.Float64(f.Name, 0, "")
	case "bool":
		fs.Bool(f.Name, false, "")
	case "duration":
		fs.Duration(f.Name, 0, "")
	case "string":
		fs.String(f.Name, "", "")
	default: // optionalBool and other custom values parse as a bool or a string
		fs.String(f.Name, "", "")
	}
	c := fs.Lookup(f.Name)
	if err := c.Value.Set(f.DefValue); err != nil && f.DefValue != "" {
		t.Fatalf("--%s: default %q: %v", f.Name, f.DefValue, err)
	}
	c.DefValue = f.DefValue
	c.Changed = false
	return c
}

// otherValue is a value for f that differs from its default.
func otherValue(t *testing.T, f *pflag.Flag) string {
	t.Helper()
	if resultFlagFiles[f.Name] {
		p := filepath.Join(t.TempDir(), f.Name)
		if err := os.WriteFile(p, []byte(f.Name), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	switch f.Value.Type() {
	case "int", "int64":
		if f.DefValue == "7" {
			return "8"
		}
		return "7"
	case "float64":
		if f.DefValue == "0.5" {
			return "0.25"
		}
		return "0.5"
	case "bool":
		if f.DefValue == "true" {
			return "false"
		}
		return "true"
	case "duration":
		return "7s"
	}
	if f.DefValue == "x" {
		return "y"
	}
	return "x"
}
