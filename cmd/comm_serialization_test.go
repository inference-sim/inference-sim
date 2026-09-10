package cmd

// CLI-surface tests for the cross-node collective serialization factor S (#1694, Part B):
// --comm-serialization-factor (the multiplier) and --enforce-eager (the regime guard).
//
// The step-time math S drives (n_steps·α_hop·S, monotone, inert at S=1) is proven directly
// on the latency model in sim/latency/network_cost_test.go. These tests pin the CLI wiring:
// the flags exist on both run and replay (INV-13 flag-surface parity), the factor is
// validated at the command boundary (R3), and --enforce-eager refuses to run without an
// explicit calibrated factor (#1694 anti-overfitting guardrail #2).

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
)

// TestCommSerializationFlags_RegisteredOnRunAndReplay pins INV-13 at the flag surface:
// both flags must exist on run and replay so a trace replays with identical flags. S is a
// model-level latency input re-supplied on both legs, not round-tripped through the header.
func TestCommSerializationFlags_RegisteredOnRunAndReplay(t *testing.T) {
	for _, name := range []string{"comm-serialization-factor", "enforce-eager"} {
		if runCmd.Flags().Lookup(name) == nil {
			t.Errorf("runCmd missing --%s flag", name)
		}
		if replayCmd.Flags().Lookup(name) == nil {
			t.Errorf("replayCmd missing --%s flag", name)
		}
	}
	// Defaults must be the inert values (INV-6): S=1, eager off.
	if f := runCmd.Flags().Lookup("comm-serialization-factor"); f != nil && f.DefValue != "1" {
		t.Errorf("--comm-serialization-factor default = %q, want \"1\"", f.DefValue)
	}
	if f := runCmd.Flags().Lookup("enforce-eager"); f != nil && f.DefValue != "false" {
		t.Errorf("--enforce-eager default = %q, want \"false\"", f.DefValue)
	}
}

// TestCommSerializationFlags_AbsentOnObserve encodes the boundary that S is a
// step-time (latency-model) input, and `blis observe` is a black-box HTTP dispatcher
// that builds no simulator and computes no step time — so neither flag belongs there,
// exactly as --kv-cache-dtype and --kv-offload-config are excluded. Registering S on
// observe would be a dead knob a user could set expecting an effect. This is the S twin
// of TestNetworkTopology_HasNoCLIFlag's observe row.
func TestCommSerializationFlags_AbsentOnObserve(t *testing.T) {
	for _, name := range []string{"comm-serialization-factor", "enforce-eager"} {
		if f := observeCmd.Flags().Lookup(name); f != nil {
			t.Errorf("observeCmd registers --%s, but observe derives no step time — the flag would be a "+
				"dead knob (same boundary as --kv-cache-dtype / --kv-offload-config)", name)
		}
	}
}

// resolveCommSerialForTest parses the given args onto a fresh command with the real flag
// registration, then runs resolveCommSerializationFactor, capturing a fatal exit via the
// logrus ExitFunc override. Returns (factor, fatal).
func resolveCommSerialForTest(t *testing.T, args []string) (result float64, fatal bool) {
	t.Helper()
	// Reset every package var the resolver reads, so cases don't leak into each other and
	// the S-inert warning (which reads latencyModelBackend/policyConfigPath) is deterministic.
	// Default to the trained-physics + node-pool shape so a valid S>1 does NOT warn here;
	// the warning paths have their own test.
	origFactor, origEager := commSerializationFactor, enforceEager
	origBackend, origPolicy := latencyModelBackend, policyConfigPath
	defer func() {
		commSerializationFactor, enforceEager = origFactor, origEager
		latencyModelBackend, policyConfigPath = origBackend, origPolicy
	}()
	latencyModelBackend = "trained-physics"
	policyConfigPath = "some-policy.yaml"

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags(%v): %v", args, err)
	}

	logger := logrus.StandardLogger()
	origExit := logger.ExitFunc
	logger.ExitFunc = func(int) { fatal = true; panic("fatal") }
	defer func() {
		logger.ExitFunc = origExit
		if r := recover(); r != nil && r != "fatal" {
			panic(r) // a real panic, not our injected fatal
		}
	}()
	result = resolveCommSerializationFactor(testCmd)
	return result, fatal
}

// TestResolveCommSerializationFactor_Default verifies BC-5 at the CLI: with no flags the
// factor is the inert 1.0 and nothing is fatal.
func TestResolveCommSerializationFactor_Default(t *testing.T) {
	got, fatal := resolveCommSerialForTest(t, nil)
	if fatal {
		t.Fatal("default (no flags) must not be fatal")
	}
	if got != 1.0 {
		t.Errorf("default factor = %v, want 1.0", got)
	}
}

// TestResolveCommSerializationFactor_Valid verifies a supplied factor > 1 passes through.
func TestResolveCommSerializationFactor_Valid(t *testing.T) {
	got, fatal := resolveCommSerialForTest(t, []string{"--comm-serialization-factor", "30"})
	if fatal {
		t.Fatal("a valid factor of 30 must not be fatal")
	}
	if got != 30.0 {
		t.Errorf("factor = %v, want 30.0", got)
	}
}

// TestResolveCommSerializationFactor_Guards verifies BC-7: a factor < 1 is rejected, and
// --enforce-eager without an explicit factor > 1 is rejected (guardrail #2 — BLIS ships no
// fitted eager magnitude). All via a fatal exit at the command boundary (R3/R6).
func TestResolveCommSerializationFactor_Guards(t *testing.T) {
	cases := []struct {
		name string
		args []string
	}{
		{"factor below 1", []string{"--comm-serialization-factor", "0.5"}},
		{"enforce-eager without factor", []string{"--enforce-eager"}},
		{"enforce-eager with factor=1", []string{"--enforce-eager", "--comm-serialization-factor", "1"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, fatal := resolveCommSerialForTest(t, tc.args)
			if !fatal {
				t.Errorf("expected a fatal exit for %s (%v)", tc.name, tc.args)
			}
		})
	}
}

// TestResolveCommSerializationFactor_EnforceEagerWithFactor verifies the intended eager
// usage — --enforce-eager WITH an explicit factor > 1 — is accepted.
func TestResolveCommSerializationFactor_EnforceEagerWithFactor(t *testing.T) {
	got, fatal := resolveCommSerialForTest(t, []string{"--enforce-eager", "--comm-serialization-factor", "25"})
	if fatal {
		t.Fatal("--enforce-eager with an explicit factor > 1 must be accepted")
	}
	if got != 25.0 {
		t.Errorf("factor = %v, want 25.0", got)
	}
}

// TestResolveCommSerializationFactor_WarnsWhenInert verifies finding #3 from the #1695
// review: an S > 1 that provably cannot fire must warn loudly (R1), not vanish silently.
// The two cheaply-knowable inert cases are a non-trained-physics backend and no node pools;
// a genuinely-live shape (trained-physics + node pools) must NOT warn.
func TestResolveCommSerializationFactor_WarnsWhenInert(t *testing.T) {
	cases := []struct {
		name       string
		backend    string
		policyPath string
		wantWarn   string // substring the warning must contain; "" = must NOT warn
	}{
		{"roofline backend", "roofline", "some-policy.yaml", "only the trained-physics backend models"},
		{"no node pools", "trained-physics", "", "no node_pools"},
		{"live shape does not warn", "trained-physics", "some-policy.yaml", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			origFactor, origEager := commSerializationFactor, enforceEager
			origBackend, origPolicy := latencyModelBackend, policyConfigPath
			defer func() {
				commSerializationFactor, enforceEager = origFactor, origEager
				latencyModelBackend, policyConfigPath = origBackend, origPolicy
			}()
			var buf bytes.Buffer
			logger := logrus.StandardLogger()
			origOut, origLevel := logger.Out, logger.Level
			logger.SetOutput(&buf)
			logger.SetLevel(logrus.WarnLevel)
			defer func() { logger.SetOutput(origOut); logger.SetLevel(origLevel) }()

			testCmd := &cobra.Command{}
			registerSimConfigFlags(testCmd)
			require.NoError(t, testCmd.ParseFlags([]string{"--comm-serialization-factor", "5"}))
			// Set the resolver's inputs AFTER ParseFlags — cobra resets bound vars
			// (latencyModelBackend, policyConfigPath) to their flag defaults during parse.
			commSerializationFactor, enforceEager = 5.0, false
			latencyModelBackend, policyConfigPath = tc.backend, tc.policyPath
			resolveCommSerializationFactor(testCmd)

			out := buf.String()
			if tc.wantWarn == "" {
				require.NotContains(t, out, "will have NO effect",
					"a live trained-physics + node-pool shape must not warn about inert S")
			} else {
				require.Contains(t, out, "will have NO effect", "an inert S>1 must warn (R1)")
				require.Contains(t, out, tc.wantWarn)
			}
		})
	}
}

// runWithCommSerialFlag runs `blis run` on a tiny fixed single-node workload, optionally
// passing --comm-serialization-factor 1, and captures stdout. It mirrors
// runSpecAndCaptureStdoutSpecFlag (parity_run_replay_test.go): a single-node run declares
// no node span, so S is inert at the default 1 — this proves the flag PLUMBING does not
// perturb the non-spanning path (INV-6). The spanning-path math is proven in
// sim/latency/network_cost_test.go. NOTE: mutates package vars + os.Stdout; not parallel.
func runWithCommSerialFlag(t *testing.T, specYAML string, seedVal, horizon int64, passS1 bool) []byte {
	t.Helper()
	tmpDir := t.TempDir()
	specPath := filepath.Join(tmpDir, "workload.yaml")
	if err := os.WriteFile(specPath, []byte(specYAML), 0644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	mcFolder, hwPath, defaultsPath := setupTrainedPhysicsTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()
	origFactor, origEager := commSerializationFactor, enforceEager
	defer func() { commSerializationFactor, enforceEager = origFactor, origEager }()

	traceOutput = ""
	workloadSpecPath = specPath
	workloadType = ""
	simulationHorizon = horizon
	seed = seedVal
	lazyGeneration = false
	requestTimeoutSecs = 300

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "")
	testCmd.Flags().BoolVar(&lazyGeneration, "lazy-generation", false, "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--model", "qwen/qwen3-14b", "--latency-model", "trained-physics",
		"--defaults-filepath", defaultsPath, "--model-config-folder", mcFolder,
		"--hardware-config", hwPath, "--hardware", "H100", "--tp", "1",
		"--total-kv-blocks", "1000", "--seed", strconv.FormatInt(seedVal, 10),
		"--workload-spec", specPath, "--horizon", strconv.FormatInt(horizon, 10),
	}
	if passS1 {
		args = append(args, "--comm-serialization-factor", "1")
	}
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	done := make(chan []byte, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		done <- buf.Bytes()
	}()
	runCmd.Run(testCmd, nil)
	_ = w.Close()
	os.Stdout = oldStdout
	return <-done
}

// TestParity_CommSerializationFactorDefault_ByteIdenticalStdout pins BC-5 (INV-6) at the
// CLI: a run with --comm-serialization-factor 1 produces byte-identical stdout to a run
// without the flag at all. The default is inert — the flag is a guarded addition, not a
// rewrite of any existing path. (This is the run-level twin of the latency-model
// TestStepTime_SerializationFactorDefaultIsInert; since run and replay thread S through the
// same resolveCommSerializationFactor, INV-13 follows from this INV-6 guarantee.)
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars and os.Stdout.
func TestParity_CommSerializationFactorDefault_ByteIdenticalStdout(t *testing.T) {
	const seed int64 = 20260909
	shape := paritySpecShapes()[0] // chatbot

	noFlag := runWithCommSerialFlag(t, shape.yaml, seed, shape.horizon, false)
	s1 := runWithCommSerialFlag(t, shape.yaml, seed, shape.horizon, true)

	if len(noFlag) == 0 {
		t.Fatal("run produced empty stdout; test is vacuous")
	}
	if !bytes.Equal(noFlag, s1) {
		t.Fatalf("BC-5: --comm-serialization-factor 1 changed stdout vs no flag\nNO-FLAG:\n%s\nS1:\n%s", noFlag, s1)
	}
}
