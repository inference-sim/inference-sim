package cmd

import (
	"bytes"
	"errors"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/spf13/cobra"
)

// resetBundleState clears the shared cmd-level package vars that the bundle,
// provenance, and periodic-interval resolvers read, so a test case never inherits
// pollution from an earlier one. registerSimConfigFlags re-defaults the flag-backed
// vars (routingPolicy, loraBundle, loraPeriodicInterval, policyConfigPath) on the
// fresh command; loraConfigPath is not a bundle flag, so reset it explicitly.
func resetBundleState() {
	loraConfigPath = ""
	defaultsFilePath = "../defaults.yaml"
}

// TestBundleExpansion_PerKnobOverride pins the knob-precedence contract (DD-B7-2,
// FR-015): a named --lora-bundle expands to its {routing, eviction, creation} triple,
// but an explicitly-set per-knob flag overrides ONLY its own knob while the other two
// still take the bundle value. It drives the exact production seam pair — resolveLoRAConfig
// (eviction/creation knobs) then resolvePolicies (routing knob), in that order per
// DD-B7-2a — and reads the resolved values.
func TestBundleExpansion_PerKnobOverride(t *testing.T) {
	cases := []struct {
		name         string
		args         []string
		wantRouting  string
		wantEviction string
		wantCreation string
	}{
		{
			name:         "bundle_only_expands_all_three",
			args:         []string{"--lora-bundle", "lora-affinity"},
			wantRouting:  "route-to-holder",
			wantEviction: "rank-aware",
			wantCreation: "pre-placement",
		},
		{
			name:         "eviction_flag_overrides_only_eviction",
			args:         []string{"--lora-bundle", "lora-affinity", "--eviction-policy", "lru"},
			wantRouting:  "route-to-holder",
			wantEviction: "lru",
			wantCreation: "pre-placement",
		},
		{
			name:         "routing_flag_overrides_only_routing",
			args:         []string{"--lora-bundle", "lora-affinity", "--routing-policy", "weighted"},
			wantRouting:  "weighted",
			wantEviction: "rank-aware",
			wantCreation: "pre-placement",
		},
		{
			name:         "creation_flag_overrides_only_creation",
			args:         []string{"--lora-bundle", "lora-affinity", "--creation-policy", "on-demand"},
			wantRouting:  "route-to-holder",
			wantEviction: "rank-aware",
			wantCreation: "on-demand",
		},
		{
			name:         "no_bundle_leaves_baseline",
			args:         nil,
			wantRouting:  "round-robin",
			wantEviction: "",
			wantCreation: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resetBundleState()
			c := &cobra.Command{}
			registerSimConfigFlags(c)
			args := append([]string{"--defaults-filepath", "../defaults.yaml"}, tc.args...)
			if err := c.ParseFlags(args); err != nil {
				t.Fatalf("ParseFlags: %v", err)
			}

			// Production order (DD-B7-2a): LoRA config (eviction/creation) resolves
			// before policies (routing). Both validate the bundle name independently.
			cfg := resolveLoRAConfig(c)
			resolvePolicies(c)

			if routingPolicy != tc.wantRouting {
				t.Errorf("routing: got %q, want %q", routingPolicy, tc.wantRouting)
			}
			if cfg.EvictionPolicy != tc.wantEviction {
				t.Errorf("eviction: got %q, want %q", cfg.EvictionPolicy, tc.wantEviction)
			}
			if cfg.CreationPolicy != tc.wantCreation {
				t.Errorf("creation: got %q, want %q", cfg.CreationPolicy, tc.wantCreation)
			}
		})
	}
	resetBundleState()
}

// TestBundleParity_RunReplayIdenticalTriple is an INV-13 parity law: the shared
// resolvers (registered on both run and replay via registerSimConfigFlags) expand the
// same --lora-bundle to the identical effective triple regardless of which command's
// flag set parsed it. It resolves the same args twice through two independent fresh
// commands (standing in for run and replay) and asserts the provenance triple matches.
// A --lora-config fixture supplies adapters so the LoRA subsystem is active and
// provenance is emitted (it is adapter-gated, INV-L1).
func TestBundleParity_RunReplayIdenticalTriple(t *testing.T) {
	resolve := func() sim.PolicyTriple {
		resetBundleState()
		c := &cobra.Command{}
		registerSimConfigFlags(c)
		if err := c.ParseFlags([]string{
			"--defaults-filepath", "../defaults.yaml",
			"--lora-config", "testdata/lora_ondemand.yaml",
			"--lora-bundle", "lora-affinity",
		}); err != nil {
			t.Fatalf("ParseFlags: %v", err)
		}
		loraConfigPath = "testdata/lora_ondemand.yaml"
		cfg := resolveLoRAConfig(c)
		resolvePolicies(c)
		prov := computeLoRAProvenance(cfg)
		if prov == nil {
			t.Fatal("bundle selection on an adapter-active run must yield non-nil provenance")
		}
		return *prov
	}

	runTriple := resolve()
	replayTriple := resolve()
	if runTriple != replayTriple {
		t.Errorf("run/replay triple mismatch (INV-13): run=%+v replay=%+v", runTriple, replayTriple)
	}
	want := sim.PolicyTriple{Routing: "route-to-holder", Eviction: "rank-aware", Creation: "pre-placement"}
	if runTriple != want {
		t.Errorf("effective triple = %+v, want %+v", runTriple, want)
	}
	resetBundleState()
}

// withAdapter returns a copy of cfg carrying one adapter so cfg.HasAdapters() is true
// (the LoRA subsystem is active). Provenance is adapter-gated (INV-L1), so the
// "present" cases must model an active subsystem.
func withAdapter(cfg sim.LoRAConfig) sim.LoRAConfig {
	cfg.Adapters = []sim.AdapterSpec{{ID: "a0", Rank: 8}}
	return cfg
}

// TestComputeLoRAProvenance pins the emission rule (DD-B7-4, INV-6,
// contracts/metrics.md): provenance is present iff the LoRA subsystem is ACTIVE
// (HasAdapters) AND a non-baseline seam or bundle is selected; when present it records
// the canonical (empty-normalized) seam names. Two guards close the invariants:
// weighted/round-robin routing alone does NOT trigger provenance (not a LoRA seam);
// and a non-baseline selection on an ADAPTER-BLIND run returns nil (INV-L1 — a LoRA
// policy is inert without adapters, so stdout stays byte-identical to no-LoRA).
func TestComputeLoRAProvenance(t *testing.T) {
	cases := []struct {
		name     string
		routing  string
		bundle   string
		cfg      sim.LoRAConfig
		wantNil  bool
		wantTrip sim.PolicyTriple
	}{
		{
			name:    "all_baseline_active_returns_nil",
			routing: "round-robin",
			cfg:     withAdapter(sim.LoRAConfig{}),
			wantNil: true,
		},
		{
			name:    "weighted_routing_alone_returns_nil",
			routing: "weighted",
			cfg:     withAdapter(sim.LoRAConfig{}),
			wantNil: true,
		},
		{
			name:    "adapter_blind_rank_aware_returns_nil",
			routing: "route-to-holder",
			bundle:  "lora-affinity",
			cfg:     sim.LoRAConfig{EvictionPolicy: "rank-aware", CreationPolicy: "pre-placement"}, // no adapters
			wantNil: true,
		},
		{
			name:     "route_to_holder_triggers_with_canonical_empties",
			routing:  "route-to-holder",
			cfg:      withAdapter(sim.LoRAConfig{}),
			wantTrip: sim.PolicyTriple{Routing: "route-to-holder", Eviction: "lru", Creation: "on-demand"},
		},
		{
			name:     "rank_aware_eviction_triggers",
			routing:  "round-robin",
			cfg:      withAdapter(sim.LoRAConfig{EvictionPolicy: "rank-aware"}),
			wantTrip: sim.PolicyTriple{Routing: "round-robin", Eviction: "rank-aware", Creation: "on-demand"},
		},
		{
			name:     "pre_placement_creation_triggers",
			routing:  "round-robin",
			cfg:      withAdapter(sim.LoRAConfig{CreationPolicy: "pre-placement"}),
			wantTrip: sim.PolicyTriple{Routing: "round-robin", Eviction: "lru", Creation: "pre-placement"},
		},
		{
			name:     "bundle_name_triggers_even_with_baseline_knobs",
			routing:  "round-robin",
			bundle:   "lora-affinity",
			cfg:      withAdapter(sim.LoRAConfig{}),
			wantTrip: sim.PolicyTriple{Routing: "round-robin", Eviction: "lru", Creation: "on-demand"},
		},
	}
	origRouting, origBundle := routingPolicy, loraBundle
	defer func() { routingPolicy, loraBundle = origRouting, origBundle }()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			routingPolicy = tc.routing
			loraBundle = tc.bundle
			got := computeLoRAProvenance(tc.cfg)
			if tc.wantNil {
				if got != nil {
					t.Errorf("want nil provenance, got %+v", got)
				}
				return
			}
			if got == nil {
				t.Fatalf("want provenance %+v, got nil", tc.wantTrip)
			}
			if *got != tc.wantTrip {
				t.Errorf("provenance = %+v, want %+v", *got, tc.wantTrip)
			}
		})
	}
}

// TestResolveLoRAPeriodicInterval_NonNegative verifies the scaffold accessor returns a
// valid non-negative interval unchanged (0 = off, positive = inert-but-recorded).
func TestResolveLoRAPeriodicInterval_NonNegative(t *testing.T) {
	orig := loraPeriodicInterval
	defer func() { loraPeriodicInterval = orig }()
	for _, v := range []int64{0, 1, 1_000_000} {
		loraPeriodicInterval = v
		if got := resolveLoRAPeriodicInterval(); got != v {
			t.Errorf("resolveLoRAPeriodicInterval() = %d, want %d", got, v)
		}
	}
}

// periodicIntervalFatalSubprocess drives resolveLoRAPeriodicInterval with a negative
// value in a subprocess; the CLI boundary must Fatalf (Principle V, R3).
func periodicIntervalFatalSubprocess(t *testing.T) {
	t.Helper()
	if os.Getenv("BLIS_TEST_SUBPROCESS") != "1" {
		return
	}
	loraPeriodicInterval = -1
	resolveLoRAPeriodicInterval() // must Fatalf before returning
	os.Exit(0)
}

// TestResolveLoRAPeriodicInterval_NegativeFatal verifies a negative
// --lora-periodic-interval-us aborts at the CLI boundary (exit 1, R3) rather than
// silently accepting an invalid interval.
func TestResolveLoRAPeriodicInterval_NegativeFatal(t *testing.T) {
	periodicIntervalFatalSubprocess(t)
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestResolveLoRAPeriodicInterval_NegativeFatal", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("expected logrus.Fatalf (exit 1), got err=%v; output:\n%s", err, out)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit 1, got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "lora-periodic-interval-us") {
		t.Errorf("fatal message must mention lora-periodic-interval-us; output:\n%s", out)
	}
}

// TestProvenance_EmittedOnLoRAActiveBundleRun is the positive end-to-end counterpart
// to the inertness laws: a real `blis run` with an ACTIVE LoRA subsystem
// (--lora-config declaring adapters) plus --lora-bundle lora-affinity MUST emit the
// policy_provenance block in stdout recording the effective route-to-holder triple
// (SC-006 reproducibility), and MUST be byte-identical across two invocations (INV-6
// determinism). Driven in a re-exec subprocess so the real cobra tree executes.
func TestProvenance_EmittedOnLoRAActiveBundleRun(t *testing.T) {
	if os.Getenv("BLIS_NOOP_SUBPROCESS") == "1" {
		rootCmd.SetArgs([]string{
			"run", "--model", "qwen/qwen3-14b", "--seed", "42",
			"--defaults-filepath", "../defaults.yaml",
			"--lora-config", "testdata/lora_ondemand.yaml",
			"--lora-bundle", "lora-affinity",
			"--lora-adapter-placement", "0=adapter-a",
		})
		_ = rootCmd.Execute()
		os.Exit(0)
	}

	run := func() (string, bool) {
		t.Helper()
		cmd := exec.Command(os.Args[0], "-test.run=TestProvenance_EmittedOnLoRAActiveBundleRun")
		cmd.Env = append(os.Environ(), "BLIS_NOOP_SUBPROCESS=1")
		var stdout bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = io.Discard // logrus diagnostics go to stderr (INV-6)
		err := cmd.Run()
		return stdout.String(), err == nil
	}

	first, ok1 := run()
	second, ok2 := run()
	if !ok1 || !ok2 {
		t.Fatalf("LoRA-active bundle run exited non-zero (ok1=%v ok2=%v); flags must parse and placement validate", ok1, ok2)
	}
	if first != second {
		t.Errorf("INV-6 VIOLATION: LoRA-active bundle run is non-deterministic.\n--- first ---\n%s\n--- second ---\n%s", first, second)
	}
	if !strings.Contains(first, "policy_provenance") {
		t.Errorf("SC-006: LoRA-active bundle run must emit policy_provenance; stdout:\n%s", first)
	}
	for _, want := range []string{"route-to-holder", "rank-aware", "pre-placement"} {
		if !strings.Contains(first, want) {
			t.Errorf("SC-006: provenance must record effective %q; stdout:\n%s", want, first)
		}
	}
}

// unknownBundleFatalSubprocess drives resolveLoRAConfig with an unknown --lora-bundle
// in a subprocess; the CLI boundary must Fatalf (Principle V) with the valid names.
func unknownBundleFatalSubprocess(t *testing.T) {
	t.Helper()
	if os.Getenv("BLIS_TEST_SUBPROCESS") != "1" {
		return
	}
	c := &cobra.Command{}
	registerSimConfigFlags(c)
	if err := c.ParseFlags([]string{"--defaults-filepath", "../defaults.yaml", "--lora-bundle", "bogus"}); err != nil {
		os.Exit(2)
	}
	defaultsFilePath = "../defaults.yaml"
	loraConfigPath = ""
	resolveLoRAConfig(c) // must Fatalf before returning
	os.Exit(0)
}

// TestUnknownBundle_Fatal verifies an unknown --lora-bundle name aborts at the CLI
// boundary (exit 1) with a message listing the valid bundle names — never a silent
// no-op or a deferred error.
func TestUnknownBundle_Fatal(t *testing.T) {
	unknownBundleFatalSubprocess(t)
	if os.Getenv("BLIS_TEST_SUBPROCESS") == "1" {
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestUnknownBundle_Fatal", "-test.v")
	cmd.Env = append(os.Environ(), "BLIS_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("expected logrus.Fatalf (exit 1), got err=%v; output:\n%s", err, out)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit 1, got %d; output:\n%s", exitErr.ExitCode(), out)
	}
	if !strings.Contains(string(out), "lora-affinity") {
		t.Errorf("fatal message must list valid bundle names (lora-affinity); output:\n%s", out)
	}
}
