package cmd

// CLI contracts for #1548: the per-ROLE MoE all-to-all backend (AC-5/AC-6) and the shared
// per-pool-flag registry it joins. The EP-placement contracts live in dp_placement_test.go
// and dp_replay_parity_test.go; the step-time physics lives in sim/latency/ep_mode_test.go.

import (
	"bytes"
	"os"
	"os/exec"
	"testing"

	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPerRoleMoECommBackendFlags_RegisteredOnRunAndReplay is INV-13 parity: the per-role
// backend is a model-level latency input supplied on the CLI (it does not round-trip
// through a trace), so both commands must accept it identically or a run cannot be
// replayed.
func TestPerRoleMoECommBackendFlags_RegisteredOnRunAndReplay(t *testing.T) {
	for _, name := range []string{"prefill-moe-comm-backend", "decode-moe-comm-backend"} {
		runFlag := runCmd.Flags().Lookup(name)
		replayFlag := replayCmd.Flags().Lookup(name)
		require.NotNilf(t, runFlag, "runCmd must have --%s", name)
		require.NotNilf(t, replayFlag, "replayCmd must have --%s", name)
		assert.Equalf(t, runFlag.DefValue, replayFlag.DefValue,
			"--%s default must match between run and replay", name)
		assert.Emptyf(t, runFlag.DefValue,
			"--%s default must be \"\" so it defers to the global --moe-comm-backend", name)
	}
}

// TestPerPoolHardwareFlags_AllRegistered guards the shared registry against naming a flag
// that does not exist: anyPerPoolHardwareFlagChanged consults cmd.Flags().Changed, which
// silently reports false for an unknown name — so a typo here would quietly disable the
// "per-pool flags set but disaggregation is off" diagnostic rather than fail loudly.
func TestPerPoolHardwareFlags_AllRegistered(t *testing.T) {
	c := &cobra.Command{}
	registerSimConfigFlags(c)
	for _, name := range perPoolHardwareFlags {
		assert.NotNilf(t, c.Flags().Lookup(name),
			"perPoolHardwareFlags names --%s, which registerSimConfigFlags does not register", name)
	}
	assert.Contains(t, perPoolHardwareFlags, "prefill-moe-comm-backend",
		"the per-role backend must join the shared per-pool registry, or setting it alone would "+
			"skip the disaggregation-disabled diagnostic")
}

// TestAnyPerPoolHardwareFlagChanged is the pure contract of the shared detector.
func TestAnyPerPoolHardwareFlagChanged(t *testing.T) {
	assert.False(t, anyPerPoolHardwareFlagChanged(func(string) bool { return false }),
		"nothing changed ⇒ false")
	assert.True(t, anyPerPoolHardwareFlagChanged(func(n string) bool { return n == "decode-moe-comm-backend" }),
		"a single per-pool flag ⇒ true")
	assert.False(t, anyPerPoolHardwareFlagChanged(func(n string) bool { return n == "tp" }),
		"a flag outside the per-pool set must not count")
}

// TestApplyPerRoleMoECommBackends is AC-5's resolution contract, driven directly so it is
// independent of the two command bodies that call it (R23: they share this one function).
func TestApplyPerRoleMoECommBackends(t *testing.T) {
	origP, origD := prefillMoECommBackend, decodeMoECommBackend
	t.Cleanup(func() { prefillMoECommBackend, decodeMoECommBackend = origP, origD })

	changedAll := func(string) bool { return true }

	t.Run("both roles resolve independently", func(t *testing.T) {
		prefillMoECommBackend, decodeMoECommBackend = "deepep_high_throughput", "deepep_low_latency"
		var p, d cluster.PoolOverrides
		require.NoError(t, applyPerRoleMoECommBackends(changedAll, true, true, &p, &d))
		assert.Equal(t, "deepep_high_throughput", p.MoECommBackend)
		assert.Equal(t, "deepep_low_latency", d.MoECommBackend)
	})

	t.Run("unset roles are left empty (inherit the global)", func(t *testing.T) {
		prefillMoECommBackend, decodeMoECommBackend = "pplx", "pplx"
		var p, d cluster.PoolOverrides
		onlyPrefill := func(n string) bool { return n == "prefill-moe-comm-backend" }
		require.NoError(t, applyPerRoleMoECommBackends(onlyPrefill, true, true, &p, &d))
		assert.Equal(t, "pplx", p.MoECommBackend)
		assert.Empty(t, d.MoECommBackend, "an unset role must inherit the global backend")
	})

	t.Run("an unknown backend is a hard error, not a silent default", func(t *testing.T) {
		prefillMoECommBackend, decodeMoECommBackend = "deepep_medium_throughput", ""
		var p, d cluster.PoolOverrides
		err := applyPerRoleMoECommBackends(func(n string) bool { return n == "prefill-moe-comm-backend" },
			true, true, &p, &d)
		require.Error(t, err, "R1: a typo'd per-role backend must surface")
		assert.Contains(t, err.Error(), "prefill-moe-comm-backend")
		assert.Contains(t, err.Error(), "deepep_medium_throughput")
		assert.Empty(t, p.MoECommBackend, "a rejected value must not be written through")
	})

	t.Run("a rejected decode value also errors", func(t *testing.T) {
		prefillMoECommBackend, decodeMoECommBackend = "", "nope"
		var p, d cluster.PoolOverrides
		err := applyPerRoleMoECommBackends(func(n string) bool { return n == "decode-moe-comm-backend" },
			true, true, &p, &d)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "decode-moe-comm-backend")
	})
}

// epPDRunArgs is a PD-disaggregated MoE run with expert parallelism on: the deployment
// shape #1548's per-role backend exists for. --dp stays 1 because PD + --dp>1 is still a
// fail-fast (#1553); EP-on is what makes the dispatch/combine term fire at DP=1, and so
// what makes the per-role selection observable at all.
func epPDRunArgs(extra ...string) []string {
	return append([]string{
		"run",
		"--model", "deepseek-ai/deepseek-v2-lite",
		"--model-config-folder", "../model_configs/deepseek-v2-lite",
		"--hardware", "H100",
		"--hardware-config", "../hardware_config.json",
		"--latency-model", "trained-physics",
		"--tp", "2",
		"--enable-expert-parallel",
		"--num-instances", "2",
		"--prefill-instances", "1",
		"--decode-instances", "1",
		"--pd-decider", "always",
		"--total-kv-blocks", "20000",
		"--rate", "8",
		"--num-requests", "24",
		"--seed", "42",
		"--defaults-filepath", "../defaults.yaml",
	}, extra...)
}

// TestRunCmd_PerRoleMoECommBackend_ChangesOutput is AC-5 end to end: selecting a DIFFERENT
// all-to-all backend for the decode pool than for the prefill pool must change the
// simulation, which a single global mode cannot express.
//
// The two backends chosen belong to different volume FAMILIES (all-gather vs modular
// all-to-all), because DeepEP high-throughput and low-latency deliberately share one
// placeholder cost today (#1568 differentiates them) — comparing those two would prove
// nothing about the plumbing. The families differ in byte volume, so a per-role difference
// is observable; the assertion is "the outputs differ", not any particular direction.
func TestRunCmd_PerRoleMoECommBackend_ChangesOutput(t *testing.T) {
	const env = "BLIS_EP_PERROLE"
	if v := os.Getenv(env); v != "" {
		var extra []string
		switch v {
		case "uniform":
			extra = []string{"--moe-comm-backend", "allgather_reducescatter"}
		case "split":
			extra = []string{"--moe-comm-backend", "allgather_reducescatter",
				"--decode-moe-comm-backend", "deepep_high_throughput"}
		}
		rootCmd.SetArgs(epPDRunArgs(extra...))
		if err := rootCmd.Execute(); err != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	// stdout ONLY, deliberately: the comparison below is an INEQUALITY, and stderr carries
	// logrus timestamps that differ between two subprocesses — comparing combined output
	// would pass vacuously whether or not the flag did anything. stdout is the deterministic
	// channel (INV-6), so a difference there is attributable to the config.
	leg := func(mode string) string {
		t.Helper()
		c := exec.Command(os.Args[0], "-test.run=^TestRunCmd_PerRoleMoECommBackend_ChangesOutput$")
		c.Env = append(os.Environ(), env+"="+mode)
		var stdout, stderr bytes.Buffer
		c.Stdout, c.Stderr = &stdout, &stderr
		if err := c.Run(); err != nil {
			t.Fatalf("%s leg failed: %v\nstdout:\n%s\nstderr:\n%s", mode, err, stdout.String(), stderr.String())
		}
		return stdout.String()
	}
	uniform, split := leg("uniform"), leg("split")
	assert.NotEqual(t, uniform, split,
		"AC-5: overriding only the DECODE pool's all-to-all backend must change the simulation; "+
			"if these match, the per-pool backend never reached the decode instances")
	// Non-vacuity: both legs must have actually served requests through both pools.
	for _, l := range []struct{ name, out string }{{"uniform", uniform}, {"split", split}} {
		assert.Contains(t, l.out, `"completed_requests"`, "%s leg produced no metrics", l.name)
		assert.NotContains(t, l.out, `"completed_requests": 0`, "%s leg completed nothing", l.name)
	}
	// A control leg: repeating the SAME configuration must reproduce stdout byte-for-byte
	// (INV-6). Without this, the inequality above could be satisfied by run-to-run noise
	// rather than by the per-role backend.
	assert.Equal(t, uniform, leg("uniform"),
		"INV-6: two identical runs must produce byte-identical stdout, or the inequality above "+
			"proves nothing about the per-role backend")
}

// TestValidatePerPoolLatencyBackends closes the latent hole #1548 makes live: a per-pool
// latency-model override must not silently opt one pool out of the DP/EP step-time physics
// the rest of the cluster uses. resolveLatencyConfig's own gate reads the GLOBAL backend
// only, so it cannot see this.
func TestValidatePerPoolLatencyBackends(t *testing.T) {
	roofline := cluster.PoolOverrides{LatencyBackend: "roofline"}
	tphys := cluster.PoolOverrides{LatencyBackend: "trained-physics"}
	none := cluster.PoolOverrides{}

	// Inactive DP/EP ⇒ the override is none of this function's business.
	assert.NoError(t, validatePerPoolLatencyBackends(false, roofline, roofline),
		"without --dp>1 or --enable-expert-parallel a per-pool roofline override is legitimate")

	// No override, or an explicit trained-physics override, is fine.
	assert.NoError(t, validatePerPoolLatencyBackends(true, none, none))
	assert.NoError(t, validatePerPoolLatencyBackends(true, tphys, tphys))

	// Either role on a DP/EP-blind backend is a hard error, and the message names the role
	// so the operator knows which flag to change.
	prefillErr := validatePerPoolLatencyBackends(true, roofline, none)
	require.Error(t, prefillErr)
	assert.Contains(t, prefillErr.Error(), "prefill-latency-model")
	decodeErr := validatePerPoolLatencyBackends(true, none, roofline)
	require.Error(t, decodeErr)
	assert.Contains(t, decodeErr.Error(), "decode-latency-model")
}
