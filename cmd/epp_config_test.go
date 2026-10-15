package cmd

import (
	"strings"
	"testing"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// eppTestCmd registers the shared run/replay flags and parses args.
func eppTestCmd(t *testing.T, args ...string) *cobra.Command {
	t.Helper()
	c := &cobra.Command{}
	registerSimConfigFlags(c)
	if err := c.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	return c
}

// expectFatal runs fn and requires a logrus.Fatalf whose message contains want.
func expectFatal(t *testing.T, want string, fn func()) {
	t.Helper()
	logger := logrus.StandardLogger()
	origExit := logger.ExitFunc
	var msg string
	hook := &captureHook{}
	logger.AddHook(hook)
	logger.ExitFunc = func(int) { msg = hook.last; panic("fatal") }
	defer func() {
		logger.ExitFunc = origExit
		logger.ReplaceHooks(make(logrus.LevelHooks))
		if r := recover(); r != "fatal" {
			t.Fatalf("expected a fatal exit containing %q, recover=%v", want, r)
		}
		if !strings.Contains(msg, want) {
			t.Errorf("fatal message %q does not contain %q", msg, want)
		}
	}()
	fn()
}

type captureHook struct{ last string }

func (h *captureHook) Levels() []logrus.Level { return []logrus.Level{logrus.FatalLevel} }
func (h *captureHook) Fire(e *logrus.Entry) error {
	h.last = e.Message
	return nil
}

func resetEPPGlobals() {
	eppIngest = nil
	prefillInstances, decodeInstances, prefillDecodeInstances, encodeInstances = 0, 0, 0, 0
	blockSizeTokens = 64
}

// --epp-config and --epp-version are registered on both run and replay (INV-13).
func TestEPPConfigFlags_RegisteredOnRunAndReplay(t *testing.T) {
	c := eppTestCmd(t)
	for _, f := range []string{"epp-config", "epp-version"} {
		if c.Flags().Lookup(f) == nil {
			t.Errorf("--%s must be registered by registerSimConfigFlags", f)
		}
	}
}

// The curvebender router ingests and fixes routing, flow control and the `epp`
// provenance block; the DeploymentConfig mapping matches the ingested config.
func TestApplyEPPConfig_CurvebenderOptimizedBaseline(t *testing.T) {
	resetEPPGlobals()
	defer resetEPPGlobals()
	path := "../sim/testdata/epp/optimized-baseline-cpu-tier.yaml"
	defer func() { eppConfigPath, eppVersionArg = "", string(sim.EPPVersionMain) }()
	c := eppTestCmd(t, "--epp-config", path, "--epp-version", "v0.10.0")
	applyEPPConfig(c, nil)
	if eppIngest == nil || routingPolicy != "weighted" || !flowControlEnabled || flowControlMaxConcurrency != 256 {
		t.Fatalf("ingest not applied: ingest=%v policy=%q fc=%v maxConc=%d", eppIngest != nil, routingPolicy, flowControlEnabled, flowControlMaxConcurrency)
	}
	p := eppProvenance()
	if p == nil || p.LLMDRouterVersion != "v0.10.0" || len(p.ConfigSHA256) != 64 || p.FlowControl == nil {
		t.Fatalf("epp provenance block: %+v", p)
	}
	var d cluster.DeploymentConfig
	d.ApplyEPPIngest(eppIngest)
	if d.FlowControlConcurrency == nil || d.FlowControlConcurrency.FailOpen || d.FlowControlMaxQueueDepth != 4096 ||
		d.FlowControlRequestTTL != 300_000_000 || d.FlowControlMaxBytes != 4<<30 || len(d.RoutingPipeline.Filters) != 2 {
		t.Errorf("deployment mapping: %+v", d)
	}
}

// Routing and flow-control flags conflict with --epp-config and are refused rather
// than silently overridden.
func TestApplyEPPConfig_ConflictingFlagsRefused(t *testing.T) {
	resetEPPGlobals()
	defer resetEPPGlobals()
	path := "../sim/testdata/epp/precise-gpu.yaml"
	defer func() { eppConfigPath = "" }()
	c := eppTestCmd(t, "--epp-config", path, "--routing-scorers", "queue-depth:1", "--max-concurrency", "8")
	expectFatal(t, "--routing-scorers, --max-concurrency", func() { applyEPPConfig(c, nil) })
}

// A single-profile config on a P/D scenario (and the reverse) is refused.
func TestApplyEPPConfig_ProfileHandlerMustMatchPools(t *testing.T) {
	resetEPPGlobals()
	defer resetEPPGlobals()
	defer func() { eppConfigPath = "" }()
	c := eppTestCmd(t, "--epp-config", "../sim/testdata/epp/pd-1p1d.yaml")
	expectFatal(t, "defines no prefill/decode pools", func() { applyEPPConfig(c, nil) })

	c = eppTestCmd(t, "--epp-config", "../sim/testdata/epp/precise-gpu.yaml")
	prefillInstances, decodeInstances = 1, 1 // as adoptKernelDeployment sets them from a P/D scenario
	expectFatal(t, "single-profile config", func() { applyEPPConfig(c, nil) })
}

// An unknown --epp-version is refused.
func TestApplyEPPConfig_UnknownVersionRefused(t *testing.T) {
	resetEPPGlobals()
	defer resetEPPGlobals()
	defer func() { eppConfigPath, eppVersionArg = "", string(sim.EPPVersionMain) }()
	c := eppTestCmd(t, "--epp-config", "../sim/testdata/epp/precise-gpu.yaml", "--epp-version", "v0.9.0")
	expectFatal(t, "unknown llm-d EPP version", func() { applyEPPConfig(c, nil) })
}

// Without --epp-config nothing changes: no ingest, no provenance option (the
// results file stays byte-identical to a pre-feature build).
func TestApplyEPPConfig_AbsentIsInert(t *testing.T) {
	resetEPPGlobals()
	defer resetEPPGlobals()
	eppConfigPath = ""
	before := routingPolicy
	applyEPPConfig(eppTestCmd(t), nil)
	if eppIngest != nil || eppProvenance() != nil || routingPolicy != before {
		t.Errorf("absent --epp-config must be inert")
	}
}

// --epp-cpu-tier-view is the explicit opt-in for the CPU-tier view; without it the
// curvebender CPU producer stays llm-d's LRU and no deviation is recorded.
func TestApplyEPPConfig_CPUTierViewOptIn(t *testing.T) {
	resetEPPGlobals()
	defer resetEPPGlobals()
	defer func() { eppConfigPath, eppCPUTierArg = "", "" }()
	path := "../sim/testdata/epp/optimized-baseline-cpu-tier.yaml"
	applyEPPConfig(eppTestCmd(t, "--epp-config", path), nil)
	if d := eppProvenance().Deviations; len(d) != 0 {
		t.Errorf("no opt-in, deviations %v", d)
	}
	applyEPPConfig(eppTestCmd(t, "--epp-config", path, "--epp-cpu-tier-view", "cpu-prefix-cache-producer"), nil)
	if d := eppProvenance().Deviations; len(d) != 1 {
		t.Errorf("opt-in must record one deviation, got %v", d)
	}
}
