package cmd

import (
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// TestSIM11_SameInvalidValueRefusedThroughEveryInput: a value of one kind that the command line
// refuses, a trace header refuses too, by the same validator (package check), and each refusal
// names the field as that input spells it and the value (SIM-11, X15; #2007). A goodput latency
// target is checked through all three inputs that carry one: --slo-ttft, the trace header and the
// workload specification. Each kind has its
// refusal pair (SIM-9): the smallest refused value and the closest accepted one, through each
// input.
func TestSIM11_SameInvalidValueRefusedThroughEveryInput(t *testing.T) {
	// The router's three timings: spans of simulated time, refused below 0 (#2007).
	timings := []struct {
		flag, key string
		header    func(v int64) *workload.TraceRouterConfig
	}{
		{"--gateway-latency", "router.gateway_latency_us", func(v int64) *workload.TraceRouterConfig { return &workload.TraceRouterConfig{GatewayLatencyUs: v} }},
		{"--epp-scrape-jitter", "router.scrape_jitter_us", func(v int64) *workload.TraceRouterConfig { return &workload.TraceRouterConfig{ScrapeJitterUs: v} }},
		{"--kv-event-delay", "router.kv_event_delay_us", func(v int64) *workload.TraceRouterConfig { return &workload.TraceRouterConfig{KVEventDelayUs: v} }},
	}
	for _, tm := range timings {
		t.Run(tm.flag, func(t *testing.T) {
			for _, c := range []struct {
				v      int64
				refuse bool
			}{{-1, true}, {0, false}} {
				cliErr := routerTimingsFlagError(tm.flag, c.v)
				hdrErr := headerError(t, &workload.TraceHeader{Version: workload.TraceVersion, TimeUnit: "microseconds", Mode: "generated", Router: tm.header(c.v)})
				assertRefusal(t, tm.flag, c.v, c.refuse, cliErr, "-1")
				assertRefusal(t, tm.key, c.v, c.refuse, hdrErr, "-1")
			}
		})
	}
	// A goodput latency target: a positive span, refused at 0, below it, and non-finite. The trace
	// header encodes an absent target as 0, so its smallest refused value is the negative one.
	t.Run("latency target", func(t *testing.T) {
		for _, c := range []struct {
			ms     float64
			refuse bool
			show   string
		}{{-1, true, "-1"}, {math.Inf(1), true, "+Inf"}, {0.001, false, ""}} {
			var cliErr error
			if math.IsInf(c.ms, 0) {
				_, cliErr = parseSLODurationFlag("standard=" + strings.Repeat("9", 30) + "h")
				// time.ParseDuration refuses an overflowing duration before the validator sees it:
				// the command line cannot carry an infinite target at all.
				if cliErr == nil {
					t.Errorf("--slo-ttft: an overflowing duration was accepted")
				}
			} else {
				_, cliErr = parseSLODurationFlag("standard=" + time.Duration(c.ms*float64(time.Millisecond)).String())
				assertRefusal(t, "standard", c.ms, c.refuse, cliErr, c.show)
			}
			targets := map[string]workload.SLODimTargets{"standard": {TTFTMs: c.ms}}
			hdrErr := headerError(t, &workload.TraceHeader{Version: workload.TraceVersion, TimeUnit: "microseconds", Mode: "generated",
				GoodputSLOTargets: targets})
			assertRefusal(t, "goodput_slo_targets.standard.ttft_ms", c.ms, c.refuse, hdrErr, c.show)
			spec := &workload.WorkloadSpec{Version: "2", Category: "language", AggregateRate: 1, GoodputSLOTargets: targets,
				Clients: []workload.ClientSpec{sim11Client()}}
			assertRefusal(t, "goodput_slo_targets.standard.ttft_ms", c.ms, c.refuse, spec.Validate(), c.show)
		}
		// 0 on the command line is refused; 0 in the header is the absent target.
		if _, err := parseSLODurationFlag("standard=0s"); err == nil {
			t.Error("--slo-ttft standard=0s: a zero latency target was accepted")
		}
	})
}

// sim11Client is the smallest valid workload client: a constant one-token prompt and output.
func sim11Client() workload.ClientSpec {
	one := workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 1}}
	return workload.ClientSpec{ID: "c", RateFraction: 1, Arrival: workload.ArrivalSpec{Process: "poisson"},
		InputDist: one, OutputDist: one}
}

// routerTimingsFlagError is validateRouterFlags' check of one flag set to v, the others 0.
func routerTimingsFlagError(flag string, v int64) error {
	g, j, d := int64(0), int64(0), int64(0)
	switch flag {
	case "--gateway-latency":
		g = v
	case "--epp-scrape-jitter":
		j = v
	default:
		d = v
	}
	return routerTimingsError("--gateway-latency", "--epp-scrape-jitter", "--kv-event-delay", g, j, d)
}

// headerError loads a one-request trace with header h, and returns the load error.
func headerError(t *testing.T, h *workload.TraceHeader) error {
	t.Helper()
	dir := t.TempDir()
	hp, dp := filepath.Join(dir, "h.yaml"), filepath.Join(dir, "d.csv")
	if err := workload.ExportTraceV2(h, []workload.TraceRecord{{InputTokens: 10, OutputTokens: 2, Status: "ok"}}, hp, dp); err != nil {
		t.Fatal(err)
	}
	_, err := workload.LoadTraceV2(hp, dp)
	return err
}

// assertRefusal checks that err refuses (naming field and value) or accepts, as wanted.
func assertRefusal(t *testing.T, field string, v any, refuse bool, err error, show string) {
	t.Helper()
	switch {
	case refuse && err == nil:
		t.Errorf("%s = %v was accepted; it must be refused", field, v)
	case refuse && (!strings.Contains(err.Error(), field+": ") || !strings.Contains(err.Error(), show)):
		t.Errorf("%s = %v: the refusal must name the field and the value, got %q", field, v, err)
	case !refuse && err != nil:
		t.Errorf("%s = %v was refused; it is the closest accepted value: %v", field, v, err)
	}
}

// TestReplayCmd_RefusesNegativeHeaderRouterTiming: a trace whose router block states a negative
// timing is refused before any simulation state is built, with a message naming the header key
// and the value, rather than panicking in the event scheduler (#2007).
func TestReplayCmd_RefusesNegativeHeaderRouterTiming(t *testing.T) {
	if os.Getenv("BLIS_SIM11_SUBPROCESS") == "1" {
		dir := t.TempDir()
		hp, dp := filepath.Join(dir, "h.yaml"), filepath.Join(dir, "d.csv")
		h := &workload.TraceHeader{Version: workload.TraceVersion, TimeUnit: "microseconds", Mode: "generated",
			Router: &workload.TraceRouterConfig{GatewayLatencyUs: -1}}
		if err := workload.ExportTraceV2(h, []workload.TraceRecord{{InputTokens: 10, OutputTokens: 2, Status: "ok"}}, hp, dp); err != nil {
			os.Exit(2)
		}
		catalogDir, defaultsPath := setupKernelTestFixturesWithDefaults(t)
		runReplayCaptureStdout(t, []string{"--trace-header", hp, "--trace-data", dp, "--catalog", catalogDir,
			"--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t), "--registry", kernelRegistryDir(t),
			"--defaults-filepath", defaultsPath})
		os.Exit(0) // must Fatalf before here
	}
	out, code := runSubprocess(t, "^TestReplayCmd_RefusesNegativeHeaderRouterTiming$", "BLIS_SIM11_SUBPROCESS=1")
	if code != 1 {
		t.Fatalf("want exit 1 (logrus.Fatalf), got %d:\n%s", code, out)
	}
	if !strings.Contains(out, "router.gateway_latency_us: -1") {
		t.Errorf("the refusal must name the header key and the value, got:\n%s", out)
	}
	if strings.Contains(out, "panic") {
		t.Errorf("the replay panicked instead of refusing the trace:\n%s", out)
	}
}

// runSubprocess runs this test binary on the tests matching run, with env added, and returns its
// combined output and exit code.
func runSubprocess(t *testing.T, run string, env ...string) (string, int) {
	t.Helper()
	c := exec.Command(os.Args[0], "-test.run="+run)
	c.Env = append(os.Environ(), env...)
	out, err := c.CombinedOutput()
	var exitErr *exec.ExitError
	switch {
	case err == nil:
		return string(out), 0
	case errors.As(err, &exitErr):
		return string(out), exitErr.ExitCode()
	}
	t.Fatalf("run %s: %v", run, err)
	return "", -1
}
