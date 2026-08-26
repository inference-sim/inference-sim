package cmd

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim/saturation"
)

// TestSaturationCLI_RunReplayParity closes INV-13 through the ACTUAL command paths:
// it executes `blis run --trace-output ... --saturation-report ...` and then
// `blis replay` over that exported trace, and requires the two saturation reports to
// be byte-identical.
//
// This is deliberately stronger than the in-process parity test
// (TestSaturationKnobs_RunReplayReportsAreByteIdentical), which drives
// resolveSaturation and saturationTracer.run directly. That test proves the shared
// downstream pipeline is deterministic given equivalent inputs; it cannot prove that
// the run and replay COMMANDS assemble equivalent inputs in the first place. Only
// executing both commands does, because everything upstream of the shared tracer --
// flag registration, trace export, trace parsing, the completed-request extractor on
// each side -- differs between them.
//
// Each leg runs in a re-exec subprocess so the real cobra tree executes, following
// the harness established by TestSaturationStdout_FinalLabelShape.
func TestSaturationCLI_RunReplayParity(t *testing.T) {
	// Subprocess leg: execute the command named by the env var.
	if leg := os.Getenv("BLIS_PARITY_LEG"); leg != "" {
		common := []string{
			"--model", "qwen/qwen3-14b",
			"--defaults-filepath", "../defaults.yaml",
			"--detectors", os.Getenv("BLIS_PARITY_DETECTORS"),
			"--saturation-config", os.Getenv("BLIS_PARITY_CONFIG"),
			"--saturation-report", os.Getenv("BLIS_PARITY_REPORT"),
		}
		var args []string
		switch leg {
		case "run":
			args = append([]string{"run", "--seed", "42", "--num-requests", "400", "--rate", "16",
				"--trace-output", os.Getenv("BLIS_PARITY_TRACE")}, common...)
		case "replay":
			args = append([]string{"replay",
				"--trace-header", os.Getenv("BLIS_PARITY_TRACE") + ".yaml",
				"--trace-data", os.Getenv("BLIS_PARITY_TRACE") + ".csv"}, common...)
		}
		rootCmd.SetArgs(args)
		if err := rootCmd.Execute(); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}

	dir := t.TempDir()
	tracePrefix := filepath.Join(dir, "trace")

	// Every detector's own knob block: ownership is enforced over the selected set,
	// so a single-detector selection may carry only its own block.
	blocks := map[string]string{
		"composite":     "composite:\n  sensitivity: 2.0\n",
		"threshold":     "threshold:\n  threshold_ms: 250\n",
		"backlog-drift": "backlog_drift:\n  slope_k: 4.0\n",
		"peak-rate":     "peak_rate:\n  threshold: 0.25\n  min_observations: 10\n  warmup_us: 1000\n",
	}

	execLeg := func(t *testing.T, leg, detectors, cfgPath, reportPath string) {
		t.Helper()
		cmd := exec.Command(os.Args[0], "-test.run=TestSaturationCLI_RunReplayParity")
		cmd.Env = append(os.Environ(),
			"BLIS_PARITY_LEG="+leg,
			"BLIS_PARITY_DETECTORS="+detectors,
			"BLIS_PARITY_CONFIG="+cfgPath,
			"BLIS_PARITY_REPORT="+reportPath,
			"BLIS_PARITY_TRACE="+tracePrefix,
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s leg (--detectors %s) failed: %v\n%s", leg, detectors, err, out)
		}
	}

	for _, detectors := range append(saturation.AllDetectorNames(), "all") {
		t.Run(detectors, func(t *testing.T) {
			body := blocks[detectors]
			if detectors == "all" {
				body = ""
				for _, n := range saturation.AllDetectorNames() {
					body += blocks[n]
				}
			}
			cfgPath := filepath.Join(dir, "sat.yaml")
			if err := os.WriteFile(cfgPath, []byte(body), 0o600); err != nil {
				t.Fatalf("write config: %v", err)
			}

			runReport := filepath.Join(dir, detectors+"-run.json")
			replayReport := filepath.Join(dir, detectors+"-replay.json")

			execLeg(t, "run", detectors, cfgPath, runReport)
			execLeg(t, "replay", detectors, cfgPath, replayReport)

			runBytes, err := os.ReadFile(runReport)
			if err != nil {
				t.Fatalf("read run report: %v", err)
			}
			replayBytes, err := os.ReadFile(replayReport)
			if err != nil {
				t.Fatalf("read replay report: %v", err)
			}
			if len(runBytes) == 0 {
				t.Fatal("the run leg wrote an empty report; the comparison would be vacuous")
			}
			if string(runBytes) != string(replayBytes) {
				t.Errorf("run and replay saturation reports differ (INV-13)\nrun     %d bytes\nreplay  %d bytes", len(runBytes), len(replayBytes))
			}
		})
	}
}
