package cmd

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestSaturationStdout_FinalLabelShape is the #1517 stdout contract, replacing the
// pre-#1517 BC-8 byte-identity test (which asserted --detectors did NOT change
// stdout — deliberately false now that the final label is spliced back in).
//
// The invariant #1517 preserves is narrower: a run WITHOUT --detectors is
// byte-identical to the historical no-feature output (the "saturation" key stays
// dropped by omitempty). A run WITH --detectors ADDS a "saturation" field — a
// detector→LABEL map — whose key set is exactly the selected roster.
//
// Each variant runs in a re-exec subprocess so the real cobra tree executes and
// os.Exit(0) suppresses the test framework's own stdout.
func TestSaturationStdout_FinalLabelShape(t *testing.T) {
	// Subprocess leg: emit the metrics stdout for the variant named by the env var.
	if v := os.Getenv("BLIS_SAT_VARIANT"); v != "" {
		args := []string{
			"run", "--model", "qwen/qwen3-14b", "--seed", "42",
			"--num-requests", "50",
			"--defaults-filepath", "../defaults.yaml",
		}
		switch v {
		case "plain":
			// no saturation flags
		case "single":
			args = append(args, "--detectors", "composite", "--saturation-report", os.Getenv("BLIS_SAT_REPORT"))
		case "bank":
			args = append(args, "--detectors", "all", "--saturation-report", os.Getenv("BLIS_SAT_REPORT"))
		}
		rootCmd.SetArgs(args)
		// Exit non-zero on a non-fatal Execute error (e.g. cobra flag parsing) so the
		// parent sees a failed subprocess rather than silently comparing partial or
		// empty stdout.
		if err := rootCmd.Execute(); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}

	runVariant := func(variant string) string {
		t.Helper()
		cmd := exec.Command(os.Args[0], "-test.run=TestSaturationStdout_FinalLabelShape")
		env := append(os.Environ(), "BLIS_SAT_VARIANT="+variant)
		if variant != "plain" {
			// Each leg writes to its own trace file so the file write is exercised
			// but never collides; the file path never appears on stdout.
			env = append(env, "BLIS_SAT_REPORT="+filepath.Join(t.TempDir(), variant+".json"))
		}
		cmd.Env = env
		var stdout bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = io.Discard // logrus diagnostics go to stderr, not part of INV-6 stdout
		if err := cmd.Run(); err != nil {
			t.Fatalf("subprocess (%s) failed: %v\nstdout:\n%s", variant, err, stdout.String())
		}
		return stdout.String()
	}

	// parseSaturation extracts the "saturation" field (nil if absent) from the
	// "=== Simulation Metrics ===" JSON block on stdout.
	parseSaturation := func(t *testing.T, stdout string) (map[string]string, bool) {
		t.Helper()
		const marker = "=== Simulation Metrics ==="
		idx := bytes.Index([]byte(stdout), []byte(marker))
		if idx < 0 {
			t.Fatalf("stdout missing metrics marker:\n%s", stdout)
		}
		jsonStr := stdout[idx+len(marker):]
		var parsed struct {
			Saturation map[string]string `json:"saturation"`
		}
		// The metrics block is the first JSON object after the marker; decode with a
		// streaming decoder so any trailing session/goodput text is ignored.
		dec := json.NewDecoder(bytes.NewReader([]byte(jsonStr)))
		if err := dec.Decode(&parsed); err != nil {
			t.Fatalf("decode metrics JSON: %v\n%s", err, jsonStr)
		}
		return parsed.Saturation, parsed.Saturation != nil
	}

	plain := runVariant("plain")
	if plain == "" {
		t.Fatal("plain run produced empty stdout")
	}

	// A run WITHOUT --detectors must carry no saturation field (omitempty drops it).
	if _, present := parseSaturation(t, plain); present {
		t.Errorf("plain run (no --detectors) must not emit a saturation field, but one was present")
	}

	// --detectors composite → a one-key map keyed exactly by "composite".
	single := runVariant("single")
	singleSat, present := parseSaturation(t, single)
	if !present {
		t.Fatalf("--detectors composite must emit a saturation field; stdout:\n%s", single)
	}
	if len(singleSat) != 1 || singleSat["composite"] == "" {
		t.Errorf("--detectors composite saturation = %v, want a single \"composite\" key with a non-empty label", singleSat)
	}

	// --detectors all → the full roster's three keys, each with a valid label.
	bankStdout := runVariant("bank")
	bankSat, present := parseSaturation(t, bankStdout)
	if !present {
		t.Fatalf("--detectors all must emit a saturation field; stdout:\n%s", bankStdout)
	}
	validLabels := map[string]bool{"STABLE": true, "BACKLOGGED": true, "OVERLOADED": true}
	for _, name := range []string{"composite", "threshold", "backlog-drift"} {
		label, ok := bankSat[name]
		if !ok {
			t.Errorf("--detectors all saturation missing key %q; got %v", name, bankSat)
			continue
		}
		if !validLabels[label] {
			t.Errorf("--detectors all: detector %q has invalid label %q", name, label)
		}
	}
	if len(bankSat) != 3 {
		t.Errorf("--detectors all saturation should have exactly 3 keys, got %d (%v)", len(bankSat), bankSat)
	}

	// INV-6: two identical --detectors all runs produce byte-identical stdout.
	if again := runVariant("bank"); again != bankStdout {
		t.Errorf("INV-6 VIOLATION: two identical --detectors all runs differ.\n--- run 1 ---\n%s\n--- run 2 ---\n%s", bankStdout, again)
	}
}
