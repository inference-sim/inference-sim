package cmd

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestSaturationBC8_StdoutByteIdenticalWithoutAndWithDetectors is the BC-8
// regression guard: a run with --detectors (single or bank) must produce stdout
// byte-identical to the same run without it. The saturation feature's only output
// is the trace FILE; nothing saturation-related may reach stdout (the MetricsOutput
// Saturation field stays unset ⇒ omitempty drops it), so adding --detectors must
// not perturb the deterministic metrics JSON.
//
// This matters specifically because #1517 will deliberately repopulate
// output.Saturation through the same BuildOutput seam (currently passed nil). If
// that wiring ever leaks onto stdout — or an accidental non-omitempty change
// does — this test fails, catching the regression before it changes every run's
// stdout.
//
// The comparison is EXACT byte-identity (not the tolerant golden comparison used
// for the cross-architecture LoRA baseline): all three runs execute on the same
// machine in the same process style, so INV-6 byte-identity across runs holds
// strictly. Each variant runs in a re-exec subprocess so the real cobra tree
// executes and os.Exit(0) suppresses the test framework's own stdout.
func TestSaturationBC8_StdoutByteIdenticalWithoutAndWithDetectors(t *testing.T) {
	// Subprocess leg: emit the metrics stdout for the variant named by the env var.
	if v := os.Getenv("BLIS_BC8_VARIANT"); v != "" {
		args := []string{
			"run", "--model", "qwen/qwen3-14b", "--seed", "42",
			"--num-requests", "50",
			"--defaults-filepath", "../defaults.yaml",
		}
		switch v {
		case "plain":
			// no saturation flags
		case "single":
			args = append(args, "--detectors", "composite", "--saturation-report", os.Getenv("BLIS_BC8_REPORT"))
		case "bank":
			args = append(args, "--detectors", "all", "--saturation-report", os.Getenv("BLIS_BC8_REPORT"))
		}
		rootCmd.SetArgs(args)
		_ = rootCmd.Execute()
		os.Exit(0)
	}

	runVariant := func(variant string) string {
		t.Helper()
		cmd := exec.Command(os.Args[0], "-test.run=TestSaturationBC8_StdoutByteIdenticalWithoutAndWithDetectors")
		env := append(os.Environ(), "BLIS_BC8_VARIANT="+variant)
		if variant != "plain" {
			// Each leg writes to its own trace file so the file write is exercised
			// but never collides; the file path never appears on stdout.
			env = append(env, "BLIS_BC8_REPORT="+filepath.Join(t.TempDir(), variant+".json"))
		}
		cmd.Env = env
		var stdout bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = io.Discard // logrus diagnostics (incl. the no-report warning) go to stderr, not part of INV-6 stdout
		if err := cmd.Run(); err != nil {
			t.Fatalf("subprocess (%s) failed: %v\nstdout:\n%s", variant, err, stdout.String())
		}
		return stdout.String()
	}

	plain := runVariant("plain")
	if plain == "" {
		t.Fatal("plain run produced empty stdout")
	}

	if got := runVariant("single"); got != plain {
		t.Errorf("BC-8 VIOLATION: --detectors composite changed stdout.\n--- plain ---\n%s\n--- single ---\n%s", plain, got)
	}
	if got := runVariant("bank"); got != plain {
		t.Errorf("BC-8 VIOLATION: --detectors all changed stdout.\n--- plain ---\n%s\n--- bank ---\n%s", plain, got)
	}
}
