package cmd

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// catalogLegEnv selects which leg of TestRunCmd_CatalogLocation the re-exec subprocess
// executes.
const catalogLegEnv = "BLIS_CATALOG_LEG"

// TestRunCmd_CatalogLocation is the CLI-level companion to the unit tests on
// catalogRootFrom / resolveCatalogRoot: it proves the catalog requirement is actually
// wired into `blis run`, not merely available as a helper (#1731).
//
//	AC-1 — `--catalog` locates the catalog; so does `BLIS_CATALOG`.
//	AC-2 — a run with NEITHER is refused, naming both forms.
//
// Each leg runs in a re-exec subprocess so the real cobra tree executes and a
// logrus.Fatalf surfaces as exit status 1. The subprocess environment always sets
// BLIS_CATALOG explicitly (to "" where the leg wants it absent), so an ambient value in
// the developer's or CI's environment cannot make the refusal leg pass vacuously.
func TestRunCmd_CatalogLocation(t *testing.T) {
	if leg := os.Getenv(catalogLegEnv); leg != "" {
		args := []string{
			"run", "--model", "qwen/qwen3-14b", "--seed", "42",
			"--num-requests", "5",
			"--defaults-filepath", "../defaults.yaml",
		}
		if leg == "flag" {
			args = append(args, "--catalog", "../model_configs")
		}
		rootCmd.SetArgs(args)
		if err := rootCmd.Execute(); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}

	tests := []struct {
		name      string
		leg       string
		env       string // BLIS_CATALOG value for the subprocess ("" = unset-equivalent)
		wantFatal bool
	}{
		{name: "flag locates the catalog", leg: "flag", wantFatal: false},
		{name: "BLIS_CATALOG locates the catalog", leg: "env", env: "../model_configs", wantFatal: false},
		{name: "neither is refused", leg: "none", wantFatal: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=TestRunCmd_CatalogLocation")
			cmd.Env = append(os.Environ(), catalogLegEnv+"="+tt.leg, catalogEnvVar+"="+tt.env)
			var stderr bytes.Buffer
			cmd.Stdout = io.Discard
			cmd.Stderr = &stderr
			err := cmd.Run()

			if !tt.wantFatal {
				if err != nil {
					t.Fatalf("expected the run to succeed, got %v\nstderr:\n%s", err, stderr.String())
				}
				return
			}
			if err == nil {
				t.Fatalf("expected the run to be refused when neither --catalog nor %s is set\nstderr:\n%s",
					catalogEnvVar, stderr.String())
			}
			// The refusal must name BOTH forms, so the operator learns either works.
			msg := stderr.String()
			if !strings.Contains(msg, "--catalog") {
				t.Errorf("refusal must name --catalog; stderr:\n%s", msg)
			}
			if !strings.Contains(msg, catalogEnvVar) {
				t.Errorf("refusal must name %s; stderr:\n%s", catalogEnvVar, msg)
			}
			// And it must NOT silently fall back to a working-directory location.
			if strings.Contains(msg, "using config from") {
				t.Errorf("refusal leg resolved a config from a working-directory default; stderr:\n%s", msg)
			}
		})
	}
}
