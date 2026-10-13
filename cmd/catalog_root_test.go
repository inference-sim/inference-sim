package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestCatalogRootFrom_Precedence is AC-1 and AC-2 of #1731 (R1/S4) as a pure law:
// --catalog and BLIS_CATALOG both locate the catalog, --catalog wins when both are set,
// and NEITHER is refused naming both forms. catalogRootFrom takes both inputs as
// arguments, so the law is table-testable without touching globals or the environment.
func TestCatalogRootFrom_Precedence(t *testing.T) {
	tests := []struct {
		name    string
		flag    string
		env     string
		want    string
		wantErr bool
	}{
		{"flag only", "/from/flag", "", "/from/flag", false},
		{"env only", "", "/from/env", "/from/env", false},
		{"both set: flag wins", "/from/flag", "/from/env", "/from/flag", false},
		{"both set to the same path", "/same", "/same", "/same", false},
		{"neither set: refused", "", "", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := catalogRootFrom(tt.flag, tt.env)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected a refusal, got %q", got)
				}
				// AC-2: the refusal must name BOTH forms so the operator knows either works.
				if !strings.Contains(err.Error(), "--catalog") {
					t.Errorf("refusal must name --catalog, got: %v", err)
				}
				if !strings.Contains(err.Error(), catalogEnvVar) {
					t.Errorf("refusal must name %s, got: %v", catalogEnvVar, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("catalogRootFrom(%q, %q) = %q, want %q", tt.flag, tt.env, got, tt.want)
			}
		})
	}
}

// TestResolveCatalogRoot_ReadsFlagAndEnv is AC-1 through the production wrapper, which
// is the only place the --catalog package var and os.Getenv(BLIS_CATALOG) are read.
func TestResolveCatalogRoot_ReadsFlagAndEnv(t *testing.T) {
	origFlag := catalogPath
	t.Cleanup(func() { catalogPath = origFlag })

	flagDir := t.TempDir()
	envDir := t.TempDir()

	// Env only.
	catalogPath = ""
	t.Setenv(catalogEnvVar, envDir)
	got, err := resolveCatalogRoot()
	if err != nil {
		t.Fatalf("env-only: unexpected error: %v", err)
	}
	if got != envDir {
		t.Errorf("env-only: got %q, want %q", got, envDir)
	}

	// Flag wins over env.
	catalogPath = flagDir
	got, err = resolveCatalogRoot()
	if err != nil {
		t.Fatalf("flag+env: unexpected error: %v", err)
	}
	if got != flagDir {
		t.Errorf("flag+env: got %q, want %q (--catalog must win)", got, flagDir)
	}

	// Neither: refused. There is no working-directory default.
	catalogPath = ""
	t.Setenv(catalogEnvVar, "")
	if got, err = resolveCatalogRoot(); err == nil {
		t.Errorf("neither --catalog nor %s set must be refused, got %q", catalogEnvVar, got)
	}
}

// TestResolveCatalogRoot_RejectsUnusableRoot: a mistyped catalog path is refused as a
// CATALOG error naming both forms, not deferred into a per-model "not in the catalog"
// message that would blame the model (R1).
func TestResolveCatalogRoot_RejectsUnusableRoot(t *testing.T) {
	origFlag := catalogPath
	t.Cleanup(func() { catalogPath = origFlag })
	t.Setenv(catalogEnvVar, "")

	fileNotDir := filepath.Join(t.TempDir(), "catalog-is-a-file")
	if err := os.WriteFile(fileNotDir, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct{ name, root string }{
		{"missing directory", filepath.Join(t.TempDir(), "no-such-catalog")},
		{"not a directory", fileNotDir},
	} {
		t.Run(tt.name, func(t *testing.T) {
			catalogPath = tt.root
			got, err := resolveCatalogRoot()
			if err == nil {
				t.Fatalf("expected refusal for %s, got %q", tt.name, got)
			}
			if !strings.Contains(err.Error(), tt.root) {
				t.Errorf("refusal must name the offending root %q, got: %v", tt.root, err)
			}
			if !strings.Contains(err.Error(), "--catalog") || !strings.Contains(err.Error(), catalogEnvVar) {
				t.Errorf("refusal must name both --catalog and %s, got: %v", catalogEnvVar, err)
			}
		})
	}
}
