package sim

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// captureStdout runs fn with os.Stdout redirected to a pipe and returns everything
// written. EmitOutput writes the stdout channel with fmt.Println, so this is the only
// way to observe the channel INV-6 protects.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		done <- buf.String()
	}()
	fn()
	_ = w.Close()
	os.Stdout = old
	return <-done
}

// TestEmitOutput_CatalogProvenance_FileOnly pins the AC-1 + AC-4 pair for #1732: the
// catalog provenance block reaches the --metrics-path file with all three fields, and
// NEVER reaches stdout (INV-6). Same file-only discipline as cache_hit_rate (#1583).
func TestEmitOutput_CatalogProvenance_FileOnly(t *testing.T) {
	m := NewMetrics()
	out := m.BuildOutput("cluster")
	if out.Catalog != nil {
		t.Fatalf("BuildOutput (the stdout shape) must not set catalog provenance; got %+v", out.Catalog)
	}

	dir := t.TempDir()
	fpath := filepath.Join(dir, "metrics.json")
	prov := &CatalogProvenance{Path: "/models/catalog", Revision: "deadbeefcafe", Dirty: true}

	stdout := captureStdout(t, func() {
		if err := m.EmitOutput(out, fpath, WithCatalogProvenance(prov)); err != nil {
			t.Errorf("EmitOutput: %v", err)
		}
	})

	// AC-4: stdout carries no trace of the provenance — not the key, not any value.
	for _, forbidden := range []string{`"catalog"`, "/models/catalog", "deadbeefcafe", `"dirty"`, `"revision"`} {
		if strings.Contains(stdout, forbidden) {
			t.Errorf("INV-6 VIOLATION: stdout contains provenance fragment %q:\n%s", forbidden, stdout)
		}
	}

	// AC-1: the file records path, revision and dirty.
	var fileOut MetricsOutput
	data, err := os.ReadFile(fpath)
	if err != nil {
		t.Fatalf("read metrics file: %v", err)
	}
	if err := json.Unmarshal(data, &fileOut); err != nil {
		t.Fatalf("unmarshal metrics file: %v", err)
	}
	if fileOut.Catalog == nil {
		t.Fatalf("metrics-path file must record catalog provenance:\n%s", data)
	}
	if fileOut.Catalog.Path != prov.Path {
		t.Errorf("file catalog path = %q, want %q", fileOut.Catalog.Path, prov.Path)
	}
	if fileOut.Catalog.Revision != prov.Revision {
		t.Errorf("file catalog revision = %q, want %q", fileOut.Catalog.Revision, prov.Revision)
	}
	if !fileOut.Catalog.Dirty {
		t.Errorf("file catalog dirty = false, want true")
	}
}

// TestEmitOutput_CatalogProvenance_DirtyFlagReachesFile is the AC-3 discriminator at the
// sim/ boundary: two emissions differing ONLY in the dirty flag produce results files
// that differ in that flag, while their stdout is byte-identical. A dirty catalog must be
// distinguishable from a reproducible run in the file, and indistinguishable on stdout.
func TestEmitOutput_CatalogProvenance_DirtyFlagReachesFile(t *testing.T) {
	dir := t.TempDir()
	const rev = "1111111111111111111111111111111111111111"

	emit := func(name string, dirty bool) (stdout string, file MetricsOutput) {
		m := NewMetrics()
		out := m.BuildOutput("cluster")
		fpath := filepath.Join(dir, name)
		stdout = captureStdout(t, func() {
			err := m.EmitOutput(out, fpath, WithCatalogProvenance(
				&CatalogProvenance{Path: dir, Revision: rev, Dirty: dirty}))
			if err != nil {
				t.Errorf("EmitOutput: %v", err)
			}
		})
		data, err := os.ReadFile(fpath)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if err := json.Unmarshal(data, &file); err != nil {
			t.Fatalf("unmarshal %s: %v", name, err)
		}
		return stdout, file
	}

	cleanStdout, cleanFile := emit("clean.json", false)
	dirtyStdout, dirtyFile := emit("dirty.json", true)

	if cleanStdout != dirtyStdout {
		t.Errorf("INV-6 VIOLATION: stdout differs between a clean and a dirty catalog:\n"+
			"--- clean ---\n%s\n--- dirty ---\n%s", cleanStdout, dirtyStdout)
	}
	if cleanFile.Catalog == nil || dirtyFile.Catalog == nil {
		t.Fatalf("both results files must record provenance; clean=%+v dirty=%+v",
			cleanFile.Catalog, dirtyFile.Catalog)
	}
	if cleanFile.Catalog.Dirty {
		t.Errorf("clean emission recorded dirty = true")
	}
	if !dirtyFile.Catalog.Dirty {
		t.Errorf("dirty emission recorded dirty = false — an experiment would be "+
			"indistinguishable from a reproducible run at revision %s", rev)
	}
}

// TestEmitOutput_CatalogProvenance_AbsentWithoutOption pins the INV-6 no-op: an
// EmitOutput call with no provenance option (every pre-#1732 call site) writes a
// results file with no catalog block at all, so omitempty keeps the file
// byte-identical to a pre-feature build. A nil provenance behaves the same, so a
// caller that resolved no catalog need not branch.
func TestEmitOutput_CatalogProvenance_AbsentWithoutOption(t *testing.T) {
	dir := t.TempDir()
	for _, tc := range []struct {
		name string
		opts []EmitOption
	}{
		{"no-option", nil},
		{"nil-provenance", []EmitOption{WithCatalogProvenance(nil)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := NewMetrics()
			out := m.BuildOutput("cluster")
			fpath := filepath.Join(dir, tc.name+".json")
			_ = captureStdout(t, func() {
				if err := m.EmitOutput(out, fpath, tc.opts...); err != nil {
					t.Errorf("EmitOutput: %v", err)
				}
			})
			data, err := os.ReadFile(fpath)
			if err != nil {
				t.Fatalf("read metrics file: %v", err)
			}
			if strings.Contains(string(data), `"catalog"`) {
				t.Errorf("results file must omit the catalog block when no provenance is "+
					"supplied:\n%s", data)
			}
		})
	}
}

// TestEmitOutput_CatalogProvenance_StdoutOnlyIsSafe pins that supplying the option with
// no output file path is a harmless no-op (file-only metadata, nothing to write) and
// leaves stdout free of the provenance.
func TestEmitOutput_CatalogProvenance_StdoutOnlyIsSafe(t *testing.T) {
	m := NewMetrics()
	out := m.BuildOutput("cluster")
	stdout := captureStdout(t, func() {
		err := m.EmitOutput(out, "", WithCatalogProvenance(
			&CatalogProvenance{Path: "/models/catalog", Revision: "abc123", Dirty: true}))
		if err != nil {
			t.Errorf("EmitOutput with no file path must not fail: %v", err)
		}
	})
	if strings.Contains(stdout, "/models/catalog") || strings.Contains(stdout, "abc123") {
		t.Errorf("INV-6 VIOLATION: provenance leaked to stdout:\n%s", stdout)
	}
}

// TestEmitOutput_CatalogProvenance_DoesNotMutateCaller pins that EmitOutput's file-only
// assignment cannot be observed by the caller — output is passed by value, so a caller
// that emits to stdout twice (per-instance then aggregate) cannot pick up provenance
// from an earlier file emission.
func TestEmitOutput_CatalogProvenance_DoesNotMutateCaller(t *testing.T) {
	m := NewMetrics()
	out := m.BuildOutput("cluster")
	fpath := filepath.Join(t.TempDir(), "metrics.json")
	_ = captureStdout(t, func() {
		err := m.EmitOutput(out, fpath, WithCatalogProvenance(
			&CatalogProvenance{Path: "/models/catalog", Revision: "abc123"}))
		if err != nil {
			t.Errorf("EmitOutput: %v", err)
		}
	})
	if out.Catalog != nil {
		t.Errorf("EmitOutput must not mutate the caller's MetricsOutput; got %+v", out.Catalog)
	}
}
