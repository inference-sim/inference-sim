package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/api"
)

// This file stands in for an `./api/...` entry in .github/workflows/ci.yml's test matrix.
// That matrix lists its packages EXPLICITLY (there is no `go test ./...` in CI) and the root
// package "." is already one of them, while `./api/...` is not. The issue that introduced the
// envelope (#1856) requires CI to fail on a stale published schema, and a GitHub App token
// without `workflows` permission cannot add the matrix entry that would run the api tests
// directly. Delete this whole file when a human adds that entry — tracked by #1866.
//
// It bridges the api suite in two layers, which are deliberately not one:
//
//   - TestCommittedAPISchemaIsNotStale and TestAPIExampleDocumentsValidate express the two
//     gates #1856 NAMES, natively and in-process, over package api's exported surface only.
//     They need nothing but this test binary, so the gates the issue requires hold even where
//     the layer below cannot run.
//   - TestAPIPackageTestsPass runs the package's REAL suite as a subprocess, so a test added
//     to api/ later is covered without anyone remembering to mirror it here. Without it, this
//     file would silently cover exactly the two tests that existed when it was written.

// TestCommittedAPISchemaIsNotStale fails when api/schema/llm-d-perf-simulator-v1.json no
// longer matches the schema derived from the Go envelope types. SchemaJSON returns the
// embedded committed file, so comparing it against a fresh GenerateSchema is exactly the
// staleness check, with no path handling.
func TestCommittedAPISchemaIsNotStale(t *testing.T) {
	derived, err := api.GenerateSchema()
	if err != nil {
		t.Fatalf("api.GenerateSchema: %v", err)
	}
	if !bytes.Equal(derived, api.SchemaJSON()) {
		t.Errorf("%s is STALE: it differs from the schema derived from the Go types in package api.\n"+
			"Regenerate it with:\n    go generate ./api/...", filepath.Join("api", api.SchemaPath))
	}
}

// TestAPIExampleDocumentsValidate validates the committed example corpus against the committed
// schema, and checks the corpus still covers every kind the envelope defines.
func TestAPIExampleDocumentsValidate(t *testing.T) {
	dir := filepath.Join("api", "examples")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}
	if got, want := len(entries), len(api.AllKinds()); got != want {
		t.Errorf("%s holds %d documents, want one per kind (%d)", dir, got, want)
	}
	for _, entry := range entries {
		path := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if err := api.ValidateDocument(data); err != nil {
			t.Errorf("%s does not validate against the committed schema: %v", path, err)
		}
	}
}

// TestAPIPackageTestsPass runs `go test ./api/...` so that EVERY test in the api package runs
// wherever this package's tests run, not just the two mirrored above. The mirrors are a fixed
// list, and a fixed list silently stops covering the package the moment someone adds a test to
// it — the failure mode is a test that looks green in CI because it never ran.
//
// A missing toolchain is a failure, not a skip: a skipped bridge is indistinguishable from a
// passing one in CI output, which is the same silence this test exists to remove.
func TestAPIPackageTestsPass(t *testing.T) {
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Fatalf("cannot locate the go tool to run the api package's own tests: %v", err)
	}
	// -count=1 because a cached PASS from an earlier identical run is a correct answer, but
	// only for code identical to this commit's; the explicit flag says that is intended.
	output, err := exec.Command(goTool, "test", "-count=1", "-timeout", "3m", "./api/...").CombinedOutput()
	if err != nil {
		t.Errorf("`go test ./api/...` failed (%v). Fix it in the api package, not here:\n%s", err, output)
	}
}
