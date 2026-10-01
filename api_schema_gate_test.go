package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/api"
)

// The two gates below belong to package api and are also tested there, in full. They are
// MIRRORED here because .github/workflows/ci.yml's test job lists its packages EXPLICITLY
// (there is no `go test ./...` in CI), and the root package "." is already one of them, while
// `./api/...` is not. The issue that introduced the envelope (#1856) requires CI to fail on a
// stale published schema, and a GitHub App token without `workflows` permission cannot add the
// matrix entry that would run the api tests directly.
//
// Replace this file with a `./api/...` entry in ci.yml's matrix (and `api` in its
// pull_request branch list, since epic #1855 targets that branch) the first time a human edits
// that workflow. Keeping both is harmless but redundant.
//
// They use only package api's exported surface, so they are a gate, not a copy of its tests.

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
