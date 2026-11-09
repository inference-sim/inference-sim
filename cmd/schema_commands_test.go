package cmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/schema"
)

// schema <document> prints the schema the pinned substrate release publishes, byte for byte,
// for every document it names.
func TestSchemaPrintsThePublishedSchema(t *testing.T) {
	for _, name := range schema.Names() {
		want, err := schema.Published(name)
		if err != nil {
			t.Fatal(err)
		}
		var out bytes.Buffer
		schemaCmd.SetOut(&out)
		if err := schemaCmd.RunE(schemaCmd, []string{name}); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !bytes.Equal(out.Bytes(), want) {
			t.Errorf("%s: printed %d bytes, the release publishes %d", name, out.Len(), len(want))
		}
		var doc map[string]any
		if err := json.Unmarshal(out.Bytes(), &doc); err != nil || doc["$defs"] == nil {
			t.Errorf("%s: not a JSON Schema with $defs: %v", name, err)
		}
	}
	if err := schemaCmd.RunE(schemaCmd, []string{"no-such-document"}); err == nil ||
		!strings.Contains(err.Error(), "documents are") {
		t.Errorf("an unknown document must be refused naming the documents, got %v", err)
	}
}

// validate accepts the repository's own scenarios, refuses an unknown key and a rule the
// schema cannot state, and tells a refused document (exit 1) from a command that could not
// run (exit 2).
func TestSchemaValidate(t *testing.T) {
	good := filepath.Join("..", "testdata", "scenarios", "deepseek-v4.1-flash-h100-1p1d-dspark5-cpu1t.yaml")
	if _, err := os.Stat(good); err != nil {
		t.Fatalf("fixture: %v", err)
	}
	if err := validateDocument("scenario", good); err != nil {
		t.Fatalf("a repository scenario must validate: %v", err)
	}
	data, err := os.ReadFile(good)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	unknown := filepath.Join(dir, "unknown.yaml")
	if err := os.WriteFile(unknown, append([]byte("no_such_key: 1\n"), data...), 0o644); err != nil {
		t.Fatal(err)
	}
	err = validateDocument("scenario", unknown)
	if !errors.Is(err, errDocumentInvalid) || !strings.Contains(err.Error(), "no_such_key") {
		t.Errorf("an unknown key must be refused by name, got %v", err)
	}
	if documentExitCode(err) != 1 {
		t.Errorf("a refused document exits 1, got %d", documentExitCode(err))
	}
	// A value the decoder reads but a validator rule refuses: the cluster has a negative node count.
	negative := filepath.Join(dir, "negative.yaml")
	if !bytes.Contains(data, []byte("  nodes: 2\n")) {
		t.Fatal("fixture no longer states the cluster's node count")
	}
	if err := os.WriteFile(negative, bytes.Replace(data, []byte("  nodes: 2\n"), []byte("  nodes: -1\n"), 1), 0o644); err != nil {
		t.Fatal(err)
	}
	err = validateDocument("scenario", negative)
	if !errors.Is(err, errDocumentInvalid) || !strings.Contains(err.Error(), "nodes") {
		t.Errorf("a negative node count must be refused by the validator, naming nodes, got %v", err)
	}
	for _, c := range []struct {
		name, document, path string
	}{
		{"a missing file", "scenario", filepath.Join(dir, "absent.yaml")},
		{"an unknown document", "nonesuch", good},
	} {
		err := validateDocument(c.document, c.path)
		if err == nil || documentExitCode(err) != 2 {
			t.Errorf("%s must fail with exit 2, got %v (%d)", c.name, err, documentExitCode(err))
		}
	}
}

// Every document with a validator has a published schema, so the two commands name the same
// documents.
func TestEveryValidatedDocumentHasASchema(t *testing.T) {
	for _, name := range validatable() {
		if _, err := schema.Published(name); err != nil {
			t.Errorf("%s has a validator but no published schema: %v", name, err)
		}
	}
}
