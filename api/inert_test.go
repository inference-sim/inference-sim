package api

import (
	"fmt"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// importPath is this package's import path.
const importPath = "github.com/inference-sim/inference-sim/api"

// TestEnvelopeIsInert is the "deliberately inert" contract of API-1 (#1856), checked rather
// than asserted in prose: no PRODUCTION file outside api/ may import the envelope yet, so this
// PR cannot have changed what any verb reads or writes — and `blis run` stdout is
// byte-identical because nothing a shipped binary executes can even see the new types.
//
// Test files are exempt, and deliberately: ../api_schema_gate_test.go imports the envelope to
// run its CI gates from a package ci.yml's explicit test matrix already lists. A test import
// cannot change what the binary does, which is the inertness that matters here.
//
// This is scaffolding for exactly one PR. API-2 (#1857) wires the envelope into `blis run`,
// which is the moment this test SHOULD be deleted — its failure there is the signal that the
// wiring landed, not a regression. Delete it; do not add an exception list.
//
// It PARSES each file's import block rather than searching its text. A substring search over
// the quoted path reads every ordinary import — aliased, dot, grouped — but misses the one
// spelling that is still a legal import of this package, a raw-string literal in backticks,
// and an inertness check with a spelling that evades it is not a check.
func TestEnvelopeIsInert(t *testing.T) {
	scanned := 0
	fileSet := token.NewFileSet()
	err := filepath.WalkDir("..", func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			switch entry.Name() {
			case "api", ".git", ".worktrees", "site", "node_modules":
				return fs.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
			return nil
		}
		parsed, err := parser.ParseFile(fileSet, path, nil, parser.ImportsOnly)
		if err != nil {
			return fmt.Errorf("parsing %s: %w", path, err)
		}
		scanned++
		for _, spec := range parsed.Imports {
			// Unquote handles both literal forms the language allows for an import path.
			imported, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				return fmt.Errorf("%s: import path %s is not a valid string literal: %w", path, spec.Path.Value, err)
			}
			if imported == importPath {
				t.Errorf("%s imports the envelope package. API-1 is inert by contract: if you are "+
					"wiring the document into a verb (API-2 onward), delete TestEnvelopeIsInert in "+
					"the same change rather than excluding this file.", path)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking the repository: %v", err)
	}
	// Guard against the walk silently finding nothing (a moved test, a skip rule that ate
	// the tree): the repository has hundreds of Go files outside api/.
	if scanned < 100 {
		t.Errorf("non-vacuity: scanned only %d Go files outside api/, so this test proved nothing", scanned)
	}
}
