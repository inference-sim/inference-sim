package kvkey

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// forbiddenHashImport reports whether an import path is a hashing primitive that
// production code must NOT import directly — all hashing must route through
// sim/internal/hash (BC-K1). Covers crypto/* (e.g. crypto/sha256), the stdlib
// hash package and its subpackages (hash, hash/fnv, hash/crc32, …).
func forbiddenHashImport(path string) bool {
	return path == "hash" ||
		strings.HasPrefix(path, "hash/") ||
		strings.HasPrefix(path, "crypto/")
}

// scanForbiddenHashImports parses every non-test .go file in dir and returns the
// count of files scanned plus any (file, import) pairs that violate BC-K1.
func scanForbiddenHashImports(t *testing.T, dir string) (scanned int, violations []string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("reading %s: %v", dir, err)
	}
	fset := token.NewFileSet()
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		p := filepath.Join(dir, name)
		f, err := parser.ParseFile(fset, p, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("parsing %s: %v", p, err)
		}
		scanned++
		for _, imp := range f.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			if forbiddenHashImport(path) {
				violations = append(violations, p+" imports "+path)
			}
		}
	}
	return scanned, violations
}

// TestNoForbiddenHashImports is the BC-K1 static gate: production code in sim/kv
// and in this box (sim/internal/kvkey) must not import crypto/*, the stdlib hash
// package, or fnv — hashing routes exclusively through sim/internal/hash. This
// promotes the "one hash source" convention into an enforced contract.
func TestNoForbiddenHashImports(t *testing.T) {
	// Test cwd is the package dir (sim/internal/kvkey).
	dirs := map[string]string{
		"sim/kv":             filepath.Join("..", "..", "kv"),
		"sim/internal/kvkey": ".",
	}
	for label, dir := range dirs {
		scanned, violations := scanForbiddenHashImports(t, dir)
		// Anti-vacuous guard: a wrong path must not let the gate silently pass.
		if scanned == 0 {
			t.Fatalf("%s (%s): scanned 0 files — check the directory path", label, dir)
		}
		for _, v := range violations {
			t.Errorf("%s: forbidden hashing import — %s (route through sim/internal/hash)", label, v)
		}
	}
}

// TestForbiddenHashImport_Discriminates guards the gate's own logic: it must flag
// the known-forbidden paths and allow the sanctioned hash source and neighbors.
func TestForbiddenHashImport_Discriminates(t *testing.T) {
	forbidden := []string{"crypto/sha256", "crypto/md5", "hash", "hash/fnv", "hash/crc32"}
	for _, p := range forbidden {
		if !forbiddenHashImport(p) {
			t.Errorf("expected %q to be forbidden", p)
		}
	}
	allowed := []string{
		"github.com/inference-sim/inference-sim/sim/internal/hash",
		"github.com/inference-sim/inference-sim/sim/internal/tokenid",
		"fmt", "strings", "encoding/hex",
	}
	for _, p := range allowed {
		if forbiddenHashImport(p) {
			t.Errorf("expected %q to be allowed", p)
		}
	}
}
