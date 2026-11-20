package artifacttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// InRepository reports whether a root is this repository's own catalog/ or registry/
// directory, the data each substrate release carries.
//
// The repository's own data is never optional: if it fails to load, the cost model and the
// data it is released with disagree, and that must fail rather than skip. A root named by
// BLIS_CATALOG or BLIS_REGISTRY can genuinely be absent, and may skip.
//
// Decided by path rather than by asking another package, so this package stays a leaf that
// every test package can import.
func InRepository(root string) bool {
	abs, err := filepath.Abs(root)
	if err != nil {
		return false
	}
	repo := repoRoot()
	if repo == "" {
		return false
	}
	for _, dir := range []string{"catalog", "registry"} {
		if abs == filepath.Join(repo, dir) || strings.HasPrefix(abs, filepath.Join(repo, dir)+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

// repoRoot finds the repository root by walking up from the working directory to the
// directory holding go.mod.
func repoRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// RequireArtifact reports a failure to load a catalog or registry artifact as either a
// skip or a test failure, depending on which of the two it actually is.
//
// An artifact under a root named by an environment variable that is NOT THERE skips: a
// candidate directory may hold only part of the data. An artifact that IS there and still
// failed to load FAILS, because that is an incompatibility between the code and the data,
// and a blanket skip over it hides it: a skip reads as a pass. An artifact in this
// repository's own catalog/ or registry/ is never optional, so any failure there fails.
//
// root is the catalog or registry root being read; path is the specific artifact whose load
// failed, which is what gets stat'ed: a root that exists says nothing about a file inside it.
func RequireArtifact(t testing.TB, root, path, what string, err error) {
	t.Helper()
	if err == nil {
		return
	}
	if InRepository(root) {
		t.Fatalf("this repository's %s failed to load %s: %v", what, path, err)
	}
	if _, statErr := os.Stat(path); os.IsNotExist(statErr) {
		t.Skipf("%s artifact %s is absent", what, path)
	}
	t.Fatalf("%s artifact %s is present but unreadable, which is an incompatibility "+
		"rather than a missing file: %v", what, path, err)
}
