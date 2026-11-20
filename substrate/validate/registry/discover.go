// Package registry locates the committed coefficient-set files in a blis-registry
// checkout, with one discovery predicate shared by every blis-schemas surface that walks
// the registry — the validate-registry command and the committed-data test — so they
// cannot drift from each other or from the registry's own gate.
package registry

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// SetPaths returns every coefficient-set file under <root>/coefficients, discovered the
// way blis-registry's own validator (validator/validate.py) discovers them: a recursive
// walk taking both .yaml and .yml, case-insensitively, de-duplicated by resolved path,
// in deterministic sorted order. Sharing that predicate is the point — a set the
// registry's Python gate sees is one a Go consumer sees too, so a .yml set, an upper-case
// extension, or a set in a subdirectory cannot pass one gate while the other never looks
// at it. A set the gate does not see is worse than one it rejects.
//
// Like any plain directory walk it does not descend into symlinked directories, so a set
// reachable only through a linked directory is out of scope; the registry stores its sets
// as regular files. The result is empty with no error when the directory is absent, so a
// caller can treat "no sets" as its own case rather than an error.
func SetPaths(root string) ([]string, error) { return yamlPaths(root, "coefficients") }

// AcceptancePaths returns every acceptance-profile set file under <root>/acceptance, by
// the same predicate as SetPaths. Empty with no error when the directory is absent.
func AcceptancePaths(root string) ([]string, error) { return yamlPaths(root, "acceptance") }

// RoutingPaths returns every routing-profile set file under <root>/routing, by the same
// discovery rule.
func RoutingPaths(root string) ([]string, error) { return yamlPaths(root, "routing") }

func yamlPaths(root, namespace string) ([]string, error) {
	dir := filepath.Join(root, namespace)
	if _, err := os.Stat(dir); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	seen := map[string]bool{}
	var paths []string
	walkErr := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		switch strings.ToLower(filepath.Ext(p)) {
		case ".yaml", ".yml":
		default:
			return nil
		}
		// De-duplicate by resolved path, as validate.py does, so two names for one file
		// (a symlink and its target) count once rather than validating the same bytes
		// twice.
		resolved := p
		if r, e := filepath.EvalSymlinks(p); e == nil {
			resolved = r
		}
		// else: an unresolvable path (e.g. a dangling symlink) deliberately falls through
		// under its literal name, so a broken set surfaces as a LoadCoefficientSet open
		// error rather than being silently dropped here.
		if seen[resolved] {
			return nil
		}
		seen[resolved] = true
		paths = append(paths, p)
		return nil
	})
	if walkErr != nil {
		return nil, walkErr
	}
	sort.Strings(paths)
	return paths, nil
}
