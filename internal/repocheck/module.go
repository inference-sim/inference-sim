package repocheck

import (
	"archive/zip"
	"bytes"
	"io/fs"
	"path/filepath"

	"golang.org/x/mod/module"
	modzip "golang.org/x/mod/zip"
)

// ModuleZipLimit is SUB-13's bound on the module zip: the Go module proxy refuses a module zip
// whose uncompressed size is above 500 MB, and the design holds the module to 100 MB.
const ModuleZipLimit = 100 << 20

// NestedModules returns every go.mod below repo other than repo's own: a nested module would cut
// its directory out of the single module (SUB-13). Vendored and hidden trees are skipped.
func NestedModules(repo string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(repo, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path != repo && (d.Name() == ".git" || d.Name() == "node_modules") {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Name() == "go.mod" && filepath.Dir(path) != repo {
			rel, _ := filepath.Rel(repo, path)
			out = append(out, filepath.ToSlash(rel))
		}
		return nil
	})
	return out, err
}

// ModuleZipSize builds the module zip the way the proxy would (golang.org/x/mod/zip, which applies
// the module's own file rules) and returns the sum of its uncompressed file sizes.
func ModuleZipSize(repo, modulePath string) (int64, error) {
	var buf bytes.Buffer
	v := module.Version{Path: modulePath, Version: "v0.0.0"}
	if err := modzip.CreateFromDir(&buf, v, repo); err != nil {
		return 0, err
	}
	zr, err := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		return 0, err
	}
	var total int64
	for _, f := range zr.File {
		total += int64(f.UncompressedSize64)
	}
	return total, nil
}
