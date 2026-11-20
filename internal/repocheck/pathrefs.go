package repocheck

import (
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
)

// enginesPath matches a path that begins at, or goes through, the engines/ directory: an import
// path, an open() argument, a documentation link, a data file's field.
var enginesPath = regexp.MustCompile(`(^|[^A-Za-z0-9_-])engines/[A-Za-z0-9_]`)

// PathReferences is the path-reference lint of X7: no file under <repo>/substrate, in any
// language or data format, names a path under engines/. A fact the substrate needs from an
// engine reaches it as a generated, engine-neutral file under substrate/registry/inputs/, or
// through the simulator. A repository without a substrate/ directory has nothing to report.
func PathReferences(repo string) ([]Violation, error) {
	root := filepath.Join(repo, "substrate")
	if _, err := os.Stat(root); os.IsNotExist(err) {
		return nil, nil
	}
	var out []Violation
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "node_modules", ".venv":
				return filepath.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if bytes.IndexByte(raw[:min(len(raw), 8192)], 0) >= 0 {
			return nil
		}
		loc := enginesPath.FindIndex(raw)
		if loc == nil {
			return nil
		}
		rel, err := filepath.Rel(repo, path)
		if err != nil {
			return err
		}
		line := 1 + bytes.Count(raw[:loc[0]], []byte{'\n'})
		out = append(out, Violation{
			Where: filepath.ToSlash(rel) + ":" + strconv.Itoa(line),
			What:  "names a path under engines/; the substrate reads no file of an engine module",
			Rule:  "X7",
		})
		return nil
	})
	return out, err
}
