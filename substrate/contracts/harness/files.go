package harness

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// GoFiles returns every non-test Go file under root, by slash-separated path relative to root.
// It skips directories whose name starts with "." or is testdata, and the directories skip names
// (relative to root, such as "cmd"). It fails rather than return nothing, since a walk that finds no
// file would make every check over it pass.
func GoFiles(root string, skip ...string) ([]string, error) {
	skipped := map[string]bool{}
	for _, s := range skip {
		skipped[filepath.ToSlash(filepath.Clean(s))] = true
	}
	var out []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			if rel != "." && (strings.HasPrefix(d.Name(), ".") || d.Name() == "testdata" || skipped[rel]) {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(rel, ".go") && !strings.HasSuffix(rel, "_test.go") {
			out = append(out, rel)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, &os.PathError{Op: "harness.GoFiles", Path: root, Err: fs.ErrNotExist}
	}
	sort.Strings(out)
	return out, nil
}
