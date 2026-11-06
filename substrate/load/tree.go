package load

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"sort"
	"strings"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
)

// tree is the data files of one catalog, registry or overlay directory: each file's path
// relative to the directory, with forward slashes, and its bytes.
type tree struct {
	// label names the tree in an error: "the embedded catalog", or the directory's path.
	label string
	files map[string][]byte
}

// rules are the file patterns of one kind of tree, and the paths inside its data
// directories that are ignored by name.
type rules struct {
	patterns []string
	ignored  func(rel string) bool
	// refused names top-level directories whose every file is refused, with the reason.
	refused map[string]string
	// extraDirs are data directories deeper than the top level that no pattern names yet:
	// every file in one must match a pattern, so a file there before its pattern exists is
	// refused rather than skipped.
	extraDirs []string
}

// inData reports whether rel is in a data directory, and whether it is an ancestor of one
// (a directory the walk must enter to reach an extra data directory).
func (r rules) inData(dirs map[string]bool, rel string) (in, ancestor bool) {
	first, _, _ := strings.Cut(rel, "/")
	if dirs[first] {
		return true, false
	}
	for _, e := range r.extraDirs {
		if rel == e || strings.HasPrefix(rel, e+"/") {
			return true, false
		}
		if strings.HasPrefix(e, rel+"/") {
			ancestor = true
		}
	}
	return false, ancestor
}

// dataDirs returns the data directories the patterns name: each pattern's first segment.
func (r rules) dataDirs() map[string]bool {
	out := map[string]bool{}
	for _, p := range r.patterns {
		first, _, _ := strings.Cut(p, "/")
		out[first] = true
	}
	return out
}

func (r rules) matches(rel string) bool {
	for _, p := range r.patterns {
		if ok, _ := path.Match(p, rel); ok {
			return true
		}
	}
	return false
}

// readTree reads every data file of fsys. It refuses, naming the file, anything that would
// make the data ambiguous (design §3.3): a file in a data directory that matches no pattern,
// a symbolic link, a data file holding a carriage return, and two paths in the data
// directories that differ only in letter case. Files outside the data directories, and the
// paths r.ignored names, are not read.
func readTree(label string, fsys fs.FS, r rules) (*tree, error) {
	t := &tree{label: label, files: map[string][]byte{}}
	dirs := r.dataDirs()
	var problems []string
	folded := map[string]string{}
	err := fs.WalkDir(fsys, ".", func(rel string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		first, _, _ := strings.Cut(rel, "/")
		if why, ok := r.refused[first]; ok {
			if !d.IsDir() {
				problems = append(problems, fmt.Sprintf("%s: %s", rel, why))
			}
			return nil
		}
		in, ancestor := r.inData(dirs, rel)
		if !in {
			if d.IsDir() && !ancestor {
				return fs.SkipDir
			}
			return nil
		}
		if r.ignored != nil && r.ignored(rel) {
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if prev, ok := folded[strings.ToLower(rel)]; ok {
			problems = append(problems, fmt.Sprintf(
				"%s and %s differ only in letter case; one of them must go", prev, rel))
		} else {
			folded[strings.ToLower(rel)] = rel
		}
		if d.Type()&fs.ModeSymlink != 0 {
			problems = append(problems, fmt.Sprintf(
				"%s is a symbolic link; data must be regular files", rel))
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if !d.Type().IsRegular() {
			problems = append(problems, fmt.Sprintf("%s is not a regular file", rel))
			return nil
		}
		if !r.matches(rel) {
			problems = append(problems, fmt.Sprintf(
				"%s is in a data directory but matches none of the data patterns %v; "+
					"rename or remove it, since a file the loader skipped would drop its change silently",
				rel, r.patterns))
			return nil
		}
		b, err := fs.ReadFile(fsys, rel)
		if err != nil {
			return err
		}
		if err := validate.CheckSize(rel, len(b)); err != nil {
			problems = append(problems, err.Error())
			return nil
		}
		if i := strings.IndexByte(string(b), '\r'); i >= 0 {
			line := 1 + strings.Count(string(b[:i]), "\n")
			problems = append(problems, fmt.Sprintf(
				"%s holds a carriage return at line %d; data files end lines with a line feed only, "+
					"so the content hash is the same on every machine", rel, line))
			return nil
		}
		t.files[rel] = b
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", label, err)
	}
	if len(problems) > 0 {
		sort.Strings(problems)
		return nil, &Error{Source: label, Problems: problems}
	}
	return t, nil
}

// dirFS opens a directory a caller named, refusing a path that is not a directory.
func dirFS(dir string) (fs.FS, error) {
	info, err := os.Stat(dir)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return nil, fmt.Errorf("%s does not exist", dir)
	case err != nil:
		return nil, err
	case !info.IsDir():
		return nil, fmt.Errorf("%s is not a directory", dir)
	}
	return os.DirFS(dir), nil
}

// Hash is the content hash of a set of data files (design §3.3): one line per file,
// "<sha256 of the file's bytes>  <path relative to the tree's root, with forward slashes>",
// sorted by path in byte order, every line ending in a line feed; the SHA-256 of that text,
// in lower-case hex.
func Hash(files map[string][]byte) string {
	paths := make([]string, 0, len(files))
	for p := range files {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	h := sha256.New()
	for _, p := range paths {
		sum := sha256.Sum256(files[p])
		fmt.Fprintf(h, "%s  %s\n", hex.EncodeToString(sum[:]), p)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// combinedHash is the hash of base data with overlays applied: the base hash when there is
// no overlay, and otherwise the SHA-256 of one line per input in the order applied,
// "<hash>  base" and then "<hash>  overlay" for each overlay, each ending in a line feed.
// It changes whenever any input or their order changes.
func combinedHash(base string, overlays []Overlay) string {
	if len(overlays) == 0 {
		return base
	}
	h := sha256.New()
	fmt.Fprintf(h, "%s  base\n", base)
	for _, o := range overlays {
		fmt.Fprintf(h, "%s  overlay\n", o.Hash)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Error is a refusal of a catalog, registry or overlay, with every problem found.
type Error struct {
	Source   string
	Problems []string
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s is refused:\n  %s", e.Source, strings.Join(e.Problems, "\n  "))
}
