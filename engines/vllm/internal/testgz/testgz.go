// Original to blis-vllm.

// Package testgz reads test data that the repository stores gzip-compressed (SUB-13: the module
// zip stays under 100 MB). A data file of 16000 bytes or more is committed as <name>.gz; a smaller
// one stays plain. Every reader names the plain file, and ReadFile and Glob find either form.
package testgz

import (
	"bytes"
	"compress/gzip"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const suffix = ".gz"

// ReadFile returns the contents of name, or of name+".gz" decompressed when name does not exist.
func ReadFile(name string) ([]byte, error) {
	b, err := os.ReadFile(name)
	if !errors.Is(err, fs.ErrNotExist) {
		return b, err
	}
	z, zerr := os.ReadFile(name + suffix)
	if zerr != nil {
		return nil, err
	}
	r, err := gzip.NewReader(bytes.NewReader(z))
	if err != nil {
		return nil, &fs.PathError{Op: "gunzip", Path: name + suffix, Err: err}
	}
	defer func() { _ = r.Close() }()
	b, err = io.ReadAll(r)
	if err != nil {
		return nil, &fs.PathError{Op: "gunzip", Path: name + suffix, Err: err}
	}
	return b, nil
}

// Exists reports whether name or name+".gz" exists.
func Exists(name string) bool {
	for _, p := range []string{name, name + suffix} {
		if _, err := os.Stat(p); err == nil {
			return true
		}
	}
	return false
}

// Glob is filepath.Glob that also matches the compressed form: it returns each match once, sorted,
// with the ".gz" suffix removed, so a caller reads every name through ReadFile.
func Glob(pattern string) ([]string, error) {
	plain, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	zipped, err := filepath.Glob(pattern + suffix)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var out []string
	for _, p := range append(plain, zipped...) {
		p = strings.TrimSuffix(p, suffix)
		if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out, nil
}
