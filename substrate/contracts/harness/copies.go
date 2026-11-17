package harness

import (
	"bytes"
	"embed"
	"fmt"
	"os"
	"path/filepath"
)

//go:embed golangci.yml oracle/check_lock.py oracle/check_mocks.py oracle/check_goldens.py
var sources embed.FS

// A Copy is a file an engine repository keeps as a copy of one of this package's files: Path is
// where it lives in the engine repository, Source the file here it is generated from (X7).
type Copy struct {
	Path   string
	Source string
}

// Copies are the files every engine repository keeps, generated from this package at the
// substrate version the repository pins.
var Copies = []Copy{
	{Path: ".golangci.yml", Source: "golangci.yml"},
	{Path: "oracle/check_lock.py", Source: "oracle/check_lock.py"},
	{Path: "oracle/check_mocks.py", Source: "oracle/check_mocks.py"},
	{Path: "oracle/check_goldens.py", Source: "oracle/check_goldens.py"},
}

// Command is how an engine repository regenerates its copies.
const Command = "go run github.com/inference-sim/blis-substrate/contracts/harness/cmd/harnessfiles"

// Content is the copy's bytes: a line saying it is generated, then the source.
func (c Copy) Content() ([]byte, error) {
	src, err := sources.ReadFile(c.Source)
	if err != nil {
		return nil, err
	}
	head := fmt.Sprintf("# Generated from blis-substrate contracts/harness/%s by `%s`; do not edit.\n", c.Source, Command)
	return append([]byte(head), src...), nil
}

// WriteCopies writes every copy under root.
func WriteCopies(root string) error {
	for _, c := range Copies {
		b, err := c.Content()
		if err != nil {
			return err
		}
		p := filepath.Join(root, filepath.FromSlash(c.Path))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(p, b, 0o644); err != nil {
			return err
		}
	}
	return nil
}

// StaleCopies returns one problem for each copy under root that is missing or differs from what
// WriteCopies would write.
func StaleCopies(root string) ([]string, error) {
	var bad []string
	for _, c := range Copies {
		want, err := c.Content()
		if err != nil {
			return nil, err
		}
		got, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(c.Path)))
		switch {
		case os.IsNotExist(err):
			bad = append(bad, fmt.Sprintf("%s is missing; run %s", c.Path, Command))
		case err != nil:
			return nil, err
		case !bytes.Equal(got, want):
			bad = append(bad, fmt.Sprintf("%s differs from contracts/harness/%s at the pinned substrate; run %s", c.Path, c.Source, Command))
		}
	}
	return bad, nil
}
