package result

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	validate "github.com/inference-sim/blis-substrate/contracts/check"
)

// File is one small input file the run read, held whole so the Result reproduces without it
// (DEV-7, SIM-10): the router's endpoint-picker configuration and manifests the deployment
// names, and each configuration file a command-line option names. Bulk data, a trace or a
// corpus, is never held here; the Result names it by sha256 only (blis-substrate#154,
// Principle 6).
type File struct {
	// Path is the file as the input names it: the deployment's router path, or the option's
	// name (a file option's value is the file's sha256, in options).
	Path string `yaml:"path"`
	// SHA256 is the lowercase hex sha256 of Content.
	SHA256 string `yaml:"sha256"`
	// Content is the file's bytes, as text.
	Content string `yaml:"content"`
}

// validateFiles checks the files an input holds: each names a path once, the list is sorted
// by path, and each sha256 is its content's.
func validateFiles(files []File, p *validate.Problems) {
	for i, f := range files {
		at := fmt.Sprintf("input.files[%d]", i)
		switch {
		case f.Path == "":
			p.Field(at+".path", "required: the path the input names the file by")
		case i > 0 && f.Path <= files[i-1].Path:
			p.Field(at+".path", "files are listed once each, sorted by path")
		}
		sum := sha256.Sum256([]byte(f.Content))
		if want := hex.EncodeToString(sum[:]); f.SHA256 != want {
			p.Field(at+".sha256", "is %q, but the content's sha256 is %q", f.SHA256, want)
		}
	}
}
