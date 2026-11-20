package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// The licence-file check (X21): each repository keeps the licence and notice files its code
// requires. Every BLIS repository has a LICENSE. An engine repository keeps a NOTICE for
// the code it derives from its upstream (vLLM's for blis-vllm; by policy SGLang's for
// blis-sglang), and the NOTICE names that upstream.
var noticeUpstream = map[string]string{
	"blis-vllm":   "vLLM",
	"blis-sglang": "SGLang",
}

// CheckLicenceFiles returns the X21 findings for a repository's root.
func CheckLicenceFiles(root, repo string) []string {
	var bad []string
	if fi, err := os.Stat(filepath.Join(root, "LICENSE")); err != nil || fi.Size() == 0 {
		bad = append(bad, "LICENSE is missing or empty (X21)")
	}
	if up, ok := noticeUpstream[repo]; ok {
		raw, err := os.ReadFile(filepath.Join(root, "NOTICE"))
		switch {
		case err != nil:
			bad = append(bad, fmt.Sprintf("NOTICE is missing; %s keeps one for the code it derives from %s (X21)", repo, up))
		case !strings.Contains(string(raw), up):
			bad = append(bad, fmt.Sprintf("NOTICE does not name %s, the upstream whose code %s derives from (X21)", up, repo))
		}
	}
	return bad
}
