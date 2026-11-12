package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// X17 (blis-schemas#105): lint is a required check of this repository. Branch protection
// requires the check run "required", and that job passes only when every job it needs
// passed, so lint is required exactly when the golangci-lint job is one of its needs.
// Known-bad input: the same workflow with lint left out of the list.
func TestLintIsAmongTheJobsTheRequiredCheckWaitsFor(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", "ci.yml"))
	if err != nil {
		t.Fatal(err)
	}
	if !requiresLint(string(raw)) {
		t.Fatal("ci.yml's required job does not need the lint job, so lint is not a required check")
	}
	without := strings.Replace(string(raw), "needs: [changes, go, lint,", "needs: [changes, go,", 1)
	if without == string(raw) {
		t.Fatal("the known-bad input did not change the workflow; update this test with the needs list")
	}
	if requiresLint(without) {
		t.Error("a required job without lint was accepted")
	}
}

var requiredNeeds = regexp.MustCompile(`(?s)\n  required:\n.*?\n    needs: \[([^\]]*)\]`)

func requiresLint(workflow string) bool {
	m := requiredNeeds.FindStringSubmatch(workflow)
	if m == nil {
		return false
	}
	for _, n := range strings.Split(m[1], ",") {
		if strings.TrimSpace(n) == "lint" {
			return true
		}
	}
	return false
}
