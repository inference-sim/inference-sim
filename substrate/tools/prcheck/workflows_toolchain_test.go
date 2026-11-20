package main

import "testing"

const setupGo = `name: ci
on: pull_request
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262
      - uses: actions/setup-go@40f1582b2485089dde7abd97c1529aa768e1baff
        with:
          go-version-file: go.mod
`

// LIF-2: a job installs Go with the go-toolchain action, which installs contracts/toolchain's
// version. A workflow or another action that installs the Go that go.mod names leaves a job on
// a different version when the two differ (inference-sim/blis-substrate#193).
func TestAJobThatInstallsGoFromGoModIsRefused(t *testing.T) {
	has(t, mustLint(t, ".github/workflows/ci.yml", setupGo), "go-toolchain")
	// A caller's own composite action is held to the same rule.
	has(t, mustLint(t, ".github/actions/go-setup/action.yml", setupGo), "go-toolchain")
	// A literal version is no better.
	lit := "name: ci\non: pull_request\njobs:\n  b:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/setup-go@40f1582b2485089dde7abd97c1529aa768e1baff\n        with:\n          go-version: '1.27.2'\n"
	has(t, mustLint(t, ".github/workflows/ci.yml", lit), "go-toolchain")
	// The go-toolchain action itself installs Go through setup-go, with a version it has read.
	none(t, mustLint(t, ".github/actions/go-toolchain/action.yml", setupGo))
	// Installing through the action passes.
	none(t, mustLint(t, ".github/workflows/ci.yml",
		"name: ci\non: pull_request\njobs:\n  b:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: inference-sim/blis-substrate/.github/actions/go-toolchain@40f1582b2485089dde7abd97c1529aa768e1baff\n"))
}

func mustLint(t *testing.T, name, src string) []string {
	t.Helper()
	got, err := LintWorkflow(name, []byte(src))
	if err != nil {
		t.Fatal(err)
	}
	return got
}
