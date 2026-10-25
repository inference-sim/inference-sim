// Original to blis-vllm.

// Command provenance prints, as JSON, the Go side of a CI run's provenance: blis-vllm's
// identity as the build records it (blisvllm.ProvenanceFor), the commit the run checks, the Go
// toolchain and the platform. CI writes it into the oracle artifact beside the traces, whose
// own provenance block is the Python side's (docs/testing.md, "Provenance"), so every result in
// the artifact can be reproduced from the two together.
//
// Usage: go run ./cmd/provenance [commit]
// The commit defaults to $GITHUB_SHA; a run outside CI passes `git rev-parse HEAD`.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"

	blisvllm "github.com/inference-sim/blis-vllm"
)

// record is one run's Go-side provenance.
type record struct {
	Engine    blisvllm.Provenance `json:"engine"`
	Commit    string              `json:"blis_vllm_commit"`
	GoVersion string              `json:"go_version"`
	Platform  string              `json:"platform"`
}

func main() {
	commit := os.Getenv("GITHUB_SHA")
	if len(os.Args) > 1 {
		commit = os.Args[1]
	}
	if commit == "" {
		fmt.Fprintln(os.Stderr, "provenance: no commit: set GITHUB_SHA or pass the commit")
		os.Exit(2)
	}
	out, err := json.MarshalIndent(record{
		Engine:    blisvllm.ProvenanceFor("vllm-" + blisvllm.Upstream.Version),
		Commit:    commit,
		GoVersion: runtime.Version(),
		Platform:  runtime.GOOS + "-" + runtime.GOARCH,
	}, "", " ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "provenance:", err)
		os.Exit(1)
	}
	fmt.Println(string(out))
}
