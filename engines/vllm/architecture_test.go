// Original to blis-vllm.

package blisvllm

import (
	"regexp"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/harness"
)

// provenanceRules: a file translated from vLLM names the upstream file and commit and keeps
// vLLM's licence (#2021 attribution rule), so the next vLLM release can be diffed function by
// function; a file moved from inference-sim or blis-latency-kernel names the commit it was taken
// from; anything else says it is original.
var provenanceRules = harness.ProvenanceRules{
	Derived: regexp.MustCompile(`^// Derived from vllm/\S+ at vLLM v\d+\.\d+\.\d+ \([0-9a-f]{40}\)`),
	Licence: []string{
		"// SPDX-License-Identifier: Apache-2.0",
		"// SPDX-FileCopyrightText: Copyright contributors to the vLLM project",
	},
	Moved:    regexp.MustCompile(`^// Moved from (inference-sim|blis-latency-kernel)/\S+ at [0-9a-f]{40}`),
	Original: "// Original to blis-vllm.",
	Release:  Upstream,
}

// TestEveryFileStatesItsProvenance: each Go file says where its logic comes from, in its first
// comment block (VLM-2), by the rules above. A file that says nothing fails, which is what keeps
// the rule from decaying as files are added. The check is blis-substrate's contracts/harness.
func TestEveryFileStatesItsProvenance(t *testing.T) {
	files, err := harness.GoFiles(".")
	if err != nil {
		t.Fatal(err)
	}
	bad, err := harness.ProvenanceProblems(".", files, provenanceRules)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range bad {
		t.Error(b)
	}
}

// TestNoClockNoGoroutines: engine code never reads a clock, sleeps or starts a goroutine
// (#2021 B3, X6). Time reaches it only as a value the simulator passes in, which is what makes
// a run reproducible and lets the oracle run vLLM's own decision clocks on the same virtual
// time. Offline tools under cmd/ may time themselves; engine packages may not.
func TestNoClockNoGoroutines(t *testing.T) {
	files, err := harness.GoFiles(".", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	bad, err := harness.ClockAndGoroutineUses(".", files)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range bad {
		t.Error(b)
	}
}

// TestHarnessCopiesAreCurrent: .golangci.yml and the oracle's guard scripts are copies of
// blis-substrate's contracts/harness at the substrate version go.mod selects (X7). Regenerate
// them with `go run github.com/inference-sim/blis-substrate/contracts/harness/cmd/harnessfiles`.
func TestHarnessCopiesAreCurrent(t *testing.T) {
	bad, err := harness.StaleCopies(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range bad {
		t.Error(b)
	}
}

// TestUpstreamRecordIsWellFormed: the release this build models is named by a release version, a tag
// and a full commit id. It checks the record's form only; that the checkout CI compares against is
// that commit is the rev-parse step of ci.yml.
func TestUpstreamRecordIsWellFormed(t *testing.T) {
	for _, p := range Upstream.Problems() {
		t.Error(p)
	}
}
