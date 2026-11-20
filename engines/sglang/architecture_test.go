// Original to blis-sglang.

package blissglang

import (
	"regexp"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/harness"
)

// provenanceRules (SGL-2): a file translated from SGLang names the upstream file and commit and
// keeps SGLang's licence, so the next SGLang release can be diffed function by function; anything
// else says it is original. SGLang's files carry "Copyright 2023-2024 SGLang Team" under the
// Apache License 2.0 (python/sglang/srt/managers/scheduler.py:1-2@297e54fd).
var provenanceRules = harness.ProvenanceRules{
	Derived: regexp.MustCompile(`^// Derived from python/sglang/\S+ at SGLang v\d+\.\d+\.\d+(\.post\d+)? \([0-9a-f]{40}\)`),
	Licence: []string{
		"// SPDX-License-Identifier: Apache-2.0",
		"// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team",
	},
	Original: "// Original to blis-sglang.",
	Release:  Upstream,
}

// TestEveryFileStatesItsProvenance: each Go file says where its logic comes from in its first
// comment block (SGL-2). A file that says nothing fails, which keeps the rule from decaying as
// files are added. The check is blis-substrate's contracts/harness.
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

// TestNoClockNoGoroutines: engine code never reads a clock, sleeps or starts a goroutine (X6).
// Time reaches it only as a value the simulator passes in. Offline tools under cmd/ may time
// themselves; engine packages may not.
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
// them with `go run github.com/inference-sim/inference-sim/substrate/contracts/harness/cmd/harnessfiles`.
func TestHarnessCopiesAreCurrent(t *testing.T) {
	bad, err := harness.StaleCopies(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range bad {
		t.Error(b)
	}
}
