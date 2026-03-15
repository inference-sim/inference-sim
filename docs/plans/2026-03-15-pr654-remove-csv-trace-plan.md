# chore: Remove Lossy CSV Trace Paths Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the `--workload traces` flag from `blis run` and the `blis convert csv-trace` subcommand, eliminating two deprecated code paths that performed lossy statistical approximation rather than faithful trace replay.

**The problem today:** The `--workload traces` mode reads legacy CSV trace files but discards per-request arrival times and token lengths, replacing them with averaged statistics — a fundamentally lossy transformation misrepresented as "replay." The `blis convert csv-trace` subcommand performs the same lossy aggregation. Both paths have been superseded by the trace v2 format (`--workload-spec` with a YAML header + CSV data), which provides faithful per-request replay. Keeping them creates confusion about what "trace replay" means in BLIS.

**What this PR removes:**
1. `--workload traces` flag path in `blis run` — the `else if workloadType == "traces"` branch in `cmd/root.go`, plus the `--workload-traces-filepath` flag registration and its variable declaration
2. `blis convert csv-trace` subcommand — the `convertCSVTraceCmd` cobra command and its variables from `cmd/convert.go`
3. `workload.ConvertCSVTrace()` function from `sim/workload/convert.go`
4. `workload.SynthesizeFromCSVTrace()` function from `sim/workload/synthesis.go`
5. All tests covering the above four deletions
6. All documentation references to the deleted flags, commands, and functions

**Why this matters:** A clean separation between generative simulation (`blis run`) and faithful replay (`blis replay`, per issue #652) requires that no lossy approximation masquerades as replay in the main run path. This PR eliminates that ambiguity.

**Architecture:** Pure deletion — no new types, no new interfaces, no behavior changes to existing paths. Removes approximately 150 lines of production code and 100 lines of tests across `sim/workload/` and `cmd/`. The task ordering is dependency-driven: remove callers before callees so each intermediate commit builds cleanly.

**Source:** GitHub issue #654

**Closes:** Fixes #654

**Behavioral Contracts:** See Part 1, Section B below.

---

## Part 1: Design Validation

### A) Executive Summary

This PR removes two deprecated code paths that were already internally documented as "lossy" and warned against in production: `--workload traces` (in `blis run`) and `blis convert csv-trace`. Both paths called `workload.ConvertCSVTrace()` which averaged per-request token lengths and ignored per-request arrival times — the opposite of trace replay.

**System position:** `cmd/root.go` → `sim/workload/synthesis.go` → `sim/workload/convert.go` is the call chain being severed. Nothing downstream changes. No `sim/` invariants are touched.

**Adjacent blocks:** The remaining convert subcommands (`servegen`, `preset`, `inference-perf`) and the remaining `blis run` workload paths (`distribution`, `workload-spec`, named presets) are unaffected. The trace v2 path (`--workload-spec` with a trace YAML) is the intended replacement and is not touched by this PR.

**No DEVIATION flags.** Issue #654 description matches current code exactly.

### B) Behavioral Contracts

**Positive Contracts (what MUST be true after this PR)**

BC-1: `csv-trace` subcommand absent from `blis convert`
- GIVEN the `blis` binary built from this PR
- WHEN `blis convert --help` output is captured
- THEN the output does not contain "csv-trace"
- MECHANISM: `convertCSVTraceCmd` and its `AddCommand` call are removed from `cmd/convert.go`

BC-2: `--workload-traces-filepath` flag absent from `blis run`
- GIVEN the `blis` binary built from this PR
- WHEN `blis run --help` output is captured
- THEN the output does not contain "--workload-traces-filepath"
- MECHANISM: flag registration line removed from `cmd/root.go` init()

BC-3: `--workload` description no longer lists "traces" as a valid type
- GIVEN the `blis` binary built from this PR
- WHEN `blis run --help` output is captured
- THEN the `--workload` flag description does not include "traces"
- MECHANISM: flag description string updated in `cmd/root.go` init()

BC-4: `ConvertCSVTrace` and `SynthesizeFromCSVTrace` absent from `sim/workload` package
- GIVEN `go build ./sim/workload/...` on this PR
- WHEN the package compiles
- THEN no exported symbol `ConvertCSVTrace` or `SynthesizeFromCSVTrace` exists
- MECHANISM: both functions deleted from their respective source files

BC-5: Remaining convert subcommands unchanged
- GIVEN valid input for `servegen`, `preset`, `inference-perf`
- WHEN those subcommands are called via the library functions `ConvertServeGen`, `ConvertPreset`, `ConvertInferencePerf`
- THEN they return valid v2 WorkloadSpec with no regression
- MECHANISM: those functions are not touched by this PR

BC-6: Documentation no longer references the deprecated paths
- GIVEN `docs/guide/workloads.md`, `docs/reference/configuration.md`, `CLAUDE.md`, `README.md`
- WHEN searching those files for `csv-trace`, `--workload traces`, `--workload-traces-filepath`
- THEN no live documentation file contains these deprecated references
- MECHANISM: explicit edits to the four documentation files in Task 5

**Negative Contracts (what MUST NOT happen)**

BC-N1: Build must not break at any intermediate commit
- GIVEN each individual commit in this PR
- WHEN `go build ./...` is run
- THEN the build succeeds (callers removed before callees, in dependency order)

BC-N2: No other workload paths regress
- GIVEN `go test ./sim/workload/... ./cmd/...` after all deletions
- WHEN the full test suite runs
- THEN zero new failures (existing `TestConvert*`, `TestCompose*`, `TestSynthesize*` tests all pass)

### C) Component Interaction

```
BEFORE:
  cmd/root.go ──── SynthesizeFromCSVTrace ──► sim/workload/synthesis.go
                                                       │
  cmd/convert.go ── ConvertCSVTrace ──────────► sim/workload/convert.go

AFTER:
  cmd/root.go ── [traces branch removed]
  cmd/convert.go ── [csv-trace command removed]
  sim/workload/synthesis.go ── [SynthesizeFromCSVTrace removed]
  sim/workload/convert.go ── [ConvertCSVTrace removed]
```

No new state ownership, no new interfaces, no API surface added. The deletion chain flows top-down: `cmd/` callers first, then `sim/workload/` callees.

**Extension Friction Assessment:** N/A — deletion PR, no new types.

### D) Deviation Log

No deviations from issue #654 description. All four code deletion categories, documentation update requirement, and test removal requirement match exactly.

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "remove traces branch from cmd/root.go" | Removes the `else if workloadType == "traces"` block, the `tracesWorkloadFilePath` variable declaration, and both flag registrations in `init()` | CORRECTION: issue lists variables separately from code logic; plan groups them as a single atomic task |
| "delete helper functions SynthesizeFromCSVTrace and ConvertCSVTrace" | Deletes them in two separate tasks (callers first) | ADDITION: dependency ordering requires cmd/ callers to be removed before sim/ callees can be safely deleted |

### E) Review Guide

- **THE TRICKY PART:** Unused-import cleanup in `convert.go` (after removing `ConvertCSVTrace`, five imports become unused: `"encoding/csv"`, `"encoding/json"`, `"io"`, `"os"`, `"strconv"`). Miss one and `golangci-lint` flags it.
- **WHAT TO SCRUTINIZE:** Task 3 — `synthesis.go` still compiles after removing its `"fmt"` import; verify with `go build`. Task 4 — `convert_test.go` retains the non-CSV-trace tests (`TestConvertPreset_*`, `TestComposeSpecs_*`, `TestConvertServeGen_*`, `TestConvertInferencePerf_*`).
- **WHAT'S SAFE TO SKIM:** Documentation updates (Task 5) — purely editorial, no logic.
- **KNOWN DEBT:** `docs/plans/archive/2026-02-16-workload-generator-design.md` and `docs/plans/pr350-phase2-gaps-plan.md` reference the old traces path. These are archived plans — their historical references are intentional and should NOT be edited in this PR.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to modify (no new files created):

| File | Change |
|------|--------|
| `cmd/convert.go` | Delete `convertCSVTraceCmd`, its 2 variables, init() bindings; update Long description |
| `cmd/root.go` | Delete `tracesWorkloadFilePath` variable, `else if workloadType == "traces"` branch, `--workload-traces-filepath` flag registration; update `--workload` flag description |
| `sim/workload/synthesis.go` | Delete `SynthesizeFromCSVTrace`; remove `"fmt"` import |
| `sim/workload/convert.go` | Delete `ConvertCSVTrace`; remove 5 now-unused imports |
| `sim/workload/convert_test.go` | Delete 4 `TestConvertCSVTrace_*` test functions |
| `docs/guide/workloads.md` | Remove "CSV traces" row from mode table; update count |
| `docs/reference/configuration.md` | Remove "CSV traces" row; remove `--workload-traces-filepath` flag row; remove it from WorkloadConfig flag list |
| `CLAUDE.md` | Remove `csv-trace` from build example and file tree description |
| `README.md` | Remove `csv-trace` usage example and file tree entry |

**Key decisions:** Task ordering is caller-before-callee to keep `go build ./...` green at every commit. No new abstractions or types are introduced.

**Confirmation:** After all tasks, `grep -r "ConvertCSVTrace\|SynthesizeFromCSVTrace\|csv-trace\|workload traces\|workload-traces-filepath" --include="*.go"` returns zero results in non-archived files.

### G) Task Breakdown

---

#### Task 1: Remove `blis convert csv-trace` subcommand from `cmd/convert.go`

**Contracts Implemented:** BC-1, BC-N1

**Files:**
- Modify: `cmd/convert.go`

**Step 1: Write failing test verifying subcommand absence**

Context: We verify that after the change, the `convert` command no longer has a `csv-trace` subcommand. `cmd/root_test.go` already exists — add the test there.

```go
// Add to cmd/root_test.go
func TestConvertCmd_NoCSVTraceSubcommand(t *testing.T) {
	// GIVEN the convert cobra command
	// WHEN listing its subcommands
	for _, sub := range convertCmd.Commands() {
		if sub.Name() == "csv-trace" {
			// THEN csv-trace must not be present
			t.Error("csv-trace subcommand should not exist after removal")
			return
		}
	}
}
```

**Step 2: Run test to verify it fails**

```bash
GOCACHE=/tmp/claude/go-cache go test github.com/inference-sim/inference-sim/cmd -run TestConvertCmd_NoCSVTraceSubcommand -v
```

Expected: FAIL — `csv-trace subcommand should not exist after removal`

**Step 3: Delete the csv-trace command, variables, and init() bindings from `cmd/convert.go`**

In `cmd/convert.go`, make these deletions:

1. Remove the section comment + variables block (lines 36-41):
```go
// --- blis convert csv-trace ---

var (
	csvTracePath    string
	csvTraceHorizon int64
)
```

2. Remove the `convertCSVTraceCmd` command definition (lines 43-57):
```go
var convertCSVTraceCmd = &cobra.Command{
	Use:   "csv-trace",
	Short: "Convert legacy CSV trace file to v2 spec",
	Run: func(cmd *cobra.Command, args []string) {
		// R3: validate numeric CLI flags at the boundary
		if csvTraceHorizon < 0 {
			logrus.Fatalf("--horizon must be >= 0 (0 = no truncation), got %d", csvTraceHorizon)
		}
		spec, err := workload.ConvertCSVTrace(csvTracePath, csvTraceHorizon)
		if err != nil {
			logrus.Fatalf("CSV trace conversion failed: %v", err)
		}
		writeSpecToStdout(spec)
	},
}
```

3. Remove the init() bindings for `convertCSVTraceCmd` (lines 144-146):
```go
	convertCSVTraceCmd.Flags().StringVar(&csvTracePath, "file", "", "Path to CSV trace file")
	convertCSVTraceCmd.Flags().Int64Var(&csvTraceHorizon, "horizon", 0, "Horizon in microseconds (0 = no truncation)")
	_ = convertCSVTraceCmd.MarkFlagRequired("file")
```

4. Remove the `convertCmd.AddCommand(convertCSVTraceCmd)` call in init() (line 158).

5. Update `convertCmd.Long` description to remove "CSV traces":

Change:
```go
Long:  "Convert external workload formats (ServeGen, inference-perf, CSV traces, presets) to v2 WorkloadSpec YAML. Output is written to stdout for piping.",
```
To:
```go
Long:  "Convert external workload formats (ServeGen, inference-perf, presets) to v2 WorkloadSpec YAML. Output is written to stdout for piping.",
```

**Step 4: Run test to verify it passes**

```bash
GOCACHE=/tmp/claude/go-cache go test github.com/inference-sim/inference-sim/cmd -run TestConvertCmd_NoCSVTraceSubcommand -v
```

Expected: PASS

Also verify build still passes (since `workload.ConvertCSVTrace` is still defined, just not called):

```bash
GOCACHE=/tmp/claude/go-cache go build ./...
```

Expected: no errors

**Step 5: Run lint check**

```bash
golangci-lint run ./cmd/...
```

Expected: zero new issues

**Step 6: Commit**

```bash
git add cmd/convert.go cmd/root_test.go
git commit -m "chore(cmd): remove blis convert csv-trace subcommand (BC-1)

- Delete convertCSVTraceCmd, csvTracePath, csvTraceHorizon variables
- Remove init() flag bindings and AddCommand call for csv-trace
- Update convertCmd.Long to remove CSV traces from description
- Add TestConvertCmd_NoCSVTraceSubcommand verifying removal

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Remove `--workload traces` branch from `cmd/root.go`

**Contracts Implemented:** BC-2, BC-3, BC-N1

**Files:**
- Modify: `cmd/root.go`

**Step 1: Write failing tests verifying flag and branch absence**

`cmd/root_test.go` already exists — add the following two tests there:

```go
package cmd

import (
	"strings"
	"testing"
)

func TestRunCmd_NoWorkloadTracesFlag(t *testing.T) {
	// GIVEN the run cobra command
	// WHEN looking up the --workload-traces-filepath flag
	f := runCmd.Flags().Lookup("workload-traces-filepath")
	// THEN the flag does not exist
	if f != nil {
		t.Error("--workload-traces-filepath flag should not exist after removal")
	}
}

func TestRunCmd_WorkloadFlagDescriptionExcludesTraces(t *testing.T) {
	// GIVEN the run cobra command
	// WHEN inspecting the --workload flag description
	f := runCmd.Flags().Lookup("workload")
	if f == nil {
		t.Fatal("--workload flag must exist")
	}
	// THEN "traces" is not in the usage string
	if strings.Contains(f.Usage, "traces") {
		t.Errorf("--workload flag description must not contain 'traces', got: %q", f.Usage)
	}
}
```

**Step 2: Run tests to verify they fail**

```bash
GOCACHE=/tmp/claude/go-cache go test github.com/inference-sim/inference-sim/cmd -run "TestRunCmd_NoWorkloadTracesFlag|TestRunCmd_WorkloadFlagDescriptionExcludesTraces" -v
```

Expected: Both FAIL

**Step 3: Delete the traces variable, branch, and flag registrations from `cmd/root.go`**

1. Remove `tracesWorkloadFilePath` variable declaration (line 40):
```go
tracesWorkloadFilePath    string    // Workload filepath for traces workload type.
```

2. Update `workloadType` comment on line 39 — remove "traces" from the inline comment:

Change:
```go
workloadType              string    // Workload type (chatbot, summarization, contentgen, multidoc, distribution, traces)
```
To:
```go
workloadType              string    // Workload type (chatbot, summarization, contentgen, multidoc, distribution)
```

3. Remove the `else if workloadType == "traces"` branch (lines 726-738 approximately):
```go
	} else if workloadType == "traces" {
		// CSV trace path → synthesize v2 spec (lossy conversion)
		if tracesWorkloadFilePath == "" {
			logrus.Fatalf("--workload-traces-filepath is required when using --workload traces")
		}
		logrus.Warn("--workload traces uses lossy CSV conversion (averaged token lengths, constant arrival). " +
			"For faithful trace replay, use --workload-spec with a trace v2 YAML file instead.")
		var err error
		spec, err = workload.SynthesizeFromCSVTrace(tracesWorkloadFilePath, simulationHorizon)
		if err != nil {
			logrus.Fatalf("Failed to convert CSV trace: %v", err)
		}
		spec.Seed = seed
	} else if workloadType == "distribution" {
```

Delete only the `else if workloadType == "traces" { ... }` block. The following `} else if workloadType == "distribution" {` becomes the new branch at that position.

4. In `init()` (around line 1158-1159), make two changes:

Remove this line entirely:
```go
runCmd.Flags().StringVar(&tracesWorkloadFilePath, "workload-traces-filepath", "", "Workload filepath for traces workload type.")
```

Update the `--workload` flag description string:

Change:
```go
runCmd.Flags().StringVar(&workloadType, "workload", "distribution", "Workload type (chatbot, summarization, contentgen, multidoc, distribution, traces)")
```
To:
```go
runCmd.Flags().StringVar(&workloadType, "workload", "distribution", "Workload type (chatbot, summarization, contentgen, multidoc, distribution)")
```

**Step 4: Run tests to verify they pass**

```bash
GOCACHE=/tmp/claude/go-cache go test github.com/inference-sim/inference-sim/cmd -run "TestRunCmd_NoWorkloadTracesFlag|TestRunCmd_WorkloadFlagDescriptionExcludesTraces" -v
```

Expected: Both PASS

Build verification (note: `workload.SynthesizeFromCSVTrace` still exists in sim/workload, just not called from cmd/ — build must pass):

```bash
GOCACHE=/tmp/claude/go-cache go build ./...
```

Expected: no errors

**Step 5: Run lint check**

```bash
golangci-lint run ./cmd/...
```

Expected: zero new issues

**Step 6: Commit**

```bash
git add cmd/root.go cmd/root_test.go
git commit -m "chore(cmd): remove --workload traces path and --workload-traces-filepath flag (BC-2, BC-3)

- Delete tracesWorkloadFilePath variable declaration
- Delete else-if workloadType==\"traces\" branch
- Remove --workload-traces-filepath flag registration from init()
- Update --workload flag description to exclude traces
- Add tests verifying flag absence and description content

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Remove `SynthesizeFromCSVTrace` from `sim/workload/synthesis.go`

**Contracts Implemented:** BC-4, BC-N1

**Files:**
- Modify: `sim/workload/synthesis.go`

**Step 1: No new test needed**

`synthesis_test.go` has no test for `SynthesizeFromCSVTrace` (confirmed: reading the file shows only `TestSynthesizeFromDistribution_*` tests). The contract BC-4 is verified by successful compilation, not by a test assertion. Proceed directly to implementation.

**Step 2: Verify the function exists before deletion (sanity check)**

```bash
grep -n "SynthesizeFromCSVTrace" sim/workload/synthesis.go
```

Expected: lines 84-96 show the function definition.

**Step 3: Delete `SynthesizeFromCSVTrace` and the `"fmt"` import**

In `sim/workload/synthesis.go`:

1. Remove the `"fmt"` import (line 3) — it is only used by `SynthesizeFromCSVTrace`. After deletion, the import block becomes empty and can be removed entirely:

Change:
```go
import "fmt"
```
To: _(delete the import line entirely)_

2. Remove `SynthesizeFromCSVTrace` function (lines 84-96):
```go
// SynthesizeFromCSVTrace creates a v2 WorkloadSpec from a CSV trace file path.
// WARNING: this is a lossy conversion — per-request arrival times and token
// lengths are replaced by aggregate statistics (mean lengths, constant rate).
// For faithful trace replay preserving per-request fidelity, use --workload-spec
// with a trace v2 YAML file (LoadTraceV2 + ReplayTraceV2Requests).
// R6: no logging — callers should warn about lossy conversion.
func SynthesizeFromCSVTrace(path string, horizon int64) (*WorkloadSpec, error) {
	spec, err := ConvertCSVTrace(path, horizon)
	if err != nil {
		return nil, fmt.Errorf("synthesizing from CSV trace: %w", err)
	}
	return spec, nil
}
```

**Step 4: Verify build passes**

```bash
GOCACHE=/tmp/claude/go-cache go build ./sim/workload/...
```

Expected: no errors (note: `ConvertCSVTrace` still exists in `convert.go` but is no longer called from `synthesis.go`)

Run existing tests:

```bash
GOCACHE=/tmp/claude/go-cache go test github.com/inference-sim/inference-sim/sim/workload -v 2>&1 | tail -20
```

Expected: all `TestSynthesizeFromDistribution_*` tests pass; `TestConvertCSVTrace_*` tests also still pass (function still exists)

**Step 5: Run lint check**

```bash
golangci-lint run ./sim/workload/...
```

Expected: zero new issues

**Step 6: Commit**

```bash
git add sim/workload/synthesis.go
git commit -m "chore(workload): remove SynthesizeFromCSVTrace helper (BC-4)

- Delete SynthesizeFromCSVTrace (thin wrapper around ConvertCSVTrace)
- Remove now-unused fmt import
- All callers in cmd/ were removed in prior commits

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Remove `ConvertCSVTrace` from `sim/workload/convert.go` and delete its tests

**Contracts Implemented:** BC-4, BC-N1, BC-N2

**Files:**
- Modify: `sim/workload/convert.go`
- Modify: `sim/workload/convert_test.go`

**Step 1: No new test needed**

The 4 `TestConvertCSVTrace_*` tests will be deleted. `BC-4` is verified by compilation. `BC-N2` is verified by the remaining tests in `convert_test.go` all passing after deletion.

**Step 2: Delete 4 `TestConvertCSVTrace_*` test functions from `sim/workload/convert_test.go`**

Remove the following four complete test functions (lines 9-105):

```go
func TestConvertCSVTrace_ValidFile_ProducesV2Spec(t *testing.T) { ... }
func TestConvertCSVTrace_EmptyFile_ReturnsError(t *testing.T) { ... }
func TestConvertCSVTrace_MalformedRow_ReturnsErrorWithLine(t *testing.T) { ... }
func TestConvertCSVTrace_HorizonTruncation(t *testing.T) { ... }
```

Also remove the unused imports from `convert_test.go`. After deletion, the remaining tests (`TestConvertPreset_*`, `TestComposeSpecs_*`, `TestConvertServeGen_*`, `TestConvertInferencePerf_*`) use:
- `"os"` ✗ (was used by `WriteFile` in csv-trace tests) — check remaining tests
- `"path/filepath"` ✗ (was used in csv-trace tests) — check remaining tests
- `"testing"` ✓

After reviewing `convert_test.go` lines 107-221, the remaining tests (`TestConvertPreset_*`, `TestComposeSpecs_*`) do NOT use `os.WriteFile` or `filepath.Join`. So the imports `"os"` and `"path/filepath"` become unused.

Update `convert_test.go` import block from:
```go
import (
	"os"
	"path/filepath"
	"testing"
)
```
To:
```go
import (
	"testing"
)
```

**Step 3: Delete `ConvertCSVTrace` and its 5 now-unused imports from `sim/workload/convert.go`**

1. Remove the `ConvertCSVTrace` function and its doc comment (lines 32-149):

```go
// ConvertCSVTrace converts a legacy CSV trace file into a v2 WorkloadSpec.
// ...
func ConvertCSVTrace(path string, horizon int64) (*WorkloadSpec, error) {
    // ... entire function body ...
}
```

2. Update the `import` block in `convert.go`. After removing `ConvertCSVTrace`, these five imports become unused:
   - `"encoding/csv"` — only used by `ConvertCSVTrace`
   - `"encoding/json"` — only used by `ConvertCSVTrace`
   - `"io"` — only used by `ConvertCSVTrace` (for `io.EOF`)
   - `"os"` — only used by `ConvertCSVTrace` (for `os.Open`)
   - `"strconv"` — only used by `ConvertCSVTrace` (for `strconv.ParseFloat`)

Change import block from:
```go
import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
)
```
To:
```go
import (
	"fmt"
	"math"
)
```

**Step 4: Verify build and tests pass**

```bash
GOCACHE=/tmp/claude/go-cache go build ./...
```

Expected: no errors

```bash
GOCACHE=/tmp/claude/go-cache go test github.com/inference-sim/inference-sim/sim/workload -v 2>&1 | tail -20
```

Expected: `TestConvertPreset_*`, `TestComposeSpecs_*`, `TestConvertServeGen_*`, `TestConvertInferencePerf_*`, `TestSynthesizeFromDistribution_*` all pass; `TestConvertCSVTrace_*` gone.

**Step 5: Run lint check**

```bash
golangci-lint run ./sim/workload/...
```

Expected: zero new issues

**Step 6: Verify grep confirms complete removal**

```bash
grep -r "ConvertCSVTrace\|SynthesizeFromCSVTrace" --include="*.go" .
```

Expected: zero matches

**Step 7: Commit**

```bash
git add sim/workload/convert.go sim/workload/convert_test.go
git commit -m "chore(workload): remove ConvertCSVTrace and its tests (BC-4, BC-N2)

- Delete ConvertCSVTrace function (lossy CSV→spec conversion)
- Remove 5 now-unused imports (csv, json, io, os, strconv)
- Delete 4 TestConvertCSVTrace_* tests from convert_test.go
- Remove os and path/filepath imports now unused in convert_test.go
- Remaining ConvertPreset, ComposeSpecs, ConvertServeGen, ConvertInferencePerf tests pass

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Update documentation

**Contracts Implemented:** BC-6

**Files:**
- Modify: `docs/guide/workloads.md`
- Modify: `docs/reference/configuration.md`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: No tests needed for documentation**

Documentation changes are verified by grep in Step 4.

**Step 2: Update `docs/guide/workloads.md`**

Remove the "CSV traces" row and update the mode count. Current line 12 says "BLIS supports four modes"; change to "three modes":

Change line 13:
```markdown
BLIS supports four modes, in order of precedence:
```
To:
```markdown
BLIS supports three modes, in order of precedence:
```

Remove line 20 (the CSV traces row from the table):
```markdown
| **CSV traces** | `--workload traces` | Replaying recorded production traffic |
```

**Step 3: Update `docs/reference/configuration.md`**

Three edits:

1. Remove the "CSV traces" row from the mode table (line 168):
```markdown
| **CSV traces** | `--workload traces` | Replay recorded traces from a CSV file. |
```

Also update the description above the table from "four workload specification modes" to "three workload specification modes" (line 161):

Change:
```markdown
BLIS supports four workload specification modes, in order of precedence:
```
To:
```markdown
BLIS supports three workload specification modes, in order of precedence:
```

2. Remove the `--workload-traces-filepath` flag row under "Trace Files" (line 249):
```markdown
| `--workload-traces-filepath` | string | "" | Path to CSV trace file (required when `--workload traces`). |
```

After this deletion, the "Trace Files" section heading at line 244 only has two rows (`--workload-spec` and `--defaults-filepath`). That's fine.

3. Remove `--workload-traces-filepath` from the WorkloadConfig flag list (line 385):

Change:
```markdown
| **WorkloadConfig** | `--workload`, `--workload-spec`, `--workload-traces-filepath`, `--defaults-filepath`, `--rate`, `--num-requests`, `--prompt-tokens*`, `--output-tokens*`, `--prefix-tokens` |
```
To:
```markdown
| **WorkloadConfig** | `--workload`, `--workload-spec`, `--defaults-filepath`, `--rate`, `--num-requests`, `--prompt-tokens*`, `--output-tokens*`, `--prefix-tokens` |
```

**Step 4: Update `CLAUDE.md`**

Three edits:

1. In the "Build and Run Commands" section (line 22), remove the `csv-trace` example:
```bash
./blis convert csv-trace --file trace.csv
```
The surrounding lines (preset and servegen examples) remain.

2. In the File Organization tree (line 207), update `cmd/convert.go` description:

Change:
```
│   ├── convert.go             # `blis convert` subcommands (servegen, csv-trace, preset, inference-perf)
```
To:
```
│   ├── convert.go             # `blis convert` subcommands (servegen, preset, inference-perf)
```

3. In the File Organization tree (line 276), update `sim/workload/convert.go` description:

Change:
```
│   ├── convert.go             # Format converters: ConvertServeGen, ConvertCSVTrace, ConvertPreset, ComposeSpecs
```
To:
```
│   ├── convert.go             # Format converters: ConvertServeGen, ConvertPreset, ComposeSpecs
```

**Step 5: Update `README.md`**

Three edits:

1. Remove the `csv-trace` usage example (around line 114-115):
```bash
# Convert a CSV request trace from production logs (requires your own trace.csv)
./blis convert csv-trace --file trace.csv
```
Remove both the comment line and the command line.

2. In the file tree (around line 161), update `cmd/convert.go` description:

Change:
```
│   ├── convert.go          # `./blis convert` subcommands (servegen, csv-trace, preset, inference-perf)
```
To:
```
│   ├── convert.go          # `./blis convert` subcommands (servegen, preset, inference-perf)
```

3. In the file tree (around line 226), update `sim/workload/convert.go` description:

Change:
```
│   ├── convert.go          # Format converters: ConvertServeGen, ConvertCSVTrace, ConvertPreset
```
To:
```
│   ├── convert.go          # Format converters: ConvertServeGen, ConvertPreset
```

**Step 6: Verify documentation references are cleaned up**

Use flag-specific patterns (prefix `--`) to avoid matching generic uses of "workload traces" as a concept (e.g., `docs/contributing/templates/design-guidelines.md:63` which mentions "workload traces" as a general noun, not a flag):

```bash
grep -r "csv-trace\|--workload traces\|--workload-traces-filepath" docs/ CLAUDE.md README.md
```

Expected: The only matches should be in archived plan files (`docs/plans/archive/`, `docs/plans/pr350-phase2-gaps-plan.md`) — those are historical and intentionally left as-is. No matches in live documentation.

**Step 7: Run build and full test suite one final time**

```bash
GOCACHE=/tmp/claude/go-cache go build ./...
GOCACHE=/tmp/claude/go-cache go test github.com/inference-sim/inference-sim/sim/workload github.com/inference-sim/inference-sim/sim/... github.com/inference-sim/inference-sim/cmd -v 2>&1 | tail -30
golangci-lint run ./...
```

Expected: build exits 0, all tests pass, lint zero new issues.

**Step 8: Commit**

```bash
git add docs/guide/workloads.md docs/reference/configuration.md CLAUDE.md README.md
git commit -m "docs: remove csv-trace and --workload traces references (BC-6)

- docs/guide/workloads.md: remove CSV traces mode row, update count to three
- docs/reference/configuration.md: remove CSV traces row, --workload-traces-filepath row, WorkloadConfig flag list entry
- CLAUDE.md: remove csv-trace example and file tree entry
- README.md: remove csv-trace example and file tree entry
- Archived plan files intentionally left unchanged (historical context)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|--------------------------|
| BC-1 | Task 1 | Unit | `TestConvertCmd_NoCSVTraceSubcommand` — cobra subcommand listing |
| BC-2 | Task 2 | Unit | `TestRunCmd_NoWorkloadTracesFlag` — cobra flag lookup |
| BC-3 | Task 2 | Unit | `TestRunCmd_WorkloadFlagDescriptionExcludesTraces` — flag usage string check |
| BC-4 | Task 3+4 | Compilation | `go build ./...` — absence of symbols |
| BC-5 | Task 4 | Unit | Existing `TestConvertPreset_*`, `TestComposeSpecs_*`, `TestConvertServeGen_*`, `TestConvertInferencePerf_*` (unmodified) |
| BC-6 | Task 5 | Manual | `grep -r "csv-trace\|--workload traces\|--workload-traces-filepath"` over docs, CLAUDE.md, README.md |
| BC-N1 | All | Compilation | `go build ./...` after each task commit |
| BC-N2 | Task 4 | Unit | Full `go test ./sim/workload/...` run after deletions |

**Golden dataset:** No output format changes — golden dataset unaffected.

**Invariant note:** This PR removes code, not behavior. No new invariant tests are needed beyond the compilation and existing-test non-regression checks above.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Missed unused import in `convert.go` | Medium | Low | lint catches it; explicit list in Task 4 | Task 4 |
| Missed `"fmt"` removal in `synthesis.go` | Medium | Low | lint catches it; explicitly called out in Task 3 | Task 3 |
| Missed `"os"` / `"path/filepath"` removal in `convert_test.go` | Medium | Low | lint catches it; explicitly called out in Task 4 | Task 4 |
| Archived plan files flagged as unclean by grep | Low | Low | grep step excludes `docs/plans/archive/` and `docs/plans/pr350-*` | Task 5 |
| Test file structure differs from assumed | Low | Low | Step 1 of Task 1 checks for existing test files before creating | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — pure deletion, zero new types
- [x] No feature creep — strictly scoped to issue #654 (no "while we're here" fixes)
- [x] No unexercised flags or interfaces — all deleted code paths removed in full
- [x] No partial implementations — each commit leaves a complete, buildable state
- [x] No breaking changes to non-deprecated paths — BC-5 verified
- [x] No hidden global state impact — removed variables are local to cmd/
- [x] All new code will pass golangci-lint — only 3 small tests added, straightforward
- [x] Shared test helpers used — no new helpers needed for deletion tests
- [x] CLAUDE.md updated — Task 5 includes CLAUDE.md
- [x] No stale references left in CLAUDE.md — Task 5 explicitly updates it
- [x] Documentation DRY — no canonical source modified (rules.md, invariants.md, etc.)
- [x] Deviation log reviewed — zero unresolved deviations
- [x] Each task produces working, testable code — no scaffolding
- [x] Task dependencies correctly ordered — callers removed before callees
- [x] All contracts mapped to specific tasks — see Section H
- [x] Golden dataset regeneration not needed — no output format changes
- [x] Construction site audit — no new structs; no construction sites to audit
- [x] Not part of a macro plan — N/A

**Antipattern rules:**
- [x] R1: No silent continues — N/A (deletion PR)
- [x] R2: No new map iteration — N/A
- [x] R3: No new numeric parameters — N/A
- [x] R4: No struct field additions — N/A
- [x] R5: No resource allocation loops — N/A
- [x] R6: No `logrus.Fatalf` added to `sim/` — N/A
- [x] R7: No new golden tests — N/A
- [x] R8: No exported maps added — N/A
- [x] R9: No new YAML fields — N/A
- [x] R10: No new YAML parsing — N/A
- [x] R11: No new divisions — N/A
- [x] R12: Golden dataset unaffected — verified
- [x] R13: No new interfaces — N/A
- [x] R14: No new multi-responsibility methods — N/A
- [x] R15: No stale PR references introduced — deletions only
- [x] R16: No new config params — N/A
- [x] R17: No routing signals changed — N/A
- [x] R18: No defaults.yaml interactions — N/A
- [x] R19: No loops added — N/A
- [x] R20: No new detectors — N/A
- [x] R21: No range over mutable slices — N/A
- [x] R22: No pre-checks added — N/A
- [x] R23: No parallel code paths — N/A

---

## Appendix: File-Level Implementation Details

### File: `cmd/convert.go`

**Purpose:** Remove `convertCSVTraceCmd`, its variables, and its init() registrations. Update Long description.

**Lines to delete:**
- Lines 36-41: section comment + variables block
- Lines 43-57: `convertCSVTraceCmd` cobra.Command definition
- Lines 144-146: init() flag bindings for csv-trace
- Line 158: `convertCmd.AddCommand(convertCSVTraceCmd)`

**Line to edit:**
- Line 17: `Long` string: remove ", CSV traces" from the list

**Resulting state:** `init()` registers flags for servegen, preset, inference-perf only. `convertCmd.AddCommand` called three times, not four.

---

### File: `cmd/root.go`

**Purpose:** Remove `tracesWorkloadFilePath`, the `else if workloadType == "traces"` block, the `--workload-traces-filepath` flag registration, and update two description strings.

**Lines to delete:**
- Line 40: `tracesWorkloadFilePath string // Workload filepath for traces workload type.`
- Lines 726-738: the `else if workloadType == "traces" { ... }` block (inclusive of the closing `}` before `else if workloadType == "distribution"`)
- Line 1159: `runCmd.Flags().StringVar(&tracesWorkloadFilePath, "workload-traces-filepath", "", ...)`

**Lines to edit:**
- Line 39: remove `, traces` from inline comment
- Line 1158: remove `, traces` from flag description string

---

### File: `sim/workload/synthesis.go`

**Purpose:** Remove `SynthesizeFromCSVTrace` and its `"fmt"` import.

**Lines to delete:**
- Line 3: `import "fmt"` (entire import statement)
- Lines 84-96: `SynthesizeFromCSVTrace` function with doc comment

**Resulting state:** File has two functions (`SynthesizeFromDistribution`, `SynthesizeFromPreset`), no imports needed.

---

### File: `sim/workload/convert.go`

**Purpose:** Remove `ConvertCSVTrace` function and 5 now-unused imports.

**Lines to delete:**
- Lines 32-149: `ConvertCSVTrace` function with doc comment

**Import block change:**
```go
// Before
import (
    "encoding/csv"
    "encoding/json"
    "fmt"
    "io"
    "math"
    "os"
    "strconv"
)

// After
import (
    "fmt"
    "math"
)
```

---

### File: `sim/workload/convert_test.go`

**Purpose:** Remove 4 `TestConvertCSVTrace_*` test functions and their now-unused imports.

**Lines to delete:**
- Lines 9-105: four test functions (ValidFile, EmptyFile, MalformedRow, HorizonTruncation)

**Import block change:**
```go
// Before
import (
    "os"
    "path/filepath"
    "testing"
)

// After
import (
    "testing"
)
```

---

### File: `cmd/root_test.go` (Task 1 addition — file already exists)

**Purpose:** Behavioral test verifying `csv-trace` subcommand absence. Add to existing `cmd/root_test.go`.

**Complete implementation:**

```go
package cmd

import "testing"

func TestConvertCmd_NoCSVTraceSubcommand(t *testing.T) {
	// GIVEN the convert cobra command
	// WHEN listing its subcommands
	for _, sub := range convertCmd.Commands() {
		if sub.Name() == "csv-trace" {
			// THEN csv-trace must not be present
			t.Error("csv-trace subcommand should not exist after removal")
			return
		}
	}
}
```

---

### File: `cmd/root_test.go` (Task 2 addition — same existing file)

**Purpose:** Two behavioral tests verifying flag absence and description update.

**Note:** `cmd/root_test.go` already has an import block. Add `"strings"` to the existing import block (it already imports `"testing"`). Do NOT replace the entire import block.

**Functions to add:**

```go
func TestRunCmd_NoWorkloadTracesFlag(t *testing.T) {
	// GIVEN the run cobra command
	// WHEN looking up the --workload-traces-filepath flag
	f := runCmd.Flags().Lookup("workload-traces-filepath")
	// THEN the flag does not exist
	if f != nil {
		t.Error("--workload-traces-filepath flag should not exist after removal")
	}
}

func TestRunCmd_WorkloadFlagDescriptionExcludesTraces(t *testing.T) {
	// GIVEN the run cobra command
	// WHEN inspecting the --workload flag usage string
	f := runCmd.Flags().Lookup("workload")
	if f == nil {
		t.Fatal("--workload flag must exist")
	}
	// THEN "traces" is not listed as a valid type
	if strings.Contains(f.Usage, "traces") {
		t.Errorf("--workload flag description must not contain 'traces', got: %q", f.Usage)
	}
}
```

All three tests (`TestConvertCmd_NoCSVTraceSubcommand`, `TestRunCmd_NoWorkloadTracesFlag`, `TestRunCmd_WorkloadFlagDescriptionExcludesTraces`) are added to the existing `cmd/root_test.go` — Task 1 adds the first, Task 2 adds the latter two.
