# SLO Priority Translation for vLLM/llm-d Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable `blis observe` to send SLO class priorities to both vLLM (via request body) and llm-d (via HTTP header) simultaneously.

**The problem today:** When `blis observe` dispatches workloads with SLO classes to real servers, it only sends the `x-gateway-inference-objective` header for llm-d. vLLM's priority scheduling feature (`--scheduling-policy priority`) expects a `priority` field in the request body with inverted semantics (lower integer = more urgent), but BLIS doesn't send it. Users testing vLLM priority scheduling must manually modify requests or use custom scripts.

**What this PR adds:**
1. **Priority inversion method** — `SLOPriorityMap.InvertForVLLM(class)` computes vLLM-compatible priority values using `maxPriority - blisPriority`, enabling automatic conversion from BLIS/llm-d semantics (higher=urgent) to vLLM semantics (lower=urgent)
2. **Dual delivery mechanism** — When `slo_class` is set in workload specs, `blis observe` injects both the `x-gateway-inference-objective` header (for llm-d) and the `priority` body field (for vLLM) in every HTTP request
3. **Universal compatibility** — Servers ignore what they don't use: vLLM ignores the header, llm-d (using vLLM FCFS backend) silently ignores the body priority field, making dual delivery safe everywhere

**Why this matters:** This unblocks testing and calibration of vLLM's priority scheduler using real workloads. Users can validate that BLIS's simulated priority preemption matches vLLM's production behavior. It also enables A/B testing between llm-d's gateway-level priorities and vLLM's scheduler-level priorities without changing workload specs.

**Architecture:** Add `InvertForVLLM()` method to `sim.SLOPriorityMap` in `sim/admission.go` (computes max priority across all tiers, returns `max - priority(class)`). In `cmd/observe.go`, extend the `Send()` method to inject `body["priority"] = InvertForVLLM(req.SLOClass)` when `req.SLOClass != ""` alongside the existing header injection. No changes to `sim.Request` or simulation behavior — this is purely an observe-time HTTP injection for real-server testing.

**Source:** GitHub issue #1208

**Closes:** Fixes #1208

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

This PR extends `blis observe` to send priority information to both vLLM and llm-d simultaneously when workloads specify SLO classes. The key insight is that vLLM and llm-d use different priority conventions (lower-is-urgent vs higher-is-urgent) and different mechanisms (body field vs header), but both tolerate the other's mechanism (vLLM FCFS ignores body priority, llm-d ignores vLLM headers). This allows dual delivery without server-side changes.

**System context:** The observe command sits at the boundary between BLIS workload generation (`sim/workload`) and real HTTP servers. It translates `sim.Request` structs with SLO classes into HTTP POST bodies and headers. This PR adds a priority field to the POST body alongside the existing header injection.

**Adjacent components:**
- `sim/workload/GenerateRequests()` — produces `sim.Request` with `SLOClass` field
- `cmd/observe.go:Send()` — builds HTTP requests from `PendingRequest` structs
- `sim.SLOPriorityMap` — maps SLO class strings to integer priorities

**No deviations from issue #1208** — the issue is unambiguous and complete.

### B) Behavioral Contracts

**Positive Contracts (Normal Operation):**

BC-1: Priority inversion for vLLM
- GIVEN an SLOPriorityMap with default priorities (critical=4, standard=3, batch=-1, sheddable=-2, background=-3)
- WHEN InvertForVLLM("critical") is called
- THEN it MUST return 0 (the lowest vLLM priority, computed as maxPriority=4 minus blisPriority=4)
- MECHANISM: Find max across all configured priorities, return `max - Priority(class)`

BC-2: Priority inversion preserves urgency ordering
- GIVEN an SLOPriorityMap with default priorities
- WHEN InvertForVLLM() is called for all five standard classes
- THEN critical MUST map to the lowest vLLM priority (0), standard to 1, and background to the highest (7), preserving the urgency ranking
- MECHANISM: Subtraction from a shared max inverts the scale while preserving relative order

BC-3: Dual delivery for non-empty SLO class
- GIVEN a PendingRequest with SLOClass="critical"
- WHEN Send() builds the HTTP request
- THEN the request MUST include both `x-gateway-inference-objective: critical` header AND `priority: 0` body field
- MECHANISM: Header injection (existing, line 179) + new body field injection

BC-4: No injection for empty SLO class
- GIVEN a PendingRequest with SLOClass=""
- WHEN Send() builds the HTTP request
- THEN the request MUST NOT include the `x-gateway-inference-objective` header or `priority` body field
- MECHANISM: Existing `if req.SLOClass != ""` guard applies to both header and body

BC-5: InvertForVLLM with custom overrides
- GIVEN an SLOPriorityMap with custom overrides `{ "batch": 0 }` (making batch non-sheddable)
- WHEN InvertForVLLM("batch") is called
- THEN it MUST compute max across overridden priorities and return `max - 0`
- MECHANISM: NewSLOPriorityMap merges overrides into defaults; InvertForVLLM sees merged state

**Negative Contracts (What MUST NOT Happen):**

BC-6: No simulation behavior change
- GIVEN any workload with SLO classes
- WHEN running `blis run` or `blis replay`
- THEN the simulated request lifecycle, KV cache allocation, routing decisions, and completion times MUST remain unchanged (this PR only affects observe-time HTTP formatting)
- MECHANISM: `sim/` packages are not modified; only `cmd/observe.go` changes

BC-7: No modification of other body fields
- GIVEN a PendingRequest with `Streaming=true, MaxOutputTokens=128, SLOClass="standard"`
- WHEN Send() injects the priority field
- THEN it MUST NOT overwrite or interfere with existing body fields (`model`, `stream`, `max_tokens`, `messages`, `prompt`, `stream_options`, `min_tokens`)
- MECHANISM: Body is a `map[string]interface{}` built incrementally; priority is added after other fields

**Error Handling Contracts:**

BC-8: Unknown SLO class uses default priority
- GIVEN a PendingRequest with SLOClass="unknown-tier"
- WHEN InvertForVLLM("unknown-tier") is called
- THEN it MUST return `maxPriority - defaultPri` (where defaultPri=3 for Standard)
- MECHANISM: SLOPriorityMap.Priority() already returns defaultPri for unknown keys; InvertForVLLM delegates to Priority()

BC-9: Empty SLOPriorityMap uses fallback
- GIVEN an SLOPriorityMap constructed with nil overrides
- WHEN InvertForVLLM(class) is called
- THEN it MUST compute max from the five default priorities (critical=4, standard=3, batch=-1, sheddable=-2, background=-3) and return the correct inversion
- MECHANISM: NewSLOPriorityMap with nil overrides produces DefaultSLOPriorityMap(); InvertForVLLM computes max from defaults

### C) Component Interaction

**Component Diagram:**

```
┌──────────────────────┐
│ sim/workload         │
│ GenerateRequests()   │────────┐
└──────────────────────┘        │
                                │ sim.Request{SLOClass}
                                ↓
┌──────────────────────────────────────────────────┐
│ cmd/observe.go                                   │
│                                                  │
│  PendingRequest ───→ RealClient.Send()          │
│   └─ SLOClass         │                          │
│                       ↓                          │
│                  Build HTTP Request              │
│                       │                          │
│                       ├─→ Header (existing)      │
│                       │   x-gateway-inference-   │
│                       │   objective: <SLOClass>  │
│                       │                          │
│                       └─→ Body (NEW)             │
│                           priority: <inverted>   │
└──────────────────────────────────────────────────┘
                                │
                                ↓
                    ┌───────────────────────┐
                    │ sim.SLOPriorityMap    │
                    │ InvertForVLLM(class)  │
                    └───────────────────────┘
                                │
                                ↓
                        vLLM Priority (int)
```

**Data flow:** Workload generation produces `sim.Request` with `SLOClass` field → observe command converts to `PendingRequest` (no change) → `RealClient.Send()` queries `SLOPriorityMap.InvertForVLLM(req.SLOClass)` when `req.SLOClass != ""` and injects the result as `body["priority"]` → HTTP POST sent to server with both header and body field.

**API Contracts:**

1. **New method signature:**
   ```go
   func (m *SLOPriorityMap) InvertForVLLM(class string) int
   ```
   - **Precondition:** `m` is non-nil (constructed via `DefaultSLOPriorityMap()` or `NewSLOPriorityMap()`)
   - **Postcondition:** Returns `maxPriority - m.Priority(class)` where `maxPriority` is the largest value in `m.priorities` or `m.defaultPri`, whichever is greater
   - **Failure mode:** None (always returns an integer; unknown classes → defaultPri path)

2. **Modified behavior (no signature change):**
   - `RealClient.Send()` in `cmd/observe.go` now checks `req.SLOClass != ""` and injects `body["priority"]` before marshaling
   - No new parameters, no new return values
   - Failure mode: If `InvertForVLLM()` panics (it won't — pure computation), Send() already has error handling for all downstream steps

**State Changes:**
- **No new mutable state** — `InvertForVLLM()` is a pure query method (reads `m.priorities` and `m.defaultPri`, computes max, returns result)
- **Existing state unchanged** — `PendingRequest.SLOClass` field already exists (added in #420); this PR only consumes it in a new way

**Extension Friction Assessment:**
- **Adding one more SLO class** requires changes in:
  1. `sim/admission.go` — add to `DefaultSLOPriorityMap()` priorities map (1 line)
  2. Documentation — update CLAUDE.md "SLO tiers" section (1 line)
  3. No changes needed in `cmd/observe.go` — InvertForVLLM() is data-driven
- **Total: 2 files** (acceptable for a cross-cutting domain concept like SLO tiers)

### D) Deviation Log

No deviations from source document (issue #1208).

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Validation enforces: all clients have `slo_class` set OR all empty (no mixing)" | Validation deferred to issue #1210 | Issue #1208 explicitly notes "To be fixed in #1210" in the validation section |

### E) Review Guide

**THE TRICKY PART:**
The priority inversion logic requires computing the maximum priority across all configured tiers (including overrides), not just using a hardcoded constant. Default max is 4 (critical), but custom overrides could set `critical=10`, changing the max and thus all inversions. The implementation must scan `m.priorities` and compare with `m.defaultPri` to handle this correctly.

**WHAT TO SCRUTINIZE:**
- BC-1 test: Verify the max computation iterates over the priorities map and compares with defaultPri (both must be considered)
- BC-5 test: Verify custom overrides are reflected in the inversion (max changes if an override exceeds the previous max)
- cmd/observe.go: Verify the body field injection happens AFTER all other fields are set but BEFORE marshaling (lines 112-146 build the body; priority injection goes around line 146-147, right before marshaling at line 153)

**WHAT'S SAFE TO SKIM:**
- The header injection (line 178-180) is unchanged — just read it for context
- The HTTP round-trip logic (lines 185-231) is untouched
- Test structure for BC-2, BC-3, BC-4, BC-7, BC-8, BC-9 is mechanical table-driven testing

**KNOWN DEBT:**
- Mixed SLO class validation (#1210) is deferred — observe currently accepts workloads where some clients have `slo_class` set and others don't, which can create ambiguous priority intent
- No integration test with a real vLLM priority server (requires Docker setup with vLLM `--scheduling-policy priority`; out of scope for unit testing)

---

## PART 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- None

**Files to modify:**
- `sim/admission.go` — Add `InvertForVLLM()` method to `SLOPriorityMap` (~15 lines: method + doc comment)
- `sim/admission_test.go` — Modify if exists, or add tests to package test file (~80 lines: 5 test cases)
- `cmd/observe.go` — Inject priority field into request body when SLOClass is set (~3 lines at line 146)
- `cmd/observe_test.go` — Add test for dual delivery mechanism (~40 lines: 1 test case with table)

**Key decisions:**
1. **Max computation strategy:** Iterate over all priorities in the map, compare with defaultPri, return the largest (handles both defaults and custom overrides)
2. **Injection point:** Add `body["priority"]` right before the JSON marshaling step (line 153) to ensure it's included in the POST body
3. **Test strategy:** Use table-driven tests for InvertForVLLM with various SLO classes; mock HTTP server for observe integration test

**Confirmation:**
- **No dead code:** Every method added is called (InvertForVLLM called by observe.go); every test exercises production code
- **All paths exercisable:** Both the empty-SLOClass path (no injection) and non-empty path (dual injection) are tested

### G) Task Breakdown

---

### Task 1: Add InvertForVLLM Method to SLOPriorityMap

**Contracts Implemented:** BC-1, BC-2, BC-5, BC-8, BC-9

**Files:**
- Modify: `sim/admission.go` (after line 117, before SLOTierPriority function)
- Test: `sim/admission_test.go` (or create if missing)

**Step 1: Write failing tests for InvertForVLLM**

Context: Test the priority inversion formula with default priorities, custom overrides, and unknown classes.

In `sim/admission_test.go`:
```go
func TestSLOPriorityMap_InvertForVLLM_DefaultPriorities(t *testing.T) {
	m := DefaultSLOPriorityMap()
	tests := []struct {
		class    string
		expected int
	}{
		{"critical", 0},     // 4 - 4 = 0 (most urgent in vLLM)
		{"standard", 1},     // 4 - 3 = 1
		{"batch", 5},        // 4 - (-1) = 5
		{"sheddable", 6},    // 4 - (-2) = 6
		{"background", 7},   // 4 - (-3) = 7 (least urgent in vLLM)
		{"unknown", 1},      // 4 - 3 (defaultPri) = 1
		{"", 1},             // 4 - 3 (defaultPri) = 1
	}
	for _, tt := range tests {
		t.Run(tt.class, func(t *testing.T) {
			got := m.InvertForVLLM(tt.class)
			if got != tt.expected {
				t.Errorf("InvertForVLLM(%q) = %d, want %d", tt.class, got, tt.expected)
			}
		})
	}
}

func TestSLOPriorityMap_InvertForVLLM_CustomOverrides(t *testing.T) {
	// Override: batch=0 (non-sheddable), critical=10 (ultra-high)
	m := NewSLOPriorityMap(map[string]int{
		"batch":    0,
		"critical": 10,
	})
	tests := []struct {
		class    string
		expected int
	}{
		{"critical", 0},    // 10 - 10 = 0 (max is now 10)
		{"standard", 7},    // 10 - 3 = 7
		{"batch", 10},      // 10 - 0 = 10
		{"sheddable", 12},  // 10 - (-2) = 12
		{"background", 13}, // 10 - (-3) = 13
	}
	for _, tt := range tests {
		t.Run(tt.class, func(t *testing.T) {
			got := m.InvertForVLLM(tt.class)
			if got != tt.expected {
				t.Errorf("InvertForVLLM(%q) = %d, want %d", tt.class, got, tt.expected)
			}
		})
	}
}

func TestSLOPriorityMap_InvertForVLLM_PreservesUrgencyOrder(t *testing.T) {
	// GIVEN default priorities
	m := DefaultSLOPriorityMap()

	// WHEN inverting all classes
	critical := m.InvertForVLLM("critical")
	standard := m.InvertForVLLM("standard")
	batch := m.InvertForVLLM("batch")
	sheddable := m.InvertForVLLM("sheddable")
	background := m.InvertForVLLM("background")

	// THEN vLLM priorities preserve urgency order (lower = more urgent)
	if !(critical < standard && standard < batch && batch < sheddable && sheddable < background) {
		t.Errorf("InvertForVLLM broke urgency order: critical=%d, standard=%d, batch=%d, sheddable=%d, background=%d",
			critical, standard, batch, sheddable, background)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/... -run TestSLOPriorityMap_InvertForVLLM -v`
Expected: FAIL with "m.InvertForVLLM undefined (type *SLOPriorityMap has no field or method InvertForVLLM)"

**Step 3: Implement InvertForVLLM method**

Context: Compute max priority across all tiers, then return `max - Priority(class)`.

In `sim/admission.go` (insert after line 117, after IsSheddable method):
```go
// InvertForVLLM converts a BLIS SLO class to a vLLM priority value.
// vLLM uses lower integers for higher urgency (min-heap), opposite of BLIS/llm-d.
// Returns maxPriority - Priority(class), where maxPriority is the highest value
// across all configured priorities and defaultPri.
//
// Example with defaults (maxPriority=4):
//   critical (4) → 0, standard (3) → 1, batch (-1) → 5, sheddable (-2) → 6, background (-3) → 7
//
// Handles custom overrides: if an override sets critical=10, maxPriority becomes 10
// and all inversions adjust accordingly.
func (m *SLOPriorityMap) InvertForVLLM(class string) int {
	maxPriority := m.defaultPri
	for _, p := range m.priorities {
		if p > maxPriority {
			maxPriority = p
		}
	}
	return maxPriority - m.Priority(class)
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run TestSLOPriorityMap_InvertForVLLM -v`
Expected: PASS (all 3 test functions, ~17 total cases)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit with contract references**

```bash
git add sim/admission.go sim/admission_test.go
git commit -m "feat(sim): add SLOPriorityMap.InvertForVLLM for vLLM priority translation (BC-1, BC-2, BC-5, BC-8, BC-9)

- Add InvertForVLLM() method to compute vLLM-compatible priorities
- Implement BC-1: critical maps to 0 (most urgent in vLLM)
- Implement BC-2: urgency order preserved after inversion
- Implement BC-5: custom overrides handled (max recomputed)
- Implement BC-8: unknown classes use defaultPri
- Implement BC-9: nil overrides use default priorities

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Inject Priority Field in Observe HTTP Requests

**Contracts Implemented:** BC-3, BC-4, BC-6, BC-7

**Files:**
- Modify: `cmd/observe.go` (around line 146, after body construction, before marshaling)

**Step 1: Write failing test for dual delivery**

Context: Test that when SLOClass is set, both header and body contain priority information.

In `cmd/observe_test.go`:
```go
package cmd

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim"
)

func TestRealClient_Send_InjectsPriorityWhenSLOClassSet(t *testing.T) {
	tests := []struct {
		name             string
		sloClass         string
		expectHeader     bool
		expectHeaderVal  string
		expectBodyField  bool
		expectBodyVal    int
	}{
		{
			name:            "critical class",
			sloClass:        "critical",
			expectHeader:    true,
			expectHeaderVal: "critical",
			expectBodyField: true,
			expectBodyVal:   0, // 4 - 4 = 0
		},
		{
			name:            "standard class",
			sloClass:        "standard",
			expectHeader:    true,
			expectHeaderVal: "standard",
			expectBodyField: true,
			expectBodyVal:   1, // 4 - 3 = 1
		},
		{
			name:            "batch class",
			sloClass:        "batch",
			expectHeader:    true,
			expectHeaderVal: "batch",
			expectBodyField: true,
			expectBodyVal:   5, // 4 - (-1) = 5
		},
		{
			name:            "empty slo_class",
			sloClass:        "",
			expectHeader:    false,
			expectHeaderVal: "",
			expectBodyField: false,
			expectBodyVal:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Mock server that captures the request
			var capturedHeader string
			var capturedBody map[string]interface{}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				capturedHeader = r.Header.Get("x-gateway-inference-objective")
				decoder := json.NewDecoder(r.Body)
				_ = decoder.Decode(&capturedBody)
				// Return minimal valid response
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id":"test","choices":[{"text":"output"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`))
			}))
			defer server.Close()

			client := &RealClient{
				httpClient: &http.Client{Timeout: 5 * time.Second},
				baseURL:    server.URL,
				modelName:  "test-model",
				apiFormat:  "completions",
			}

			req := &PendingRequest{
				RequestID:       1,
				InputTokens:     10,
				MaxOutputTokens: 20,
				Model:           "test-model",
				Streaming:       false,
				SLOClass:        tt.sloClass,
				Prompt:          "test prompt",
			}

			ctx := context.Background()
			_, err := client.Send(ctx, req)
			if err != nil {
				t.Fatalf("Send() error = %v", err)
			}

			// Verify header
			if tt.expectHeader {
				if capturedHeader != tt.expectHeaderVal {
					t.Errorf("header x-gateway-inference-objective = %q, want %q", capturedHeader, tt.expectHeaderVal)
				}
			} else {
				if capturedHeader != "" {
					t.Errorf("header x-gateway-inference-objective should be empty, got %q", capturedHeader)
				}
			}

			// Verify body priority field
			if tt.expectBodyField {
				priority, ok := capturedBody["priority"]
				if !ok {
					t.Errorf("body missing 'priority' field")
				} else {
					// JSON numbers decode as float64
					priorityInt := int(priority.(float64))
					if priorityInt != tt.expectBodyVal {
						t.Errorf("body['priority'] = %d, want %d", priorityInt, tt.expectBodyVal)
					}
				}
			} else {
				if _, ok := capturedBody["priority"]; ok {
					t.Errorf("body should not contain 'priority' field when SLOClass is empty")
				}
			}

			// Verify other body fields are not disturbed (BC-7)
			if capturedBody["model"] != "test-model" {
				t.Errorf("body['model'] was disturbed: got %v", capturedBody["model"])
			}
			if capturedBody["stream"] != false {
				t.Errorf("body['stream'] was disturbed: got %v", capturedBody["stream"])
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestRealClient_Send_InjectsPriority -v`
Expected: FAIL with "body missing 'priority' field" for non-empty SLOClass cases

**Step 3: Inject priority field into request body**

Context: After building the body map but before marshaling, check if SLOClass is set and inject the inverted priority.

In `cmd/observe.go` (around line 146, after the endpoint switch and before marshaling):
```go
	// Set prompt/messages and endpoint based on API format.
	var endpoint string
	switch c.apiFormat {
	case "chat":
		endpoint = c.baseURL + "/v1/chat/completions"
		body["messages"] = []map[string]string{{"role": "user", "content": req.Prompt}}
	default: // "completions"
		endpoint = c.baseURL + "/v1/completions"
		body["prompt"] = req.Prompt
	}

	// Request usage data in streaming responses (required for token count extraction).
	if req.Streaming {
		body["stream_options"] = map[string]interface{}{"include_usage": true}
	}

	// Inject vLLM priority field when SLO class is set (dual delivery: header + body).
	// vLLM priority scheduling (--scheduling-policy priority) uses lower integers for
	// higher urgency. BLIS/llm-d use higher integers. InvertForVLLM converts.
	if req.SLOClass != "" {
		sloMap := sim.DefaultSLOPriorityMap()
		body["priority"] = sloMap.InvertForVLLM(req.SLOClass)
	}

	bodyBytes, err := json.Marshal(body)
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -run TestRealClient_Send_InjectsPriority -v`
Expected: PASS (4 test cases: critical, standard, batch, empty)

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit with contract references**

```bash
git add cmd/observe.go cmd/observe_test.go
git commit -m "feat(observe): inject vLLM priority field when slo_class set (BC-3, BC-4, BC-6, BC-7)

- Implement BC-3: dual delivery (header + body) for non-empty SLOClass
- Implement BC-4: no injection when SLOClass is empty
- Implement BC-6: no simulation behavior change (observe-only)
- Implement BC-7: existing body fields unaffected

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|-------------------------|
| BC-1 | Task 1 | Unit | TestSLOPriorityMap_InvertForVLLM_DefaultPriorities — critical→0 |
| BC-2 | Task 1 | Unit | TestSLOPriorityMap_InvertForVLLM_PreservesUrgencyOrder — ordering preserved |
| BC-5 | Task 1 | Unit | TestSLOPriorityMap_InvertForVLLM_CustomOverrides — max recomputed |
| BC-8 | Task 1 | Unit | TestSLOPriorityMap_InvertForVLLM_DefaultPriorities — unknown→defaultPri path |
| BC-9 | Task 1 | Unit | TestSLOPriorityMap_InvertForVLLM_DefaultPriorities — nil overrides use defaults |
| BC-3 | Task 2 | Integration | TestRealClient_Send_InjectsPriorityWhenSLOClassSet — dual delivery |
| BC-4 | Task 2 | Integration | TestRealClient_Send_InjectsPriorityWhenSLOClassSet — empty SLOClass case |
| BC-6 | Task 2 | Integration | (verified by unchanged sim/ packages — no test needed) |
| BC-7 | Task 2 | Integration | TestRealClient_Send_InjectsPriorityWhenSLOClassSet — other fields checked |

**Test types:**
- **Unit:** `sim/admission_test.go` tests for InvertForVLLM method in isolation
- **Integration:** `cmd/observe_test.go` tests for full HTTP request construction with mock server

**Shared test infrastructure:**
- Use `httptest.NewServer` for mock HTTP server in cmd tests (standard library, no new dependencies)
- No golden dataset updates needed (this PR doesn't change simulator output)

**Lint requirements:**
- `golangci-lint run ./...` must pass with zero new issues
- Pre-existing issues (if any) are ignored per project guidelines

**Test naming convention:**
- Follows BDD-style: `TestType_Method_Scenario` (e.g., `TestSLOPriorityMap_InvertForVLLM_DefaultPriorities`)

**Test isolation:**
- Each test case is independent (no shared mutable state)
- Table-driven tests for multiple scenarios of the same behavior

**No invariant tests needed:**
This PR doesn't touch request lifecycle, KV cache, or simulation metrics. The existing invariant tests (INV-1 through INV-12) remain valid and unchanged.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|------------|--------|------------|------|
| Priority inversion formula incorrect (off-by-one, wrong max) | Medium | High | Table-driven tests with all 5 default classes + custom overrides; verify against issue #1208 formula | Task 1 |
| Body field injection breaks existing fields (overwrite model, stream, etc.) | Low | High | Test verifies other body fields unchanged (BC-7) | Task 2 |
| vLLM FCFS rejects requests with priority field | Low | Medium | Issue #1208 verified vLLM FCFS silently ignores priority (see FCFSRequestQueue implementation) | N/A (design) |
| Custom SLO priority overrides not reflected in inversion | Medium | Medium | Test with custom overrides (BC-5); max computation must scan all priorities | Task 1 |
| Header injection regresses (accidentally removed during refactor) | Low | Medium | Test verifies both header and body present (BC-3) | Task 2 |

---

## PART 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — InvertForVLLM is the minimal method needed
- [x] No feature creep — strictly implements issue #1208; mixed validation deferred to #1210
- [x] No unexercised flags or interfaces — InvertForVLLM called by observe.go
- [x] No partial implementations — full dual delivery (header + body) implemented
- [x] No breaking changes — observe behavior is additive (only affects HTTP formatting)
- [x] No hidden global state impact — InvertForVLLM is stateless (reads map, computes, returns)
- [x] All new code will pass golangci-lint — tested in each task
- [x] Shared test helpers used — uses standard `httptest.NewServer`, no duplication
- [x] CLAUDE.md updated if needed — no new files/packages; CLI flags unchanged; plan completed (update CLAUDE.md "Recent Changes" after merge)
- [x] No stale references in CLAUDE.md — no references to remove
- [x] Documentation DRY — no canonical sources modified (no rules/invariants/principles changes)
- [x] Deviation log reviewed — zero deviations except intentional deferral of #1210
- [x] Each task produces working code — Task 1: method + tests pass; Task 2: injection + tests pass
- [x] Task dependencies ordered — Task 1 (method) must complete before Task 2 (usage)
- [x] All contracts mapped to tasks — see Test Strategy table
- [x] Golden dataset regeneration — not needed (no simulation output changes)
- [x] Construction site audit — no struct fields added; PendingRequest.SLOClass already exists
- [x] Macro plan status — not part of a macro plan (standalone issue)

**Antipattern rules (subset applicable to this PR):**
- [x] R1: No silent continue/return — InvertForVLLM has no early returns
- [x] R2: Map iteration order — priorities map iteration only for max computation (order-independent)
- [x] R3: Numeric validation — no new CLI flags or constructors with numeric params
- [x] R4: Construction site audit — no new fields added to existing structs
- [x] R5: Resource allocation rollback — no resource allocation in this PR
- [x] R6: No logrus.Fatalf in sim/ — InvertForVLLM is in sim/ but never calls logrus
- [x] R7: Invariant tests — no golden tests added; no invariant changes
- [x] R8: Exported mutable maps — SLOPriorityMap.priorities already unexported (per existing code)
- [x] R9: YAML float64 pointers — no YAML config changes
- [x] R10: YAML strict parsing — no YAML schema changes
- [x] R11: Division guards — no division operations added
- [x] R12: Golden dataset regen — not applicable (no dataset changes)
- [x] R13: Interfaces work for 2+ implementations — no new interfaces
- [x] R14: Single-module methods — InvertForVLLM is single-responsibility (priority conversion)
- [x] R15: Stale PR references — no PR references in code
- [x] R16: Config grouped by module — no config changes
- [x] R17: Routing scorer signals — not applicable (no scorer changes)
- [x] R18: CLI flag overrides — no CLI flag changes
- [x] R19: Unbounded retry loops — no retry logic added
- [x] R20: Degenerate inputs — InvertForVLLM handles empty/unknown classes via defaultPri
- [x] R21: Range over shrinking slices — no slice iteration added
- [x] R22: Pre-check consistency — not applicable (no pre-checks)
- [x] R23: Parallel transformations — not applicable (no parallelism)

---

## APPENDIX: File-Level Implementation Details

### File: `sim/admission.go`

**Purpose:** Add InvertForVLLM method to SLOPriorityMap type for vLLM priority conversion.

**Location:** Insert after line 117 (after IsSheddable method, before SLOTierPriority function).

**Complete Implementation:**

```go
// InvertForVLLM converts a BLIS SLO class to a vLLM priority value.
// vLLM uses lower integers for higher urgency (min-heap), opposite of BLIS/llm-d.
// Returns maxPriority - Priority(class), where maxPriority is the highest value
// across all configured priorities and defaultPri.
//
// Example with defaults (maxPriority=4):
//   critical (4) → 0, standard (3) → 1, batch (-1) → 5, sheddable (-2) → 6, background (-3) → 7
//
// Handles custom overrides: if an override sets critical=10, maxPriority becomes 10
// and all inversions adjust accordingly.
func (m *SLOPriorityMap) InvertForVLLM(class string) int {
	maxPriority := m.defaultPri
	for _, p := range m.priorities {
		if p > maxPriority {
			maxPriority = p
		}
	}
	return maxPriority - m.Priority(class)
}
```

**Key Implementation Notes:**
- **RNG usage:** None (pure computation, no randomness)
- **Metrics:** None (method doesn't record metrics)
- **Event ordering:** N/A (not an event-driven method)
- **State mutation:** None (pure query method, reads `m.priorities` and `m.defaultPri`)
- **Error handling:** None (always returns an integer; unknown classes delegate to Priority() which returns defaultPri)

**Behavioral subtleties:**
- The max computation must include both `m.defaultPri` and all values in `m.priorities` to handle the case where `defaultPri > max(priorities)` (rare but possible with extreme custom overrides)
- Map iteration order doesn't matter because we're computing a max (commutative operation)

---

### File: `sim/admission_test.go`

**Purpose:** Test InvertForVLLM method with default priorities, custom overrides, and edge cases.

**Complete Implementation:**

(See Task 1 Step 1 for full test code — 3 test functions:
`TestSLOPriorityMap_InvertForVLLM_DefaultPriorities`,
`TestSLOPriorityMap_InvertForVLLM_CustomOverrides`,
`TestSLOPriorityMap_InvertForVLLM_PreservesUrgencyOrder`)

---

### File: `cmd/observe.go`

**Purpose:** Inject vLLM priority field into HTTP request body when SLOClass is set.

**Location:** Insert after line 151 (after stream_options injection, before marshaling at line 153).

**Complete Implementation:**

```go
	// Request usage data in streaming responses (required for token count extraction).
	if req.Streaming {
		body["stream_options"] = map[string]interface{}{"include_usage": true}
	}

	// Inject vLLM priority field when SLO class is set (dual delivery: header + body).
	// vLLM priority scheduling (--scheduling-policy priority) uses lower integers for
	// higher urgency. BLIS/llm-d use higher integers. InvertForVLLM converts.
	// Safe for all servers: vLLM FCFS silently ignores priority, llm-d ignores body priority.
	if req.SLOClass != "" {
		sloMap := sim.DefaultSLOPriorityMap()
		body["priority"] = sloMap.InvertForVLLM(req.SLOClass)
	}

	bodyBytes, err := json.Marshal(body)
```

**Key Implementation Notes:**
- **RNG usage:** None
- **Metrics:** None (HTTP formatting doesn't record metrics directly; metrics recorded in RequestRecord)
- **Event ordering:** N/A (HTTP client, not DES)
- **State mutation:** Modifies local `body` map before marshaling (ephemeral, not shared state)
- **Error handling:** None needed (InvertForVLLM always returns an integer; marshaling errors already handled at line 154-157)

**Behavioral subtleties:**
- `DefaultSLOPriorityMap()` is instantiated per request (no mutable state, safe for concurrent observe operations)
- The header injection (line 178-180) already exists; this PR adds the body field in parallel
- Body field injected AFTER all other fields to avoid ordering dependencies in the map

---

### File: `cmd/observe_test.go`

**Purpose:** Test dual delivery mechanism (header + body) and verify other fields unaffected.

**Complete Implementation:**

(See Task 2 Step 1 for full test code — 1 test function with 4 table cases:
`TestRealClient_Send_InjectsPriorityWhenSLOClassSet`)

---

## Execution Handoff

This plan is ready for execution. Two options:

**Option 1: Subagent-Driven Development (in current session)**
- Invoke `superpowers:subagent-driven-development @docs/plans/pr1208-slo-priority-translation-plan.md`
- Fresh subagent per task
- Code review between tasks
- Fast iteration

**Option 2: Inline Execution (recommended for this 2-task PR)**
- Invoke `superpowers:executing-plans @docs/plans/pr1208-slo-priority-translation-plan.md`
- Continuous execution (stops only on test/lint failure)
- Single commit per task (2 commits total)

After execution completes, invoke `superpowers:verification-before-completion` to run build/test/lint verification gate, then use `/commit-commands:commit-push-pr` to create the pull request with behavioral contracts in the description.
