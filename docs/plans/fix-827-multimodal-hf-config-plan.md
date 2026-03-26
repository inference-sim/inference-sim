# Implementation Plan: Fix HuggingFace Config Parser for Multimodal Models

**Goal:** Enable `isHFConfig` to recognize multimodal model configs where transformer fields are nested in `text_config`

**Source:** Issue #827

**Closes:** #827

---

## Behavioral Contracts

### BC-1: Top-level field validation (existing behavior preserved)
**GIVEN** a config.json with `num_hidden_layers` or `hidden_size` at the top level
**WHEN** `isHFConfig` validates the JSON
**THEN** it returns `true`

### BC-2: Nested text_config field validation (new behavior)
**GIVEN** a config.json with `text_config.num_hidden_layers` or `text_config.hidden_size` (multimodal model structure)
**WHEN** `isHFConfig` validates the JSON
**THEN** it returns `true`

### BC-3: Invalid config rejection (existing behavior preserved)
**GIVEN** a config.json without transformer fields at top level OR in `text_config`
**WHEN** `isHFConfig` validates the JSON
**THEN** it returns `false`

### BC-4: Backward compatibility
**GIVEN** all existing text-only model configs
**WHEN** `isHFConfig` validates them
**THEN** they continue to be recognized as valid (no regressions)

---

## Deviation Log

| Item | Deviation | Reason |
|------|-----------|--------|
| Scope | Issue #827 describes a "parser failure" but the actual bug is in the validation layer (`isHFConfig` in `cmd/hfconfig.go`), not the parser (`ParseHFConfig` in `sim/latency/config.go`) | `ParseHFConfig` already handles `text_config` pivoting correctly (lines 113-119). The validation gate rejects configs before parsing ever runs. Only `isHFConfig` needs fixing. |
| BC-4 Coverage | BC-4 (backward compatibility) is not tested by a new test in this PR | BC-4 is already covered by existing tests (`TestResolveModelConfig_LocalHit`, `TestIsHFConfig` with top-level fields). The new Task 3 test verifies BC-2 end-to-end, not BC-4. |
| Parameter Extraction | Plan does not include end-to-end test through `ParseHFConfig` → `GetModelConfigFromHF` to verify extracted parameters | Task 3 integration test focuses on validation/resolution chain. Parameter extraction correctness is assumed based on existing `ParseHFConfig` pivot logic (already in production, lines 113-119 of `sim/latency/config.go`). |

---

## Tasks

### Task 1: Write failing tests for multimodal config validation (BC-2, BC-3)
**Test:** Add table-driven test cases to `cmd/hfconfig_test.go` in `TestIsHFConfig`:
```go
{"multimodal with text_config num_hidden_layers", `{"text_config": {"num_hidden_layers": 48}}`, true},
{"multimodal with text_config hidden_size", `{"text_config": {"hidden_size": 5120}}`, true},
{"multimodal with both text_config fields", `{"text_config": {"num_hidden_layers": 48, "hidden_size": 5120, "num_attention_heads": 40}}`, true},
{"multimodal without expected fields", `{"text_config": {"other_field": 123}, "vision_config": {"hidden_size": 1408}}`, false},
{"vision_config only (no text_config)", `{"vision_config": {"num_hidden_layers": 34, "hidden_size": 1408}}`, false},
{"text_config is not an object (string)", `{"text_config": "not_an_object"}`, false},
{"text_config is not an object (null)", `{"text_config": null}`, false},
{"deeply nested text_config", `{"text_config": {"text_config": {"num_hidden_layers": 48}}}`, false},
```

**Run:** `go test ./cmd/... -run TestIsHFConfig -v`
**Expected:** FAIL (multimodal cases not yet handled)

**Commit:** None (test-only task, commit with implementation)

---

### Task 2: Implement nested text_config field checking (BC-2)
**Implementation:** Update `isHFConfig` in `cmd/hfconfig.go`:
```go
func isHFConfig(data []byte) bool {
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		return false
	}

	// Check for fields present in every HuggingFace transformer config.json
	// Try top level first (text-only models)
	_, hasLayers := m["num_hidden_layers"]
	_, hasHidden := m["hidden_size"]
	if hasLayers || hasHidden {
		return true
	}

	// Fall back to text_config.* for multimodal models (Llama4ForConditionalGeneration, etc.)
	if textCfg, ok := m["text_config"].(map[string]interface{}); ok {
		_, hasLayers = textCfg["num_hidden_layers"]
		_, hasHidden = textCfg["hidden_size"]
		return hasLayers || hasHidden
	}

	return false
}
```

**Run:** `go test ./cmd/... -run TestIsHFConfig -v`
**Expected:** PASS (all test cases pass)

**Lint:** `golangci-lint run ./cmd/...`
**Expected:** No issues

**Commit:**
```bash
git add cmd/hfconfig.go cmd/hfconfig_test.go
git commit -m "fix(cmd): recognize multimodal HF configs with nested text_config

- Implement BC-1, BC-2, BC-3: isHFConfig checks both top-level and text_config.* fields
- Add table-driven tests for multimodal config validation
- Fixes #827

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Add end-to-end integration test (BC-4)
**Test:** Add integration test to verify multimodal config survives full resolution chain:
```go
func TestResolveModelConfig_MultimodalConfig(t *testing.T) {
	tmpDir := t.TempDir()
	localDir := filepath.Join(tmpDir, modelConfigsDir, "llama4-test")
	if err := os.MkdirAll(localDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Write a multimodal config (text_config structure)
	multimodalConfig := `{
		"architectures": ["Llama4ForConditionalGeneration"],
		"model_type": "llama4",
		"text_config": {
			"num_hidden_layers": 48,
			"hidden_size": 5120,
			"num_attention_heads": 40,
			"num_key_value_heads": 8
		},
		"vision_config": {
			"num_hidden_layers": 34,
			"hidden_size": 1408
		}
	}`
	if err := os.WriteFile(filepath.Join(localDir, hfConfigFile), []byte(multimodalConfig), 0o644); err != nil {
		t.Fatal(err)
	}

	// Mock HF fetch to fail (safety net - local config should be found first)
	old := fetchHFConfigFunc
	fetchHFConfigFunc = func(_, _ string) (string, error) {
		return "", fmt.Errorf("test should not reach HF fetch - local config should be found")
	}
	t.Cleanup(func() { fetchHFConfigFunc = old })

	defaultsFile := filepath.Join(tmpDir, "defaults.yaml")
	dir, err := resolveModelConfig("test-org/llama4-test", "", defaultsFile)
	if err != nil {
		t.Fatalf("multimodal config should be recognized: %v", err)
	}
	expected := filepath.Join(tmpDir, modelConfigsDir, "llama4-test")
	if dir != expected {
		t.Errorf("expected %s, got %s", expected, dir)
	}
}
```

**Run:** `go test ./cmd/... -run TestResolveModelConfig_MultimodalConfig -v`
**Expected:** PASS

**Commit:**
```bash
git add cmd/hfconfig_test.go
git commit -m "test(cmd): add integration test for multimodal config resolution

- Verify multimodal configs (text_config structure) pass full resolution chain
- Verifies BC-2 end-to-end (multimodal config acceptance through full resolution path)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Update error message for clarity
**Implementation:** Update the error message in `fetchHFConfigFromURL` (line 205-207) to mention nested configs:
```go
if !isHFConfig(body) {
	return "", fmt.Errorf("response from %s is valid JSON but does not contain expected "+
		"HuggingFace config fields (num_hidden_layers, hidden_size, or text_config with these fields). "+
		"The model may not exist or the response is an error page", url)
}
```

And in `resolveModelConfig` (line 62):
```go
logrus.Warnf("--latency-model: config at %s exists but lacks expected HuggingFace fields (num_hidden_layers, hidden_size, or text_config.*); trying HuggingFace fetch", localPath)
```

**Run:** `go test ./cmd/...`
**Expected:** All tests pass

**Lint:** `golangci-lint run ./cmd/...`
**Expected:** No issues

**Commit:**
```bash
git add cmd/hfconfig.go
git commit -m "docs(cmd): clarify error messages for multimodal config validation

- Mention text_config.* fallback in validation error messages
- Helps users understand why multimodal configs are now accepted

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Sanity Checklist

- [ ] All tests pass: `go test ./cmd/... && go test ./...`
- [ ] Lint passes: `golangci-lint run ./...`
- [ ] Build succeeds: `go build -o blis main.go`
- [ ] Manual verification: Config from issue #827 is recognized
- [ ] No structural tests: All tests check observable behavior (BC validation)
- [ ] No regressions: Existing text-only model configs still work
- [ ] Error messages updated to reflect new behavior
- [ ] Antipattern rules: R1 (no silent continue/return) ✓, R2 (map key sorting) N/A, R3 (input validation) ✓ (via tests), R4 (construction sites) N/A (no new structs), R5 (resource rollback) N/A, R6 (no logrus.Fatalf in sim/) N/A (cmd/ only), R7 (invariant tests) ✓ (behavioral tests), R8-10 (exported maps, YAML pointers, strict parsing) N/A (no config structs modified), R11-23 N/A or verified by lint

---

## Notes

**PR Size:** Medium tier (2 files: `cmd/hfconfig.go`, `cmd/hfconfig_test.go`)

**Rationale:** While only 2 files are changed, this PR modifies behavioral logic in the `isHFConfig` validation function (adding nested `text_config` checking). Per `docs/contributing/pr-workflow.md`, Small tier requires "only mechanical changes AND no behavioral logic changes." The behavioral logic change qualifies this as Medium tier despite the low file count.

**Extension Type:** Bug fix (parser limitation), not a new feature

**Affected Invariants:** None (this is a parser fix, not a simulator behavior change)

**Backward Compatibility:** Fully backward compatible — all existing text-only model configs continue to work. Multimodal configs are now also recognized.

**Test Strategy:**
- Unit tests for `isHFConfig` with multimodal structures (table-driven)
- Integration test for full resolution chain with multimodal config
- No golden tests needed (all tests check behavioral contracts)

**Design Alignment:** This fix aligns with the existing `ParseHFConfig` behavior (lines 113-119 in `sim/latency/config.go`), which already pivots to `text_config` when present. The validation layer now matches the parsing layer's understanding of multimodal configs.

**Semantic Notes:**
- **Field precedence:** `ParseHFConfig` merges `text_config` fields into the top-level map, with `text_config` values overwriting any duplicate top-level keys. This is correct for BLIS because it models text generation latency, not vision processing. For example, if both top-level and `text_config` define `torch_dtype`, the text_config value wins.
- **Quantization preservation:** `quantization_config` lives at the top level (not inside `text_config` for Llama 4 Scout and similar models), so it is preserved correctly through the pivot.
- **Double unmarshal:** Both `json.Valid()` and `isHFConfig` parse the same JSON independently. This is pre-existing behavior (not introduced by this PR). The validation uses simple key existence checks, while parsing performs full extraction. The semantic gap (validation accepts if fields exist; parsing extracts values) is intentional and correct.
- **Vision encoder weights:** Not accounted for in KV capacity estimation (`computeModelWeightBytes` only uses text_config parameters). This is a known limitation for multimodal models—GPU memory consumed by vision encoder is not modeled. Acceptable for this PR scope (parser fix); may warrant future work.
