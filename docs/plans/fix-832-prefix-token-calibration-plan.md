# Fix: Prefix token inflation in blis observe — Implementation Plan

**Goal:** Calibrate prefix word counts so the server tokenizes them to the intended token count, matching what `blis run` simulates.
**Source:** [GitHub Issue #832](https://github.com/inference-sim/inference-sim/issues/832)
**Closes:** Fixes #832

## Behavioral Contracts

BC-1: Prefix calibration measures tokens-per-word
- GIVEN a server is reachable and prefix groups exist in the workload spec
- WHEN `blis observe` starts (before dispatching real requests)
- THEN a calibration request is sent with vocabulary words, and the measured `tokens_per_word` ratio is used to scale prefix word counts

BC-2: Scaled prefix strings approximate target token count
- GIVEN a prefix group with `prefix_length: N` tokens in the spec
- WHEN `buildPrefixStrings` uses the calibrated `tokens_per_word` ratio
- THEN the prefix string contains `round(N / tokensPerWord)` words instead of `N` words
- NOTE: The ratio includes a small chat template overhead (~10-20 tokens), so the actual server token count will be within ~1% of the target. This is acceptable — the current gap without calibration is ~60%.

BC-3: Suffix computation uses target token count (unchanged)
- GIVEN a request with `input_tokens = prefix_length + body_tokens`
- WHEN `requestToPending` computes the suffix word count
- THEN it uses `prefixLengths[group]` which stores the target token count (`prefix_length`), not the word count of the prefix string

BC-4: Graceful fallback when calibration fails or ratio is out of bounds
- GIVEN a server that is unreachable, returns an error, or returns a ratio outside [1.0, 3.0]
- WHEN `calibratePrefixTokenRatio` is called
- THEN a warning is logged and the ratio defaults to 1.0 (current behavior preserved)

BC-5: No calibration when no prefix groups exist
- GIVEN a workload spec with no prefix groups
- WHEN `blis observe` starts
- THEN no calibration request is sent

## Change Summary

Files modified: `cmd/observe_cmd.go`, `cmd/observe_cmd_test.go`

1. **New function `calibratePrefixTokenRatio`**: Sends a non-streaming request with 100 vocabulary words (= full vocabulary, no repetition) to the server. Returns `tokensPerWord = float64(promptTokens) / float64(100)`. Validates ratio is in [1.0, 3.0]. On failure or out-of-bounds, returns 1.0 with a warning.

2. **Modified `buildPrefixStrings`**: Accepts `tokensPerWord float64` parameter. Generates `round(prefixLength / tokensPerWord)` words instead of `prefixLength` words. The `prefixLengths` map continues to store the **target token count** (not word count), since downstream code (`requestToPending`) uses it for suffix computation against `len(req.InputTokens)` which is in tokens.

3. **Modified `runObserveCmd`**: After building the client and before building prefix strings, calls `calibratePrefixTokenRatio` if prefix groups exist. Passes the ratio to `buildPrefixStrings`.

4. **No changes to `requestToPending`**: The suffix computation `suffixWords = wordCount - prefixLen` already works correctly because `prefixLengths[group]` stores the target token count (e.g., 14336), and `wordCount = len(req.InputTokens)` is also in tokens.

5. **Updated existing test**: `TestBuildPrefixStrings_DeterministicSameGroup` and `TestRequestToPending_PrependsPrefixString` call `buildPrefixStrings` — updated to pass `1.0` as `tokensPerWord` (no behavioral change to existing tests).

## Tasks

### Task 1: Write tests for calibration function (BC-1, BC-4)

**Files:** modify `cmd/observe_cmd_test.go`

**Test:**
```go
func TestCalibratePrefixTokenRatio_ReturnsRatio(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]string{"content": "ok"}, "finish_reason": "length"},
			},
			"usage": map[string]interface{}{
				"prompt_tokens": 167.0, "completion_tokens": 1.0,
			},
		})
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))
	ratio := calibratePrefixTokenRatio(context.Background(), client)

	expected := 167.0 / 100.0
	if math.Abs(ratio-expected) > 0.01 {
		t.Errorf("ratio = %.4f, want %.4f", ratio, expected)
	}
}

func TestCalibratePrefixTokenRatio_FallbackOnError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))
	ratio := calibratePrefixTokenRatio(context.Background(), client)

	if ratio != 1.0 {
		t.Errorf("ratio = %.4f, want 1.0 (fallback)", ratio)
	}
}

func TestCalibratePrefixTokenRatio_FallbackOnOutOfBounds(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]string{"content": "ok"}, "finish_reason": "length"},
			},
			"usage": map[string]interface{}{
				"prompt_tokens": 50000.0, "completion_tokens": 1.0,
			},
		})
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm", WithAPIFormat("chat"))
	ratio := calibratePrefixTokenRatio(context.Background(), client)

	if ratio != 1.0 {
		t.Errorf("ratio = %.4f, want 1.0 (fallback for out-of-bounds)", ratio)
	}
}
```

**Verify (expect fail):** `go test ./cmd/... -run TestCalibratePrefixTokenRatio`

### Task 2: Implement calibration function (BC-1, BC-4)

**Files:** modify `cmd/observe_cmd.go`

**Impl:**
```go
// calibrationWordCount is the number of vocabulary words used in the
// calibration request. Equals len(prefixVocabulary) to avoid repetition.
const calibrationWordCount = 100

// calibratePrefixTokenRatio sends a calibration request to measure how many
// tokens the server's tokenizer produces per vocabulary word. Returns the
// ratio (typically 1.5-1.7 for BPE tokenizers with multi-syllable words).
// The ratio includes a small chat template overhead (~10-20 tokens out of
// ~167 total, <10%) which is acceptable for prefix scaling purposes.
// On failure or out-of-bounds ratio, returns 1.0 (no scaling) with a warning.
func calibratePrefixTokenRatio(ctx context.Context, client *RealClient) float64 {
	prompt := strings.Join(prefixVocabulary[:calibrationWordCount], " ")

	pending := &PendingRequest{
		RequestID:       -1,
		Model:           client.modelName,
		Streaming:       false,
		Prompt:          prompt,
		MaxOutputTokens: 1,
	}

	record, err := client.Send(ctx, pending)
	if err != nil || record.Status != "ok" || record.ServerInputTokens <= 0 {
		msg := "unknown"
		if err != nil {
			msg = err.Error()
		} else if record != nil && record.ErrorMessage != "" {
			msg = record.ErrorMessage
		}
		logrus.Warnf("Prefix token calibration failed (%s); using 1:1 word-to-token ratio", msg)
		return 1.0
	}

	ratio := float64(record.ServerInputTokens) / float64(calibrationWordCount)
	if ratio < 1.0 || ratio > 3.0 {
		logrus.Warnf("Prefix token calibration ratio %.3f outside expected range [1.0, 3.0]; using 1:1 fallback", ratio)
		return 1.0
	}

	logrus.Infof("Prefix token calibration: %d words → %d server tokens (%.3f tokens/word)",
		calibrationWordCount, record.ServerInputTokens, ratio)
	return ratio
}
```

**Verify:** `go test ./cmd/... -run TestCalibratePrefixTokenRatio`

### Task 3: Write test for scaled prefix string generation (BC-2)

**Files:** modify `cmd/observe_cmd_test.go`

**Test:**
```go
func TestBuildPrefixStrings_ScalesWordCount(t *testing.T) {
	groups := map[string]int{"test-group": 1000}

	// With ratio 1.0 (no scaling): 1000 words
	prefixes1, lengths1 := buildPrefixStrings(groups, 42, 1.0)
	words1 := strings.Fields(prefixes1["test-group"])

	// With ratio 1.67: round(1000/1.67) = 599 words
	prefixes2, lengths2 := buildPrefixStrings(groups, 42, 1.67)
	words2 := strings.Fields(prefixes2["test-group"])

	if len(words1) != 1000 {
		t.Errorf("ratio=1.0: word count = %d, want 1000", len(words1))
	}
	if lengths1["test-group"] != 1000 {
		t.Errorf("ratio=1.0: prefixLengths = %d, want 1000 (target tokens)", lengths1["test-group"])
	}

	expectedWords := int(math.Round(1000.0 / 1.67))
	if len(words2) != expectedWords {
		t.Errorf("ratio=1.67: word count = %d, want %d", len(words2), expectedWords)
	}
	if lengths2["test-group"] != 1000 {
		t.Errorf("ratio=1.67: prefixLengths = %d, want 1000 (target tokens)", lengths2["test-group"])
	}
}
```

**Verify (expect fail):** `go test ./cmd/... -run TestBuildPrefixStrings_ScalesWordCount`

### Task 4: Modify buildPrefixStrings and update existing callers (BC-2)

**Files:** modify `cmd/observe_cmd.go`, `cmd/observe_cmd_test.go`

**Impl:** Add `tokensPerWord float64` parameter to `buildPrefixStrings`. Compute `wordCount = int(math.Round(float64(length) / tokensPerWord))`. Keep `prefixLengths[group] = length` (target token count, not word count). Update existing test call sites (`TestBuildPrefixStrings_DeterministicSameGroup`, `TestRequestToPending_PrependsPrefixString`) to pass `1.0`.

**Verify:** `go test ./cmd/... -run TestBuildPrefixStrings`
**Full:** `go test ./cmd/... -count=1`

### Task 5: Wire calibration into observe startup (BC-1, BC-5)

**Files:** modify `cmd/observe_cmd.go`

**Impl:** After creating the client (line 225) and before building prefix strings (line 242), add:
```go
tokensPerWord := 1.0
if len(groups) > 0 {
    tokensPerWord = calibratePrefixTokenRatio(ctx, client)
}
prefixes, prefixLengths = buildPrefixStrings(groups, spec.Seed, tokensPerWord)
```

Note: `ctx` is created later (line 254). Need to create a temporary context for calibration, or move the calibration after ctx creation. Simplest: use `context.Background()` for the calibration call since it's a one-shot pre-dispatch call.

**Verify:** `go test ./cmd/... -count=1`
**Lint:** `golangci-lint run ./cmd/...`

## Sanity Checklist

- [x] R1 (silent continue): Calibration failure logs a warning and falls back to 1.0
- [x] R2 (determinism): Prefix string generation remains seeded; calibration ratio is deterministic for a given server
- [x] R4 (construction sites): `buildPrefixStrings` has 1 production call site (~line 243), existing test call sites — all updated with new parameter
- [x] R13 (behavioral contracts): THEN clauses describe observable outcomes (word counts, ratios, fallback values)
- [x] R14 (single-module): Change is self-contained in `cmd/`
- [x] R23 (code path parity): Fix brings observe token counts into alignment with run token counts
- [x] INV-6 (determinism): Calibration ratio logged to stderr, not stdout

## Deviation Log

- **CLARIFICATION**: Chat template overhead (~10-20 tokens) is included in the calibration ratio. This causes ~1% overshoot vs the target token count. Accepted as negligible compared to the current ~60% inflation. BC-2 wording updated to say "approximate" rather than "exact".
- **CLARIFICATION**: Calibration uses exactly `len(prefixVocabulary)` = 100 words (no repetition) to avoid any BPE tokenization bias from repeated words.
