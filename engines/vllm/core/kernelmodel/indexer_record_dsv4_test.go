package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// openDSV4Indexer opens the committed DeepSeek-V4-Pro B200 fixture with
// attention_config.indexer_kv_dtype dtype ("" leaves it unstated).
func openDSV4Indexer(t *testing.T, dtype string) (*Model, error) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "kerneltests", "testdata", "aisimulate", "deepseek-v4-pro-b200-fp4-vllm-tp8.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	if dtype != "" {
		body = strings.Replace(body, "      block_size:", "      attention_config: {indexer_kv_dtype: "+dtype+"}\n      block_size:", 1)
		if !strings.Contains(body, "indexer_kv_dtype") {
			t.Fatal("the fixture has no block_size line to put the key before")
		}
	}
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return Open("v.yaml", r)
}

// indexRecords counts the KV layers of each key record the indexer's cache is registered with: a
// key-only MLA spec of one head, one state per compress_ratio tokens.
func indexRecords(t *testing.T, m *Model) map[int]int {
	t.Helper()
	runs, err := m.KVLayers()
	if err != nil {
		t.Fatal(err)
	}
	got := map[int]int{}
	for _, r := range runs {
		if r.Spec == "mla" && r.NumKVHeads == 1 && r.TokensPerState > 1 && (r.HeadSize == 132 || r.HeadSize == 68) {
			got[r.HeadSize] += r.Count
		}
	}
	return got
}

// Under indexer_kv_dtype mxfp4, DeepSeek-V4's indexer stores two key values to the byte and one
// UE8M0 scale per 32 values, 68 bytes a key against FP8's 132
// (vllm/models/deepseek_v4/attention.py:1020-1029@4ba787ec). On a Blackwell part the deployment
// opens, every indexer layer registers the 68-byte record, and a long-context decode, whose
// indexer scan reads every compressed key, is cheaper than at the FP8 default.
func TestDeepSeekV4MXFP4IndexerKeysAre68Bytes(t *testing.T) {
	fp8, err := openDSV4Indexer(t, "")
	if err != nil {
		t.Fatal(err)
	}
	base := indexRecords(t, fp8)
	if base[132] == 0 || base[68] != 0 {
		t.Fatalf("unstated: indexer records %v, want FP8's 132 only", base)
	}
	m, err := openDSV4Indexer(t, "mxfp4")
	if err != nil {
		t.Fatalf("indexer_kv_dtype mxfp4 on b200: %v", err)
	}
	if got := indexRecords(t, m); got[68] != base[132] || got[132] != 0 {
		t.Errorf("mxfp4: indexer records %v, want %d layers of 68 bytes and none of 132", got, base[132])
	}
	long := []ReqShape{{Scheduled: 1, Computed: 60_000, PromptLen: 50_000}}
	if a, b := m.StepTimeOfShapes(long), fp8.StepTimeOfShapes(long); a >= b {
		t.Errorf("mxfp4 prices a 60k-token decode at %d us, fp8 at %d us: the smaller keys must read less", a, b)
	}
}

// bf16 is not a DeepSeek indexer dtype in vLLM v0.33.0 (DSA_INDEXER_KV_DTYPES is fp8 and mxfp4,
// vllm/v1/attention/backends/mla/indexer.py:57-67@4ba787ec), so it is still refused.
func TestDeepSeekV4BF16IndexerIsStillRefused(t *testing.T) {
	m, err := openDSV4Indexer(t, "bf16")
	if err == nil {
		_, err = m.KVLayers()
	}
	if err == nil || !strings.Contains(err.Error(), "is not supported by the DeepSeek sparse indexer") {
		t.Errorf("indexer_kv_dtype bf16: %v, want vLLM's refusal", err)
	}
}
