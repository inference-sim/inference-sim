// Original to blis-sglang.

package rules

import (
	"fmt"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/enginefacts"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"gopkg.in/yaml.v3"
)

func block(t *testing.T, s string) deployment.EngineBlock {
	t.Helper()
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(s), &n); err != nil {
		t.Fatal(err)
	}
	return deployment.NewEngineBlock(n.Content[0])
}

// Only the release this build models has rules; a scenario naming another is refused by the
// kernel's opener because Lookup answers nil.
func TestLookupAnswersForTheModelledReleaseOnly(t *testing.T) {
	if Lookup("0.5.23") == nil || Lookup("0.5.22") != nil || Lookup("") != nil {
		t.Errorf("Lookup answers for 0.5.23 only; got %v %v %v", Lookup("0.5.23"), Lookup("0.5.22"), Lookup(""))
	}
	if got := V0_5_23.EngineVersion(); got != "0.5.23" {
		t.Errorf("EngineVersion %q", got)
	}
}

// An engine block key that is not one of ServerArgs' fields is a finding that names it (X1): a
// vLLM name does not pass for SGLang's.
func TestEngineBlockKeysAreServerArgsFields(t *testing.T) {
	ok := V0_5_23.CheckEngineBlock("pools[0]", block(t, "mem_fraction_static: 0.8\nmax_running_requests: 64\n"))
	if !ok.OK() {
		t.Errorf("SGLang's own names were refused: %v", ok)
	}
	bad := V0_5_23.CheckEngineBlock("pools[0]", block(t, "mem_fraction_static: 0.8\nmax_num_seqs: 64\n"))
	if len(bad.Errors()) != 1 || !strings.Contains(bad.Error(), "max_num_seqs") || !strings.Contains(bad.Error(), "pools[0].engine.max_num_seqs") {
		t.Errorf("want one finding at pools[0].engine.max_num_seqs, got %v", bad)
	}
}

func TestDraftTokensAreTheVerifiedTokens(t *testing.T) {
	if n := V0_5_23.DraftTokens(block(t, "speculative_algorithm: EAGLE\nspeculative_num_draft_tokens: 4\n")); n != 4 {
		t.Errorf("EAGLE with 4 draft tokens: %d", n)
	}
	if n := V0_5_23.DraftTokens(block(t, "speculative_num_draft_tokens: 4\n")); n != 0 {
		t.Errorf("draft tokens without an algorithm drafts nothing; got %d", n)
	}
	if n := V0_5_23.DraftTokens(deployment.EngineBlock{}); n != 0 {
		t.Errorf("no block drafts %d", n)
	}
}

func TestCustomAllReduceWidths(t *testing.T) {
	for tp, want := range map[int]bool{1: false, 2: true, 3: false, 4: true, 6: true, 8: true, 16: false} {
		if got := V0_5_23.CustomAllReduceSupportsWidth(tp); got != want {
			t.Errorf("tp %d: %v, want %v", tp, got, want)
		}
	}
}

// DeepSeek-R1's KV is one MLA latent of 576 elements a layer: 61 layers x 576 x 2 bytes is 70,272
// bytes a token, and a page of 64 tokens is 64 times that. Every layer shares one block table.
func TestKVCacheGroupsOfAnMLAModel(t *testing.T) {
	var layers []enginefacts.KVCacheLayer
	for i := 0; i < 61; i++ {
		layers = append(layers, enginefacts.KVCacheLayer{Name: fmt.Sprintf("layers.%d.kv", i),
			Cache: model.Cache{Name: "kv", Kind: model.CacheKV, Owned: true, BytesPerEntry: 576 * 2}})
	}
	for _, page := range []int{1, 64} {
		got, err := V0_5_23.KVCacheGroups(enginefacts.KVCacheGroupsInput{Layers: layers, EngineBlockSize: page})
		if err != nil {
			t.Fatal(err)
		}
		if len(got.Groups) != 1 || got.Packed || got.BytesPerBlock != 70272*page {
			t.Errorf("page %d: %d groups packed=%v bytes %d; want one group of 70272 x page", page, len(got.Groups), got.Packed, got.BytesPerBlock)
		}
	}
	if _, err := V0_5_23.KVCacheGroups(enginefacts.KVCacheGroupsInput{EngineBlockSize: 1}); err == nil {
		t.Error("no caches were grouped")
	}
	win := []enginefacts.KVCacheLayer{{Name: "layers.0.kv", Cache: model.Cache{Kind: model.CacheWindow, Window: 128}}}
	if _, err := V0_5_23.KVCacheGroups(enginefacts.KVCacheGroupsInput{Layers: win, EngineBlockSize: 1}); err == nil || !strings.Contains(err.Error(), "#94") {
		t.Errorf("a window cache is a hybrid pool, refused naming its issue: %v", err)
	}
}

func TestCacheRecords(t *testing.T) {
	for in, want := range map[string]string{"": "", "auto": "", "bfloat16": "", "fp8_e4m3": "fp8", "fp8_e5m2": "fp8"} {
		if got := V0_5_23.ResolveCacheDType("", 9, 8, in); got != want {
			t.Errorf("%q -> %q, want %q", in, got, want)
		}
	}
}
