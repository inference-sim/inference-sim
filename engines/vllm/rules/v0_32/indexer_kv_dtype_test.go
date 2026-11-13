// Original to blis-vllm.

package v0_32

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/rules"

	"github.com/inference-sim/blis-vllm/rules/pack"
	"github.com/inference-sim/blis-vllm/rules/v0_31"
)

type indexerCase struct {
	Requested string `json:"requested"`
	Family    int    `json:"family"`
	Resolved  string `json:"resolved"`
	WidthV4   int    `json:"width_v4"`
	WidthV41  int    `json:"width_v41"`
	Error     *struct {
		Class   string `json:"class"`
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// ResolveIndexerKVDType and IndexerRecordBytes give, for each requested dtype on Hopper (family 90)
// and Blackwell datacenter (100), what vLLM's own functions give: testdata/vllm_indexer_kv_dtype.json
// is the output of oracle/indexer/oracle.py, which runs AttentionConfig.resolve_indexer_kv_dtype,
// dsa_indexer_uses_fp4, DeepseekV4Indexer's record-width statement and V4.1's
// _indexer_k_cache_head_dim unmodified (CI regenerates it and compares). A ValueError's text is
// compared whole; the Literal check is pydantic's, whose message is compared.
func TestIndexerKVDTypeIsWhatVLLMsOwnFunctionsGive(t *testing.T) {
	raw, err := os.ReadFile("testdata/vllm_indexer_kv_dtype.json")
	if err != nil {
		t.Fatal(err)
	}
	var table struct {
		Cases        []indexerCase `json:"cases"`
		IndexHeadDim int           `json:"index_head_dim"`
	}
	if err := json.Unmarshal(raw, &table); err != nil {
		t.Fatal(err)
	}
	if len(table.Cases) != 12 {
		t.Fatalf("%d cases, want 6 dtypes on 2 families", len(table.Cases))
	}
	p := Pack()
	errors, ok := 0, 0
	for _, c := range table.Cases {
		for _, arch := range []string{rules.ArchDeepseekV4, rules.ArchDeepseekV41} {
			name := arch + "/" + c.Requested + "/" + map[int]string{90: "sm90", 100: "sm100"}[c.Family]
			t.Run(name, func(t *testing.T) {
				got, known, err := p.ResolveIndexerKVDType(arch, c.Family/10, c.Requested)
				if !known {
					t.Fatal("the pack has no rule for a DeepSeek architecture")
				}
				if c.Error != nil {
					errors++
					if err == nil {
						t.Fatalf("resolved to %q, vLLM raises %s: %s", got, c.Error.Class, c.Error.Message)
					}
					if c.Error.Class == "ValueError" && err.Error() != c.Error.Message {
						t.Errorf("error %q\nvLLM    %q", err, c.Error.Message)
					}
					if c.Error.Class == "ValidationError" && err.Error() != c.Error.Message {
						t.Errorf("error %q\npydantic %q", err, c.Error.Message)
					}
					return
				}
				ok++
				if err != nil || got != c.Resolved {
					t.Fatalf("resolved to %q (%v), vLLM resolves to %q", got, err, c.Resolved)
				}
				if w := pack.IndexerRecordBytes(got, table.IndexHeadDim); w != c.WidthV4 || w != c.WidthV41 {
					t.Errorf("record %d bytes, vLLM V4 %d and V4.1 %d", w, c.WidthV4, c.WidthV41)
				}
			})
		}
	}
	// Both outcomes occur, so a table that went all one way would not pass for evidence.
	if errors == 0 || ok == 0 {
		t.Errorf("the table has %d refusals and %d resolutions", errors, ok)
	}
}

// Other architectures, and v0.31.0's pack, carry no rule: the value is returned as stated and the
// caller decides.
func TestIndexerKVDTypeRuleIsDeepSeekAndV032Only(t *testing.T) {
	if got, known, err := Pack().ResolveIndexerKVDType("MiniMaxM3ForCausalLM", 10, "bf16"); known || err != nil || got != "bf16" {
		t.Errorf("another architecture: %q %v %v", got, known, err)
	}
	if got, known, err := v0_31.Pack().ResolveIndexerKVDType(rules.ArchDeepseekV4, 10, "fp8"); known || err != nil || got != "fp8" {
		t.Errorf("v0.31.0: %q %v %v", got, known, err)
	}
	if !strings.Contains(Pack().DSAIndexer.Default, "fp8") {
		t.Error("auto does not resolve to fp8")
	}
}
