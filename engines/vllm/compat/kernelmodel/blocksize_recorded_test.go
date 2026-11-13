package kernelmodel

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/vllm"
)

// recordedRun is one InferenceX benchmark row with the block size its run used
// (testdata/recorded_block_sizes, written by fetch.py from the public API's server log of each row).
type recordedRun struct {
	Row               int    `json:"row"`
	Model             string `json:"model"`
	Hardware          string `json:"hardware"`
	LogTP             int    `json:"log_tp"`
	SpeculativeMethod string `json:"speculative_method"`
	CacheDType        string `json:"cache_dtype"`
	BlockSize         int    `json:"block_size"`
	EngineVersion     string `json:"engine_version"`
	// Evidence says where the run got its block size:
	//   engine_line  the log has "Setting kv cache block size to N": vLLM picked N itself;
	//   silent       vLLM v0.18.0 or later logged neither that line nor a block_size argument, so it
	//                kept the default 16 (vllm/platforms/interface.py:757-781@5df63a72 logs the line
	//                only for another size). A log cannot tell that from an explicit --block-size 16;
	//   args         the log's non-default arguments state block_size N: an operator's flag, which
	//                says nothing about what vLLM picks when none is passed.
	Evidence string `json:"evidence"`
}

// catalogModel maps InferenceX's model slug to the catalog model the table is keyed by. llama70b is
// Llama-3.3-70B-Instruct-FP8 in InferenceX and has no run with a server log, so it is not mapped.
var catalogModel = map[string]string{
	"gptoss120b": "gpt-oss-120b", "minimaxm2.5": "minimax-m2.5", "kimik2.5": "kimi-k2.5",
	"minimaxm3": "minimax-m3", "dsv4": "deepseek-v4-pro", "dsv41flash": "deepseek-v4.1-flash",
}

// operatorFlag32 is why the minimax-m2.5 Blackwell rows that passed --block-size 32 differ from the
// resolution: vLLM picks a size only when none is passed (vllm/platforms/interface.py:757-781@5df63a72),
// and the table's pick for minimax-m2.5 on compute capability 10.x is 16.
const operatorFlag32 = "the run passed --block-size 32 (its log's non-default arguments state it); vLLM's own pick on Blackwell is 16"

// operatorFlagDiffers are the recorded rows whose resolved size differs from the size the run used,
// keyed by InferenceX row id. Only a row whose evidence is "args" may be listed: a size vLLM chose
// that the resolution does not reproduce is a finding about the table and fails the test.
var operatorFlagDiffers = map[int]string{
	194852: operatorFlag32, 194862: operatorFlag32, 411043: operatorFlag32, 411048: operatorFlag32,
	411049: operatorFlag32, 417629: operatorFlag32, 417631: operatorFlag32,
}

type tally struct{ reproduced, refused, differs int }

func (a tally) String() string {
	return fmt.Sprintf("reproduced %d, refused %d, differs %d", a.reproduced, a.refused, a.differs)
}

// TestResolutionAgainstTheRecordedRuns resolves each recorded run's block size as the engine would
// had the run stated none, and compares. The classes are kept apart because only some can validate
// the resolution:
//
//	engine_line  vLLM logged the size it picked: the resolution must reproduce it where it resolves;
//	silent       vLLM kept the default 16: the resolution must give 16 where it resolves;
//	args         an operator's flag: a different resolved size is not a disagreement with vLLM, and
//	             each is named by row in operatorFlagDiffers.
//
// A refusal cites #68. A difference in an engine_line or silent row is not allow-listable: it is a
// finding about the table, and the test fails. The inputs are what a log records: the tensor-parallel
// width and KV cache dtype of its engine configuration, the chip, and whether the run speculates
// with a method that builds draft layers; the rows do not record num_spec_tokens, so one token
// stands for any positive count. The coverage by model is logged.
func TestResolutionAgainstTheRecordedRuns(t *testing.T) {
	raw, err := os.ReadFile("testdata/recorded_block_sizes/recorded_block_sizes.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Unobtainable []struct {
			Row    int    `json:"row"`
			Reason string `json:"reason"`
		} `json:"unobtainable"`
		Runs []recordedRun `json:"runs"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	data := repos().Data
	chipOf := map[string]string{"gb200": "gb200-nvl72"}
	byClass := map[string]*tally{"engine_line": {}, "silent": {}, "args": {}}
	byModel := map[string]map[string]*tally{}
	for _, r := range f.Runs {
		name := r.Hardware
		if c, ok := chipOf[name]; ok {
			name = c
		}
		chip, err := data.Chip(name)
		if err != nil {
			t.Fatalf("row %d: %v", r.Row, err)
		}
		model, ok := catalogModel[r.Model]
		if !ok {
			t.Fatalf("row %d: no catalog model for slug %q", r.Row, r.Model)
		}
		method := r.SpeculativeMethod
		if method == "none" {
			method = ""
		}
		bs, err := vllm.ResolveBlockSize(vllm.BlockSizeInput{
			Model: model, Capability: chip.ComputeCapability, TP: r.LogTP, KVCacheDType: r.CacheDType,
			Speculative: vllm.SpeculationAddsAttentionLayers(method, 1),
		})
		c := byClass[r.Evidence]
		if c == nil {
			t.Fatalf("row %d: evidence %q", r.Row, r.Evidence)
		}
		m := byModel[model]
		if m == nil {
			m = map[string]*tally{"engine_line": {}, "silent": {}, "args": {}}
			byModel[model] = m
		}
		switch {
		case err != nil:
			if !strings.Contains(err.Error(), "blis-vllm/issues/68") {
				t.Errorf("row %d: refused, but not citing #68: %v", r.Row, err)
			}
			c.refused++
			m[r.Evidence].refused++
		case bs.Size == r.BlockSize:
			c.reproduced++
			m[r.Evidence].reproduced++
		default:
			c.differs++
			m[r.Evidence].differs++
			why, listed := operatorFlagDiffers[r.Row]
			switch {
			case r.Evidence != "args":
				t.Errorf("InferenceX row %d (%s %s tp%d kv %s, vLLM %s): the log shows %s %d, and the table resolves %d: "+
					"a finding about the table, not an operator's flag", r.Row, model, r.Hardware, r.LogTP, r.CacheDType, r.EngineVersion, r.Evidence, r.BlockSize, bs.Size)
			case !listed:
				t.Errorf("InferenceX row %d (%s %s tp%d): the run passed --block-size %d and the table resolves %d, and no entry of operatorFlagDiffers names the row",
					r.Row, model, r.Hardware, r.LogTP, r.BlockSize, bs.Size)
			default:
				_ = why
			}
		}
	}
	for row := range operatorFlagDiffers {
		found := false
		for _, r := range f.Runs {
			found = found || (r.Row == row && r.Evidence == "args")
		}
		if !found {
			t.Errorf("operatorFlagDiffers names row %d, which is not an args row of the recorded runs", row)
		}
	}
	// Coverage by model, in the test log and in the blis-vllm#68 comment.
	models := make([]string, 0, len(byModel))
	for k := range byModel {
		models = append(models, k)
	}
	sort.Strings(models)
	for _, k := range models {
		t.Logf("%-20s engine_line: %s | silent: %s | args: %s", k, byModel[k]["engine_line"], byModel[k]["silent"], byModel[k]["args"])
	}
	for _, e := range []struct {
		class string
		want  tally
	}{
		{"engine_line", tally{14, 0, 0}},
		{"silent", tally{31, 0, 0}},
		{"args", tally{0, 56, 7}},
	} {
		if got := *byClass[e.class]; got != e.want {
			t.Errorf("%s: %s; want %s", e.class, got, e.want)
		}
	}
	if len(f.Unobtainable) != 44 || len(f.Runs) != 108 {
		t.Errorf("%d runs and %d unobtainable rows; want 108 and 44", len(f.Runs), len(f.Unobtainable))
	}
}

// Provenance of the recorded rows: each carries the sha256 and byte count of the decoded UTF-8
// serverLog string the API returned (the exact bytes fetch.py classifies), the date it was fetched,
// and the file carries the note fetch.py writes, so re-running fetch.py --from-rows reproduces it.
// Rows 411031 (args) and 420106 (engine_line) have digests known from the API, stable across
// repeat fetches; they are checked here offline when the file holds them, and `fetch.py --self-test`
// checks them against the network.
func TestRecordedRowsCarryTheirProvenance(t *testing.T) {
	raw, err := os.ReadFile("testdata/recorded_block_sizes/recorded_block_sizes.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Note string `json:"note"`
		Runs []struct {
			Row       int    `json:"row"`
			Evidence  string `json:"evidence"`
			LogSHA256 string `json:"log_sha256"`
			LogBytes  int    `json:"log_bytes"`
			Fetched   string `json:"fetched"`
		} `json:"runs"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(f.Note, "fetch.py writes this file") {
		t.Errorf("the file's note %q is not the one fetch.py writes", f.Note)
	}
	known := map[int]struct {
		evidence, sha string
		bytes         int
	}{
		411031: {"args", "388dfb966c896b83cf47956614a78007fa0941675892173b6a77e330c65e900e", 126004},
		420106: {"engine_line", "7718bd95e60c95bc4279a1cc6760b32e0ab7413060b7a706cd184ef1f09a88cd", 100832},
	}
	for _, r := range f.Runs {
		if len(r.LogSHA256) != 64 || r.LogBytes <= 0 || len(r.Fetched) != 10 {
			t.Errorf("row %d: log_sha256 %q, log_bytes %d, fetched %q: provenance is incomplete", r.Row, r.LogSHA256, r.LogBytes, r.Fetched)
		}
		if k, ok := known[r.Row]; ok && (r.Evidence != k.evidence || r.LogSHA256 != k.sha || r.LogBytes != k.bytes) {
			t.Errorf("row %d: %s %s %d, want %s %s %d", r.Row, r.Evidence, r.LogSHA256, r.LogBytes, k.evidence, k.sha, k.bytes)
		}
	}
}
