package kernelmodel

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
	"github.com/inference-sim/inference-sim/engines/vllm/internal/testgz"
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

func recordedCapabilityOf() func(string) (string, error) {
	data := repos().Data
	return func(name string) (string, error) {
		chip, err := data.Chip(name)
		if err != nil {
			return "", err
		}
		return chip.ComputeCapability, nil
	}
}

type recordedRuns struct {
	Unobtainable []struct {
		Row    int    `json:"row"`
		Reason string `json:"reason"`
	} `json:"unobtainable"`
	Runs []recordedRun `json:"runs"`
}

func readRecordedRuns(t *testing.T) recordedRuns {
	t.Helper()
	raw, err := testgz.ReadFile("testdata/recorded_block_sizes/recorded_block_sizes.json")
	if err != nil {
		t.Fatal(err)
	}
	var f recordedRuns
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// recordedRunInput is the catalog model and the resolver input a recorded run's log determines: the
// tensor-parallel width and KV cache dtype of its engine configuration, the chip, and whether the
// run speculates with a method that builds draft layers. The rows do not record num_spec_tokens, so
// one token stands for any positive count.
func recordedRunInput(t *testing.T, capabilityOf func(chip string) (string, error), r recordedRun) (string, vllm.BlockSizeInput) {
	t.Helper()
	chipOf := map[string]string{"gb200": "gb200-nvl72"}
	name := r.Hardware
	if c, ok := chipOf[name]; ok {
		name = c
	}
	capability, err := capabilityOf(name)
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
	return model, vllm.BlockSizeInput{
		Model: model, Capability: capability, TP: r.LogTP, KVCacheDType: r.CacheDType,
		Speculative: vllm.SpeculationAddsAttentionLayers(method, 1),
	}
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
	f := readRecordedRuns(t)
	capabilityOf := recordedCapabilityOf()
	byClass := map[string]*tally{"engine_line": {}, "silent": {}, "args": {}}
	byModel := map[string]map[string]*tally{}
	for _, r := range f.Runs {
		model, in := recordedRunInput(t, capabilityOf, r)
		bs, err := vllm.ResolveBlockSizeAsRecorded(in)
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
			if !strings.Contains(err.Error(), "inference-sim/issues/2425") {
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
	// Coverage by model, in the test log and in the inference-sim/inference-sim#2425 comment.
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
		{"args", tally{5, 51, 7}},
	} {
		if got := *byClass[e.class]; got != e.want {
			t.Errorf("%s: %s; want %s", e.class, got, e.want)
		}
	}
	if len(f.Unobtainable) != 44 || len(f.Runs) != 108 {
		t.Errorf("%d runs and %d unobtainable rows; want 108 and 44", len(f.Runs), len(f.Unobtainable))
	}
}

// TestRecordedRunsAtThePin is the same 108 recorded runs resolved by the pin's table. The logs come
// from vLLM 0.18 to 0.22, whose default block size is 16 where the pin's is 64 (commit 732e49d44), so
// the pin cannot equal a logged 16 or 32 by construction; TestResolutionAgainstTheRecordedRuns holds the
// oracle to the logs with the default they were produced under. What the pin's table must satisfy is
// the law relating the two: for each run the pin resolves, the as-recorded table either refuses
// (a case the default change newly decides) or gives a size the pin's is a multiple of, and the pin's
// size equals the logged one where the logged size does not depend on the default (the 256 of
// DeepSeek-V4, which its backends state as their smallest accepted size). Every refusal cites #68.
// Counts are of all 108 runs: resolved at the pin and equal to the log, resolved and different because
// the default changed (the 52 include the 7 operator-flag rows), refused (minimax-m3 and the speculative
// DeepSeek-V4 runs, as recorded).
func TestRecordedRunsAtThePin(t *testing.T) {
	f := readRecordedRuns(t)
	capabilityOf := recordedCapabilityOf()
	var equal, defaultChanged, refused int
	for _, r := range f.Runs {
		model, in := recordedRunInput(t, capabilityOf, r)
		pin, perr := vllm.ResolveBlockSize(in)
		rec, rerr := vllm.ResolveBlockSizeAsRecorded(in)
		if perr != nil {
			refused++
			if !strings.Contains(perr.Error(), "inference-sim/issues/2425") {
				t.Errorf("row %d: refused at the pin, but not citing #68: %v", r.Row, perr)
			}
			if rerr == nil {
				t.Errorf("row %d (%s %s): the as-recorded table resolves %d and the pin refuses; the default change cannot remove a resolution", r.Row, model, r.Hardware, rec.Size)
			}
			continue
		}
		if rerr == nil && (pin.Size < rec.Size || pin.Size%rec.Size != 0) {
			t.Errorf("row %d (%s %s): the pin resolves %d and the as-recorded table %d, want a multiple", r.Row, model, r.Hardware, pin.Size, rec.Size)
		}
		if pin.Size == r.BlockSize {
			equal++
			continue
		}
		defaultChanged++
		if rerr == nil && rec.Size == pin.Size {
			t.Errorf("row %d (%s %s): the log shows %d and both tables resolve %d: the difference is not the default change", r.Row, model, r.Hardware, r.BlockSize, pin.Size)
		}
	}
	t.Logf("%d runs at the pin: equal to the log %d, different because the default changed %d, refused %d", len(f.Runs), equal, defaultChanged, refused)
	if equal != 5 || defaultChanged != 52 || refused != 51 {
		t.Errorf("equal %d, default changed %d, refused %d; want 5, 52, 51", equal, defaultChanged, refused)
	}
}

// Provenance of the recorded rows: each carries the sha256 and byte count of the decoded UTF-8
// serverLog string the API returned (the exact bytes fetch.py classifies), the date it was fetched,
// and the file carries the note fetch.py writes, so re-running fetch.py --from-rows reproduces it.
// Rows 411031 (args) and 420106 (engine_line) have digests known from the API, stable across
// repeat fetches; they are checked here offline when the file holds them, and `fetch.py --self-test`
// checks them against the network.
func TestRecordedRowsCarryTheirProvenance(t *testing.T) {
	raw, err := testgz.ReadFile("testdata/recorded_block_sizes/recorded_block_sizes.json")
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
