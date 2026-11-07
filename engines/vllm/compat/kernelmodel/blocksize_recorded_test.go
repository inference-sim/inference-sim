package kernelmodel

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/vllm"
)

// recordedRun is one InferenceX benchmark row with the block size its run used
// (testdata/recorded_block_sizes, written by fetch.py from the public API's server log of each row).
type recordedRun struct {
	Row               int    `json:"row"`
	Model             string `json:"model"`
	GPU               string `json:"gpu"`
	TP                int    `json:"tp"`
	SpeculativeMethod string `json:"speculative_method"`
	CacheDType        string `json:"cache_dtype"`
	BlockSize         int    `json:"block_size"`
	// Evidence is "engine_line" when the engine logged "Setting kv cache block size to N" for
	// the run, which is vLLM's own pick, or "args" when the run passed --block-size N, an
	// operator's flag that says nothing about what vLLM picks when none is passed.
	Evidence string `json:"evidence"`
}

// reason is why the eleven minimax-m2.5 Blackwell runs differ. Each passed --block-size 32 (an
// operator's flag); vLLM's own pick for minimax-m2.5 on compute capability 10.x is 16, the size
// every backend accepts (the FlashInfer backend, oracle/blocksize/TABLE.tsv), because it picks a
// size only when none is passed (vllm/platforms/interface.py:757-781@5df63a72).
const reason = "the run passed --block-size 32; vLLM's own pick on Blackwell is 16"

// disagreements are the recorded runs whose resolved size differs from the recorded one, each
// with why. Keyed by run name; a run not listed here that disagrees fails the test.
var disagreements = map[int]string{
	411031: reason,
	411032: reason,
	411033: reason,
	411034: reason,
	411035: reason,
	411039: reason,
	417607: reason,
	417608: reason,
	417610: reason,
	417612: reason,
	417613: reason,
}

// TestResolutionAgainstTheRecordedRuns resolves each recorded run's block size as the engine
// would had the run stated none, and compares. Two classes of run are kept apart because only one
// can validate the resolution:
//
//	engine_line   the engine logged the size it chose: the resolution must reproduce it where it
//	              resolves;
//	args          the run passed --block-size: an operator's flag, so a different resolved size is
//	              not a disagreement with vLLM. The flag is checked for consistency only and each
//	              difference is named.
//
// Two InferenceX rows have no server log in the public API (fetch.py lists them under
// "unobtainable"), and thirteen more runs of deepseek-v4.1-flash with an mtp drafter, whose
// engine line another source recorded, have no row the public API lists today, so none of the
// fifteen is in the file. Those are the only recorded engine-logged sizes missing here, and the
// resolution refuses that model with a speculative drafter, citing #68, so they could not be
// reproduced by it in any case.
//
// The inputs are what a run records: model, GPU, tensor-parallel width, KV dtype and whether it
// speculates with a method that builds draft layers. The snapshots do not record num_spec_tokens,
// so one speculative token stands for any positive count.
func run(r recordedRun) string {
	return fmt.Sprintf("InferenceX row %d (%s %s tp%d)", r.Row, r.Model, r.GPU, r.TP)
}

func TestResolutionAgainstTheRecordedRuns(t *testing.T) {
	raw, err := os.ReadFile("testdata/recorded_block_sizes/recorded_block_sizes.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Unobtainable []int         `json:"unobtainable"`
		Runs         []recordedRun `json:"runs"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Unobtainable) != 2 {
		t.Errorf("the file lists %d unobtainable rows, want 2", len(f.Unobtainable))
	}
	data := repos().Data
	chipOf := map[string]string{"gb200": "gb200-nvl72"}
	type class struct{ reproduced, refused, disagrees int }
	got := map[string]*class{"engine_line": {}, "args": {}}
	for _, r := range f.Runs {
		name := r.GPU
		if c, ok := chipOf[name]; ok {
			name = c
		}
		chip, err := data.Chip(name)
		if err != nil {
			t.Fatalf("%s: %v", run(r), err)
		}
		method := r.SpeculativeMethod
		if method == "none" {
			method = ""
		}
		bs, err := vllm.ResolveBlockSize(vllm.BlockSizeInput{
			Model: r.Model, Capability: chip.ComputeCapability, TP: r.TP, KVCacheDType: r.CacheDType,
			Speculative: vllm.SpeculationAddsAttentionLayers(method, 1),
		})
		c := got[r.Evidence]
		switch {
		case err != nil:
			if !strings.Contains(err.Error(), "blis-vllm/issues/68") {
				t.Errorf("%s: refused, but not citing #68: %v", run(r), err)
			}
			c.refused++
		case bs.Size == r.BlockSize:
			c.reproduced++
		default:
			c.disagrees++
			why, ok := disagreements[r.Row]
			if !ok {
				t.Errorf("%s (%s): resolves %d, the run used %d, and no allow-list entry explains it", run(r), r.Evidence, bs.Size, r.BlockSize)
			} else if r.Evidence == "engine_line" {
				t.Errorf("%s: the engine logged %d and the resolution gives %d; %s", run(r), r.BlockSize, bs.Size, why)
			}
		}
	}
	for _, e := range []struct {
		evidence                      string
		reproduced, refused, disagree int
	}{
		{"engine_line", 10, 0, 0},
		{"args", 0, 79, 11},
	} {
		c := got[e.evidence]
		t.Logf("%s: reproduced %d, refused %d, disagrees %d", e.evidence, c.reproduced, c.refused, c.disagrees)
		if c.reproduced != e.reproduced || c.refused != e.refused || c.disagrees != e.disagree {
			t.Errorf("%s: reproduced %d, refused %d, disagrees %d; want %d, %d, %d",
				e.evidence, c.reproduced, c.refused, c.disagrees, e.reproduced, e.refused, e.disagree)
		}
	}
	for row := range disagreements {
		found := false
		for _, r := range f.Runs {
			found = found || r.Row == row
		}
		if !found {
			t.Errorf("the allow-list names row %d, which is not a recorded row", row)
		}
	}
}
