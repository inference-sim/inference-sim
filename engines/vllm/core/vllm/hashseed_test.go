package vllm

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"os"
	"strconv"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// A deployment's env PYTHONHASHSEED roots every block-hash chain, so the hashes the engine
// publishes in its KV events are the ones vLLM publishes under that variable. The expected hashes
// are vLLM's own, from testdata written by oracle/gen_seeded_block_hashes.py: the same 64 tokens
// in blocks of 16, as the cases' "plain" chain, published as the low 64 bits big-endian in
// decimal (maybe_convert_block_hash, vllm/v1/core/kv_cache_utils.py:96-99@5df63a72).
func TestKVEventHashesFollowThePythonHashSeed(t *testing.T) {
	raw, err := os.ReadFile("../kv/testdata/seeded_block_hashes.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Cases []struct {
			Seed   *string  `json:"pythonhashseed"`
			Name   string   `json:"name"`
			Tokens []int32  `json:"tokens"`
			Hashes []string `json:"hashes"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	first := map[string]bool{}
	for _, c := range f.Cases {
		if c.Name != "plain" {
			continue
		}
		extra := "block_size: 16\nnum_gpu_blocks: 64\nmax_model_len: 256\nkv_events: {enable_kv_cache_events: true}\n"
		label := "unset"
		if c.Seed != nil {
			extra += "env: {PYTHONHASHSEED: " + strconv.Quote(*c.Seed) + "}\n"
			label = "seed=" + *c.Seed
		}
		t.Run(label, func(t *testing.T) {
			e := mustEngine(t, extra)
			e.EnableKVEvents()
			r := engine.Request{ID: "r", Prompt: c.Tokens, Output: []int32{7}}
			if err := e.Add(r, 0); err != nil {
				t.Fatal(err)
			}
			var stored []string
			now := engine.Time(0)
			for steps := 0; steps < 100 && e.s.HasUnfinished(); steps++ {
				b, err := e.Schedule(now)
				if err != nil {
					t.Fatal(err)
				}
				if b.Empty() {
					break
				}
				now += e.StepTime(b)
				res, err := e.Complete(b.ID, engine.Outcome{}, now)
				if err != nil {
					t.Fatal(err)
				}
				for _, ev := range res.KVEvents {
					if !ev.Removed {
						stored = append(stored, ev.Hash)
					}
				}
			}
			if len(stored) < len(c.Hashes) {
				t.Fatalf("%d blocks stored, vLLM hashes %d", len(stored), len(c.Hashes))
			}
			for i, h := range c.Hashes {
				b, err := hex.DecodeString(h)
				if err != nil {
					t.Fatal(err)
				}
				want := strconv.FormatUint(binary.BigEndian.Uint64(b[len(b)-8:]), 10)
				if stored[i] != want {
					t.Fatalf("block %d: published %s, vLLM publishes %s", i, stored[i], want)
				}
			}
			first[stored[0]] = true
		})
	}
	if len(first) < 5 {
		t.Errorf("only %d distinct first hashes across the seeds", len(first))
	}
}
