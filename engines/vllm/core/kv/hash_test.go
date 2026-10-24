package kv

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

type hashCase struct {
	Name      string     `json:"name"`
	BlockSize int        `json:"block_size"`
	Tokens    []int64    `json:"tokens"`
	Extra     [][]string `json:"extra"`
	Salt      string     `json:"salt"`
	Hashes    []string   `json:"hashes"`
}

// testdata/block_hashes.json was written by vLLM v0.32.0's own hash_block_tokens with
// sha256 and init_none_hash (vllm/v1/core/kv_cache_utils.py:695-725@5df63a72), so these
// cases check the Go hasher against vLLM's bytes, not against itself.
func loadHashCases(t *testing.T) (string, []hashCase) {
	t.Helper()
	raw, err := os.ReadFile("testdata/block_hashes.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		NoneHash string     `json:"none_hash"`
		Cases    []hashCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) == 0 {
		t.Fatal("no cases")
	}
	return f.NoneHash, f.Cases
}

func TestNoneHashIsVLLMs(t *testing.T) {
	want, _ := loadHashCases(t)
	if got := hex.EncodeToString(NoneHash()); got != want {
		t.Fatalf("NONE_HASH = %s, vLLM's is %s", got, want)
	}
}

func TestBlockHashesAreVLLMs(t *testing.T) {
	_, cases := loadHashCases(t)
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			var keys []ExtraKey
			for _, e := range c.Extra {
				keys = append(keys, ExtraKey(e))
			}
			hashes := BlockHashes(c.Tokens, c.BlockSize, BlockKeys{Extra: keys, CacheSalt: c.Salt})
			if len(hashes) != len(c.Hashes) {
				t.Fatalf("%d hashes, vLLM has %d", len(hashes), len(c.Hashes))
			}
			for i, h := range hashes {
				if got := hex.EncodeToString(h); got != c.Hashes[i] {
					t.Errorf("block %d: %s, vLLM's is %s", i, got, c.Hashes[i])
				}
			}
		})
	}
}

// A LoRA adapter is part of every block's key, so the same prompt under two adapters shares
// no cached block: the inference-sim bug #2021 B9 lists ("same prompt with two adapters must
// not share cache").
func TestAnAdapterChangesEveryBlockHash(t *testing.T) {
	toks := []int64{1, 2, 3, 4, 5, 6, 7, 8}
	base := BlockHashes(toks, 4, BlockKeys{})
	a := BlockHashes(toks, 4, BlockKeys{Extra: []ExtraKey{{"lora", "a", "/a"}}})
	b := BlockHashes(toks, 4, BlockKeys{Extra: []ExtraKey{{"lora", "b", "/b"}}})
	for i := range base {
		if string(base[i]) == string(a[i]) || string(a[i]) == string(b[i]) {
			t.Fatalf("block %d hashes alike across adapters", i)
		}
	}
}
