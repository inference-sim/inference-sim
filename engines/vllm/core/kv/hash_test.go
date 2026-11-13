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
// sha256 and init_none_hash (vllm/v1/core/kv_cache_utils.py:695-725@5df63a72), through
// oracle/gen_block_hashes.py, which records its provenance in the file, so these cases check
// the Go hasher against vLLM's bytes, not against itself. The frame-boundary and large-block
// cases pickle to more than 64 KiB, which CPython writes in several frames.
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

// A set PYTHONHASHSEED roots every chain at sha256(pickle(seed)), however it reads: "0", a number,
// a word, and the empty string, which is set (os.getenv returns "" and resolve_none_hash_seed
// tests for None). testdata/seeded_block_hashes.json was written by vLLM's own init_none_hash and
// hash_block_tokens under each value (oracle/gen_seeded_block_hashes.py).
func TestSeededBlockHashesAreVLLMs(t *testing.T) {
	raw, err := os.ReadFile("testdata/seeded_block_hashes.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Cases []struct {
			Seed     *string    `json:"pythonhashseed"`
			Name     string     `json:"name"`
			BlockSz  int        `json:"block_size"`
			Tokens   []int64    `json:"tokens"`
			Extra    [][]string `json:"extra"`
			NoneHash string     `json:"none_hash"`
			Hashes   []string   `json:"hashes"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) < 10 {
		t.Fatalf("%d seeded cases", len(f.Cases))
	}
	seen := map[string]bool{}
	for _, c := range f.Cases {
		label := "unset"
		if c.Seed != nil {
			label = "seed=" + *c.Seed
		}
		t.Run(label+"/"+c.Name, func(t *testing.T) {
			if got := hex.EncodeToString(NoneHashFor(c.Seed)); got != c.NoneHash {
				t.Errorf("NONE_HASH = %s, vLLM's is %s", got, c.NoneHash)
			}
			var keys []ExtraKey
			for _, e := range c.Extra {
				keys = append(keys, ExtraKey(e))
			}
			got := BlockHashes(c.Tokens, c.BlockSz, BlockKeys{Extra: keys, Seed: c.Seed})
			if len(got) != len(c.Hashes) {
				t.Fatalf("%d hashes, vLLM has %d", len(got), len(c.Hashes))
			}
			for i := range got {
				if hex.EncodeToString(got[i]) != c.Hashes[i] {
					t.Fatalf("block %d: %x, vLLM %s", i, got[i], c.Hashes[i])
				}
			}
			seen[c.Hashes[0]] = true
		})
	}
	// The seeds really differ: a hasher that ignored the seed would pass one case but not ten.
	if len(seen) < 10 {
		t.Errorf("only %d distinct first hashes across the seeded cases", len(seen))
	}
}
