// Original to blis-vllm.

package vllm

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-vllm/core/kv"
)

// vLLM keys a block's hash on the LoRA adapter's name and path (_gen_lora_extra_hash_keys,
// vllm/v1/core/kv_cache_utils.py:578-595@5df63a72), not on lora_int_id. core/kv/testdata/block_hashes.json
// holds vLLM's own hashes for a name whose path differs from it, and for the same name pointed at
// another adapter, over the same tokens (oracle/gen_block_hashes.py). The key the engine builds for
// a request (loraExtraKey, from Adapter and AdapterPath) must reproduce them.
func TestTheLoRABlockKeyIsVLLMsNameAndPath(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "kv", "testdata", "block_hashes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Cases []struct {
			Name      string     `json:"name"`
			BlockSize int        `json:"block_size"`
			Tokens    []int64    `json:"tokens"`
			Extra     [][]string `json:"extra"`
			Hashes    []string   `json:"hashes"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	first := func(keys []kv.ExtraKey, tokens []int64, bs int) string {
		return hex.EncodeToString(kv.BlockHashes(tokens, bs, kv.BlockKeys{Extra: keys})[0])
	}
	seen := 0
	var v1 string
	for _, c := range f.Cases {
		if c.Name != "lora-path-differs-from-name" && c.Name != "lora-name-repointed" {
			continue
		}
		seen++
		name, path := c.Extra[0][1], c.Extra[0][2]
		if name == path {
			t.Fatalf("%s: the case's path equals its name, so it cannot tell the two apart", c.Name)
		}
		got := first(loraExtraKey(name, path), c.Tokens, c.BlockSize)
		if got != c.Hashes[0] {
			t.Errorf("%s: the engine's key hashes to %s, vLLM's is %s", c.Name, got, c.Hashes[0])
		}
		// Known bad: keying on the name alone (as the engine did before the path reached it)
		// is not vLLM's hash.
		if bad := first(loraExtraKey(name, ""), c.Tokens, c.BlockSize); bad == c.Hashes[0] {
			t.Errorf("%s: a key on the name alone gives vLLM's hash, so the case cannot catch it", c.Name)
		}
		if c.Name == "lora-path-differs-from-name" {
			v1 = got
		} else if got == v1 {
			t.Errorf("re-pointing %q at another adapter left the hash unchanged", name)
		}
	}
	if seen != 2 {
		t.Fatalf("block_hashes.json has %d of the two LoRA-path cases", seen)
	}
}

// A request's adapter path reaches the block key: the same prompt under the same adapter name
// hits the cached blocks when the path is the same, and misses them when the name was re-pointed
// at another adapter (vLLM keys on both). Keying on the name alone makes the second case hit.
func TestARepointedAdapterDoesNotHitTheEarlierBlocks(t *testing.T) {
	for _, c := range []struct {
		name       string
		path2      string
		wantCached int
	}{{"same path", "/loras/sql-v1", 32}, {"re-pointed", "/loras/sql-v2", 0}} {
		t.Run(c.name, func(t *testing.T) {
			e, err := build(t, "max_loras: 2\n")()
			if err != nil {
				t.Fatal(err)
			}
			a, b := req("x", 48, 1), req("x", 48, 1)
			a.ID, b.ID = "first", "second"
			a.Adapter, a.AdapterPath = "sql", "/loras/sql-v1"
			b.Adapter, b.AdapterPath = "sql", c.path2
			var last engine.Batch
			for _, r := range []engine.Request{a, b} {
				if err := e.Add(r, 0); err != nil {
					t.Fatal(err)
				}
				last, err = e.Schedule(0)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := e.Complete(last.ID, engine.Outcome{}, 1); err != nil {
					t.Fatal(err)
				}
			}
			if got := last.Reqs[0].CachedTokens; got != c.wantCached {
				t.Errorf("the second request cached %d tokens, want %d", got, c.wantCached)
			}
		})
	}
}
