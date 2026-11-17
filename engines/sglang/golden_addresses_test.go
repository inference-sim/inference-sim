// Original to blis-sglang.

package blissglang

import (
	"bytes"
	"encoding/json"
	"io/fs"
	"math/big"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// addressBound is 2^40: no counter, size or index an oracle trace records reaches it, and a
// user-space address on linux-x86_64 is above it.
var addressBound = new(big.Int).Lsh(big.NewInt(1), 40)

// Integers a trace records above addressBound that are not addresses, by the key that holds them:
// a KV event's page hashes (64-bit, python/sglang/srt/mem_cache/events.py@297e54fd), and the
// lock uuids a tree component hands out from its own range of counters
// (mem_cache/unified_cache/components/base.py:62-65@297e54fd).
var (
	hashKeys     = []string{"block_hashes", "parent_block_hash"}
	lockUUIDMaps = []string{"component_lock_uuids", "component_host_lock_uuids"}
)

// addressLike is each integer in a trace that may be a memory address: above addressBound, not
// an object's id() recorded as {"_id_of": class}, and not held under a key whose values are known
// to be large (hashKeys, lockUUIDMaps). Each is named by its path in the trace.
func addressLike(raw []byte) ([]string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	var out []string
	var walk func(x any, path []string)
	walk = func(x any, path []string) {
		switch x := x.(type) {
		case map[string]any:
			if _, ok := x["_id_of"]; ok {
				return
			}
			for k, y := range x {
				walk(y, append(slices.Clone(path), k))
			}
		case []any:
			for _, y := range x {
				walk(y, path)
			}
		case json.Number:
			n, ok := new(big.Int).SetString(x.String(), 10)
			if !ok || n.CmpAbs(addressBound) <= 0 {
				return
			}
			if len(path) > 0 && slices.Contains(hashKeys, path[len(path)-1]) {
				return
			}
			if len(path) > 1 && slices.Contains(lockUUIDMaps, path[len(path)-2]) {
				return
			}
			out = append(out, strings.Join(path, ".")+"="+x.String())
		}
	}
	walk(v, nil)
	return out, nil
}

// No golden trace records a memory address: an address changes from run to run, so a trace that
// holds one cannot be reproduced, and it names the oracle process's memory.
func TestNoGoldenRecordsAMemoryAddress(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("oracle", "harness.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		GoldenDirs  []string `json:"golden_dirs"`
		GoldenFiles []string `json:"golden_files"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	files := slices.Clone(cfg.GoldenFiles)
	for _, dir := range cfg.GoldenDirs {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			continue // a golden directory another item adds
		}
		err := filepath.WalkDir(filepath.Clean(dir), func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !d.IsDir() && strings.HasSuffix(p, ".json") {
				files = append(files, p)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(files) == 0 {
		t.Fatal("no golden trace found under oracle/harness.json's golden_dirs")
	}
	for _, p := range files {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		found, err := addressLike(b)
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		for _, f := range found {
			t.Errorf("%s: %s may be a memory address; record it as {\"_id_of\": ...} or by its count", p, f)
		}
	}
}

// The guard finds an address wherever a trace holds one, and passes the large integers that are
// not addresses.
func TestTheAddressGuardFindsAnAddress(t *testing.T) {
	bad := []byte(`{"calls": [{"ret": {"k_data_ptrs": [140234567890432], "page": 7}},
		{"args": {"buffer": {"_t": "Tensor", "data_ptr": 94811234567168}}}]}`)
	found, err := addressLike(bad)
	if err != nil {
		t.Fatal(err)
	}
	if len(found) != 2 {
		t.Errorf("found %v; want the two addresses", found)
	}
	good := []byte(`{"calls": [{"ret": {"block_hashes": [-9105683110569361220], "parent_block_hash": 7325169757385981536,
		"component_host_lock_uuids": {"FULL": 200000000000001}, "key": {"_id_of": "HybridCacheController"}, "size": 1099511627776}}]}`)
	if found, err := addressLike(good); err != nil || len(found) != 0 {
		t.Errorf("found %v (%v); want none", found, err)
	}
}
