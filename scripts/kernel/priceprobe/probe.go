// Command priceprobe prices every cost-model fixture at a fixed set of steps and prints one
// JSON line per fixture. scripts/kernel/price_change_report.py runs it in two builds, the
// last inference-sim release that runs a fixture's old engine release (v0.15.0) and this
// checkout, and reports the difference (release plan §3.3, "their price changes are
// reported").
//
// It uses only kernel APIs both builds have (blis-vllm compat/kernelmodel's Open and
// Kernel(), the contracts kernel's StepTime, SequenceVariableBytes and FixedBytes), so this
// file builds in both. repos.go, the one file that differs, builds the adapter's roots: from
// directories at v0.15.0, through blis-substrate's loader here. The steps are those blis-vllm's own fixture move reported
// (blis-vllm fe56e4b): decode batches of 1, 32 and 256 sequences at context 4096, one
// prefill of 2048 tokens, the per-sequence KV bytes at 1024 tokens, and the fixed bytes.
//
// Usage: go run ./scripts/kernel/priceprobe -scenarios <dir> [-catalog <dir>] [-registry <dir>]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// Price is one fixture's prices. Err is set, and the prices are zero, when the fixture does
// not open.
type Price struct {
	Fixture    string           `json:"fixture"`
	StepNs     map[string]int64 `json:"step_ns,omitempty"`
	KVBytes    int64            `json:"kv_bytes_1024,omitempty"`
	FixedBytes int64            `json:"fixed_bytes,omitempty"`
	Err        string           `json:"error,omitempty"`
}

func main() {
	scenarios := flag.String("scenarios", "", "directory of fixtures (searched recursively)")
	catalog := flag.String("catalog", "", "catalog directory (empty: the embedded copy, where the build has one)")
	registry := flag.String("registry", "", "registry directory (empty: the embedded copy, where the build has one)")
	flag.Parse()
	if *scenarios == "" {
		fmt.Fprintln(os.Stderr, "priceprobe: -scenarios is required")
		os.Exit(2)
	}
	var files []string
	err := filepath.WalkDir(*scenarios, func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && filepath.Ext(p) == ".yaml" {
			files = append(files, p)
		}
		return err
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "priceprobe:", err)
		os.Exit(1)
	}
	sort.Strings(files)
	enc := json.NewEncoder(os.Stdout)
	for _, path := range files {
		rel, _ := filepath.Rel(*scenarios, path)
		p := price(filepath.Dir(path), filepath.Base(path), *catalog, *registry)
		p.Fixture = filepath.ToSlash(rel)
		if err := enc.Encode(p); err != nil {
			fmt.Fprintln(os.Stderr, "priceprobe:", err)
			os.Exit(1)
		}
	}
}

func price(dir, name, catalog, registry string) (p Price) {
	defer func() {
		if r := recover(); r != nil {
			p = Price{Err: fmt.Sprintf("panic: %v", r)}
		}
	}()
	repos, err := reposFor(dir, catalog, registry)
	if err != nil {
		return Price{Err: err.Error()}
	}
	m, err := kernelmodel.Open(name, repos)
	if err != nil {
		return Price{Err: err.Error()}
	}
	k := m.Kernel()
	p.StepNs = map[string]int64{}
	for _, n := range []int{1, 32, 256} {
		reqs := make([]kernelmodel.ReqShape, n)
		for i := range reqs {
			reqs[i] = kernelmodel.ReqShape{Scheduled: 1, Computed: 4096, PromptLen: 1024}
		}
		p.StepNs[fmt.Sprintf("decode_%d", n)] = m.StepTimeOfShapes(reqs)
	}
	p.StepNs["prefill_2048"] = m.StepTimeOfShapes([]kernelmodel.ReqShape{{Scheduled: 2048, Computed: 0, PromptLen: 2048}})
	p.KVBytes = k.SequenceVariableBytes(1024)
	p.FixedBytes = k.FixedBytes().Total()
	return p
}
