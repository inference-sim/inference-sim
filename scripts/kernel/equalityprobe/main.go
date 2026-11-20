// Command equalityprobe prices every scenario under a directory, pool by pool, at a fixed set of
// steps, and prints one JSON line per pool: the step prices, the stage and data-parallel step
// times, the per-sequence and fixed memory, the KV budget and memory, the host costs, adaptive
// verification, the kernel's Resolved() and Provenance(), the engine assumptions and sources,
// and a refusal's text. It is the price-equality check of the engine-settings review set
// (inference-sim/blis-substrate#58): build it against two module sets, run both over the same
// scenario directories, and compare the outputs with -compare, which reports every field that
// differs and exits 1 when a price, a figure, a refusal or a record differs. A provenance field
// may be allowed to differ in order only (-order-only).
//
// Every pool of every scenario is priced: pools are opened from 0 up until the scenario states
// no more. A file that does not decode as a scenario is recorded once, with its error.
//
// It uses only blis-vllm compat/kernelmodel APIs that both the M2 release's substrate and engine
// (the pins stack.json names on main before this review set) and this checkout's pins have, so the
// same file builds in both.
//
// Usage: go run ./scripts/kernel/equalityprobe -scenarios <dir> > out.jsonl
//
//	go run ./scripts/kernel/equalityprobe -compare old.jsonl,new.jsonl -order-only origins,prov,assumed
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/engines/vllm/core/kernelmodel"
	sim "github.com/inference-sim/inference-sim/engines/vllm/core/runconfig"
	"github.com/inference-sim/inference-sim/substrate/load"
)

type Price struct {
	Fixture   string            `json:"fixture"`
	Pool      int               `json:"pool"`
	StepNs    map[string]int64  `json:"step_ns,omitempty"`
	Stages    [][]int64         `json:"stages,omitempty"`
	DP        [][]int64         `json:"dp,omitempty"`
	KVBytes   map[string]int64  `json:"kv_bytes,omitempty"`
	Fixed     any               `json:"fixed,omitempty"`
	Budget    any               `json:"budget,omitempty"`
	BudgetErr string            `json:"budget_err,omitempty"`
	Resolved  any               `json:"resolved,omitempty"`
	Origins   any               `json:"origins,omitempty"`
	Assumed   []string          `json:"assumed,omitempty"`
	Prov      any               `json:"prov,omitempty"`
	Sources   map[string]string `json:"sources,omitempty"`
	Verified  [][]int           `json:"verified,omitempty"`
	Other     map[string]any    `json:"other,omitempty"`
	Err       string            `json:"error,omitempty"`
}

func main() {
	scenarios := flag.String("scenarios", "", "directory of scenarios (searched recursively)")
	pair := flag.String("compare", "", "old.jsonl,new.jsonl: compare two outputs instead of pricing")
	orderOnly := flag.String("order-only", "", "comma-separated fields allowed to differ in order only")
	flag.Parse()
	if *pair != "" {
		old, new, ok := strings.Cut(*pair, ",")
		if !ok {
			fmt.Fprintln(os.Stderr, "equalityprobe: -compare takes old.jsonl,new.jsonl")
			os.Exit(2)
		}
		os.Exit(runCompare(old, new, *orderOnly))
	}
	if *scenarios == "" {
		fmt.Fprintln(os.Stderr, "equalityprobe: -scenarios is required")
		os.Exit(2)
	}
	var files []string
	if err := filepath.WalkDir(*scenarios, func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && filepath.Ext(p) == ".yaml" {
			files = append(files, p)
		}
		return err
	}); err != nil {
		fmt.Fprintln(os.Stderr, "equalityprobe:", err)
		os.Exit(1)
	}
	sort.Strings(files)
	d, err := kernelmodel.OpenData(load.Options{})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	for _, path := range files {
		rel, _ := filepath.Rel(*scenarios, path)
		for pool := 0; ; pool++ {
			p, _ := price(filepath.Dir(path), filepath.Base(path), d, pool)
			p.Fixture = filepath.ToSlash(rel)
			p.Pool = pool
			// Past the last pool the loader names the pool count; every pool before it is priced.
			if pool > 0 && strings.Contains(p.Err, "pool(s); pool") {
				break
			}
			if err := enc.Encode(p); err != nil {
				fmt.Fprintln(os.Stderr, rel, err)
			}
			// A file that is not a scenario has no pools to try.
			if strings.Contains(p.Err, "decoding the scenario") || strings.Contains(p.Err, "pool(s); pool") {
				break
			}
		}
	}
}

func safe(v any) any { return clean(reflect.ValueOf(v)) }

// clean turns v into JSON-able data, dropping func and chan values.
func clean(v reflect.Value) any {
	if !v.IsValid() {
		return nil
	}
	switch v.Kind() {
	case reflect.Func, reflect.Chan, reflect.UnsafePointer:
		return nil
	case reflect.Pointer, reflect.Interface:
		if v.IsNil() {
			return nil
		}
		return clean(v.Elem())
	case reflect.Struct:
		out := map[string]any{}
		for i := 0; i < v.NumField(); i++ {
			f := v.Type().Field(i)
			if !f.IsExported() {
				continue
			}
			out[f.Name] = clean(v.Field(i))
		}
		return out
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			return nil
		}
		out := make([]any, v.Len())
		for i := range out {
			out[i] = clean(v.Index(i))
		}
		return out
	case reflect.Map:
		out := map[string]any{}
		for _, k := range v.MapKeys() {
			out[fmt.Sprint(k.Interface())] = clean(v.MapIndex(k))
		}
		return out
	case reflect.Float32, reflect.Float64:
		f := v.Float()
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return fmt.Sprint(f)
		}
		return f
	}
	return v.Interface()
}

func req(prompt, progress, sched int) *sim.Request {
	r := &sim.Request{ID: "r", InputTokens: make([]sim.TokenID, prompt), OutputTokens: make([]sim.TokenID, 4096)}
	r.ProgressIndex = int64(progress)
	r.NumNewTokens = sched
	return r
}

func price(dir, name string, d *load.Data, pool int) (p Price, more bool) {
	defer func() {
		if r := recover(); r != nil {
			p = Price{Err: fmt.Sprintf("panic: %v", r)}
		}
	}()
	repos := kernelmodel.Repos{Scenarios: dir, Data: d}
	m, err := kernelmodel.OpenPool(name, repos, pool)
	if err != nil {
		return Price{Err: err.Error()}, false
	}
	more = true
	k := m.Kernel()
	p.StepNs = map[string]int64{}
	shapes := map[string][]kernelmodel.ReqShape{}
	for _, n := range []int{1, 8, 32, 256} {
		for _, ctx := range []int{512, 4096, 32768} {
			reqs := make([]kernelmodel.ReqShape, n)
			for i := range reqs {
				reqs[i] = kernelmodel.ReqShape{Scheduled: 1, Computed: ctx, PromptLen: min(ctx, 1024)}
			}
			shapes[fmt.Sprintf("decode_%d_ctx%d", n, ctx)] = reqs
			reqs4 := make([]kernelmodel.ReqShape, n)
			for i := range reqs4 {
				reqs4[i] = kernelmodel.ReqShape{Scheduled: 4, Computed: ctx, PromptLen: min(ctx, 1024)}
			}
			shapes[fmt.Sprintf("verify4_%d_ctx%d", n, ctx)] = reqs4
		}
	}
	for _, t := range []int{1, 17, 512, 2048, 8192} {
		shapes[fmt.Sprintf("prefill_%d", t)] = []kernelmodel.ReqShape{{Scheduled: t, Computed: 0, PromptLen: t}}
		shapes[fmt.Sprintf("chunk_%d_at4096", t)] = []kernelmodel.ReqShape{{Scheduled: t, Computed: 4096, PromptLen: 4096 + t}}
		shapes[fmt.Sprintf("cached_%d", t)] = []kernelmodel.ReqShape{{Scheduled: t, Computed: 0, PromptLen: t + 1024, CachedTokens: 1024}}
	}
	mixed := []kernelmodel.ReqShape{{Scheduled: 1024, Computed: 0, PromptLen: 2048}}
	for i := 0; i < 31; i++ {
		mixed = append(mixed, kernelmodel.ReqShape{Scheduled: 1, Computed: 3000, PromptLen: 1024})
	}
	shapes["mixed_1024_31"] = mixed
	keys := make([]string, 0, len(shapes))
	for k := range shapes {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, key := range keys {
		p.StepNs[key] = m.StepTimeOfShapes(shapes[key])
		p.Stages = append(p.Stages, m.StageTimesOfShapes(shapes[key]))
		p.Verified = append(p.Verified, m.VerifiedDraftsOfShapes(shapes[key]))
	}
	// Through the simulator's own request translation (speculation widens a decode here).
	for _, n := range []int{1, 32} {
		batch := make([]*sim.Request, n)
		for i := range batch {
			batch[i] = req(1024, 2048, 1)
		}
		p.StepNs[fmt.Sprintf("sim_decode_%d", n)] = m.StepTime(batch)
	}
	p.StepNs["sim_prefill_1024"] = m.StepTime([]*sim.Request{req(1024, 0, 1024)})
	p.DP = append(p.DP, m.StepTimesAcrossDPOfShapes([][]kernelmodel.ReqShape{shapes["decode_8_ctx4096"], shapes["prefill_512"]}))
	p.KVBytes = map[string]int64{}
	for _, n := range []int{1, 1024, 65536} {
		p.KVBytes[fmt.Sprint(n)] = k.SequenceVariableBytes(n)
	}
	p.Fixed = safe(k.FixedBytes())
	if b, err := m.KVBudget(); err != nil {
		p.BudgetErr = err.Error()
	} else {
		p.Budget = safe(b)
	}
	p.Resolved = safe(k.Resolved())
	p.Origins = safe(k.Provenance())
	p.Assumed = m.EngineAssumptions()
	p.Prov = safe(m.Provenance())
	p.Sources = m.EngineSources()
	p.Other = map[string]any{
		"queue":  m.QueueingTime(req(1024, 0, 1024)),
		"out":    m.OutputTokenProcessingTime(),
		"post":   m.PostDecodeFixedOverhead(),
		"dpw":    m.DataParallelWidth(),
		"stages": m.PipelineStages(),
	}
	if s, err := m.Settings(); err == nil {
		p.Other["settings"] = safe(s)
	} else {
		p.Other["settings_err"] = err.Error()
	}
	if kv, err := m.KVMemory(); err == nil {
		p.Other["kvmemory"] = safe(kv)
	} else {
		p.Other["kvmemory_err"] = err.Error()
	}
	return p, true
}
