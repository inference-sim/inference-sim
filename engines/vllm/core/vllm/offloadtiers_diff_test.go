package vllm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/engine"

	kvtier "github.com/inference-sim/blis-vllm/core/connector/tiering"
	"github.com/inference-sim/blis-vllm/internal/upstream"
)

// The fs tier through the engine against vLLM: each scenario under oracle/scenarios/offloading-tiering
// is driven through engine.Engine (Add, Schedule, Complete) and compared, step by step, with the trace
// vLLM v0.33.0's own Scheduler, OffloadingConnector and TieringOffloadingManager with an fs tier
// produced for it (oracle/run_offloading.py, run_tiered; core/connector/tiering/testdata/golden-engine).
//
// The oracle's worker mock completes a CPU<->GPU job a stated number of steps after it submits it, and
// its fs tier finishes every job its steps submitted, and resolves every lookup batch they flushed,
// before the next step. The driver is the simulator for the first (it reports each CPU transfer done
// before the schedule that the oracle's worker reports it at), and settles the second itself
// (settleTiers), because the engine reports a tier job as a transfer with the first schedule after the
// job was submitted, so a job the step's output submitted would otherwise finish a step later than the
// oracle's does (blis-vllm#190 names that substitution; offloadtiers_test.go drives the transfers).

// tierScenario is the part of an offloading scenario this driver reads.
type tierScenario struct {
	Config struct {
		MaxNumSeqs                int    `json:"max_num_seqs"`
		MaxNumBatchedTokens       int    `json:"max_num_batched_tokens"`
		BlockSize                 int    `json:"block_size"`
		NumBlocks                 int    `json:"num_blocks"`
		MaxModelLen               int    `json:"max_model_len"`
		LongPrefillTokenThreshold int    `json:"long_prefill_token_threshold"`
		EnablePrefixCaching       *bool  `json:"enable_prefix_caching"`
		SchedulingPolicy          string `json:"scheduling_policy"`
		TensorParallelSize        int    `json:"tensor_parallel_size"`
		KVPage                    struct {
			NumKVHeads int `json:"num_kv_heads"`
			HeadSize   int `json:"head_size"`
		} `json:"kv_page"`
		Transfer struct {
			LoadDelay  *int `json:"load_delay"`
			StoreDelay *int `json:"store_delay"`
		} `json:"transfer"`
		Connector struct {
			ExtraConfig map[string]any `json:"extra_config"`
		} `json:"connector"`
	} `json:"config"`
	Requests []struct {
		ID               string         `json:"id"`
		ArrivalStep      int            `json:"arrival_step"`
		Prompt           []int64        `json:"prompt"`
		Output           []int64        `json:"output"`
		MaxTokens        int            `json:"max_tokens"`
		Priority         int            `json:"priority"`
		AbortStep        *int           `json:"abort_step"`
		LoadDelay        *int           `json:"load_delay"`
		StoreDelay       *int           `json:"store_delay"`
		KVTransferParams map[string]any `json:"kv_transfer_params"`
	} `json:"requests"`
	MaxSteps int `json:"max_steps"`
}

// tierGolden is the part of the oracle's trace this driver reads; a step is kept as the JSON it is.
type tierGolden struct {
	Provenance struct {
		Tier           int      `json:"tier"`
		Kind           string   `json:"kind"`
		VLLMVersion    string   `json:"vllm_version"`
		VLLMCommit     string   `json:"vllm_commit"`
		ScenarioSHA256 string   `json:"scenario_sha256"`
		MockVersion    string   `json:"mock_version"`
		CPUChunks      int      `json:"cpu_chunks"`
		Tiers          []string `json:"secondary_tiers"`
	} `json:"provenance"`
	Steps    []map[string]any `json:"steps"`
	Finished map[string]struct {
		Reason       string `json:"reason"`
		OutputTokens int    `json:"output_tokens"`
	} `json:"finished"`
}

// tierEngineBlock is the engine block of a scenario: its scheduler, its CPU tier of cpu_bytes_to_use, and
// its fs tiers, each priced on an NVMe device the way a deployment names it (device_class).
func tierEngineBlock(t *testing.T, sc *tierScenario) string {
	t.Helper()
	c := sc.Config
	ex := c.Connector.ExtraConfig
	if ex["spec_name"] != "TieringOffloadingSpec" {
		t.Fatalf("spec_name %v: this driver runs TieringOffloadingSpec", ex["spec_name"])
	}
	// One KV cache group of one full-attention layer: a block's bytes are K and V, float32.
	perBlock := int64(2*c.BlockSize*max(c.KVPage.NumKVHeads, 1)*max(c.KVPage.HeadSize, 1)*4) * 1
	off := map[string]any{"connector": "OffloadingConnector", "spec": "TieringOffloadingSpec",
		"cpu_bytes": int64(ex["cpu_bytes_to_use"].(float64)), "bytes_per_block": perBlock,
		"workers": max(c.TensorParallelSize, 1)}
	if v, ok := ex["blocks_per_chunk"].(float64); ok {
		off["block_size"] = int(v) * c.BlockSize
	}
	if v, ok := ex["offload_prompt_only"].(bool); ok {
		off["offload_prompt_only"] = v
	}
	var tiers []map[string]any
	for _, raw := range ex["secondary_tiers"].([]any) {
		tier := map[string]any{}
		for k, v := range raw.(map[string]any) {
			if f, ok := v.(float64); ok {
				v = int(f)
			}
			tier[k] = v
		}
		tier["root_dir"] = "/" + strings.TrimPrefix(tier["root_dir"].(string), "@root/")
		tier["device_class"] = "nvme_gen4"
		tiers = append(tiers, tier)
	}
	off["secondary_tiers"] = tiers
	out, err := yaml.Marshal(map[string]any{"kv_offload": off})
	if err != nil {
		t.Fatal(err)
	}
	caching := c.EnablePrefixCaching == nil || *c.EnablePrefixCaching
	policy := c.SchedulingPolicy
	if policy == "" {
		policy = "fcfs"
	}
	return fmt.Sprintf("block_size: %d\nmax_num_seqs: %d\nmax_num_batched_tokens: %d\nmax_model_len: %d\nnum_gpu_blocks: %d\n"+
		"enable_prefix_caching: %t\nscheduling_policy: %s\nlong_prefill_token_threshold: %d\n",
		c.BlockSize, c.MaxNumSeqs, c.MaxNumBatchedTokens, c.MaxModelLen, c.NumBlocks, caching, policy,
		c.LongPrefillTokenThreshold) + string(out)
}

// tierRun drives sc through the engine for as many steps as the oracle ran, and returns the steps it
// observed in the oracle's own fields.
func tierRun(t *testing.T, sc *tierScenario, steps int) []map[string]any {
	t.Helper()
	eng, err := build(t, tierEngineBlock(t, sc))()
	if err != nil {
		t.Fatal(err)
	}
	e := eng.(*Engine)
	reqs := sc.Requests
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].ArrivalStep != reqs[j].ArrivalStep {
			return reqs[i].ArrivalStep < reqs[j].ArrivalStep
		}
		return reqs[i].ID < reqs[j].ID
	})
	delay := func(id string, store bool) int {
		for _, r := range reqs {
			if r.ID == id {
				if store && r.StoreDelay != nil {
					return *r.StoreDelay
				}
				if !store && r.LoadDelay != nil {
					return *r.LoadDelay
				}
			}
		}
		d := sc.Config.Transfer.LoadDelay
		if store {
			d = sc.Config.Transfer.StoreDelay
		}
		if d == nil {
			return 1
		}
		return *d
	}
	var out []map[string]any
	due := map[engine.TransferID]int{} // the step before whose schedule the simulator reports a CPU transfer done
	next, now := 0, engine.Time(0)
	for step := 0; step < steps+3; step++ {
		for ; next < len(reqs) && reqs[next].ArrivalStep <= step; next++ {
			r := reqs[next]
			if r.AbortStep != nil {
				t.Fatalf("%s: abort_step is not driven here", r.ID)
			}
			er := engine.Request{ID: r.ID, Priority: r.Priority, MaxOutputTokens: r.MaxTokens}
			for _, p := range r.Prompt {
				er.Prompt = append(er.Prompt, int32(p))
			}
			for _, o := range r.Output {
				er.Output = append(er.Output, int32(o))
			}
			if r.KVTransferParams != nil {
				raw, err := json.Marshal(r.KVTransferParams)
				if err != nil {
					t.Fatal(err)
				}
				er.Tags = map[string]string{TagKVTransferParams: string(raw)}
			}
			if err := e.Add(er, now); err != nil {
				t.Fatal(err)
			}
		}
		if !e.s.HasRequests() {
			if next == len(reqs) {
				break
			}
			out = append(out, map[string]any{"step": step, "idle": true})
			now += 1000
			continue
		}
		// The oracle's worker reports a job complete at the batch that executes at its step.
		var ready []engine.TransferID
		for id, at := range due {
			if at <= step {
				ready = append(ready, id)
			}
		}
		sort.Slice(ready, func(i, j int) bool { return ready[i] < ready[j] })
		for _, id := range ready {
			delete(due, id)
			if err := e.TransferDone(id, true, now); err != nil {
				t.Fatal(err)
			}
		}
		b, err := e.Schedule(now)
		if err != nil {
			t.Fatal(err)
		}
		rec := map[string]any{"step": step}
		sched := map[string]any{}
		for _, s := range b.Reqs {
			sched[s.ID] = s.Tokens
		}
		rec["scheduled"] = sched
		rec["preempted"] = b.Preempted
		loads, stores := []any{}, []any{}
		for _, tr := range b.TransfersStarted {
			if tr.Tier != "cpu" {
				continue
			}
			job := e.off.jobs[tr.ID]
			row := []any{job, tr.Request, tr.Blocks}
			if tr.Kind == engine.TransferLoad {
				loads = append(loads, row)
				due[tr.ID] = step + delay(tr.Request, false)
			} else {
				stores = append(stores, row)
				due[tr.ID] = step + 1 + delay(tr.Request, true)
			}
		}
		sortRows(loads)
		sortRows(stores)
		var fins []string
		if !b.Empty() {
			now += e.StepTime(b)
			res, err := e.Complete(b.ID, engine.Outcome{}, now)
			if err != nil {
				t.Fatal(err)
			}
			for _, f := range res.Finished {
				fins = append(fins, f.ID)
			}
		}
		// The jobs the step submitted, in its schedule and in its output (a batch that runs no model applies
		// its output inside Schedule), before the transport settles.
		tierJobs := tierJobRows(e)
		settleTiers(e)
		sort.Strings(fins)
		rec["finished"] = fins
		rec["aborted"] = []string{}
		rec["free_blocks"] = e.s.KV.Pool.NumFree()
		running := append([]string{}, e.s.Running()...)
		sort.Strings(running)
		rec["running"] = running
		nr, nw := e.s.Counts()
		rec["num_running"], rec["num_waiting"] = nr, nw
		rec["waiting_for_remote_kv"] = e.s.WaitingForRemoteKV()
		rec["recv_landed"] = e.s.RecvLanded()
		rec["delay_freed"] = e.s.DelayFreed()
		rec["kv_holding_waiting"] = e.s.Holding()
		fc := e.s.FetchCounts()
		rec["kv_fetch_stages"] = fc[:]
		rec["num_requests"] = e.s.NumRequests()
		chunks, free, evictable := e.off.conn.CPU()
		cpu := []any{}
		for _, ch := range chunks {
			cpu = append(cpu, []any{fmt.Sprintf("%x/%d", ch.Key.HashPrefix(), ch.Key.Group()), ch.ID, ch.RefCnt})
		}
		rec["offload"] = map[string]any{
			"loads": loads, "stores": stores, "jobs": e.off.conn.Jobs(), "pending_push": e.off.conn.HasPendingPushWork(),
			"cpu":       map[string]any{"chunks": cpu, "free": free, "evictable": evictable},
			"tier_jobs": tierJobs, "tier_files": tierFileRows(e),
		}
		out = append(out, rec)
	}
	return out
}

func sortRows(rows []any) {
	sort.Slice(rows, func(i, j int) bool { return rows[i].([]any)[0].(int) < rows[j].([]any)[0].(int) })
}

// tierJobRows are the fs tiers' queued jobs as [job id, tier index, "promotion" or "cascade", chunks,
// request].
func tierJobRows(e *Engine) []any {
	rows := []any{}
	if e.off.tiers == nil {
		return rows
	}
	for i, b := range e.off.tiers.tiers {
		for _, j := range b.fs.Jobs() {
			kind := "promotion"
			if j.Store {
				kind = "cascade"
			}
			rows = append(rows, []any{j.ID, i, kind, j.Chunks, j.Req})
		}
	}
	return rows
}

// settleTiers is the oracle's transport rule: every queued tier job runs, in submission order, and
// every lookup batch resolves, before the next step.
func settleTiers(e *Engine) {
	for _, b := range e.off.tiers.tiers {
		b.fs.RunAll()
		b.fs.ResolveLookups()
	}
}

// tierFileRows are each fs tier's block files by their last three path components (the key's
// directory, group directory and file name): the part of a name the run's layout (model, dtype, layer
// names) does not enter.
func tierFileRows(e *Engine) []any {
	rows := []any{}
	for _, b := range e.off.tiers.tiers {
		rows = append(rows, blockFileSuffixes(func(yield func(string)) {
			for p := range b.files.FileStore.(kvtier.MemFiles) {
				yield(p)
			}
		}))
	}
	return rows
}

func blockFileSuffixes(each func(func(string))) []any {
	var names []string
	each(func(p string) { names = append(names, lastParts(p, 3)) })
	sort.Strings(names)
	out := []any{}
	for _, n := range names {
		out = append(out, n)
	}
	return out
}

func lastParts(p string, n int) string {
	parts := strings.Split(p, "/")
	if len(parts) > n {
		parts = parts[len(parts)-n:]
	}
	return strings.Join(parts, "/")
}

// tierCompared are the fields of a step the engine is compared on; "offload" is compared by its own fields.
var tierCompared = []string{"scheduled", "preempted", "finished", "aborted", "free_blocks", "running", "num_running",
	"num_waiting", "waiting_for_remote_kv", "recv_landed", "delay_freed", "kv_holding_waiting", "kv_fetch_stages",
	"num_requests"}

var tierOffloadCompared = []string{"loads", "stores", "jobs", "pending_push", "cpu", "tier_jobs", "tier_files"}

// normalizeJSON makes a value comparable with one read from a trace: JSON's types, and a nil list
// as an empty one.
func normalizeJSON(v any) any {
	raw, _ := json.Marshal(v)
	var out any
	_ = json.Unmarshal(raw, &out)
	return scrubNil(out)
}

func scrubNil(v any) any {
	switch x := v.(type) {
	case nil:
		return []any{}
	case []any:
		for i := range x {
			x[i] = scrubNil(x[i])
		}
	case map[string]any:
		for k := range x {
			x[k] = scrubNil(x[k])
		}
	}
	return v
}

// jobRowsByBlocks reduces the oracle's [job id, request, CPU chunks, [GPU blocks, group sizes, indices]]
// to what the engine reports of a transfer: [job id, request, GPU blocks].
func jobRowsByBlocks(rows any) any {
	out := []any{}
	list, _ := rows.([]any)
	for _, r := range list {
		row := r.([]any)
		out = append(out, []any{row[0], row[1], float64(len(row[3].([]any)[0].([]any)))})
	}
	return out
}

// normalizeFiles replaces each tier's block file names in a trace's step by their last three components.
func normalizeFiles(rows any) any {
	tiers, _ := rows.([]any)
	out := []any{}
	for _, tier := range tiers {
		names := []any{}
		for _, n := range tier.([]any) {
			names = append(names, lastParts(n.(string), 3))
		}
		sort.Slice(names, func(i, j int) bool { return names[i].(string) < names[j].(string) })
		out = append(out, names)
	}
	return out
}

// tierDivergence names the first step and field at which got differs from want.
func tierDivergence(want []map[string]any, got []map[string]any) string {
	for i := 0; i < max(len(want), len(got)); i++ {
		if i >= len(want) {
			return fmt.Sprintf("step %d: the engine ran a step the oracle did not (scheduled %v)", i, got[i]["scheduled"])
		}
		if i >= len(got) {
			return fmt.Sprintf("step %d: the oracle ran a step the engine did not (scheduled %v)", i, want[i]["scheduled"])
		}
		w, g := normalizeJSON(want[i]).(map[string]any), normalizeJSON(got[i]).(map[string]any)
		if w["idle"] != nil || g["idle"] != nil {
			if !reflect.DeepEqual(w["idle"], g["idle"]) {
				return fmt.Sprintf("step %d: idle oracle %v, engine %v", i, w["idle"], g["idle"])
			}
			continue
		}
		// The tier's decisions first: they explain the rest.
		wo, _ := w["offload"].(map[string]any)
		go_, _ := g["offload"].(map[string]any)
		for _, f := range tierOffloadCompared {
			wv, gv := wo[f], go_[f]
			switch f {
			case "tier_files":
				wv = normalizeFiles(wv)
			case "loads", "stores":
				wv = jobRowsByBlocks(wv)
			}
			if !reflect.DeepEqual(wv, gv) {
				return fmt.Sprintf("step %d, offload.%s: oracle %v, engine %v", i, f, wv, gv)
			}
		}
		for _, f := range tierCompared {
			if !reflect.DeepEqual(w[f], g[f]) {
				return fmt.Sprintf("step %d, field %s: oracle %v, engine %v", i, f, w[f], g[f])
			}
		}
	}
	return ""
}

func TestFSTierEngineAgainstVLLM(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("..", "..", "oracle", "scenarios", "offloading-tiering", "tiering-fs-*.json"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no engine-level fs scenarios (%v)", err)
	}
	var mocks struct {
		Version string `json:"version"`
	}
	rawMocks, err := os.ReadFile(filepath.Join("..", "..", "oracle", "mocks.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(rawMocks, &mocks); err != nil || mocks.Version == "" {
		t.Fatalf("oracle/mocks.json states no version: %v", err)
	}
	for _, path := range paths {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var sc tierScenario
			if err := json.Unmarshal(raw, &sc); err != nil {
				t.Fatal(err)
			}
			var want tierGolden
			g, err := os.ReadFile(filepath.Join("..", "connector", "tiering", "testdata", "golden-engine", name+".json"))
			if err != nil {
				t.Fatalf("%v (record it with oracle/run_offloading.py)", err)
			}
			if err := json.Unmarshal(g, &want); err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(raw)
			p := want.Provenance
			if p.ScenarioSHA256 != hex.EncodeToString(sum[:]) || p.VLLMCommit != upstream.Commit ||
				strings.SplitN(p.VLLMVersion, "+", 2)[0] != upstream.Version || p.MockVersion != mocks.Version ||
				p.Tier != 4 || p.Kind != "tiered" {
				t.Fatalf("golden provenance %+v does not match this scenario, vLLM %s and mock %s",
					p, upstream.Version, mocks.Version)
			}
			got := tierRun(t, &sc, len(want.Steps))
			if d := tierDivergence(want.Steps, got); d != "" {
				t.Fatalf("earliest divergence from vLLM %s: %s", upstream.Version, d)
			}
			// What each request finished with.
			for id := range want.Finished {
				found := false
				for _, st := range got {
					for _, fid := range toStrings(st["finished"]) {
						found = found || fid == id
					}
				}
				if !found {
					t.Errorf("request %s finished in the oracle and not in the engine", id)
				}
			}
		})
	}
}

func toStrings(v any) []string {
	s, _ := v.([]string)
	return s
}
