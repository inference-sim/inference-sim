// Package vllmtest builds the vLLM v0.33.0 engine for the simulator's own tests: from the
// per-instance settings the cluster tests state (runconfig.SimConfig) and a latency model that
// prices steps, with no catalog, registry or scenario. It is test support: nothing but _test.go
// files links it (repocheck, X7).
package vllmtest

import (
	"cmp"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"

	runconfig "github.com/inference-sim/inference-sim/engines/vllm/core/runconfig"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
)

// Options are the parts of an engine a SimConfig does not state.
type Options struct {
	// Connector, when set, gives the engine a P/D connector in the kv_both role, which a request
	// with RemotePrefill or RemoteDecode needs, and the placement every transfer is stated against.
	Connector bool
	// Index is the instance's index: with Connector or a GPU stated, instance N sits on node N of
	// its pool, as a deployment places it.
	Index int
	// Extra is engine-block lines (vLLM setting names) added after the ones the config states.
	Extra string
}

// pricer prices a batch by the config's latency model, as the model prices the requests it holds.
type pricer struct{ m runconfig.LatencyModel }

func (p pricer) StepTime(b engine.Batch) engine.Duration {
	reqs := make([]*runconfig.Request, len(b.Reqs))
	for i, s := range b.Reqs {
		reqs[i] = &runconfig.Request{ID: s.ID, InputTokens: make([]runconfig.TokenID, s.PromptLen),
			ProgressIndex: int64(s.Computed), NumNewTokens: s.Tokens}
	}
	return p.m.StepTime(reqs)
}

// Block is the engine block cfg states, in vLLM's setting names.
func Block(cfg runconfig.SimConfig, o Options) (yaml.Node, error) {
	var b strings.Builder
	line := func(f string, a ...any) { fmt.Fprintf(&b, f+"\n", a...) }
	line("block_size: %d", max(cfg.BlockSizeTokens, 1))
	line("num_gpu_blocks: %d", cfg.TotalKVBlocks)
	line("max_num_seqs: %d", cfg.MaxNumSeqs)
	line("max_num_batched_tokens: %d", cfg.MaxNumBatchedTokens)
	// An unstated window is the model's, which a caller resolves: the longest sequence the cache holds
	// stands in for "no cap".
	line("max_model_len: %d", cmp.Or(cfg.MaxModelLen, max(cfg.TotalKVBlocks-1, 1)*max(cfg.BlockSizeTokens, 1)))
	line("use_v2_model_runner: false")
	line("async_scheduling: false")
	line("pipeline_parallel_size: 1")
	line("enable_prefix_caching: %t", !cfg.PrefixCachingDisabled)
	if cfg.LongPrefillTokenThreshold > 0 {
		line("long_prefill_token_threshold: %d", cfg.LongPrefillTokenThreshold)
	}
	if cfg.Scheduler == "priority-fcfs" || cfg.Scheduler == "priority" {
		line("scheduling_policy: priority")
	}
	if lc := cfg.LoRAConfig; lc.AdapterCapacity != nil || lc.HasAdapters() {
		line("max_loras: %d", max(1, ptrOr(lc.AdapterCapacity, len(lc.Adapters))))
	}
	if o.Connector {
		line("kv_transfer: {connector: NixlConnector, kv_role: kv_both}")
		line("placement: {hardware: %s, fabric: test-fabric, tensor: %d, nodes: %d, node: %d}",
			gpuOf(cfg), max(cfg.TP, 1), o.Index+1, o.Index)
	}
	b.WriteString(o.Extra)
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(b.String()), &n); err != nil {
		return n, err
	}
	return *n.Content[0], nil
}

func ptrOr(p *int, d int) int {
	if p != nil {
		return *p
	}
	return d
}

func gpuOf(cfg runconfig.SimConfig) string {
	if cfg.GPU != "" {
		return cfg.GPU
	}
	return "test-gpu"
}

// Engine is the v0.33.0 engine for cfg, its steps priced by cfg.LatencyModel.
func Engine(cfg runconfig.SimConfig, o Options) (engine.Engine, error) {
	if cfg.LatencyModel == nil {
		return nil, fmt.Errorf("vllmtest: SimConfig.LatencyModel is nil")
	}
	blk, err := Block(cfg, o)
	if err != nil {
		return nil, err
	}
	f := vllm.Factory{Pricer: func(engine.Spec) (vllm.Pricer, error) { return pricer{cfg.LatencyModel}, nil }}
	return f.New(engine.Spec{Kind: vllm.Kind, UpstreamVersion: vllm.Version, Config: blk, Seed: cfg.Seed})
}
