package cmd

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"
	"github.com/inference-sim/blis-substrate/load"
	"gopkg.in/yaml.v3"

	sim "github.com/inference-sim/inference-sim/sim"

	blissglang "github.com/inference-sim/blis-sglang"
	sglanglower "github.com/inference-sim/blis-sglang/lower"
	sglangrules "github.com/inference-sim/blis-sglang/rules"
	"github.com/inference-sim/blis-sglang/sglangengine"
)

// The SGLang engine kind. blis-sglang builds the engine from a host's pool (sglangengine.Host); this
// file is the host. It opens the scenario's pool through the latency kernel, with SGLang's rules
// (blis-sglang/rules) and its lowering of the pool (blis-sglang/lower), and hands the engine the
// kernel and the facts SGLang's resolution reads of the chip and the model.

// engineKindSGLang is the SGLang engine kind.
const engineKindSGLang = "sglang"

// sglangContextOverheadGiB is the device memory CUDA and the driver hold before a model loads, on
// the H200: vLLM's start-up reads 137.64 GiB free of the 139.8 GiB the driver reports
// (blis-substrate catalog/hardware/h200.yaml, _comment_memory). SGLang reads the same free figure
// before it loads the weights (get_available_gpu_memory, tp_worker.py@297e54fd). It is measured on
// the H200 only, so any other chip is refused until its figure is.
const sglangContextOverheadGiB = 2.16

type sglangHost struct{ data *load.Data }

// sglangRepos is the roots and engine hooks a SGLang scenario opens with.
func sglangRepos(dir string, data *load.Data) latencykernel.Repos {
	return latencykernel.Repos{
		Scenarios:     dir,
		Data:          data,
		Rules:         latencykernel.RulesFrom(sglangrules.Lookup),
		RulesKind:     sglangrules.Kind,
		RulesVersions: sglangrules.Versions,
		DraftModel:    func(*deployment.Deployment, int) (string, error) { return "", nil },
		Lower: func(in latencykernel.Inputs) (*plan.Lowered, error) {
			return sglanglower.Pool(sglanglower.PoolInput{Deployment: in.Deployment, PoolIndex: in.PoolIndex, Model: in.Model, Chip: in.Chip})
		},
	}
}

// Open implements sglangengine.Host: the pool the spec selects, opened from the scenario file the
// spec's roots name.
func (h sglangHost) Open(s engine.Spec) (sglangengine.Pool, error) {
	path := s.Roots[sglangengine.RootScenario]
	if path == "" {
		return sglangengine.Pool{}, fmt.Errorf("sglang engine: Roots[%q] must name the scenario file", sglangengine.RootScenario)
	}
	in, err := latencykernel.OpenInputs(filepath.Base(path), sglangRepos(filepath.Dir(path), h.data), s.Pool)
	if err != nil {
		return sglangengine.Pool{}, err
	}
	k, err := latencykernel.New(in)
	if err != nil {
		return sglangengine.Pool{}, err
	}
	sglangKernels.put(s.Pool, k)
	pin := sglanglower.PoolInput{Deployment: in.Deployment, PoolIndex: in.PoolIndex, Model: in.Model, Chip: in.Chip}
	args, err := sglanglower.Args(pin)
	if err != nil {
		return sglangengine.Pool{}, err
	}
	chip, err := sglanglower.ChipOf(in.Chip)
	if err != nil {
		return sglangengine.Pool{}, err
	}
	if in.Chip.Name != "h200" {
		return sglangengine.Pool{}, fmt.Errorf("sglang engine: the memory CUDA holds before a model loads is measured on the h200 only, not %q (inference-sim/blis-sglang#49)", in.Chip.Name)
	}
	facts, err := sglanglower.ModelFactsOf(in.Model)
	if err != nil {
		return sglangengine.Pool{}, err
	}
	ctx := in.Model.Global.ContextWindow
	if v, ok := args["context_length"]; ok {
		if n, isInt := v.(int); isInt && n > 0 {
			ctx = n
		}
	}
	if ctx <= 0 {
		return sglangengine.Pool{}, fmt.Errorf("sglang engine: the model graph states no context window and the engine block no context_length")
	}
	return sglangengine.Pool{
		Kernel: k, Chip: chip, Model: facts, Args: args,
		PreModelLoadGiB: in.Chip.MemoryGiB - sglangContextOverheadGiB,
		ContextLen:      ctx, VocabSize: in.Model.Global.VocabSize,
		// The catalog carries no tokenizer. The simulator assigns each request its output and ends
		// it at the script's end (engine.Request.Output), so the end-of-sequence id is one no script
		// holds: the first id past the vocabulary.
		EOSTokenIDs: []int64{int64(in.Model.Global.VocabSize)},
	}, nil
}

// newSGLangEngine builds one SGLang engine from spec over the run's opened data.
func newSGLangEngine(spec engine.Spec, data *load.Data) (engine.Engine, error) {
	return sglangengine.FactoryFor(sglangHost{data: data}).New(spec)
}

// sglangEngineRelease is the SGLang release this binary runs, as blis-sglang records it.
func sglangEngineRelease() (kind, version, commit string) {
	return engineKindSGLang, blissglang.Upstream.Version, blissglang.Upstream.Commit
}

// sglangRealScenario is the SGLang scenario file a run reads its engine from, set once the run's
// host-side state has been adopted from the scenario's vLLM-form twin (adoptSGLangTwin).
var sglangRealScenario string

// engineScenarioPath is the scenario file the engine factory reads: the SGLang scenario when the
// run adopted one, else the run's scenario.
func engineScenarioPath() string {
	if sglangRealScenario != "" {
		return sglangRealScenario
	}
	return filepath.Join(kernelScenarioDir, kernelScenario)
}

// adoptSGLangTwin points the run's host-side adoption at a twin of the SGLang scenario. The host
// still takes a deployment's model, hardware, parallelism, the simulated KV store and the host
// overheads from blis-vllm's adapter, which decodes only a vLLM engine block. The twin is the same
// scenario with engine {kind: vllm, version: 0.33.0} and an engine block holding what the adapter
// requires, max_model_len (the SGLang block's context_length). The engine itself is built from the
// SGLang scenario (engineScenarioPath), so SGLang's scheduler, KV pool and step pricing decide every
// step; the twin supplies only the host-side facts and the host overheads' coefficients, which are
// the registry's and engine-independent. This is a named substitution: any host-side setting the
// twin resolves by vLLM's rules is vLLM's, not SGLang's.
func adoptSGLangTwin() error {
	if sglangRealScenario != "" {
		return nil
	}
	real := filepath.Join(kernelScenarioDir, kernelScenario)
	raw, err := os.ReadFile(real)
	if err != nil {
		return err
	}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	var out bytes.Buffer
	enc := yaml.NewEncoder(&out)
	enc.SetIndent(2)
	for {
		var doc map[string]any
		if err := dec.Decode(&doc); err != nil {
			if err.Error() == "EOF" {
				break
			}
			return fmt.Errorf("scenario %s: %w", real, err)
		}
		switch doc["kind"] {
		case "Scenario":
			doc["engine"] = map[string]any{"kind": "vllm", "version": "0.33.0"}
		case "Deployment":
			pools, _ := doc["pools"].([]any)
			for _, p := range pools {
				pm, _ := p.(map[string]any)
				blk, _ := pm["engine"].(map[string]any)
				ctx, ok := blk["context_length"]
				if !ok {
					return fmt.Errorf("scenario %s: the SGLang engine block must state context_length: the catalog graph carries no context window (inference-sim/blis-sglang#49)", real)
				}
				// The adapter needs a block size for a sparse-MLA model; the twin states SGLang's
				// page_size when the block does, else 64, the page SGLang resolves for DSA models.
				page := blk["page_size"]
				if page == nil {
					page = 64
				}
				twin := map[string]any{"max_model_len": ctx, "block_size": page}
				// stream_interval is the one engine setting the host itself reads (how many tokens
				// the API server holds before streaming them); vLLM's and SGLang's field mean the
				// same, so the twin carries the pool's value.
				if v, ok := blk["stream_interval"]; ok {
					twin["stream_interval"] = v
				}
				pm["engine"] = twin
			}
		}
		if err := enc.Encode(doc); err != nil {
			return err
		}
	}
	if err := enc.Close(); err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "blis-sglang-twin-")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, kernelScenario), out.Bytes(), 0o644); err != nil {
		return err
	}
	// The scenario's companion files (the router's endpoint-picker configuration, the offload
	// tiers' files) are named relative to its directory, so the twin's directory holds each
	// of them too, as a link to the file beside the scenario.
	siblings, err := os.ReadDir(kernelScenarioDir)
	if err != nil {
		return err
	}
	for _, e := range siblings {
		if e.Name() == kernelScenario {
			continue
		}
		src, err := filepath.Abs(filepath.Join(kernelScenarioDir, e.Name()))
		if err != nil {
			return err
		}
		if err := os.Symlink(src, filepath.Join(dir, e.Name())); err != nil {
			return err
		}
	}
	sglangRealScenario, kernelScenarioDir = real, dir
	return nil
}

// sglangKernels are the kernels the SGLang engines of the run opened, by pool index. The Result's
// resolutions and kernel provenance are read from them, so they state what SGLang's rules resolved
// and what its kernel priced, not what the host's vLLM-form twin resolved.
var sglangKernels = &sglangKernelSet{byPool: map[int]*latencykernel.Kernel{}}

type sglangKernelSet struct {
	mu     sync.Mutex
	byPool map[int]*latencykernel.Kernel
}

func (s *sglangKernelSet) put(pool int, k *latencykernel.Kernel) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.byPool[pool] = k
}

// reset forgets the kernels of an earlier run in this process.
func (s *sglangKernelSet) reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.byPool = map[int]*latencykernel.Kernel{}
}

// pools are the opened pool indices, ascending.
func (s *sglangKernelSet) pools() []int {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]int, 0, len(s.byPool))
	for i := range s.byPool {
		out = append(out, i)
	}
	sort.Ints(out)
	return out
}

func (s *sglangKernelSet) get(pool int) *latencykernel.Kernel {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.byPool[pool]
}

// sglangTwinSubstitutions are the named gaps between an SGLang run and its scenario that remain
// while the host adopts its side of the deployment through blis-vllm's adapter (adoptSGLangTwin).
// They are stated in the Result beside the pool's kernel evidence, so a point that carries them is
// not read as priced end to end by SGLang's host.
func sglangTwinSubstitutions() []sim.Substitution {
	return []sim.Substitution{
		{Setting: "host_side_facts", Stated: "the SGLang engine block", Priced: "a vLLM-form twin of the scenario",
			Reason: "the host takes the model, hardware, parallelism and request-bounding facts through blis-vllm's adapter, which decodes only a vLLM engine block; the engine itself, its KV pool and its step pricing are SGLang's (inference-sim/blis-sglang#49)"},
		{Setting: "host_overheads", Stated: "SGLang's API-server and detokenizer path", Priced: "the registry's engine-independent host-overhead coefficients",
			Reason: "post_decode_fixed_overhead and output_token_processing_time are read from the registry's cost-model-host-overheads set, measured under vLLM's server; no SGLang-specific measurement exists (inference-sim/blis-sglang#49)"},
	}
}

// sglangProvenance is a pool's kernel evidence as the Result records it, from the pool's SGLang
// kernel (the same reading blis-vllm's adapter makes of its own).
func sglangProvenance(k *latencykernel.Kernel) sim.KernelPoolProvenance {
	p := sim.KernelPoolProvenance{
		MethodCounts:        map[vocab.Method]int{},
		NotEvidenced:        []kernel.CoefficientOrigin{},
		Assumptions:         []kernel.CoefficientOrigin{},
		SkippedRules:        []string{},
		SkippedRuleFindings: []sim.RuleFinding{},
		Substitutions:       sglangTwinSubstitutions(),
	}
	origins := k.Provenance()
	for _, o := range origins {
		method := vocab.Method(o.Method)
		p.MethodCounts[method]++
		switch {
		case o.Set == latencykernel.KernelAssumptionSet:
			p.Assumptions = append(p.Assumptions, o)
		case !method.Evidenced() && method != vocab.MethodNotCharged:
			p.NotEvidenced = append(p.NotEvidenced, o)
		}
	}
	p.Evidenced, p.Total, _ = k.Evidence()
	p.EvidenceSource = sim.EvidenceFromKernel
	return p
}

// sglangProvenanceByRole is the Result's kernel section for an SGLang run: the colocated pool's
// SGLang kernel. An SGLang run has one colocated pool (a disaggregated deployment is refused).
func sglangProvenanceByRole() map[string]sim.KernelPoolProvenance {
	out := map[string]sim.KernelPoolProvenance{}
	for _, i := range sglangKernels.pools() {
		out[string(deployment.RoleColocated)] = sglangProvenance(sglangKernels.get(i))
	}
	return out
}

// sglangResolutions is the Result's resolutions for an SGLang run: SGLang's own resolution of each
// setting the cost model resolves, per pool.
func sglangResolutions(dep deployment.Deployment) []result.Resolution {
	labels := result.PoolLabels(dep)
	var out []result.Resolution
	for _, i := range sglangKernels.pools() {
		out = append(out, result.ResolutionsOfPool(sglangKernels.get(i).Resolved(), labels[i])...)
	}
	result.SortResolutions(out)
	return out
}
