package cmd

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
	"github.com/inference-sim/inference-sim/substrate/contracts/vocab"
	latencykernel "github.com/inference-sim/inference-sim/substrate/kernel"
	"github.com/inference-sim/inference-sim/substrate/load"
	"gopkg.in/yaml.v3"

	sim "github.com/inference-sim/inference-sim/sim"

	blissglang "github.com/inference-sim/inference-sim/engines/sglang"
	sglanglower "github.com/inference-sim/inference-sim/engines/sglang/lower"
	sglangrules "github.com/inference-sim/inference-sim/engines/sglang/rules"
	"github.com/inference-sim/inference-sim/engines/sglang/sglangengine"
)

// The SGLang engine kind. the SGLang engine builds the engine from a host's pool (sglangengine.Host); this
// file is the host. It opens the scenario's pool through the latency kernel, with SGLang's rules
// (engines/sglang/rules) and its lowering of the pool (engines/sglang/lower), and hands the engine the
// kernel and the facts SGLang's resolution reads of the chip and the model.

// engineKindSGLang is the SGLang engine kind.
const engineKindSGLang = "sglang"

// sglangContextOverheadGiB is the device memory CUDA and the driver hold before a model loads, on
// each chip: the total the driver reports less the free figure vLLM's start-up logs on that chip
// (the substrate catalog/hardware/<chip>.yaml, _comment_memory, each with its InferenceX run). On
// the H200 vLLM reads 137.64 GiB free of the 139.8 GiB, 2.16 GiB. SGLang reads the same free figure
// before it loads the weights (get_available_gpu_memory, tp_worker.py@297e54fd), but no SGLang log
// states it, so it is vLLM's measurement carried over. A chip with no entry is refused until its
// figure is measured.
var sglangContextOverheadGiB = map[string]float64{
	"h100":        79.19 - 76.0,
	"h200":        2.16,
	"b200":        178.35 - 175.93,
	"b300":        267.69 - 265.24,
	"gb200-nvl72": 184.0 - 180.15,
	"gb300":       276.62 - 273.04,
}

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
	overhead, ok := sglangContextOverheadGiB[in.Chip.Name]
	if !ok {
		return sglangengine.Pool{}, fmt.Errorf("sglang engine: the memory CUDA holds before a model loads is not measured on the chip %q (inference-sim/inference-sim#2344)", in.Chip.Name)
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
		PreModelLoadGiB: in.Chip.MemoryGiB - overhead,
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

// sglangEngineRelease is the SGLang release this binary runs, as the SGLang engine records it.
func sglangEngineRelease() (kind, version, commit string) {
	return engineKindSGLang, blissglang.Upstream.Version, blissglang.Upstream.Commit
}

// sglangEngineModel is the SGLang engine as this build runs it, for `blis version`. SGLang has no
// separate kernel-behaviour pin: the one upstream release the engine models
// (blissglang.Upstream) is both the release it prices and the tree its rules were ported from, so
// both members of the record name it.
//
// An SGLang pool is colocated only: the engine refuses prefill/decode disaggregation
// (engines/sglang/core/sglang, Engine.Add, names inference-sim/inference-sim#2351).
func sglangEngineModel() engineReport {
	upstream := builtEngines()[engineKindSGLang]
	return engineReport{
		Kind:           upstream.Kind,
		ModeledVersion: upstream.Version,
		ModeledCommit:  upstream.Commit,
		BuildVersion:   upstream.Version,
		BuildCommit:    upstream.Commit,
		Roles:          []deployment.Role{deployment.RoleColocated},
	}
}

// sglangRealScenario is the SGLang scenario file a run reads its engine from, set once the run's
// host-side state has been adopted from the scenario's vLLM-form twin (adoptSGLangTwin).
var sglangRealScenario string

// sglangSpeculation is what the first pool of the run's SGLang scenario states of speculative
// decoding, read by adoptSGLangTwin: the algorithm, and the draft width (speculative_num_draft_tokens
// less the verified token, else speculative_num_steps; zero when the block states neither).
var sglangSpeculation struct {
	Algorithm string
	Drafts    int
}

// sglangSpeculationRefusal is the refusal for an acceptance flag on an SGLang scenario that
// speculates: the host resolves acceptance from the vLLM-form speculative block the twin
// carries, which states no drafts, so it cannot resolve this scenario's (inference-sim/inference-sim#2355).
func sglangSpeculationRefusal(flag string) error {
	drafts := ""
	if sglangSpeculation.Drafts > 0 {
		drafts = fmt.Sprintf(", %d drafts", sglangSpeculation.Drafts)
	}
	return fmt.Errorf("--%s: scenario %q speculates through the SGLang engine (speculative_algorithm %s%s), and the host cannot yet resolve an SGLang scenario's acceptance (inference-sim/inference-sim#2355)",
		flag, kernelScenario, sglangSpeculation.Algorithm, drafts)
}

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
// overheads from the vLLM engine's adapter, which decodes only a vLLM engine block. The twin is the same
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
	sglangSpeculation.Algorithm, sglangSpeculation.Drafts = "", 0
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
			doc["engine"] = map[string]any{"kind": "vllm"}
		case "Deployment":
			pools, _ := doc["pools"].([]any)
			for i, p := range pools {
				pm, _ := p.(map[string]any)
				blk, _ := pm["engine"].(map[string]any)
				if i == 0 {
					recordSGLangSpeculation(blk)
				}
				ctx, ok := blk["context_length"]
				if !ok {
					return fmt.Errorf("scenario %s: the SGLang engine block must state context_length: the catalog graph carries no context window (inference-sim/inference-sim#2344)", real)
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

// recordSGLangSpeculation records the speculative settings of the first pool's SGLang engine block.
func recordSGLangSpeculation(blk map[string]any) {
	alg, _ := blk["speculative_algorithm"].(string)
	if alg == "" {
		return
	}
	sglangSpeculation.Algorithm = alg
	if n, ok := blk["speculative_num_draft_tokens"].(int); ok && n > 1 {
		sglangSpeculation.Drafts = n - 1
	} else if n, ok := blk["speculative_num_steps"].(int); ok {
		sglangSpeculation.Drafts = n
	}
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
// while the host adopts its side of the deployment through the vLLM engine's adapter (adoptSGLangTwin).
// They are stated in the Result beside the pool's kernel evidence, so a point that carries them is
// not read as priced end to end by SGLang's host.
func sglangTwinSubstitutions() []sim.Substitution {
	return []sim.Substitution{
		{Setting: "host_side_facts", Stated: "the SGLang engine block", Priced: "a vLLM-form twin of the scenario",
			Reason: "the host takes the model, hardware, parallelism and request-bounding facts through the vLLM engine's adapter, which decodes only a vLLM engine block; the engine itself, its KV pool and its step pricing are SGLang's (inference-sim/inference-sim#2344)"},
		{Setting: "host_overheads", Stated: "SGLang's API-server and detokenizer path", Priced: "the registry's engine-independent host-overhead coefficients",
			Reason: "post_decode_fixed_overhead and output_token_processing_time are read from the registry's cost-model-host-overheads set, measured under vLLM's server; no SGLang-specific measurement exists (inference-sim/inference-sim#2344)"},
	}
}

// planSubstitutions are the overrides the engine's lowering stated, as machine-readable substitutions on the
// point: a setting the engine resolved that the cost model prices as something else (the SGLang engine's
// Lowered.Overrides, which the kernel reports in its resolution). A grade reads this list, not the
// assumptions' text, as it does for the vLLM engine's pool substitutions.
func planSubstitutions(overrides []kernel.Override) []sim.Substitution {
	out := make([]sim.Substitution, 0, len(overrides))
	for _, o := range overrides {
		out = append(out, sim.Substitution{Setting: o.Field, Stated: o.Requested, Priced: o.Resolved, Reason: o.Reason})
	}
	return out
}

// sglangProvenance is a pool's kernel evidence as the Result records it, from the pool's SGLang
// kernel (the same reading the vLLM engine's adapter makes of its own).
func sglangProvenance(k *latencykernel.Kernel) sim.KernelPoolProvenance {
	p := sim.KernelPoolProvenance{
		MethodCounts:        map[vocab.Method]int{},
		NotEvidenced:        []kernel.CoefficientOrigin{},
		Assumptions:         []kernel.CoefficientOrigin{},
		SkippedRules:        []string{},
		SkippedRuleFindings: []sim.RuleFinding{},
		Substitutions:       append(sglangTwinSubstitutions(), planSubstitutions(k.Resolved().Overrides)...),
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
