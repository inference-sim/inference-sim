// Package kernelmodel adapts blis-latency-kernel to BLIS's sim.LatencyModel interface.
//
// # What this package is, and what it deliberately is not
//
// It is a translation layer. Every latency number it returns comes from the kernel; this
// package computes no cost of its own. The point of the exercise is to measure the kernel
// as committed in blis-latency-kernel, so any arithmetic added here would contaminate the
// result it is meant to measure.
//
// Three things do need care, and they are the whole of this file's substance:
//
//   - UNITS. BLIS's sim.LatencyModel is int64 MICROSECONDS ("ticks", see
//     sim/latency_model.go). The kernel returns time.Duration, which is nanoseconds.
//     Every boundary crossing converts, and nothing else in this file may see a Duration.
//
//   - THE >= 1 POSTCONDITION. BLIS requires StepTime >= 1 for every input including an
//     empty batch, because 0 stalls the simulation clock and violates INV-3. The kernel
//     charges an empty batch its host per-step cost, which is non-zero today, but relying
//     on a coefficient staying non-zero would make a clock invariant depend on a registry
//     value. So the floor is explicit here.
//
//   - BATCH TRANSLATION. BLIS's Request and the kernel's ReqShape carry the same facts
//     under different names. Getting this mapping wrong is silent: pricing a decode as a
//     chunked-prefill tail changes the kernel's choice of attention law, and the
//     simulation still runs.
//
// # Why the kernel is constructed here rather than through its own harness
//
// blis-latency-kernel has internal/harness.Open, which does exactly this loading. Go
// forbids importing another module's internal packages, so it cannot be reused. The Open
// below makes the same sequence of calls against the same public blis-schemas loaders; it
// is loading, not modelling, and it holds no cost logic.
package kernelmodel

import (
	"fmt"
	"path/filepath"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	schemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"

	"github.com/inference-sim/inference-sim/sim"
)

// Repos locates the three artifact repositories a kernel is built from.
type Repos struct {
	Scenarios string // directory holding scenario YAML files
	Catalog   string // blis-catalog root
	Registry  string // blis-registry root
}

// Model adapts a kernel to sim.LatencyModel.
//
// decodeThreshold and smBudget are fixed at construction because they are engine and
// hardware configuration rather than per-step state, and re-deriving them inside the hot
// StepTime path would cost more than storing them.
type Model struct {
	k *latencykernel.Kernel

	decodeThreshold int
	smBudget        int

	// Host costs are invariant per kernel, so they are converted to ticks once. StepTime
	// is called once per simulated step; a Duration-to-ticks division per call is wasted.
	outputTokenTicks int64
	completionTicks  int64

	// Retained for KVBudget (capacity.go), which needs the engine settings and the chip's
	// memory to turn the kernel's byte answers into a block count.
	scenario      *scenario.Scenario
	poolIndex     int
	chipMemoryGiB float64
	isMoE         bool
}

var _ sim.LatencyModel = (*Model)(nil)

// DecodeThreshold is the scheduled-token count at or below which the engine's classifier
// treats a request as a decode. It mirrors blis-latency-kernel's own constant; the kernel
// takes it per batch rather than holding it, because it is engine configuration.
const DecodeThreshold = 8

// Open builds a kernel from a scenario file and adapts it.
//
// The loading sequence mirrors blis-latency-kernel's internal/harness.Open. It is
// duplicated because Go forbids cross-module internal imports, and it is kept to loading
// so that duplication carries no modelling risk: if these two ever disagree, the
// disagreement is about which file was read, not about what a step costs.
func Open(scenario string, r Repos) (*Model, error) {
	sc, err := schemas.LoadScenario(filepath.Join(r.Scenarios, scenario))
	if err != nil {
		return nil, err
	}
	graph, err := schemas.LoadModelGraph(
		filepath.Join(r.Catalog, "models", sc.Model, "graph.yaml"))
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", sc.Model, err)
	}
	chip, err := schemas.LoadChip(
		filepath.Join(r.Catalog, "hardware", sc.Hardware+".yaml"))
	if err != nil {
		return nil, fmt.Errorf("hardware %q: %w", sc.Hardware, err)
	}
	var fabric *hardware.Fabric
	if sc.Fabric != "" {
		if fabric, err = schemas.LoadFabric(
			filepath.Join(r.Catalog, "networks", sc.Fabric+".yaml")); err != nil {
			return nil, fmt.Errorf("fabric %q: %w", sc.Fabric, err)
		}
	}
	sets := make([]*coefficient.Set, 0, len(sc.Coefficients))
	for _, name := range sc.Coefficients {
		set, err := schemas.LoadCoefficientSet(
			filepath.Join(r.Registry, "coefficients", name+".yaml"))
		if err != nil {
			return nil, fmt.Errorf("coefficient set %q: %w", name, err)
		}
		sets = append(sets, set)
	}
	devices, err := schemas.LoadStorageDevices(
		filepath.Join(r.Catalog, "devices", "storage.yaml"))
	if err != nil {
		return nil, err
	}
	pack := rules.Lookup(sc.EngineVersion)
	if pack == nil {
		return nil, fmt.Errorf("no engine rules for version %q; known versions are %v",
			sc.EngineVersion, rules.Versions())
	}
	k, err := latencykernel.New(latencykernel.Inputs{
		Scenario: sc, PoolIndex: 0, Model: graph, Chip: chip, Fabric: fabric,
		Devices: devices, Coefficients: sets, Rules: pack,
	})
	if err != nil {
		return nil, err
	}
	m := New(k, chip.SMCount)
	m.scenario = sc
	m.poolIndex = 0
	m.chipMemoryGiB = chip.MemoryGiB
	m.isMoE = hasGroupedGEMM(graph)
	return m, nil
}

// hasGroupedGEMM reports whether any layer kind routes tokens through experts. It decides
// the DP scaling in KVBudget, mirroring latency.CalculateKVBlocks' IsMoE gate: vLLM runs dp
// independent EngineCores for an MoE model, each holding a full KV budget.
func hasGroupedGEMM(g *model.Graph) bool {
	for _, kind := range g.LayerKinds {
		for _, n := range kind.Nodes {
			if n.Op == model.OpGroupedGEMM {
				return true
			}
		}
	}
	return false
}

// New adapts an already-built kernel. Exported so a caller that constructed a kernel
// some other way -- a test with a hand-built coefficient set, say -- can adapt it without
// going through the filesystem.
func New(k *latencykernel.Kernel, smBudget int) *Model {
	return &Model{
		k:                k,
		decodeThreshold:  DecodeThreshold,
		smBudget:         smBudget,
		outputTokenTicks: ticks(k.OutputTokenOverhead()),
		completionTicks:  ticks(k.CompletionOverhead()),
	}
}

// Kernel exposes the adapted kernel. A caller that needs a memory or provenance answer
// asks the kernel directly rather than having this adapter grow a passthrough per method.
func (m *Model) Kernel() *latencykernel.Kernel { return m.k }

// StepTime prices one forward pass over the scheduled batch.
//
// StepEstimate.Expected is read rather than either band edge. The kernel reports a band
// -- Overlap sums the max over resources within each layer, NoOverlap sums every resource
// -- and Expected is the edge its own measured evidence selects, so this consumer does not
// re-decide. An earlier version of this comment claimed Overlap "is what a real engine
// achieves"; NVIDIA's FPM dataset, which measures one synchronized whole-forward iteration
// at a known batch and KV-token count, contradicts that. Over 219 points spanning two
// models, two parts and five parallelism topologies:
//
//	edge       mean|err|   signed
//	Overlap      14.70%   -10.80%
//	NoOverlap    10.16%    -0.73%
//
// NoOverlap is closer on four of the five cells and is nearly unbiased, where Overlap
// carries a one-sided deficit of the same magnitude BLIS shows end to end. The physical
// reason is PIECEWISE cudagraph mode: attention runs eagerly between captured segments,
// so per-layer overlap is structurally limited.
//
// The one dissenting cell is pure-tp2 -- the whole model on two GPUs, where the expert
// weight read dominates a single resource and per-stage max is the right composition.
// Even there NoOverlap wins at batch >= 128. That is a regime boundary worth revisiting
// with a resource-aware blend, not a reason to keep the optimistic edge everywhere.
//
// FPM is used only to choose between the kernel's own two edges. It is not fitted
// against, and the InferenceX corpus BLIS is scored on is never used for either.
func (m *Model) StepTime(batch []*sim.Request) int64 {
	b := kernel.Batch{
		Reqs:            make([]kernel.ReqShape, 0, len(batch)),
		DecodeThreshold: m.decodeThreshold,
		SMBudget:        m.smBudget,
	}
	for _, req := range batch {
		b.Reqs = append(b.Reqs, shapeOf(req))
	}
	// max(1) upholds BLIS's postcondition without depending on a registry value staying
	// non-zero. See the package comment.
	return max(1, ticks(m.k.StepTime(b).Expected))
}

// QueueingTime is the host work before a request can be scheduled: tokenization and
// prompt preprocessing. It excludes queue waiting, which depends on engine busyness and
// is the simulator's own concern rather than a pure function's.
func (m *Model) QueueingTime(req *sim.Request) int64 {
	return ticks(m.k.AdmissionOverhead(int(req.InputLen())))
}

// OutputTokenProcessingTime is per-emitted-token host work: incremental detokenization
// and streaming.
func (m *Model) OutputTokenProcessingTime() int64 { return m.outputTokenTicks }

// PostDecodeFixedOverhead is fixed per-request work at completion.
func (m *Model) PostDecodeFixedOverhead() int64 { return m.completionTicks }

// shapeOf translates one BLIS request into the kernel's request shape.
//
// The mapping, and why each field is what it is:
//
//	Scheduled    <- NumNewTokens. Tokens this step, set by FormBatch. Under speculation
//	                it is 1+accepted, which is why a value of 1 does not imply a decode.
//	Computed     <- ProgressIndex. Tokens already computed. BLIS advances it after the
//	                step, so at batch formation it is the pre-step value the kernel wants.
//	PromptLen    <- InputLen(). Computed against PromptLen is what separates a
//	                chunked-prefill tail from a decode, and the two use different
//	                attention laws.
//	CachedTokens <- 0. A prefix-cache hit is already reflected in Computed: BLIS's
//	                batch_formation sets numNewTokens = InputLen() - ProgressIndex, and
//	                the cache-aware allocation path is what advances ProgressIndex. So the
//	                hit is excluded from Scheduled before the adapter sees it, and passing
//	                a cached count here would subtract the same tokens twice. The kernel's
//	                field exists for a caller that has NOT yet subtracted it.
//
//	                Note what this does NOT rest on: BLIS implements prefix caching
//	                unconditionally, in sim/kv, not behind a flag. An earlier version of
//	                this comment said BLIS "subtracts hits before setting NumNewTokens",
//	                which is the right conclusion from a premise that would have been
//	                false if caching were the thing being relied on.
func shapeOf(req *sim.Request) kernel.ReqShape {
	return kernel.ReqShape{
		Scheduled:    req.NumNewTokens,
		Computed:     int(req.ProgressIndex),
		PromptLen:    int(req.InputLen()),
		CachedTokens: 0,
	}
}

// ticks converts a kernel Duration to BLIS's int64 microsecond tick.
//
// Truncation rather than rounding, to match BLIS's existing backends: both call
// clampToInt64 on a float microsecond value. The largest error is 1 tick, which on the
// smallest step in the evaluation corpus (553 us for an empty batch) is under 0.2%.
func ticks(d interface{ Microseconds() int64 }) int64 { return d.Microseconds() }

// Engine reports the engine settings of the pool this kernel prices.
//
// A caller configuring a simulator needs the same block size, token budget and sequence cap
// the kernel resolved against, or the two would describe different deployments: a scheduler
// admitting 512 sequences while the kernel priced a 256-sequence engine is not a
// disagreement the numbers would reveal.
//
// It returns the scenario's own values rather than a copy with defaults filled in, and
// errors when a value a simulator requires is absent, so a missing setting is a failure
// rather than a silent zero.
func (m *Model) Engine() (scenario.Engine, error) {
	e := m.scenario.Pools[m.poolIndex].Engine
	if e.BlockSize <= 0 {
		return e, fmt.Errorf("kernelmodel: scenario states no block_size")
	}
	if e.MaxNumSeqs <= 0 {
		return e, fmt.Errorf("kernelmodel: scenario states no max_num_seqs")
	}
	if e.MaxNumBatchedTokens <= 0 {
		return e, fmt.Errorf("kernelmodel: scenario states no max_num_batched_tokens")
	}
	return e, nil
}

// DataParallelWidth is the pool's attention data-parallel width.
//
// vLLM runs this many independent EngineCores, each with its own sequence cap, token budget
// and KV budget, and splits requests disjointly across them. A single-instance simulator
// modelling the aggregate must scale all three.
func (m *Model) DataParallelWidth() int {
	return m.scenario.Pools[m.poolIndex].Parallel.DP
}

// StepEstimate exposes the kernel's full band for one batch: the overlap edge, the serialized
// edge, the binding resource and the per-resource breakdown.
//
// StepTime returns only the overlap edge, because sim.LatencyModel is a single int64. A
// caller diagnosing WHY a step costs what it does needs the rest, and asking the kernel again
// through this method is cheaper and less error-prone than rebuilding the batch translation.
func (m *Model) StepEstimate(batch []*sim.Request) kernel.StepEstimate {
	b := kernel.Batch{
		Reqs:            make([]kernel.ReqShape, 0, len(batch)),
		DecodeThreshold: m.decodeThreshold,
		SMBudget:        m.smBudget,
	}
	for _, req := range batch {
		b.Reqs = append(b.Reqs, shapeOf(req))
	}
	return m.k.StepTime(b)
}

// Deployment reports the scenario facts an ALTERNATIVE latency backend needs to be
// constructed for the same deployment this kernel models.
//
// It exists for cmd/kernelscore's four-estimator comparison, which scores the kernel against
// BLIS's own roofline and trained-physics backends on one subset. Those backends are
// configured from a HuggingFace config.json and a hardware-calibration entry rather than from
// the catalog and registry, so the harness needs the scenario's model name, chip, tensor
// width and the two precisions -- and must read them from the SAME scenario the kernel was
// opened from, or the arms would describe different deployments.
//
// Nothing here is a latency coefficient: this is deployment identity, not calibration.
func (m *Model) Deployment() Deployment {
	p := m.scenario.Pools[m.poolIndex]
	return Deployment{
		Model:        m.scenario.Model,
		Hardware:     m.scenario.Hardware,
		TP:           p.Parallel.TP,
		DP:           p.Parallel.DP,
		Quantization: p.Engine.Quantization,
		CacheDType:   p.Engine.CacheDType,
		// The chip's total memory, which is what vLLM resolves its batch defaults from.
		DeviceMemoryGiB: m.chipMemoryGiB,
		// Tri-state in the scenario, resolved here to the engine's default. vLLM caches
		// unless told not to, so nil means ON and only an explicit false disables it.
		PrefixCachingDisabled: p.Engine.EnablePrefixCaching != nil &&
			!*p.Engine.EnablePrefixCaching,
		BlockSize:  int64(p.Engine.BlockSize),
		GPUMemUtil: p.Engine.GPUMemoryUtilization,
	}
}

// Deployment is the scenario identity an alternative backend is built from.
type Deployment struct {
	Model        string
	Hardware     string
	TP           int
	DP           int
	Quantization string
	CacheDType   string
	// PrefixCachingDisabled is the engine's --no-enable-prefix-caching, already resolved
	// from the scenario's tri-state against vLLM's default of ON.
	PrefixCachingDisabled bool
	// DeviceMemoryGiB is the chip's total memory. vLLM resolves max_num_seqs and
	// max_num_batched_tokens from it for a deployment that passes neither, so reproducing
	// that resolution needs it.
	DeviceMemoryGiB float64
	BlockSize       int64
	GPUMemUtil      float64
}
