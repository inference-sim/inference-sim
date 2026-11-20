package latencykernel

import (
	"fmt"
	"path/filepath"
	"reflect"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// Repos locates the roots a scenario's names resolve against, and the engine rules its
// engine_version selects.
//
// A scenario names its model, hardware, fabric and coefficient sets without saying where
// they live — it describes a deployment, not a filesystem — so a caller supplies the roots.
// It names its engine release the same way, without the rules: those belong to the engine
// model (blis-vllm), not to this cost model, so a caller supplies them too.
type Repos struct {
	Scenarios string // directory holding scenario files
	// Data is the catalog and registry the scenario's names resolve against, opened and
	// checked by package load: the embedded copy (load.Embedded), or paths and overlays
	// (load.Open). Required. Open records into it every coefficient its lookups return,
	// for the selected-entry report (Data.Report).
	Data *load.Data

	// Rules returns the engine rules for a release, or nil for a release it has none for:
	// for vLLM, the packs of github.com/inference-sim/blis-vllm/rules (its Lookup, wrapped
	// as RulesFrom does). Required: the kernel no longer looks rules up itself.
	Rules func(version string) rules.EngineRules
	// RulesKind is the engine kind the supplied Rules are for, as a scenario's engine block
	// names it (engine: {kind, version}). Open refuses a scenario whose engine block names
	// another kind, because a release number alone does not say whose rules answer it:
	// {kind: sglang, version: X} would otherwise pair with the vLLM rules for X. Empty means
	// "vllm", the only engine kind a Repos could carry before this field existed, so every
	// existing caller, none of which sets it, keeps working unchanged. A scenario with no
	// engine block (the legacy engine_version form) names no kind and is not checked.
	RulesKind string
	// RulesVersions lists the releases Rules answers for, for the error a scenario naming
	// another one gets. Optional.
	RulesVersions func() []string
	// DraftModel names the catalog model a pool's engine block names as its drafter, "" for
	// none: the engine model's reading of a block the kernel does not open itself. Open loads
	// that model's graph as Inputs.Draft. Required, as Rules is.
	DraftModel func(dep *deployment.Deployment, pool int) (string, error)
	// Lower is the engine's lowering of one pool: from the loaded documents, the pool's plan
	// with every setting the engine resolved (Inputs.Lowered). Open sets Inputs.Lowered from
	// it; a caller of OpenInputs that lowers the pool itself leaves it nil.
	Lower func(in Inputs) (*plan.Lowered, error)
}

// legacyRulesKind is the engine kind Repos.RulesKind means when it is empty.
const legacyRulesKind = "vllm"

// RulesFrom adapts a lookup returning a concrete rules type, such as blis-vllm/rules'
// Lookup returning *rules.Pack, to Repos.Rules. It keeps a nil result nil: a nil *Pack in a
// non-nil interface would read as rules that exist. That holds also when R is itself an
// interface type whose value holds a nil pointer.
func RulesFrom[R interface {
	comparable
	rules.EngineRules
}](lookup func(version string) R) func(string) rules.EngineRules {
	return func(version string) rules.EngineRules {
		r := lookup(version)
		if isNil(r) {
			return nil
		}
		return r
	}
}

// rulesAnswer asks a rules value the one question every value answers, its release, and
// reports a value that cannot: a forwarding struct around a nil rules value is not nil, so
// isNil passes it, but its first call dereferences nil. Asking once here turns that panic,
// which would come from deep inside pricing, into a refusal at construction.
func rulesAnswer(r rules.EngineRules) (err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("EngineVersion panicked: %v", p)
		}
	}()
	r.EngineVersion()
	return nil
}

// isNil reports whether v is nil or an interface holding a nil pointer, map, slice, func or
// channel: a value that cannot answer for a release.
func isNil(v any) bool {
	if v == nil {
		return true
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan, reflect.Interface:
		return rv.IsNil()
	}
	return false
}

// Open resolves one scenario file into a kernel.
//
// The path-based half of the constructor pair. New takes documents a caller already holds;
// Open takes a filename and the roots its names resolve against, which is what a simulator
// or a scoring command has. Both end in New, so both inherit its validation and neither can
// price a deployment the other would reject.
//
// It is exported because the alternative is every consumer reimplementing it. Go forbids
// importing another module's internal packages, so while this lived in internal/harness,
// inference-sim carried a copy of the sequence and said so in a comment, and
// blis-config-search and every future backend repo would have needed their own. The
// duplication was tolerable only because it was confined to loading — but "which files a
// scenario implies" is this package's answer to give, not each caller's to guess.
//
// Every artifact is loaded through blis-schemas' own loaders. The engine rules are the ones
// the caller's Repos.Rules returns for the scenario's own engine_version, so a scenario
// pinned to an older release cannot silently get current behaviour, and a release the
// caller has no rules for is refused.
//
// It prices the deployment's FIRST pool, which is what a colocated deployment has. Use
// OpenPool to price one pool of a disaggregated one.
func Open(scenario string, r Repos) (*Kernel, error) {
	return OpenPool(scenario, r, 0)
}

// OpenPool resolves ONE pool of a scenario file into a kernel.
//
// A disaggregated deployment states a prefill pool and a decode pool, and the two differ in
// the quantities that set step time: tensor-parallel width, whether expert parallelism is
// on, the engine's token budget. One kernel prices one pool — each runs its own engine with
// its own settings — so a caller serving the roles separately opens one kernel per pool.
// Pricing both from pool 0 would charge the decode pool the prefill pool's parallelism, and
// the simulation would still run.
//
// Open is this function at pool 0. Both exist because the common case should not have to
// name an index, and the disaggregated case cannot be served without one.
func OpenPool(scenario string, r Repos, poolIndex int) (*Kernel, error) {
	return OpenPoolIn(scenario, r, poolIndex, "")
}

// OpenPoolIn is OpenPool for an instance placed in the cluster's node pool called nodePool
// (SIM-17): the kernel is priced and sized by that node pool's chip. See OpenInputsIn.
func OpenPoolIn(scenario string, r Repos, poolIndex int, nodePool string) (*Kernel, error) {
	in, err := OpenInputsIn(scenario, r, poolIndex, nodePool)
	if err != nil {
		return nil, err
	}
	k, err := New(in)
	if err != nil {
		return nil, err
	}
	k.RecordSelected(r.Data)
	return k, nil
}

// RecordSelected records into d every coefficient this kernel's lookups returned while it
// was built, for d's selected-entry report (design §3.3, VAL-6), with what the report carries
// of each (the kernel implementations it is valid for, its fit range, and whether it is
// unidentified or was read outside its fit range).
//
// It also registers the kernel with d as an extrapolation observer, so a report read later
// says which entries a step priced by then read outside their fit range. Extrapolation is a
// fact about the steps a run priced, and nothing has been priced when OpenPool records the
// entries; the simulator does not record again at the end of a run. Records accumulate in d
// (load.Data.RecordEntry), so a simulator that opens one kernel per engine instance or pool
// calls RecordSelected once per kernel, and a flag any kernel raised stays raised. A caller
// that builds a kernel with New from documents it took from d calls RecordSelected itself.
func (k *Kernel) RecordSelected(d *load.Data) {
	extrapolated := extrapolatedEntries(k.trust)
	for i, r := range k.selected {
		d.RecordEntry(r.Set, r.Entry, r.Primitive, r.Running, r.Match, extrapolated[i])
	}
	d.RecordImplementations(k.running)
	d.Observe(k)
}

// Extrapolated implements load.ExtrapolationObserver: whether any step this kernel priced
// read the selected entry outside the range its fit covered.
func (k *Kernel) Extrapolated(set, name string, scope coefficient.Scope, method string, impls []coefficient.KernelImpl) bool {
	extrapolated := extrapolatedEntries(k.trust)
	for i, r := range k.selected {
		if extrapolated[i] && r.Set == set && r.Entry.Name == name && r.Entry.Scope.Key() == scope.Key() &&
			string(r.Entry.Method) == method && coefficient.DescribeImpls(r.Entry.KernelImpls) == coefficient.DescribeImpls(impls) {
			return true
		}
	}
	return false
}

// OpenInputs resolves one pool of a scenario file into the documents New prices it from,
// without building the kernel.
//
// It is OpenPool's loading half, exported for a caller that needs the documents as well as
// the kernel. A scoring harness that configures a rival backend for the same deployment
// needs the model's name, the chip's memory and the model's expert geometry -- deployment
// identity, which the kernel deliberately does not re-export (https://github.com/inference-sim/blis-schemas/issues/36 draws that
// line: identity for comparison is the harness's concern, not the cost model's). Reading
// them from these Inputs, and building the kernel with New(in), means the harness holds the
// SAME chip and graph the kernel priced, not a second load of files that could resolve
// differently.
//
// What it must not be used for is reaching configuration the kernel resolved: widths and
// backends are Resolved's to report, and the pool's settings are Deployment's. Reading
// those from Inputs instead would rebuild the second answer to "which pool is this" that
// the interface exists to remove.
func OpenInputs(scenario string, r Repos, poolIndex int) (Inputs, error) {
	return OpenInputsIn(scenario, r, poolIndex, "")
}

// OpenInputsIn is OpenInputs for an instance placed in the cluster's node pool called
// nodePool (design SIM-17): the chip, and with it the packaging, the memory and every price,
// is the pool's own, read through Cluster.PoolHardware. poolIndex still names the deployment
// pool (a role, such as prefill or decode); the two axes are independent, so a prefill pool
// and a decode pool may each be opened in any node pool.
//
// An empty nodePool is the cluster's own chip, which is what OpenInputs gives and what a
// scenario that declares no cluster.pools has for every pool. A name the cluster's pools do
// not list is refused, naming the pools it does list.
//
// Inputs.Scenario is a copy whose Cluster.Hardware is the node pool's chip, so every reader
// that resolves the cluster against the chip (Packaging, validation) sees one chip. The
// caller's scenario is not changed. A cluster that states gpus_per_node or gpus_per_rack
// must state values the node pool's chip also accepts; Packaging refuses a contradiction.
func OpenInputsIn(scenario string, r Repos, poolIndex int, nodePool string) (Inputs, error) {
	if r.Data == nil {
		return Inputs{}, fmt.Errorf("%s: no data: Repos.Data is unset. Open the catalog and "+
			"registry with load.Embedded or load.Open", scenario)
	}
	sc, dep, err := contracts.LoadBundle(filepath.Join(r.Scenarios, scenario))
	if err != nil {
		return Inputs{}, err
	}
	// Bounds-checked here as well as in New, so the error can name the FILE. New sees
	// documents and reports "outside the deployment's N pool(s)"; a caller that passed a
	// filename needs to know which of several scenarios lacks the pool it asked for.
	if poolIndex < 0 || poolIndex >= len(dep.Pools) {
		return Inputs{}, fmt.Errorf("%s states %d pool(s); pool %d was requested",
			scenario, len(dep.Pools), poolIndex)
	}
	// The documents come from r.Data, which checked them when it was opened; the kernel
	// copies what it keeps (New), so the shared documents are never changed.
	graph, err := r.Data.Model(sc.Model)
	if err != nil {
		return Inputs{}, err
	}
	chipName := sc.Cluster.Hardware
	if nodePool != "" {
		var ok bool
		if chipName, ok = sc.Cluster.PoolHardware(nodePool); !ok {
			names := make([]string, len(sc.Cluster.Pools))
			for i, p := range sc.Cluster.Pools {
				names[i] = p.Name
			}
			return Inputs{}, fmt.Errorf("%s: node pool %q is not a pool of the cluster, which lists %v "+
				"(cluster.pools)", scenario, nodePool, names)
		}
		if len(sc.Cluster.Pools) == 0 {
			// PoolHardware answers any name for a cluster without pools; asking for one by
			// name here is a mistake worth naming rather than silently pricing as the cluster.
			return Inputs{}, fmt.Errorf("%s: node pool %q was requested, but the cluster declares no "+
				"cluster.pools", scenario, nodePool)
		}
	}
	chip, err := r.Data.Chip(chipName)
	if err != nil {
		if nodePool != "" {
			return Inputs{}, fmt.Errorf("node pool %q: %w", nodePool, err)
		}
		return Inputs{}, err
	}
	if chipName != sc.Cluster.Hardware {
		view := *sc
		view.Cluster.Hardware = chipName
		sc = &view
	}
	var fabric *hardware.Fabric
	if sc.Cluster.Fabric != "" {
		if fabric, err = r.Data.Fabric(sc.Cluster.Fabric); err != nil {
			return Inputs{}, err
		}
	}
	sets := make([]*coefficient.Set, 0, len(sc.Coefficients))
	for _, name := range sc.Coefficients {
		set, err := r.Data.CoefficientSet(name)
		if err != nil {
			return Inputs{}, err
		}
		sets = append(sets, set)
	}
	devices := r.Data.StorageDevices()
	if r.Rules == nil {
		return Inputs{}, fmt.Errorf("%s: no engine rules source: Repos.Rules is unset, and the "+
			"kernel looks no rules up itself. Pass the engine model's rules for the scenario's "+
			"engine_version (for vLLM, github.com/inference-sim/blis-vllm/rules)", scenario)
	}
	if sc.Engine != nil {
		have := r.RulesKind
		if have == "" {
			have = legacyRulesKind
		}
		if sc.Engine.Kind != have {
			return Inputs{}, fmt.Errorf("%s: the scenario's engine is kind %q, but Repos.Rules are "+
				"for kind %q (version %q); rules of one engine do not describe another's release",
				scenario, sc.Engine.Kind, have, sc.Engine.Version)
		}
	}
	er := r.Rules(sc.EngineRelease())
	if isNil(er) {
		known := ""
		if r.RulesVersions != nil {
			known = fmt.Sprintf("; known versions are %v", r.RulesVersions())
		}
		return Inputs{}, fmt.Errorf(
			"no engine rules for version %q%s. A layout cannot be resolved without them, "+
				"and a nearby version would misprice whatever changed between the two",
			sc.EngineRelease(), known)
	}
	if r.DraftModel == nil {
		return Inputs{}, fmt.Errorf("%s: no engine model: Repos.DraftModel is unset, and the kernel "+
			"opens no engine block itself. Pass the engine model's reading of the block (for "+
			"vLLM, github.com/inference-sim/blis-vllm/compat/kernelmodel)", scenario)
	}
	name, err := r.DraftModel(dep, poolIndex)
	if err != nil {
		return Inputs{}, fmt.Errorf("%s: pool %d's engine block: %w", scenario, poolIndex, err)
	}
	var draft *model.Graph
	if name != "" {
		if draft, err = r.Data.Model(name); err != nil {
			return Inputs{}, fmt.Errorf("draft model: %w", err)
		}
	}
	in := Inputs{
		Scenario: sc, Deployment: dep, PoolIndex: poolIndex,
		Model: graph, Draft: draft, Chip: chip, Fabric: fabric,
		Devices: devices, Coefficients: sets, Rules: er,
	}
	if r.Lower != nil {
		if in.Lowered, err = r.Lower(in); err != nil {
			return Inputs{}, fmt.Errorf("%s: pool %d: %w", scenario, poolIndex, err)
		}
	}
	return in, nil
}
