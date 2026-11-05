package latencykernel

import (
	"fmt"
	"path/filepath"
	"reflect"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
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
	// RulesVersions lists the releases Rules answers for, for the error a scenario naming
	// another one gets. Optional.
	RulesVersions func() []string
	// EngineSettings decodes a pool's engine block into the settings the kernel prices, the
	// engine model's reading of a block the kernel does not open itself (for vLLM,
	// blis-latency-kernel/vllmsettings.OfPool). Required, as Rules is.
	EngineSettings func(dep *deployment.Deployment, pool int) (enginesettings.Settings, error)
}

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
	in, err := OpenInputs(scenario, r, poolIndex)
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
// was built, for d's selected-entry report (design §3.3, VAL-6). OpenPool calls it; a caller
// that builds a kernel with New from documents it took from d calls it itself.
func (k *Kernel) RecordSelected(d *load.Data) {
	for _, r := range k.selected {
		d.RecordCoefficient(r.Set, r.Entry.Name, r.Entry.Scope.Describe(), string(r.Entry.Method))
	}
}

// OpenInputs resolves one pool of a scenario file into the documents New prices it from,
// without building the kernel.
//
// It is OpenPool's loading half, exported for a caller that needs the documents as well as
// the kernel. A scoring harness that configures a rival backend for the same deployment
// needs the model's name, the chip's memory and the model's expert geometry -- deployment
// identity, which the kernel deliberately does not re-export (blis-schemas#36 draws that
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
	if r.Data == nil {
		return Inputs{}, fmt.Errorf("%s: no data: Repos.Data is unset. Open the catalog and "+
			"registry with load.Embedded or load.Open", scenario)
	}
	sc, dep, err := LoadBundle(filepath.Join(r.Scenarios, scenario))
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
	chip, err := r.Data.Chip(sc.Cluster.Hardware)
	if err != nil {
		return Inputs{}, err
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
	er := r.Rules(sc.EngineVersion)
	if isNil(er) {
		known := ""
		if r.RulesVersions != nil {
			known = fmt.Sprintf("; known versions are %v", r.RulesVersions())
		}
		return Inputs{}, fmt.Errorf(
			"no engine rules for version %q%s. A layout cannot be resolved without them, "+
				"and a nearby version would misprice whatever changed between the two",
			sc.EngineVersion, known)
	}
	if r.EngineSettings == nil {
		return Inputs{}, fmt.Errorf("%s: no engine settings source: Repos.EngineSettings is unset, "+
			"and the kernel opens no engine block itself. Pass the engine model's decoding (for "+
			"vLLM, blis-latency-kernel/vllmsettings.OfPool)", scenario)
	}
	eng, err := r.EngineSettings(dep, poolIndex)
	if err != nil {
		return Inputs{}, fmt.Errorf("%s: pool %d's engine block: %w", scenario, poolIndex, err)
	}
	var draft *model.Graph
	if name := draftModelName(eng); name != "" {
		if draft, err = r.Data.Model(name); err != nil {
			return Inputs{}, fmt.Errorf("draft model: %w", err)
		}
	}
	return Inputs{
		Scenario: sc, Deployment: dep, PoolIndex: poolIndex,
		Model: graph, Draft: draft, Chip: chip, Fabric: fabric,
		Devices: devices, Coefficients: sets, Rules: er, Engine: &eng,
	}, nil
}

// LoadBundle reads a scenario and the deployment applied to it from one file, as two YAML
// documents separated by `---`.
//
// Deprecated: use blis-schemas' LoadBundle, which this calls. It moved there because it is
// format loading, not pricing (inference-sim#2060).
func LoadBundle(path string) (*scenario.Scenario, *deployment.Deployment, error) {
	return contracts.LoadBundle(path)
}
