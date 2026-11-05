package contracts

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/connector"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/evaluation"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// Loading is separate from validation on purpose. A parse failure and a validation
// failure need different responses — the first is a malformed file, the second a
// well-formed document that says something wrong — and a combined call would report
// them through one channel.
//
// Every loader rejects unknown fields. A misspelled key that parsed silently would
// leave a document that validates while omitting the setting its author intended,
// which is the failure mode hardest to notice: the document looks right and the
// estimate is wrong for a reason nothing reports.

// LoadScenario reads a scenario document. It does not validate; call Validate.
func LoadScenario(path string) (*scenario.Scenario, error) {
	var s scenario.Scenario
	if err := decodeStrict(path, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// LoadBundle reads a scenario and the deployment applied to it from one file, as two YAML
// documents separated by `---`, with the same strict decoding as LoadScenario and
// LoadDeployment. A measurement row names a deployment by a single filename, which is why
// the pair shares a file.
//
// It does not run Validate; call that for the documents' rules. It does refuse, naming the
// file, a shape no caller can use: an empty file, a file with no deployment document, and a
// deployment with no pools. Every caller reads at least one pool,
// several as Pools[0] without a length check of their own, so refusing a pool-less
// deployment here turns what would be an index panic far from the file into an error that
// names it.
func LoadBundle(path string) (*scenario.Scenario, *deployment.Deployment, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("opening %s: %w", path, err)
	}
	defer f.Close()

	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)

	var sc scenario.Scenario
	if err := dec.Decode(&sc); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, nil, fmt.Errorf("%s is empty: a kernel needs a scenario and the "+
				"deployment applied to it", path)
		}
		return nil, nil, fmt.Errorf("decoding the scenario in %s: %w", path, err)
	}

	var dep deployment.Deployment
	if err := dec.Decode(&dep); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, nil, fmt.Errorf("%s holds a scenario but no deployment: the pools "+
				"a kernel prices live in a second YAML document, separated by `---`", path)
		}
		return nil, nil, fmt.Errorf("decoding the deployment in %s: %w", path, err)
	}

	if len(dep.Pools) == 0 {
		return nil, nil, fmt.Errorf(
			"the deployment in %s declares no pools: there is nothing to price", path)
	}

	return &sc, &dep, nil
}

// LoadDeployment reads a deployment document — the tunable configuration applied to a
// scenario. It does not validate; call Validate.
func LoadDeployment(path string) (*deployment.Deployment, error) {
	var d deployment.Deployment
	if err := decodeStrict(path, &d); err != nil {
		return nil, err
	}
	return &d, nil
}

// LoadModelGraph reads a derived model graph.
func LoadModelGraph(path string) (*model.Graph, error) {
	var g model.Graph
	if err := decodeStrict(path, &g); err != nil {
		return nil, err
	}
	return &g, nil
}

// LoadModelIdentity reads a model entry's identity manifest, blis-catalog's
// models/<name>/model.yaml: the name the entry claims and the provenance of the vendor
// config beside it. Unlike a chip, a fabric or a workload, this file carries its name IN
// the document — the catalog requires that name to match the directory — so the loader
// does not stamp it; the caller checks it against the directory, as it does for a graph.
func LoadModelIdentity(path string) (*model.Identity, error) {
	var id model.Identity
	if err := decodeStrict(path, &id); err != nil {
		return nil, err
	}
	return &id, nil
}

// LoadChip reads a chip descriptor from blis-catalog's hardware namespace. The file
// carries no name — identity is the filename — so the loader stamps Chip.Name from the
// path stem. The Chip type tags Name `yaml:"-"`, so the stamp is the field's only source:
// there is no in-file name to prefer, hence no `if Name == ""` guard, and an in-file
// `name` is rejected under strict decode (#27).
func LoadChip(path string) (*hardware.Chip, error) {
	var c hardware.Chip
	if err := decodeStrict(path, &c); err != nil {
		return nil, err
	}
	c.Name = stem(path)
	return &c, nil
}

// LoadFabric reads a fabric descriptor from blis-catalog's networks namespace, naming it
// from the filename as LoadChip does: the stamp is Fabric.Name's only source, and an
// in-file `name` is a rejected unknown field (#27).
func LoadFabric(path string) (*hardware.Fabric, error) {
	var f hardware.Fabric
	if err := decodeStrict(path, &f); err != nil {
		return nil, err
	}
	f.Name = stem(path)
	return &f, nil
}

// LoadCoefficientSet reads a coefficient set from blis-registry.
func LoadCoefficientSet(path string) (*coefficient.Set, error) {
	var s coefficient.Set
	if err := decodeStrict(path, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// LoadConnectorRegistry reads a KV-connector registry. It does not validate; call
// Validate on the result.
func LoadConnectorRegistry(path string) (*connector.Registry, error) {
	var r connector.Registry
	if err := decodeStrict(path, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

// LoadAcceptanceProfileSet reads an acceptance-profile set from blis-registry's
// acceptance namespace. It does not validate; call Validate.
func LoadAcceptanceProfileSet(path string) (*acceptance.ProfileSet, error) {
	var s acceptance.ProfileSet
	if err := decodeStrict(path, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// LoadRoutingProfileSet reads a routing-profile set from blis-registry's routing
// namespace. It does not validate; call Validate.
func LoadRoutingProfileSet(path string) (*routing.ProfileSet, error) {
	var s routing.ProfileSet
	if err := decodeStrict(path, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// LoadWorkload reads a traffic shape from blis-catalog's workloads namespace. The file
// carries no name — identity is the filename — so the loader stamps Shape.Name from the
// path stem. The Shape type tags Name `yaml:"-"`, so the stamp is the field's only
// source: there is no in-file name to prefer, hence no `if Name == ""` guard, and an
// in-file `name` is rejected under strict decode. Chip and Fabric now follow this same
// rule (#27), so the three by-filename entities are single-sourced identically.
//
// A replay workload (a file whose top-level key is replay:) is not a shape; it is refused
// here with a pointer to LoadCatalogWorkload, which reads either form.
func LoadWorkload(path string) (*workload.Shape, error) {
	w, err := LoadCatalogWorkload(path)
	if err != nil {
		return nil, err
	}
	if w.Shape == nil {
		return nil, fmt.Errorf("%s is a replay workload, not a shape; load it with LoadCatalogWorkload", path)
	}
	return w.Shape, nil
}

// LoadCatalogWorkload reads one catalog workloads/<name>.yaml in either form: a traffic
// shape, or a replay workload when the file's top-level key is replay:. Both are named
// from the path stem, and both decode strictly. A replay file may carry the catalog's
// `_comment` keys at its top level, as a shape may.
func LoadCatalogWorkload(path string) (*workload.CatalogWorkload, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(b, &doc); err != nil {
		return nil, fmt.Errorf("decoding %s: %w", path, err)
	}
	if !topLevelKey(&doc, "replay") {
		var s workload.Shape
		if err := decodeStrict(path, &s); err != nil {
			return nil, err
		}
		s.Name = stem(path)
		return &workload.CatalogWorkload{Shape: &s}, nil
	}
	top := doc.Content[0]
	if err := validate.RejectUnknownKeys(top, catalogReplayFile{}, validate.IgnoreCatalogComments); err != nil {
		return nil, fmt.Errorf("decoding %s: %w", path, err)
	}
	validate.StripCatalogComments(top)
	// The comment keys are gone; decode what remains with the strict decoder, so a typo
	// nested anywhere in the replay is still an unknown field.
	clean, err := yaml.Marshal(top)
	if err != nil {
		return nil, fmt.Errorf("decoding %s: %w", path, err)
	}
	var f catalogReplayFile
	dec := yaml.NewDecoder(bytes.NewReader(clean))
	dec.KnownFields(true)
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("decoding %s: %w", path, err)
	}
	if f.Replay == nil {
		return nil, fmt.Errorf("decoding %s: replay: is empty", path)
	}
	return &workload.CatalogWorkload{Replay: &workload.CatalogReplay{
		Name: stem(path), ReplayDefinition: *f.Replay}}, nil
}

// catalogReplayFile is the catalog replay file form.
type catalogReplayFile struct {
	Replay *workload.ReplayDefinition `yaml:"replay"`
}

// topLevelKey reports whether a document's top-level mapping has key.
func topLevelKey(doc *yaml.Node, key string) bool {
	if doc.Kind != yaml.DocumentNode || len(doc.Content) == 0 || doc.Content[0].Kind != yaml.MappingNode {
		return false
	}
	m := doc.Content[0]
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == key {
			return true
		}
	}
	return false
}

// LoadEvaluationRun reads a measured run.
func LoadEvaluationRun(path string) (*evaluation.Run, error) {
	var r evaluation.Run
	if err := decodeStrict(path, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

// LoadMeasuredRun reads a measured run from blis-catalog's measurements/<source>/. It
// does not validate; call ValidateWithSources with the registry beside it.
func LoadMeasuredRun(path string) (*measurement.MeasuredRun, error) {
	var r measurement.MeasuredRun
	if err := decodeStrict(path, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

// LoadResult reads a Result document, JSON or YAML (result.Unmarshal), refusing an
// unknown key in either.
func LoadResult(path string) (*result.Result, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	var r result.Result
	if err := result.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("decoding %s: %w", path, err)
	}
	return &r, nil
}

// LoadMeasurementSources reads the licence registry, blis-catalog's
// measurements/sources.yaml.
func LoadMeasurementSources(path string) (*measurement.Registry, error) {
	var r measurement.Registry
	if err := decodeStrict(path, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

// LoadStorageDevices reads the offload-tier classes from blis-catalog's devices
// namespace.
//
// The file is a mapping of tier name to device facts rather than a list, because a tier
// is referred to by name from a deployment's offload block. So this loader is not a plain
// decodeStrict: it walks the mapping and stamps each device's Name from its key, which
// is the only place that key survives into the loaded value.
//
// Sorted by name on return, so two loads of one file yield the same order and a caller
// comparing kernels built from it is comparing configuration rather than map iteration.
func LoadStorageDevices(path string) ([]*hardware.StorageDevice, error) {
	var byName map[string]*hardware.StorageDevice
	if err := decodeStrict(path, &byName); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(byName))
	for name := range byName {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]*hardware.StorageDevice, 0, len(names))
	for _, name := range names {
		d := byName[name]
		if d == nil {
			return nil, fmt.Errorf("%s: tier %q has no device facts", path, name)
		}
		// The key is the tier's identity; a name inside the value would be a second
		// source of truth for it, so the type declares none and it is set here.
		d.Name = name
		out = append(out, d)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: declares no storage tier", path)
	}
	return out, nil
}

// decodeStrict reads one YAML document into dst, rejecting any field the target type
// does not declare. The error names the file, because a caller loading a directory
// needs to know which one failed.
func decodeStrict(path string, dst any) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("opening %s: %w", path, err)
	}
	defer f.Close()

	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)
	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("decoding %s: %w", path, err)
	}
	return nil
}

// stem returns a file's base name without its extension, which is how the catalog
// names a chip, a fabric or a workload.
func stem(path string) string {
	base := filepath.Base(path)
	return strings.TrimSuffix(base, filepath.Ext(base))
}
