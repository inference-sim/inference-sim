// Package dataset holds the checks of catalog and registry data files that are shared by
// the validator commands and by the loader (package load), so a file is judged the same way
// in CI and when a run loads it (design §3.3, SUB-9, X15).
//
// Each Check function takes one file's path relative to its catalog or registry directory
// and its bytes, parses it with the strict contracts loader, and runs its validators. A
// parse failure is the returned error; a document that parses and is wrong is reported in
// the returned problems.
package dataset

import (
	"fmt"
	"path"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// CheckModelGraph checks models/<name>/graph.yaml: the graph validates, and it names itself
// what its directory is called.
func CheckModelGraph(rel string, b []byte) (*model.Graph, *validate.Problems, error) {
	g, err := contracts.ParseModelGraph(rel, b)
	if err != nil {
		return nil, nil, err
	}
	p := g.Validate()
	if dir := path.Base(path.Dir(rel)); g.Name != dir {
		p.Field("name", "graph names itself %q but its directory is %q", g.Name, dir)
	}
	return g, p, nil
}

// CheckModelIdentity checks models/<name>/model.yaml: the identity validates, and it names
// itself what its directory is called.
func CheckModelIdentity(rel string, b []byte) (*model.Identity, *validate.Problems, error) {
	id, err := contracts.ParseModelIdentity(rel, b)
	if err != nil {
		return nil, nil, err
	}
	p := id.Validate()
	if dir := path.Base(path.Dir(rel)); id.Name != dir {
		p.Field("name", "model.yaml names itself %q but its directory is %q", id.Name, dir)
	}
	return id, p, nil
}

// CheckChip checks hardware/<name>.yaml.
func CheckChip(rel string, b []byte) (*hardware.Chip, *validate.Problems, error) {
	c, err := contracts.ParseChip(rel, b)
	if err != nil {
		return nil, nil, err
	}
	return c, c.Validate(), nil
}

// CheckFabric checks networks/<name>.yaml.
func CheckFabric(rel string, b []byte) (*hardware.Fabric, *validate.Problems, error) {
	f, err := contracts.ParseFabric(rel, b)
	if err != nil {
		return nil, nil, err
	}
	return f, f.Validate(), nil
}

// CheckStorageDevices checks devices/<file>.yaml, each tier's problems under its name.
func CheckStorageDevices(rel string, b []byte) ([]*hardware.StorageDevice, *validate.Problems, error) {
	ds, err := contracts.ParseStorageDevices(rel, b)
	if err != nil {
		return nil, nil, err
	}
	p := &validate.Problems{}
	for _, d := range ds {
		p.Merge(d.Name, d.Validate())
	}
	return ds, p, nil
}

// CheckWorkload checks workloads/<name>.yaml, a shape or a replay.
func CheckWorkload(rel string, b []byte) (*workload.CatalogWorkload, *validate.Problems, error) {
	w, err := contracts.ParseCatalogWorkload(rel, b)
	if err != nil {
		return nil, nil, err
	}
	return w, w.Validate(), nil
}

// CheckMeasurementSources checks measurements/sources.yaml.
func CheckMeasurementSources(rel string, b []byte) (*measurement.Registry, *validate.Problems, error) {
	r, err := contracts.ParseMeasurementSources(rel, b)
	if err != nil {
		return nil, nil, err
	}
	return r, r.Validate(), nil
}

// CheckCoefficientSet checks coefficients/<name>.yaml. A scenario names a set by its file
// stem, so the set must name itself that too: otherwise a set's identity in an overlay,
// in provenance and in the scenario would be three different strings.
func CheckCoefficientSet(rel string, b []byte) (*coefficient.Set, *validate.Problems, error) {
	s, err := contracts.ParseCoefficientSet(rel, b)
	if err != nil {
		return nil, nil, err
	}
	p := s.Validate()
	if stem := Stem(rel); s.Name != stem {
		p.Field("name", "the set names itself %q but its file is %s", s.Name, path.Base(rel))
	}
	return s, p, nil
}

// CheckAcceptanceSet checks acceptance/<name>.yaml, with the same name rule.
func CheckAcceptanceSet(rel string, b []byte) (*acceptance.ProfileSet, *validate.Problems, error) {
	s, err := contracts.ParseAcceptanceProfileSet(rel, b)
	if err != nil {
		return nil, nil, err
	}
	p := s.Validate()
	if stem := Stem(rel); s.Name != stem {
		p.Field("name", "the set names itself %q but its file is %s", s.Name, path.Base(rel))
	}
	return s, p, nil
}

// CheckRoutingSet checks routing/<name>.yaml, with the same name rule.
func CheckRoutingSet(rel string, b []byte) (*routing.ProfileSet, *validate.Problems, error) {
	s, err := contracts.ParseRoutingProfileSet(rel, b)
	if err != nil {
		return nil, nil, err
	}
	p := s.Validate()
	if stem := Stem(rel); s.Name != stem {
		p.Field("name", "the set names itself %q but its file is %s", s.Name, path.Base(rel))
	}
	return s, p, nil
}

// Stem is a file's base name without its extension: the identity of a chip, a fabric, a
// workload or a set.
func Stem(rel string) string {
	base := path.Base(rel)
	return strings.TrimSuffix(base, path.Ext(base))
}

// Catalog is a catalog's parsed data, keyed by identity.
type Catalog struct {
	Models     map[string]*model.Graph
	Identities map[string]*model.Identity
	Chips      map[string]*hardware.Chip
	Fabrics    map[string]*hardware.Fabric
	Devices    []*hardware.StorageDevice
	Workloads  map[string]*workload.CatalogWorkload
	Sources    *measurement.Registry
}

// Registry is a registry's parsed data, keyed by file stem.
type Registry struct {
	Sets       map[string]*coefficient.Set
	Acceptance map[string]*acceptance.ProfileSet
	Routing    map[string]*routing.ProfileSet
}

// Problem is one finding against one file.
type Problem struct {
	File    string
	Message string
}

func (p Problem) String() string { return p.File + ": " + p.Message }

// collect records a parse error or every error-severity problem against file, and reports
// whether the file is usable.
func collect(out *[]Problem, file string, p *validate.Problems, err error) bool {
	if err != nil {
		*out = append(*out, Problem{file, err.Error()})
		return false
	}
	for _, e := range p.Errors() {
		*out = append(*out, Problem{file, e.String()})
	}
	return p.OK()
}

// CheckCatalog parses and checks every file of a catalog, given as paths relative to the
// catalog directory. The caller decides which files are data; a path this function does
// not recognize is reported.
func CheckCatalog(files map[string][]byte) (*Catalog, []Problem) {
	c := &Catalog{
		Models: map[string]*model.Graph{}, Identities: map[string]*model.Identity{},
		Chips: map[string]*hardware.Chip{}, Fabrics: map[string]*hardware.Fabric{},
		Workloads: map[string]*workload.CatalogWorkload{},
	}
	var out []Problem
	for _, rel := range sortedKeys(files) {
		b := files[rel]
		dir, _, _ := strings.Cut(rel, "/")
		switch {
		case dir == "models" && path.Base(rel) == "graph.yaml":
			g, p, err := CheckModelGraph(rel, b)
			if collect(&out, rel, p, err) {
				c.Models[path.Base(path.Dir(rel))] = g
			}
		case dir == "models" && path.Base(rel) == "model.yaml":
			id, p, err := CheckModelIdentity(rel, b)
			if collect(&out, rel, p, err) {
				c.Identities[path.Base(path.Dir(rel))] = id
			}
		case dir == "hardware":
			v, p, err := CheckChip(rel, b)
			if collect(&out, rel, p, err) {
				c.Chips[v.Name] = v
			}
		case dir == "networks":
			v, p, err := CheckFabric(rel, b)
			if collect(&out, rel, p, err) {
				c.Fabrics[v.Name] = v
			}
		case dir == "devices":
			v, p, err := CheckStorageDevices(rel, b)
			if collect(&out, rel, p, err) {
				c.Devices = append(c.Devices, v...)
			}
		case dir == "workloads":
			v, p, err := CheckWorkload(rel, b)
			if collect(&out, rel, p, err) {
				c.Workloads[v.Name()] = v
			}
		case rel == "measurements/sources.yaml":
			v, p, err := CheckMeasurementSources(rel, b)
			if collect(&out, rel, p, err) {
				c.Sources = v
			}
		case strings.HasPrefix(rel, "ledger/published/"):
			// The ledger's published role lists. Their checks belong to the ledger's own
			// package, which this loader calls once the ledger defines them.
		default:
			out = append(out, Problem{rel, "is not a kind of catalog file this validator knows"})
		}
	}
	// An entry needs both halves: the graph the runtime prices and the identity that
	// records where its vendor configuration came from.
	for name := range c.Models {
		if _, ok := c.Identities[name]; !ok && !failed(out, "models/"+name+"/model.yaml") {
			out = append(out, Problem{"models/" + name + "/", "has a graph.yaml but no model.yaml"})
		}
	}
	for name := range c.Identities {
		if _, ok := c.Models[name]; !ok && !failed(out, "models/"+name+"/graph.yaml") {
			out = append(out, Problem{"models/" + name + "/", "has a model.yaml but no graph.yaml"})
		}
	}
	seen := map[string]bool{}
	for _, d := range c.Devices {
		if seen[d.Name] {
			out = append(out, Problem{"devices/", fmt.Sprintf("storage tier %q is defined in two files", d.Name)})
		}
		seen[d.Name] = true
	}
	sortProblems(out)
	return c, out
}

// CheckRegistry parses and checks every file of a registry, and then the rules that span
// files: VAL-4's rule that no two entries a lookup could both return are left to load order.
func CheckRegistry(files map[string][]byte) (*Registry, []Problem) {
	r := &Registry{Sets: map[string]*coefficient.Set{}, Acceptance: map[string]*acceptance.ProfileSet{},
		Routing: map[string]*routing.ProfileSet{}}
	var out []Problem
	for _, rel := range sortedKeys(files) {
		b := files[rel]
		dir, _, _ := strings.Cut(rel, "/")
		switch dir {
		case "coefficients":
			v, p, err := CheckCoefficientSet(rel, b)
			if collect(&out, rel, p, err) {
				r.Sets[Stem(rel)] = v
			}
		case "acceptance":
			v, p, err := CheckAcceptanceSet(rel, b)
			if collect(&out, rel, p, err) {
				r.Acceptance[Stem(rel)] = v
			}
		case "routing":
			v, p, err := CheckRoutingSet(rel, b)
			if collect(&out, rel, p, err) {
				r.Routing[Stem(rel)] = v
			}
		default:
			out = append(out, Problem{rel, "is not a kind of registry file this validator knows"})
		}
	}
	out = append(out, CheckOrdered(r.Sets)...)
	sortProblems(out)
	return r, out
}

func failed(ps []Problem, file string) bool {
	for _, p := range ps {
		if p.File == file {
			return true
		}
	}
	return false
}
