package load

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/pointrecord"
	"github.com/inference-sim/blis-substrate/contracts/spec/routertable"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// The readers below return documents shared by every caller of one Data. They are read
// only: a caller that needs to change one copies it first.

// Model returns a model's graph. A model the catalog in use does not have is refused,
// naming the catalog's source and the path its entry belongs at (design §3.3).
func (d *Data) Model(name string) (*model.Graph, error) {
	if g := d.catalog.Models[name]; g != nil {
		return g, nil
	}
	return nil, d.missing(d.Catalog, "model", name, "models/"+name+"/graph.yaml", sortedKeys(d.catalog.Models))
}

// ModelIdentity returns a model's model.yaml.
func (d *Data) ModelIdentity(name string) (*model.Identity, error) {
	if id := d.catalog.Identities[name]; id != nil {
		return id, nil
	}
	return nil, d.missing(d.Catalog, "model", name, "models/"+name+"/model.yaml", sortedKeys(d.catalog.Identities))
}

// Chip returns a chip by its catalog identity.
func (d *Data) Chip(name string) (*hardware.Chip, error) {
	if c := d.catalog.Chips[name]; c != nil {
		return c, nil
	}
	return nil, d.missing(d.Catalog, "hardware", name, "hardware/"+name+".yaml", sortedKeys(d.catalog.Chips))
}

// Fabric returns a fabric by its catalog identity.
func (d *Data) Fabric(name string) (*hardware.Fabric, error) {
	if f := d.catalog.Fabrics[name]; f != nil {
		return f, nil
	}
	return nil, d.missing(d.Catalog, "fabric", name, "networks/"+name+".yaml", sortedKeys(d.catalog.Fabrics))
}

// StorageDevices returns every storage tier, sorted by name.
func (d *Data) StorageDevices() []*hardware.StorageDevice {
	return append([]*hardware.StorageDevice(nil), d.catalog.Devices...)
}

// Workload returns a catalog workload, a shape or a replay.
func (d *Data) Workload(name string) (*workload.CatalogWorkload, error) {
	if w := d.catalog.Workloads[name]; w != nil {
		return w, nil
	}
	return nil, d.missing(d.Catalog, "workload", name, "workloads/"+name+".yaml", sortedKeys(d.catalog.Workloads))
}

// Router returns a router-table entry by its name, the file stem under routers/.
func (d *Data) Router(name string) (*routertable.Entry, error) {
	if e := d.catalog.Routers[name]; e != nil {
		return e, nil
	}
	return nil, d.missing(d.Catalog, "router entry", name, "routers/"+name+".yaml", sortedKeys(d.catalog.Routers))
}

// Routers lists the router table's entries, sorted by name.
func (d *Data) Routers() []*routertable.Entry {
	var out []*routertable.Entry
	for _, name := range sortedKeys(d.catalog.Routers) {
		out = append(out, d.catalog.Routers[name])
	}
	return out
}

// LookupRouter finds the entry for a router, its mode and its session-affinity setting
// (nil when the run configured none): the entry that says which policy of llm-d's router
// stands in for it, or that none does, on the router build the run used (revision, a full
// commit). The outcome is routertable.OutcomeMapped only when an entry has the key and covers
// the revision; for any other outcome the point stays router_unsupported (G5). The entry is
// returned whenever one has the key, covered or not.
func (d *Data) LookupRouter(router, mode string, affinityTTLS *int, revision string) (*routertable.Entry, routertable.Outcome) {
	return routertable.Resolve(d.Routers(), router, mode, affinityTTLS, revision)
}

// Models lists the catalog's models, sorted.
func (d *Data) Models() []string { return sortedKeys(d.catalog.Models) }

// Chips lists the catalog's chips, sorted.
func (d *Data) Chips() []string { return sortedKeys(d.catalog.Chips) }

// CoefficientSet returns a coefficient set by name and records it as loaded.
func (d *Data) CoefficientSet(name string) (*coefficient.Set, error) {
	s := d.registry.Sets[name]
	if s == nil {
		return nil, d.missing(d.Registry, "coefficient set", name, "coefficients/"+name+".yaml", sortedKeys(d.registry.Sets))
	}
	d.markLoaded("coefficients/" + name + ".yaml")
	return s, nil
}

// AcceptanceSet returns an acceptance-profile set by name and records it as loaded.
func (d *Data) AcceptanceSet(name string) (*acceptance.ProfileSet, error) {
	s := d.registry.Acceptance[name]
	if s == nil {
		return nil, d.missing(d.Registry, "acceptance-profile set", name, "acceptance/"+name+".yaml", sortedKeys(d.registry.Acceptance))
	}
	d.markLoaded("acceptance/" + name + ".yaml")
	return s, nil
}

// AcceptanceSets returns every acceptance-profile set, in name order, and records each as
// loaded: the registry layer of acceptance.Resolve consults them all.
func (d *Data) AcceptanceSets() []*acceptance.ProfileSet {
	var out []*acceptance.ProfileSet
	for _, name := range sortedKeys(d.registry.Acceptance) {
		d.markLoaded("acceptance/" + name + ".yaml")
		out = append(out, d.registry.Acceptance[name])
	}
	return out
}

// RoutingSet returns a routing-profile set by name and records it as loaded.
func (d *Data) RoutingSet(name string) (*routing.ProfileSet, error) {
	s := d.registry.Routing[name]
	if s == nil {
		return nil, d.missing(d.Registry, "routing-profile set", name, "routing/"+name+".yaml", sortedKeys(d.registry.Routing))
	}
	d.markLoaded("routing/" + name + ".yaml")
	return s, nil
}

// RoutingSets returns every routing-profile set, in name order, and records each as loaded.
func (d *Data) RoutingSets() []*routing.ProfileSet {
	var out []*routing.ProfileSet
	for _, name := range sortedKeys(d.registry.Routing) {
		d.markLoaded("routing/" + name + ".yaml")
		out = append(out, d.registry.Routing[name])
	}
	return out
}

func (d *Data) missing(p Provenance, what, name, rel string, have []string) error {
	src := "the embedded release"
	if p.Source == SourcePath {
		src = p.Path
	}
	if len(p.Overlays) > 0 {
		src += " with overlays"
		for _, o := range p.Overlays {
			src += " " + o.Path
		}
	}
	return fmt.Errorf("%s %q is not in the data in use (%s); its entry belongs at %s, and the data has %v",
		what, name, src, rel, have)
}

func (d *Data) markLoaded(rel string) {
	d.mu.Lock()
	d.loaded[rel] = true
	d.mu.Unlock()
}

// CheckRouterSubstitutions refuses a point record whose router_differs substitution the router
// table does not support (design §11.3, §11.4): the entry it names must exist, belong to the
// router the point names, cover the build of the router the point says the run used, and map to
// the policy the point says BLIS simulated, and carry the entry's differences and bound. A router mapped for one build is not mapped for
// another, so a record that relies on an entry at a build it does not list is refused.
func (d *Data) CheckRouterSubstitutions(r *pointrecord.Record) error {
	var bad []string
	for i, s := range r.Substitutions {
		if s.Kind != pointrecord.SubRouterDiffers {
			continue
		}
		at := fmt.Sprintf("substitutions[%d]", i)
		e, err := d.Router(s.Entry)
		switch {
		case err != nil:
			bad = append(bad, fmt.Sprintf("%s: the router table has no entry %q", at, s.Entry))
		case e.Router != s.Subject:
			bad = append(bad, fmt.Sprintf("%s: entry %q is of router %q, not %q", at, s.Entry, e.Router, s.Subject))
		case !e.Covers(s.Revision):
			bad = append(bad, fmt.Sprintf("%s: entry %q does not cover revision %s of %s: what the table shows for another build is not inherited, and the point is router_unsupported (G5)", at, s.Entry, s.Revision, s.Subject))
		case string(e.Policy) != s.Policy:
			bad = append(bad, fmt.Sprintf("%s: entry %q maps to policy %q, not %q", at, s.Entry, e.Policy, s.Policy))
		case !reflect.DeepEqual(s.Difference, e.Statements()):
			bad = append(bad, fmt.Sprintf("%s: the differences are not entry %q's: a point carries the entry's statements as they stand, so a reader sees what the mapping rests on", at, s.Entry))
		case e.Bound == nil && s.Bound != nil:
			bad = append(bad, fmt.Sprintf("%s: the bound is not the part of entry %q's bound that covers the point's regime: the entry states none", at, s.Entry))
		case e.Bound == nil:
			// An entry without a bound is checked by the validator, which requires one of a mapping.
		case s.Regime == nil:
			bad = append(bad, fmt.Sprintf("%s: no regime: a point carries the part of entry %q's bound that covers its own regime, which cannot be checked without it", at, s.Entry))
		case e.Bound.For(*s.Regime) == nil:
			bad = append(bad, fmt.Sprintf("%s: entry %q has no bound for this regime: the point is router_unsupported (G5), not router_differs", at, s.Entry))
		case !reflect.DeepEqual(s.Bound, e.Bound.For(*s.Regime)):
			bad = append(bad, fmt.Sprintf("%s: the bound is not the part of entry %q's bound that covers the point's regime: a point carries exactly the entry's ratios whose regime contains it, as they stand, and none for another regime", at, s.Entry))
		}
	}
	if len(bad) > 0 {
		return fmt.Errorf("%s", strings.Join(bad, "; "))
	}
	return nil
}
