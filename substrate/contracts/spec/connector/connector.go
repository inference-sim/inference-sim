// Package connector describes the KV-connector registry: which engine connector names a
// deployment may use, which engine interface each implements, and which engine model in
// an engine module reproduces its behaviour.
//
// A connector name in a deployment (offload.connector, pd_transfer.connector) is looked up
// here instead of in a list compiled into a simulator, so a new connector that behaves like
// an existing one is data, not code. The rule that keeps the registry honest
// (inference-sim/inference-sim#2021 B10): implementing an engine's connector interface is not
// the same as behaving like one of its connectors. An entry names the model it behaves like
// and the evidence that earned it, which for a built-in connector is a differential test
// against the engine release. A connector with no such evidence is not registered.
//
// This package holds the shape only. The entries live with the engine model that owns the
// behaviour (the engine module that owns the engine model).
package connector

import (
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// Upstream names the engine interface a connector implements, as the engine names it. Which
// interfaces exist is the engine module's to say, and it checks the name.
type Upstream string

// Role is what a deployment uses a connector for.
type Role string

const (
	// RoleOffload: moving KV between the device and an offload tier (offload.connector).
	RoleOffload Role = "offload"
	// RolePDTransfer: moving KV from a prefill engine to a decode engine
	// (pd_transfer.connector).
	RolePDTransfer Role = "pd_transfer"
	// RoleComposite: composing other connectors, as the engine's MultiConnector does.
	RoleComposite Role = "composite"
)

var roles = map[Role]bool{RoleOffload: true, RolePDTransfer: true, RoleComposite: true}

// Valid reports whether r is a recognized role.
func (r Role) Valid() bool { return roles[r] }

// Registry is a set of connector entries, by the name a deployment uses.
type Registry struct {
	Kind string `yaml:"kind"` // "ConnectorRegistry"
	Name string `yaml:"name"`
	// Connectors maps a connector name, exactly as the engine registers it (its
	// KVConnectorFactory.register_connector), to its entry.
	Connectors map[string]Entry `yaml:"connectors"`
}

// Entry is one connector.
type Entry struct {
	// Upstream is the engine interface the connector implements.
	Upstream Upstream `yaml:"upstream"`
	// BehavesLike names the engine model that reproduces the connector's behaviour, in the
	// engine module that owns the registry.
	BehavesLike string `yaml:"behaves_like"`
	// Roles are what a deployment may use the connector for.
	Roles []Role `yaml:"roles"`
	// Evidence is how the entry earned BehavesLike, such as a differential test against the engine release.
	Evidence string `yaml:"evidence"`
}

// Lookup returns the entry for a connector name.
func (r *Registry) Lookup(name string) (Entry, bool) {
	e, ok := r.Connectors[name]
	return e, ok
}

// Serves reports whether the named connector is registered for a role.
func (r *Registry) Serves(name string, role Role) bool {
	e, ok := r.Lookup(name)
	if !ok {
		return false
	}
	for _, x := range e.Roles {
		if x == role {
			return true
		}
	}
	return false
}

// Names returns every registered connector name, sorted.
func (r *Registry) Names() []string {
	out := make([]string, 0, len(r.Connectors))
	for n := range r.Connectors {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Validate performs field-level validation of a registry.
func (r *Registry) Validate() *validate.Problems {
	p := &validate.Problems{}
	if r.Kind != "ConnectorRegistry" {
		p.Field("kind", "must be ConnectorRegistry, got %q", r.Kind)
	}
	if strings.TrimSpace(r.Name) == "" {
		p.Field("name", "required")
	}
	if len(r.Connectors) == 0 {
		p.Field("connectors", "a registry with no connectors registers nothing")
	}
	for _, name := range r.Names() {
		e := r.Connectors[name]
		if strings.TrimSpace(name) == "" {
			p.Field("connectors", "a connector name must not be blank")
			continue
		}
		// A name is the engine's own, matched exactly, so a padded one would never match.
		if name != strings.TrimSpace(name) {
			p.Field("connectors", "connector name %q has surrounding spaces; Lookup matches the engine's name exactly", name)
			continue
		}
		at := "connectors." + name
		if strings.TrimSpace(string(e.Upstream)) == "" {
			p.Field(at+".upstream", "required: the engine interface the connector implements")
		}
		if strings.TrimSpace(e.BehavesLike) == "" {
			p.Field(at+".behaves_like", "required: the engine model that reproduces this connector")
		}
		if len(e.Roles) == 0 {
			p.Field(at+".roles", "required: what a deployment may use the connector for")
		}
		seen := map[Role]bool{}
		for _, role := range e.Roles {
			switch {
			case !role.Valid():
				p.Field(at+".roles", "%q is not one of offload, pd_transfer, composite", role)
			case seen[role]:
				p.Field(at+".roles", "%q is listed twice", role)
			}
			seen[role] = true
		}
		if strings.TrimSpace(e.Evidence) == "" {
			p.Field(at+".evidence", "required: an entry earns behaves_like only by evidence, such as a differential test against the engine release")
		}
	}
	return p
}
