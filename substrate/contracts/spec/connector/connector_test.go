package connector

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func valid() *Registry {
	return &Registry{Kind: "ConnectorRegistry", Name: "builtin", Connectors: map[string]Entry{
		"OffloadingConnector": {
			Upstream: "KVConnectorBase_V1", BehavesLike: "offloading",
			Roles: []Role{RoleOffload}, Evidence: "diff-test @ vLLM v0.32.0",
		},
		"NixlConnector": {
			Upstream: "KVConnectorBase_V1", BehavesLike: "nixl-pull",
			Roles: []Role{RolePDTransfer}, Evidence: "diff-test @ vLLM v0.32.0",
		},
	}}
}

func TestAValidRegistryValidates(t *testing.T) {
	if p := valid().Validate(); !p.OK() {
		t.Fatalf("refused:\n%s", p.Error())
	}
}

// One failure mode each: a registry entry earns behaves_like only by evidence (#2021 B10),
// so an entry with no model, no evidence or an interface the vocabulary does not know is
// refused, as is a document that is not a registry.
func TestEachFailureModeIsRefused(t *testing.T) {
	for _, c := range []struct {
		name   string
		edit   func(*Registry)
		inPath string
	}{
		{"wrong kind", func(r *Registry) { r.Kind = "Registry" }, "kind"},
		{"no name", func(r *Registry) { r.Name = "" }, "name"},
		{"no connectors", func(r *Registry) { r.Connectors = nil }, "connectors"},
		{"an empty connector name", func(r *Registry) {
			r.Connectors[" "] = r.Connectors["NixlConnector"]
		}, "connectors"},
		{"a connector name with surrounding spaces", func(r *Registry) {
			r.Connectors[" NixlConnector "] = r.Connectors["NixlConnector"]
		}, "connectors"},
		{"no upstream interface", func(r *Registry) {
			e := r.Connectors["NixlConnector"]
			e.Upstream = " "
			r.Connectors["NixlConnector"] = e
		}, "connectors.NixlConnector.upstream"},
		{"no behaves_like", func(r *Registry) {
			e := r.Connectors["NixlConnector"]
			e.BehavesLike = ""
			r.Connectors["NixlConnector"] = e
		}, "connectors.NixlConnector.behaves_like"},
		{"no roles", func(r *Registry) {
			e := r.Connectors["NixlConnector"]
			e.Roles = nil
			r.Connectors["NixlConnector"] = e
		}, "connectors.NixlConnector.roles"},
		{"an unknown role", func(r *Registry) {
			e := r.Connectors["NixlConnector"]
			e.Roles = []Role{"caching"}
			r.Connectors["NixlConnector"] = e
		}, "connectors.NixlConnector.roles"},
		{"a repeated role", func(r *Registry) {
			e := r.Connectors["NixlConnector"]
			e.Roles = []Role{RolePDTransfer, RolePDTransfer}
			r.Connectors["NixlConnector"] = e
		}, "connectors.NixlConnector.roles"},
		{"no evidence", func(r *Registry) {
			e := r.Connectors["NixlConnector"]
			e.Evidence = " "
			r.Connectors["NixlConnector"] = e
		}, "connectors.NixlConnector.evidence"},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := valid()
			c.edit(r)
			p := r.Validate()
			if p.OK() {
				t.Fatal("accepted")
			}
			if !strings.Contains(p.Error(), c.inPath) {
				t.Fatalf("no finding at %s:\n%s", c.inPath, p.Error())
			}
		})
	}
}

// Lookup answers by the name a deployment's connector key uses.
func TestLookup(t *testing.T) {
	r := valid()
	if e, ok := r.Lookup("NixlConnector"); !ok || e.BehavesLike != "nixl-pull" {
		t.Fatalf("Lookup(NixlConnector) = %+v, %v", e, ok)
	}
	if _, ok := r.Lookup("LMCacheConnectorV1"); ok {
		t.Error("an unregistered connector was found")
	}
	if !r.Serves("OffloadingConnector", RoleOffload) {
		t.Error("OffloadingConnector does not serve offload")
	}
	if r.Serves("OffloadingConnector", RolePDTransfer) {
		t.Error("OffloadingConnector serves pd_transfer, a role its entry does not list")
	}
	if r.Serves("LMCacheConnectorV1", RoleOffload) {
		t.Error("an unregistered connector serves a role")
	}
}

// The wire form: strict, with the keys the design names.
func TestTheWireForm(t *testing.T) {
	doc := `kind: ConnectorRegistry
name: builtin
connectors:
  OffloadingConnector:
    upstream: KVConnectorBase_V1
    behaves_like: offloading
    roles: [offload]
    evidence: diff-test @ vLLM v0.32.0
`
	var r Registry
	dec := yaml.NewDecoder(strings.NewReader(doc))
	dec.KnownFields(true)
	if err := dec.Decode(&r); err != nil {
		t.Fatal(err)
	}
	if p := r.Validate(); !p.OK() {
		t.Fatalf("refused:\n%s", p.Error())
	}
	bad := strings.Replace(doc, "behaves_like", "behaves_as", 1)
	dec = yaml.NewDecoder(strings.NewReader(bad))
	dec.KnownFields(true)
	if err := dec.Decode(&Registry{}); err == nil {
		t.Error("a misspelled key loaded")
	}
}
