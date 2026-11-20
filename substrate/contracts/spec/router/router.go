// Package router describes the request router in front of a deployment, so that a
// measured run records what chose each request's replica.
//
// An Endpoint Picker (EPP) is configured by a plugin file whose contents decide routing,
// and two runs under one EPP version can route differently. So the schema pins the
// config by digest and leaves reading its bytes to the consumer that simulates it.
package router

import "github.com/inference-sim/blis-substrate/contracts/check"

// Kind names how requests reach the engines.
type Kind string

const (
	// KindDirect: the client addresses an engine, or a plain load balancer whose choice
	// does not depend on request content.
	KindDirect Kind = "direct"
	// KindEPP: a Gateway API Inference Extension endpoint picker chooses the replica.
	KindEPP Kind = "epp"
)

var kinds = map[Kind]bool{KindDirect: true, KindEPP: true}

// Valid reports whether k is a recognized router kind.
func (k Kind) Valid() bool { return kinds[k] }

// Router is the routing layer of one measured deployment.
type Router struct {
	Kind Kind `yaml:"kind"`
	EPP  *EPP `yaml:"epp,omitempty"`
}

// EPP is an endpoint picker and the config it ran with.
type EPP struct {
	Version string    `yaml:"version"`
	Image   string    `yaml:"image,omitempty"`
	Config  ConfigRef `yaml:"config"`
	// FlowControl summarizes the flow-control settings, when the config enables them.
	FlowControl string `yaml:"flow_control,omitempty"`
}

// ConfigRef names a config file by path and digest.
type ConfigRef struct {
	Path   string `yaml:"path"`
	SHA256 string `yaml:"sha256"`
}

// Validate performs field-level validation of a router.
func (r *Router) Validate() *validate.Problems {
	p := &validate.Problems{}
	switch {
	case r.Kind == "":
		p.Field("kind", "required: one of %v", AllKinds())
	case !r.Kind.Valid():
		p.Field("kind", "%q is not one of %v", r.Kind, AllKinds())
	case r.Kind == KindEPP && r.EPP == nil:
		p.Field("epp", "required for kind epp: the picker's version and config digest")
	case r.Kind == KindDirect && r.EPP != nil:
		p.Field("epp", "a direct router has no endpoint picker")
	}
	if e := r.EPP; e != nil {
		if e.Version == "" {
			p.Field("epp.version", "required")
		}
		if e.Config.Path == "" {
			p.Field("epp.config.path", "required")
		}
		p.SHA256Field("epp.config.sha256", e.Config.SHA256)
	}
	return p
}

// AllKinds returns every router kind, sorted.
func AllKinds() []string { return []string{string(KindDirect), string(KindEPP)} }
