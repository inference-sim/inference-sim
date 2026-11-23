package engine

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

// Spec is how a simulator hands an engine its configuration without reading it: the
// engine's kind, the upstream release it models, and the configuration block itself,
// kept as YAML only the engine decodes.
//
// The kind is what keeps engine identity out of the simulator. A simulator looks the kind
// up in the factory registry and passes the rest through; a second engine registers
// another kind and needs no change here or in the simulator.
type Spec struct {
	// Kind names the engine model, as its factory registers it.
	Kind string
	// UpstreamVersion is the release of the real engine the configuration was written
	// for, such as "0.32.0". An engine refuses a release it does not model rather than
	// running it as another; the version is a check, not a selector.
	UpstreamVersion string
	// Config is the engine's configuration block, exactly as the document gave it. A
	// factory decodes the node itself (Node.Decode) rather than a re-encoding of it, so the
	// node's positions put a configuration error at its line in the document.
	Config yaml.Node
	// Roots are the directories the engine may read further inputs from, by role: a
	// model catalog, a coefficient registry, the directory of the file the
	// configuration came from. A simulator passes the paths; the engine opens them.
	Roots map[string]string
	// Pool selects one pool of a multi-pool document, such as the prefill or the decode
	// pool of a disaggregated deployment. Each pool is one engine.
	Pool int
	// Seed is the run's seed, for an engine that draws anything at random.
	Seed int64
}

// Factory builds engines of one kind.
type Factory interface {
	// Kind is the kind the factory builds.
	Kind() string
	// New builds one engine from its spec.
	New(s Spec) (Engine, error)
}

var (
	mu        sync.RWMutex
	factories = map[string]Factory{}
)

// Register makes a factory available under its kind. It panics on an empty or a
// duplicate kind: two factories for one kind means one silently wins, and which one
// would depend on import order.
func Register(f Factory) {
	mu.Lock()
	defer mu.Unlock()
	if f == nil || f.Kind() == "" {
		panic("engine: a factory must declare a kind")
	}
	if _, dup := factories[f.Kind()]; dup {
		panic(fmt.Sprintf("engine: duplicate factory for kind %q", f.Kind()))
	}
	factories[f.Kind()] = f
}

// New builds an engine through the factory registered for the spec's kind.
func New(s Spec) (Engine, error) {
	mu.RLock()
	f, ok := factories[s.Kind]
	mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("engine: no factory for kind %q; registered kinds are [%s]",
			s.Kind, strings.Join(Kinds(), ", "))
	}
	return f.New(s)
}

// Kinds returns every registered kind, sorted.
func Kinds() []string {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]string, 0, len(factories))
	for k := range factories {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
