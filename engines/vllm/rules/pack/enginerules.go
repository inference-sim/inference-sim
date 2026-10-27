// Moved from inference-sim/blis-schemas/rules/enginerules.go at 70246460a379f8b1898fbf6c36a3091364118950 (the Pack's EngineRules methods).

package pack

import (
	"sort"

	"github.com/inference-sim/blis-schemas/rules"
)

var _ rules.EngineRules = (*Pack)(nil)

// EngineVersion implements EngineRules.
func (p *Pack) EngineVersion() string { return p.Version }

// Check implements EngineRules: every rule of the pack, in order.
func (p *Pack) Check(in rules.Input) *rules.Problems {
	out := &rules.Problems{}
	for _, r := range p.Rules {
		r.Check(in, out)
	}
	return out
}

// RuleNames implements EngineRules.
func (p *Pack) RuleNames() []string { return p.Names() }

// SpeculativeMethodAccepted implements EngineRules.
func (p *Pack) SpeculativeMethodAccepted(method string) bool { return p.SpeculativeMethods[method] }

// SpeculativeMethodNames implements EngineRules.
func (p *Pack) SpeculativeMethodNames() []string { return sortedKeys(p.SpeculativeMethods) }

// AsyncCompatibleSpeculative implements EngineRules.
func (p *Pack) AsyncCompatibleSpeculative(method string) (compatible, known bool) {
	if p.AsyncCompatibleSpec == nil {
		return false, false
	}
	return p.AsyncCompatibleSpec[method], true
}

// EngramOnHostByDefault implements EngineRules. It reads the EngramCPUOffloadDefault field,
// which a method of the same name could not coexist with.
func (p *Pack) EngramOnHostByDefault() bool { return p.EngramCPUOffloadDefault }

// DCPCommBackendNames implements EngineRules. It returns a copy, so a caller cannot edit
// the pack.
func (p *Pack) DCPCommBackendNames() map[string]bool {
	if p.DCPCommBackends == nil {
		return nil
	}
	out := make(map[string]bool, len(p.DCPCommBackends))
	for k, v := range p.DCPCommBackends {
		out[k] = v
	}
	return out
}

// MegaMoEBackend implements EngineRules, normalizing the name as the engine does.
func (p *Pack) MegaMoEBackend(name string) bool { return p.MegaMoEBackends[NormalizeMoEBackend(name)] }

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k, ok := range m {
		if ok {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}
