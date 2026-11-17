// Original to blis-vllm.
// Moved unchanged from blis-substrate's kernel/rulespanic.go at
// 00817b10798cbc00f560e51921c30c7084fa5e0c, with the rules calls it guards (blis-substrate#58).

package lowering

import (
	subrules "github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// rulesPanic is a panic raised inside the caller's rules while a pool is lowered.
// Lower recovers only this type, so a panic in the lowering's own code still surfaces as a panic.
type rulesPanic struct{ v any }

// panicTagged forwards every subrules.EngineRules method to the caller's rules and re-raises a
// panic from one of them as a rulesPanic. Its method set is the interface's, so a method
// added to subrules.EngineRules fails to compile here until it is forwarded too.
type panicTagged struct{ subrules.EngineRules }

func tag() {
	if p := recover(); p != nil {
		if _, ok := p.(rulesPanic); ok {
			panic(p)
		}
		panic(rulesPanic{p})
	}
}

func (r panicTagged) EngineVersion() string {
	defer tag()
	return r.EngineRules.EngineVersion()
}

func (r panicTagged) Check(in subrules.Input) *subrules.Problems {
	defer tag()
	return r.EngineRules.Check(in)
}

func (r panicTagged) RuleNames() []string {
	defer tag()
	return r.EngineRules.RuleNames()
}

func (r panicTagged) SpeculativeMethodAccepted(method string) bool {
	defer tag()
	return r.EngineRules.SpeculativeMethodAccepted(method)
}

func (r panicTagged) SpeculativeMethodNames() []string {
	defer tag()
	return r.EngineRules.SpeculativeMethodNames()
}

func (r panicTagged) AsyncCompatibleSpeculative(method string) (bool, bool) {
	defer tag()
	return r.EngineRules.AsyncCompatibleSpeculative(method)
}

func (r panicTagged) EngramOnHostByDefault() bool {
	defer tag()
	return r.EngineRules.EngramOnHostByDefault()
}

func (r panicTagged) DCPCommBackendNames() map[string]bool {
	defer tag()
	return r.EngineRules.DCPCommBackendNames()
}

func (r panicTagged) MegaMoEBackend(name string) bool {
	defer tag()
	return r.EngineRules.MegaMoEBackend(name)
}

func (r panicTagged) SequenceParallelMoE(backend string, expertParallel bool, tp, dp int) bool {
	defer tag()
	return r.EngineRules.SequenceParallelMoE(backend, expertParallel, tp, dp)
}

func (r panicTagged) ModelSequenceParallel(architecture string, pl deployment.Parallelism, moeBackend string) bool {
	defer tag()
	return r.EngineRules.ModelSequenceParallel(architecture, pl, moeBackend)
}

func (r panicTagged) CustomAllReduceSupportsWidth(tp int) bool {
	defer tag()
	return r.EngineRules.CustomAllReduceSupportsWidth(tp)
}

func (r panicTagged) TritonFetchWithholdsSMs(pageBytes int) int {
	defer tag()
	return r.EngineRules.TritonFetchWithholdsSMs(pageBytes)
}

func (r panicTagged) DBOEngages(enabled bool, totalTokens int, uniformDecode bool) bool {
	defer tag()
	return r.EngineRules.DBOEngages(enabled, totalTokens, uniformDecode)
}

func (r panicTagged) EffectiveAll2AllBackend(backend string) string {
	defer tag()
	return r.EngineRules.EffectiveAll2AllBackend(backend)
}

func (r panicTagged) ResolveCacheDType(architecture string, family, tp int, requested string) string {
	defer tag()
	return r.EngineRules.ResolveCacheDType(architecture, family, tp, requested)
}

func (r panicTagged) WindowPagesPerRequest(c model.Cache, engineBlock, inFlight, extraRetained, maxModelLen int) int {
	defer tag()
	return r.EngineRules.WindowPagesPerRequest(c, engineBlock, inFlight, extraRetained, maxModelLen)
}

func (r panicTagged) KVCacheGroups(in subrules.KVCacheGroupsInput) (subrules.KVCacheGrouping, error) {
	defer tag()
	return r.EngineRules.KVCacheGroups(in)
}

func (r panicTagged) CheckEngineBlock(at string, b deployment.EngineBlock) *subrules.Problems {
	defer tag()
	return r.EngineRules.CheckEngineBlock(at, b)
}

func (r panicTagged) CheckConnectorOptions(at string, b deployment.EngineBlock) *subrules.Problems {
	defer tag()
	return r.EngineRules.CheckConnectorOptions(at, b)
}

func (r panicTagged) DraftTokens(b deployment.EngineBlock) int {
	defer tag()
	return r.EngineRules.DraftTokens(b)
}

func (r panicTagged) CUDAGraphCaptureSizes(in subrules.CUDAGraphCaptureInput) (subrules.CUDAGraphCapture, bool, error) {
	defer tag()
	return r.EngineRules.CUDAGraphCaptureSizes(in)
}

func (r panicTagged) ResolveDPLoadBalancing(pl deployment.Parallelism) deployment.DPLoadBalancing {
	defer tag()
	return r.EngineRules.ResolveDPLoadBalancing(pl)
}

// ResolvedAll2AllBackend is not part of subrules.EngineRules: a pack that binds a default
// all2all_backend to passthrough beside a self-communicating MoE backend offers it, and the
// wrapper passes it on (resolvedAll2AllBackend falls back to EffectiveAll2AllBackend otherwise).
func (r panicTagged) ResolvedAll2AllBackend(backend, moeBackend string) string {
	defer tag()
	return resolvedAll2AllBackend(r.EngineRules, backend, moeBackend)
}

// SelfCommunicatingMoEBackend is likewise offered by a pack that knows the binding.
func (r panicTagged) SelfCommunicatingMoEBackend(moeBackend string) (self, known bool) {
	defer tag()
	if b, ok := r.EngineRules.(interface {
		SelfCommunicatingMoEBackend(string) (bool, bool)
	}); ok {
		return b.SelfCommunicatingMoEBackend(moeBackend)
	}
	return false, false
}
