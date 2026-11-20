package rules

import (
	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// Problems is the accumulating finding list a rule reports into, and Problem, Severity and
// its two values the findings in it. They are exported here so an engine release's rules can
// live in another module: a Rule's Check takes a Problems, and a type in an internal package
// cannot be named from outside this module.
type (
	Problems = validate.Problems
	Problem  = validate.Problem
	Severity = validate.Severity
)

const (
	SeverityError   = validate.SeverityError
	SeverityWarning = validate.SeverityWarning
)

// EngineRules is everything a consumer asks of one engine release's rules: which values it
// accepts, how it resolves what a deployment leaves unstated, and its version-scoped checks.
// The rules live with the engine model that owns the release, and a cost model or a
// validator takes them from its caller (Bundle.Rules): this schema registers none of its own.
//
// A consumer that needs a fact the interface does not name asks for a method to be added
// here, rather than type-asserting to a particular implementation: an assertion silently
// skips the fact for any other.
type EngineRules interface {
	// EngineVersion is the release the rules describe, such as "1.2.0".
	EngineVersion() string

	// Check runs every rule of the release over a document set.
	Check(in Input) *Problems
	// RuleNames returns the release's rule names, sorted.
	RuleNames() []string

	// CheckEngineBlock is the release's field checks of one pool's engine block, which this
	// schema holds unopened: whether the block decodes, and the bounds its own types
	// declare. at is the pool's path within the deployment, such as "pools[0]", and each
	// finding is at a path below at + ".engine".
	CheckEngineBlock(at string, b deployment.EngineBlock) *Problems
	// CheckConnectorOptions is the release's checks of an offload's connector options, which
	// this schema holds unopened as it does an engine block. at is the offload block's path,
	// "offload", and each finding is at a path below at + ".connector_options".
	CheckConnectorOptions(at string, b deployment.EngineBlock) *Problems
	// DraftTokens is the speculative tokens a pool's engine block drafts per step, zero when
	// it drafts none or does not decode.
	DraftTokens(b deployment.EngineBlock) int

	// SpeculativeMethodAccepted reports whether the release accepts a draft method, and
	// SpeculativeMethodNames lists the accepted ones, sorted, for an error message.
	SpeculativeMethodAccepted(method string) bool
	SpeculativeMethodNames() []string
	// AsyncCompatibleSpeculative reports whether a draft method leaves async scheduling
	// on, and known whether the release states a list at all.
	AsyncCompatibleSpeculative(method string) (compatible, known bool)

	// EngramOnHostByDefault is whether an Engram table lives in host memory when the
	// deployment does not say.
	EngramOnHostByDefault() bool
	// DCPCommBackendNames are the decode-context-parallel collective sets the release
	// accepts, nil when it states none.
	DCPCommBackendNames() map[string]bool
	// MegaMoEBackend reports whether an MoE backend name runs a model's mega-MoE path.
	MegaMoEBackend(name string) bool

	SequenceParallelMoE(backend string, expertParallel bool, tp, dp int) bool
	ModelSequenceParallel(architecture string, pl deployment.Parallelism, moeBackend string) bool
	CustomAllReduceSupportsWidth(tp int) bool
	TritonFetchWithholdsSMs(pageBytes int) int
	DBOEngages(enabled bool, totalTokens int, uniformDecode bool) bool
	EffectiveAll2AllBackend(backend string) string
	ResolveCacheDType(architecture string, family, tp int, requested string) string
	WindowPagesPerRequest(c model.Cache, engineBlock, inFlight, extraRetained, maxModelLen int) int
	KVCacheGroups(in KVCacheGroupsInput) (KVCacheGrouping, error)
	CUDAGraphCaptureSizes(in CUDAGraphCaptureInput) (CUDAGraphCapture, bool, error)
	ResolveDPLoadBalancing(pl deployment.Parallelism) deployment.DPLoadBalancing
}
