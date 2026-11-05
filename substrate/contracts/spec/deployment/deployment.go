// Package deployment describes the mutable configuration a user — or an optimizer
// sweeping a configuration space — chooses against a fixed Scenario: how a model is
// laid out across the available hardware (pools, each a role at a parallelism layout
// with its own engine settings), the KV offload hierarchy below HBM, and
// prefill-to-decode KV transfer.
//
// It is the tunable half of what a single `blis run` consumes. The immutable problem
// — the model, the workload (a distributional shape or a concrete trace reference), the
// available-hardware inventory, and the coefficient/engine-version references that
// identify what a prediction is fitted against — lives in a Scenario. A Scenario and a
// Deployment together describe one run; the operating point (load level) is a run-level
// sweep axis that is neither.
//
// Three rules shape the schema, each preventing a class of error rather than
// catching it later:
//
// Derived quantities have no field. Expert-parallel width is the product of the
// tensor-, prefill-context- and data-parallel widths, because that is the group the
// engine builds. A deployment that could state EP independently could describe a
// layout no engine would run, so the field does not exist.
//
// Requested settings are distinct from resolved ones. A collective backend and an
// async-scheduling preference are requests: crossing a node boundary without
// multi-node NVLink disqualifies an SM-consuming all-reduce kernel whatever the
// field says, and async scheduling has several disqualifying conditions. The
// resolver reports what it resolved, so a reader is never misled by a request.
//
// Engine settings are per pool. A prefill/decode-disaggregated deployment runs two
// kinds of engine with different backends and different graph modes; one shared
// engine block would describe neither.
package deployment

import "math"

// Deployment is the tunable configuration applied to a Scenario: the pools that lay
// the model out, the offload hierarchy, and prefill-to-decode transfer.
type Deployment struct {
	Kind string `yaml:"kind"` // "Deployment"
	Name string `yaml:"name"`

	Pools []Pool `yaml:"pools"`

	Offload    *Offload    `yaml:"offload,omitempty"`
	PDTransfer *PDTransfer `yaml:"pd_transfer,omitempty"`

	// Router names the inputs of the llm-d router (the Endpoint Picker) that routes the
	// deployment's requests. Absent means the router's built-in configuration in standalone
	// mode over every pod of the deployment.
	Router *Router `yaml:"router,omitempty"`
}

// Router names the llm-d router's own inputs, exactly as a real deployment gives them to the
// router, so one set of files configures both the real router and the simulated one. Paths are
// relative to the directory of the file that holds the deployment.
//
// There is deliberately no field for a routing decision: scorers, weights, admission and flow
// control are the router's configuration, carried by the files named here and never
// restated in this schema.
type Router struct {
	// EPPFlags are the router's command-line flags, verbatim, one entry per flag (a Helm chart's
	// epp.flags map rendered as --name=value). They select the pool (--pool-name or
	// --endpoint-selector) and may name the configuration with --config-file. The schema checks
	// their form only; a consumer refuses flags it cannot honor.
	EPPFlags []string `yaml:"epp_flags,omitempty"`
	// EndpointPickerConfig is the EndpointPickerConfig file, as the router reads it with
	// --config-file. Naming the file here and in EPPFlags is an error.
	EndpointPickerConfig string `yaml:"endpoint_picker_config,omitempty"`
	// Manifests are Kubernetes manifest files. The router reads the InferencePool,
	// InferenceObjective and InferenceModelRewrite objects in them; other kinds are ignored.
	Manifests []string `yaml:"manifests,omitempty"`
}

// Role distinguishes the engines of a disaggregated deployment.
type Role string

const (
	RoleColocated Role = "colocated"
	RolePrefill   Role = "prefill"
	RoleDecode    Role = "decode"
)

var roles = map[Role]bool{RoleColocated: true, RolePrefill: true, RoleDecode: true}

// Valid reports whether r is a recognized role.
func (r Role) Valid() bool { return roles[r] }

// RoleLabel is llm-d's pod label for an engine's disaggregation role. It is derived from the
// pool's Role (RoleLabelValue); a pool's labels may repeat it only with that value.
const RoleLabel = "llm-d.ai/role"

// RoleLabelValue is the llm-d.ai/role value of a pool of role r: prefill-decode for a colocated
// pool (it serves both phases), else the role itself.
func (r Role) RoleLabelValue() string {
	if r == RoleColocated {
		return "prefill-decode"
	}
	return string(r)
}

// Pool is the nodes serving one role: a role, a node count, a parallelism layout, and
// settings. It may hold more than one engine, each running that layout with those
// settings — llm-d runs a LeaderWorkerSet of `replicas` groups, each group one engine
// over `size` nodes, so a 36-node prefill pool of dp 8 is 36 one-node engines. Nodes is
// therefore the pool's whole extent, which one engine's layout may occupy only part of.
type Pool struct {
	Role     Role        `yaml:"role"`
	Nodes    int         `yaml:"nodes"`
	Parallel Parallelism `yaml:"parallel"`
	// Engine is the pool's engine settings, held unopened: the engine model a scenario's
	// engine_version selects opens it (EngineBlock).
	Engine EngineBlock `yaml:"engine"`
	// Labels are the Kubernetes labels on the pool's model-server pods, as the pool's pod
	// template sets them. The router selects and filters endpoints by them (an InferencePool
	// selector, the llm-d.ai/role label of a disaggregated deployment). llm-d.ai/role is derived
	// from Role (RoleLabelValue); given here, it must have that value.
	Labels map[string]string `yaml:"labels,omitempty"`
}

// Parallelism is the layout. EP is absent: see the package comment.
type Parallelism struct {
	TP int `yaml:"tp"`
	PP int `yaml:"pp"`
	DP int `yaml:"dp"`
	// DPLocal is the data-parallel rank count resident on one node. It must divide
	// the node's GPU count.
	DPLocal int `yaml:"dp_local,omitempty"`

	// DataParallelLoadBalancing is how the pool's data-parallel ranks are exposed, which
	// decides what a router sees and chooses between. Empty is internal, the default.
	// See DPLoadBalancing.
	DataParallelLoadBalancing DPLoadBalancing `yaml:"data_parallel_load_balancing,omitempty"`

	EnableExpertParallel bool `yaml:"enable_expert_parallel"`

	// PCP and DCP are prefill- and decode-context parallel width: sharding one
	// sequence across ranks. DCP is the axis that shards KV; TP shards it only
	// through KV-head count.
	PCP int `yaml:"pcp,omitempty"`
	DCP int `yaml:"dcp,omitempty"`
}

// ExpertParallelWidth returns the derived EP degree, or 1 when expert parallelism
// is off. It is a function rather than a field precisely so no file can disagree
// with it.
//
// The width is tp x pcp x dp: the engine lays ranks out DP x PP x PCP x TP and builds
// the expert group over the DP, PCP and TP axes together (initialize_model_parallel,
// in every release). An unset or non-positive width counts as one, the
// same floor every other reader of these fields applies.
//
// An earlier form was tp x max(dp, pcp). That agrees with the product exactly when one
// of dp and pcp is at most 1, which was every deployment this schema admitted while
// prefill-context parallelism and data parallelism could not be combined — so the
// product changes no width that could previously validate. It differs once both exceed
// 1, which the engine supports from e6dc16cebd and llm-d deploys (DP 4 x PCP 8 is an
// expert group of 32, not 8).
//
// The product saturates at the largest int instead of wrapping, so an absurd layout
// reads as an absurdly wide group rather than a small or negative one.
func (p Parallelism) ExpertParallelWidth() int {
	if !p.EnableExpertParallel {
		return 1
	}
	width := 1
	for _, w := range []int{p.TP, p.PCP, p.DP} {
		if w < 1 {
			w = 1
		}
		if width > math.MaxInt/w {
			return math.MaxInt
		}
		width *= w
	}
	return width
}

// Offload is the KV tier hierarchy below HBM.
type Offload struct {
	Tiers []Tier `yaml:"tiers"`
	// EvictionPolicy names the KV-offload cache policy. The engine registers its in-tree
	// policies with a CachePolicyFactory (lru, arc) and admits out-of-tree ones
	// through a module path — the same extensible-registry shape as the connector and
	// quantization vocabularies. "lru" is the default only when the field is ABSENT; a
	// SUPPLIED name the factory does not know raises at launch rather than falling back. So
	// the v0_29 rules pack enumerates the in-tree set and the `eviction-policy-known` rule
	// WARNS on an unknown value (a probable typo, caught before launch) rather than erroring
	// (it may be a valid out-of-tree policy). It is validated by a rule, not here, because the
	// legal set is a version fact that belongs with the engine's.
	EvictionPolicy string `yaml:"eviction_policy,omitempty"`
	// PrefetchDepth is how many steps ahead a fetch is issued. It decides whether a
	// tier hit is hidden or exposed, and it is a scheduler property rather than a
	// hardware one.
	PrefetchDepth int `yaml:"prefetch_depth,omitempty"`
	// Connector names the offload implementation, whose spec determines whether a
	// fetch consumes SMs or a copy engine.
	Connector string `yaml:"connector,omitempty"`
	// Spec names the offloading spec, e.g. a CPU-only spec or a tiering spec.
	Spec string `yaml:"spec,omitempty"`
	// ConnectorOptions are the offload connector's own options, held unopened like a pool's
	// engine block: the engine model opens and checks them
	// (rules.EngineRules.CheckConnectorOptions).
	ConnectorOptions EngineBlock `yaml:"connector_options,omitempty"`
}

// Tier is one offload level: a device class from the catalog, and how much of it.
// Where the Scenario cluster declares a storage inventory, the device class must be one
// it lists (ValidateAgainstCluster checks this). A cluster that declares none states no
// inventory to draw from, so the tier is unconstrained — the shape that predates the
// inventory, kept valid; declaring storage is optional and opts a cluster into the check.
type Tier struct {
	Device string `yaml:"tier"`
	Bytes  int64  `yaml:"bytes"`
}

// PDTransfer describes prefill-to-decode KV movement. Bandwidth is absent: it
// resolves from the fabric times a registry efficiency coefficient, and is not a
// free parameter.
type PDTransfer struct {
	Connector string `yaml:"connector"`
}

// DPLoadBalancing is how a data-parallel pool's ranks are exposed to clients, the engine's
// three launch modes (engine/arg_utils.py:536-538, 1225-1242):
//
//   - internal, the default: one API endpoint, and the engine's core client balances
//     every request across all ranks (DPLBAsyncMPClient, v1/engine/core_client.py:
//     155-166);
//   - hybrid (--data-parallel-hybrid-lb): one endpoint per node, balancing across that
//     node's ranks; a router chooses the node;
//   - external (--data-parallel-external-lb, or --data-parallel-multi-port-external-lb,
//     which launches one such server per local rank): every rank serves its own
//     endpoint (DPAsyncMPClient), and a router chooses the rank.
//
// The api server count is not part of it: --api-server-count sets how many frontend
// processes share an endpoint (entrypoints/cli/serve.py:105-127), not which ranks a
// router can reach.
type DPLoadBalancing string

const (
	DPLBInternal DPLoadBalancing = "internal"
	DPLBHybrid   DPLoadBalancing = "hybrid"
	DPLBExternal DPLoadBalancing = "external"
)

var dpLoadBalancings = map[DPLoadBalancing]bool{DPLBInternal: true, DPLBHybrid: true, DPLBExternal: true}

// Valid reports whether m is a recognized mode; empty is the default, internal.
func (m DPLoadBalancing) Valid() bool { return m == "" || dpLoadBalancings[m] }

// Mode returns the mode with its default applied.
func (m DPLoadBalancing) Mode() DPLoadBalancing {
	if m == "" {
		return DPLBInternal
	}
	return m
}
