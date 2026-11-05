package deployment

import (
	"fmt"
	"math/big"
	"math/bits"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// Validate performs FIELD-LEVEL validation of a deployment IN ISOLATION: required
// fields, positive counts, internal arithmetic consistency, and the structural
// invariants that hold whatever engine version the scenario names. The checks that
// couple a deployment to the cluster it runs on live in ValidateAgainstCluster,
// because the cluster is a Scenario property this type does not carry.
//
// A structural invariant is one that follows from what parallelism is, not from an
// engine's current choices. That expert-parallel width is derived rather than stated
// is a consequence of how an expert group is built, so it belongs here. Version-scoped
// rules — whether a backend name is supported, which conditions disable async
// scheduling — do not; a rules pack keyed by engine version owns them.
func (d *Deployment) Validate() *validate.Problems {
	p := &validate.Problems{}

	if d.Kind != "Deployment" {
		p.Field("kind", "must be %q, got %q", "Deployment", d.Kind)
	}
	if d.Name == "" {
		p.Field("name", "required")
	}

	d.validatePools(p)
	d.validateOffload(p)
	d.validateRouter(p)
	return p
}

// validateRouter checks that the router inputs are well formed: named files are non-empty
// paths, flags are flags, and the configuration is named once.
func (d *Deployment) validateRouter(p *validate.Problems) {
	for i, pool := range d.Pools {
		if v, ok := pool.Labels[RoleLabel]; ok && pool.Role.Valid() && v != pool.Role.RoleLabelValue() {
			p.Field(fmt.Sprintf("pools[%d].labels.%s", i, RoleLabel),
				"%q contradicts the pool's role %q, which sets it to %q", v, pool.Role, pool.Role.RoleLabelValue())
		}
		for k, v := range pool.Labels {
			if !validLabelKey(k) {
				p.Field(fmt.Sprintf("pools[%d].labels", i), "%q is not a valid Kubernetes label key", k)
			}
			if !validLabelValue(v) {
				p.Field(fmt.Sprintf("pools[%d].labels.%s", i, k), "%q is not a valid Kubernetes label value", v)
			}
		}
	}
	r := d.Router
	if r == nil {
		return
	}
	if r.EndpointPickerConfig != "" && strings.TrimSpace(r.EndpointPickerConfig) == "" {
		p.Field("router.endpoint_picker_config", "must name a file")
	}
	for i, f := range r.EPPFlags {
		if !strings.HasPrefix(f, "-") {
			p.Field(fmt.Sprintf("router.epp_flags[%d]", i), "%q is not a flag; give each flag as --name=value", f)
		}
		if r.EndpointPickerConfig != "" && (isFlag(f, "config-file") || isFlag(f, "config-text")) {
			p.Field("router.endpoint_picker_config", "the configuration is also named by the EPP flag %q; name it once", f)
		}
	}
	for i, m := range r.Manifests {
		if strings.TrimSpace(m) == "" {
			p.Field(fmt.Sprintf("router.manifests[%d]", i), "must name a file")
		}
	}
}

// validLabelKey reports whether k is a Kubernetes label key: an optional prefix (a lowercase
// DNS-1123 subdomain) and a slash, then a name of at most 63 characters.
func validLabelKey(k string) bool {
	name := k
	if i := strings.LastIndexByte(k, '/'); i >= 0 {
		prefix := k[:i]
		name = k[i+1:]
		if !dns1123Subdomain(prefix) {
			return false
		}
	}
	return name != "" && len(name) <= 63 && labelChars(name, "-_.") && alnum(name[0]) && alnum(name[len(name)-1])
}

// isFlag reports whether arg sets the command-line flag name (--name or --name=value).
func isFlag(arg, name string) bool {
	return arg == "--"+name || strings.HasPrefix(arg, "--"+name+"=")
}

// dns1123Subdomain reports whether s is a lowercase RFC 1123 subdomain, as Kubernetes requires of
// a label key's prefix: at most 253 characters of dot-separated labels, each 1 to 63 characters of
// lowercase letters, digits and '-', starting and ending with a letter or digit.
func dns1123Subdomain(s string) bool {
	if s == "" || len(s) > 253 {
		return false
	}
	for _, label := range strings.Split(s, ".") {
		if label == "" || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
			return false
		}
		for i := 0; i < len(label); i++ {
			if c := label[i]; !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-') {
				return false
			}
		}
	}
	return true
}

// validLabelValue reports whether v is a Kubernetes label value: empty, or at most 63
// characters that start and end alphanumeric.
func validLabelValue(v string) bool {
	return v == "" || (len(v) <= 63 && labelChars(v, "-_.") && alnum(v[0]) && alnum(v[len(v)-1]))
}

func labelChars(s, extra string) bool {
	for i := 0; i < len(s); i++ {
		if !alnum(s[i]) && !strings.ContainsRune(extra, rune(s[i])) {
			return false
		}
	}
	return true
}

func alnum(c byte) bool { return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' }

// ClusterConstraints is the subset of a Scenario's cluster that deployment validation
// reads: the node and per-node GPU counts, and the declared storage inventory. It is a
// deployment-local type that mirrors those facts rather than importing scenario.Cluster,
// so spec/deployment stays independent of spec/scenario. Named fields also stop the two
// same-typed counts from being transposed at a call site, which a positional
// (nodes, gpusPerNode int) pair invited.
type ClusterConstraints struct {
	Nodes       int
	GPUsPerNode int
	Storage     []string
}

// ValidateAgainstCluster adds the field-level checks that couple a deployment to the
// available-hardware inventory it is placed on: that its pools' node counts sum to the
// nodes the cluster declares, that each data-parallel-local width divides a node's GPU
// count, that no engine needs more GPUs than its pool owns or places a local replica
// past the end of a node, and that every offload tier names a storage class the cluster
// actually lists.
//
// Pools fill the cluster exactly rather than take a subset of it: a deployment lays out
// the whole cluster it is handed, so the inventory is that cluster's full extent and not
// a pool to sub-select nodes from. This exact-fill rule is the placement contract the
// pre-split Scenario enforced, carried here unchanged — relaxing it to allow partial use
// of the declared hardware would be a behavior change, not part of this value-preserving
// split.
//
// They are separate from Validate because they need the cluster, which is a Scenario
// property; the composition layer (blisschemas.Validate) supplies it as a
// ClusterConstraints when both documents are present.
func (d *Deployment) ValidateAgainstCluster(c ClusterConstraints) *validate.Problems {
	p := &validate.Problems{}

	total := 0
	for i, pool := range d.Pools {
		total += pool.Nodes
		if pool.Parallel.DPLocal > 0 && c.GPUsPerNode > 0 &&
			c.GPUsPerNode%pool.Parallel.DPLocal != 0 {
			p.Field(fmt.Sprintf("pools[%d].parallel.dp_local", i),
				"%d does not divide gpus_per_node %d", pool.Parallel.DPLocal, c.GPUsPerNode)
		}
		validateRankFit(p, fmt.Sprintf("pools[%d]", i), pool, c)
	}
	if len(d.Pools) > 0 && total != c.Nodes {
		p.Field("pools",
			"pool node counts sum to %d but the cluster declares %d", total, c.Nodes)
	}

	// Each offload tier draws from the cluster's declared storage inventory, as the Tier
	// doc states. The check fires only when the cluster lists storage: a cluster that
	// declares none states no inventory to constrain against, so a deployment offloading
	// against it is left as it was before the inventory existed, rather than rejected.
	if d.Offload != nil && len(c.Storage) > 0 {
		declared := make(map[string]bool, len(c.Storage))
		for _, s := range c.Storage {
			declared[s] = true
		}
		for i, t := range d.Offload.Tiers {
			if t.Device != "" && !declared[t.Device] {
				p.Field(fmt.Sprintf("offload.tiers[%d].tier", i),
					"%q is not in the cluster storage inventory %v", t.Device, c.Storage)
			}
		}
	}
	return p
}

// validateRankFit refuses an engine that needs more GPUs than it has been given. Every
// rank is one device — the engine's worker binds cuda:local_rank — so a layout's
// footprint is its rank count; nothing in the divisibility or node-sum checks relates the
// two, which is how tp 8 with pcp 4 on a single 8-GPU node validated while needing 32.
//
// Two bounds, each refusing only what no launch can run.
//
// The engine against its pool: pp x tp x pcp x dp ranks must not exceed the GPUs in the
// pool's nodes. This is the engine's own world_size_across_dp. DCP and expert parallelism
// add no ranks — DCP reuses the tensor-parallel ranks, and an expert group is built over
// ranks that already exist — so neither enters the product. It is an upper bound because
// a pool may hold several engines: llm-d runs a LeaderWorkerSet of `replicas` groups, each
// one engine over `size` nodes, so a 36-node pool of dp 8 is 36 engines of one node each.
//
// The local replicas against one node, when dp_local is stated. The engine places local
// data-parallel replica i at devices [i x world, i x world + local_world), where world is
// pp x tp x pcp and local_world is the share of one replica on one node
// (get_physical_gpu_ids_for_local_dp_rank). That share depends on the engine's node
// count n, which the schema does not carry and the launch chooses:
//
//   - n is 1 for each process llm-d starts — one engine per pod, joined by
//     --data-parallel-start-rank — so local_world is the whole replica.
//   - For n above 1, the engine splits each replica over n / (dp / dp_local) nodes
//     (nnodes_within_dp); that must divide world, which the multiprocessing executor
//     asserts, and local_world is world divided by it.
//
// So the bound asserted is: some n from 1 to the pool's node count gives a split under
// which every local replica's devices lie on the node. That refuses exactly what no
// launch can place — including a split that would need more nodes than the pool owns —
// and nothing the engine runs. It is the same in a release that rounds n / (dp /
// dp_local) down, and later releases, which require it to divide: rounding n down to a
// multiple gives the same split, so the set of placeable layouts is identical.
//
// When dp_local is unset the engine infers it from the launch, so no per-node verdict is
// possible and none is given.
//
// A width the checks above already reported as malformed describes no layout, so each
// bound declines to run over it rather than invent a count and file a second problem
// naming the wrong field. That matters for more than tidiness: two negative widths
// multiply to a large positive one. An unset pcp is the one zero that is meaningful — it
// means off, and counts as one. A cluster or pool that states no room constrains nothing,
// the way an undeclared storage inventory does. The per-node bound also declines when
// the pool bound has already refused the layout: an engine that does not fit at all does
// not need a second explanation of why it does not fit on a node.
//
// Counts are exact (math/big) rather than int: a width absurd enough to overflow must still
// compare correctly against a pool that is also absurdly large, and a message must never
// print a clamped number as though it were the real one.
func validateRankFit(p *validate.Problems, at string, pool Pool, c ClusterConstraints) {
	pl := pool.Parallel
	if pl.TP < 1 || pl.PP < 1 || pl.DP < 1 || pl.PCP < 0 ||
		pool.Nodes < 1 || c.GPUsPerNode < 1 {
		return
	}
	pcp := pl.PCP
	if pcp < 1 {
		pcp = 1
	}
	world := product(pl.PP, pl.TP, pcp)
	ranks := new(big.Int).Mul(world, big.NewInt(int64(pl.DP)))
	gpus := product(pool.Nodes, c.GPUsPerNode)
	if ranks.Cmp(gpus) > 0 {
		p.Field(at+".parallel",
			"needs %s GPUs (pp %d x tp %d x pcp %d x dp %d, one per rank) but the pool's %d node(s) of %d GPUs provide %s",
			ranks, pl.PP, pl.TP, pcp, pl.DP, pool.Nodes, c.GPUsPerNode, gpus)
		return
	}

	// dp_local unset, above dp, or negative states no replica count to place.
	if pl.DPLocal < 1 || pl.DPLocal > pl.DP {
		return
	}
	gpn := big.NewInt(int64(c.GPUsPerNode))
	last := new(big.Int).Mul(big.NewInt(int64(pl.DPLocal-1)), world)
	if last.Cmp(gpn) >= 0 {
		// The last local replica would start past the node under every split.
		p.Field(at+".parallel.dp_local",
			"%d local replicas of %s GPUs (pp %d x tp %d x pcp %d) are placed one after another from device 0, so the last would start at device %s, but a node has %d",
			pl.DPLocal, world, pl.PP, pl.TP, pcp, last, c.GPUsPerNode)
		return
	}
	// The last replica starts on the node; its share must end there too. Nothing here
	// fits a machine word by assumption — the pool's capacity is itself an exact product
	// and may exceed one, so world may too — so the search is exact throughout.
	groups := pl.DP / pl.DPLocal // the engine's data_parallel_node_size
	maxSplit := pool.Nodes / groups
	if maxSplit < 1 {
		maxSplit = 1 // n = 1 always gives a split of 1
	}
	room := new(big.Int).Sub(gpn, last) // positive: last < gpn
	switch placeable([]int{pl.PP, pl.TP, pcp}, world, int64(maxSplit), room) {
	case placeNo:
		p.Field(at+".parallel",
			"no node count up to the pool's %d lets the engine place %d local replicas of %s GPUs (pp %d x tp %d x pcp %d) on a %d-GPU node: replica i starts at device i x %s, so the last leaves %d GPUs for its share of the replica, and every split the pool allows (a divisor of %s over at most %d nodes) leaves a larger share",
			pool.Nodes, pl.DPLocal, world, pl.PP, pl.TP, pcp, c.GPUsPerNode, world, room, world, maxSplit)
	case placeUnknown:
		// The search is exact, but bounded so that no input can make validation hang.
		// No layout a cluster could hold reaches the bound. One that does is neither
		// refused without proof nor admitted in silence: it is reported as undecided.
		p.Warnf("%s.parallel: could not decide within the search limit whether %d local replicas of %s GPUs (pp %d x tp %d x pcp %d) can be placed on a %d-GPU node; this layout's per-node fit is unverified",
			at, pl.DPLocal, world, pl.PP, pl.TP, pcp, c.GPUsPerNode)
	}
}

type placement int

const (
	placeYes placement = iota
	placeNo
	placeUnknown
)

// placeableSearchLimit bounds the divisor walk in placeable. Real layouts need a handful
// of steps; the limit exists only so that no input can make validation hang. It is a
// variable so a test can lower it to exercise the undecided path.
var placeableSearchLimit = 1 << 22

// placeable decides whether some split k — a divisor of world, at most maxSplit — leaves a
// per-node share world/k of at most room. That is: does world have a divisor in
// [ceil(world/room), min(maxSplit, world)]?
//
// It is decided exactly, not by sampling. world is the product of the given widths, each
// an int, so each is factored exactly (factorInto) and the divisors of world no larger
// than the upper end are walked, stopping at the first that reaches the lower end. Every
// candidate fits a uint64, because it is at most maxSplit. The walk is bounded by
// placeableSearchLimit; past that the answer is placeUnknown, which the caller reports
// rather than treating as either verdict.
func placeable(widths []int, world *big.Int, maxSplit int64, room *big.Int) placement {
	// lo = ceil(world / room), the smallest split whose share fits.
	lo, rem := new(big.Int).QuoRem(world, room, new(big.Int))
	if rem.Sign() != 0 {
		lo.Add(lo, big.NewInt(1))
	}
	hi := big.NewInt(maxSplit)
	if world.Cmp(hi) < 0 {
		hi.Set(world)
	}
	if lo.Cmp(hi) > 0 {
		return placeNo
	}
	low, high := lo.Uint64(), hi.Uint64() // both at most maxSplit

	exp := map[uint64]int{}
	for _, w := range widths {
		if !factorInto(uint64(w), exp) {
			return placeUnknown
		}
	}
	primes := make([]uint64, 0, len(exp))
	for q := range exp {
		primes = append(primes, q)
	}
	// Largest primes first: a divisor reaching low is found in fewer steps.
	sort.Slice(primes, func(i, j int) bool { return primes[i] > primes[j] })

	steps, exhausted := 0, false
	var walk func(i int, k uint64) bool
	walk = func(i int, k uint64) bool {
		if k >= low {
			return true // every k walked is at most high
		}
		if i == len(primes) {
			return false
		}
		if steps++; steps > placeableSearchLimit {
			exhausted = true
			return false
		}
		q := primes[i]
		for e := 0; e <= exp[q]; e++ {
			if walk(i+1, k) {
				return true
			}
			if exhausted || k > high/q {
				return false
			}
			k *= q
		}
		return false
	}
	switch {
	case walk(0, 1):
		return placeYes
	case exhausted:
		return placeUnknown
	default:
		return placeNo
	}
}

// factorInto adds the prime factorisation of n (n >= 1) to exp, exactly, and reports
// whether it finished. Small primes are divided out first; what remains is split by
// Pollard's rho until every part is prime. Primality is big.Int.ProbablyPrime(0), which
// is documented to be exact for inputs below 2^64, so no factor is misclassified.
//
// Rho is not guaranteed to split a composite in any fixed time, and a validator must not
// hang, so it runs on a budget. If the budget runs out, factorInto reports false and the
// caller treats the placement as undecided — reported, never guessed.
func factorInto(n uint64, exp map[uint64]int) bool {
	for _, q := range []uint64{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} {
		for n%q == 0 {
			exp[q]++
			n /= q
		}
	}
	if n == 1 {
		return true
	}
	if new(big.Int).SetUint64(n).ProbablyPrime(0) {
		exp[n]++
		return true
	}
	d, ok := pollardRho(n)
	if !ok {
		return false
	}
	return factorInto(d, exp) && factorInto(n/d, exp)
}

// Pollard's rho budget. A composite below 2^63 with no factor under 38 is split in about
// n^(1/4) steps — under 60,000 at the top of the range — so 2^20 steps per constant is a
// margin of more than fifteen, and eight constants bound the worst case, a number every
// constant fails on, to about a second. rhoSteps is a variable so a test can exhaust it.
const rhoConstants = 8

var rhoSteps = 1 << 20

// pollardRho looks for a non-trivial factor of n, which must be composite, odd, and free
// of the small primes factorInto removes; for such n the first polynomial almost always
// succeeds. It reports false only if every constant within the budget fails.
func pollardRho(n uint64) (uint64, bool) {
	mulmod := func(a, b uint64) uint64 {
		hi, lo := bits.Mul64(a, b)
		return bits.Rem64(hi, lo, n)
	}
	for c := uint64(1); c <= rhoConstants; c++ {
		next := func(x uint64) uint64 {
			// x*x mod n + c, without overflow: both terms are below n, and n < 2^64.
			v, carry := bits.Add64(mulmod(x, x), c, 0)
			if carry != 0 || v >= n {
				v -= n
			}
			return v
		}
		x, y, d := uint64(2), uint64(2), uint64(1)
		for i := 0; d == 1 && i < rhoSteps; i++ {
			x = next(x)
			y = next(next(y))
			diff := x - y
			if y > x {
				diff = y - x
			}
			d = gcd(diff, n)
		}
		if d != 1 && d != n {
			return d, true
		}
	}
	return 0, false
}

func gcd(a, b uint64) uint64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// product multiplies non-negative ints exactly.
func product(xs ...int) *big.Int {
	out := big.NewInt(1)
	for _, x := range xs {
		out.Mul(out, big.NewInt(int64(x)))
	}
	return out
}

func (d *Deployment) validatePools(p *validate.Problems) {
	if len(d.Pools) == 0 {
		p.Field("pools", "at least one pool is required")
		return
	}

	roles := map[Role]int{}
	for i, pool := range d.Pools {
		at := fmt.Sprintf("pools[%d]", i)
		if !pool.Role.Valid() {
			p.Field(at+".role", "%q is not a recognized role", pool.Role)
		}
		roles[pool.Role]++
		if pool.Nodes < 1 {
			p.Field(at+".nodes", "must be at least 1")
		}
		validateParallelism(p, at, pool.Parallel)
	}

	// A disaggregated deployment needs both halves; one alone has nowhere to send or
	// receive KV, and a colocated pool alongside them describes two designs at once.
	if roles[RolePrefill] > 0 && roles[RoleDecode] == 0 {
		p.Field("pools", "a prefill pool with no decode pool has nowhere to send KV")
	}
	if roles[RoleDecode] > 0 && roles[RolePrefill] == 0 {
		p.Field("pools", "a decode pool with no prefill pool has no source of KV")
	}
	if roles[RoleColocated] > 0 && (roles[RolePrefill] > 0 || roles[RoleDecode] > 0) {
		p.Field("pools", "a colocated pool alongside disaggregated pools describes two deployments")
	}
	if roles[RolePrefill] > 0 && d.PDTransfer == nil {
		p.Field("pd_transfer", "required for a disaggregated deployment")
	}
}

func validateParallelism(p *validate.Problems, at string, pl Parallelism) {
	for field, v := range map[string]int{"tp": pl.TP, "pp": pl.PP, "dp": pl.DP} {
		if v < 1 {
			p.Field(at+".parallel."+field, "must be at least 1, got %d", v)
		}
	}
	if pl.PCP < 0 || pl.DCP < 0 {
		p.Field(at+".parallel", "context-parallel widths must not be negative")
	}
	// Whether prefill-context parallelism may be combined with data parallelism is
	// not a field check: it depends on the engine release. An earlier release refuses it; the
	// engine supports it from e6dc16cebd, and llm-d deploys it (DP 4 x PCP 8). A
	// release that refuses it says so in its rules pack.
	validateDecodeContextParallel(p, at, pl)
	// A load-balancing mode other than internal exposes ranks to a router; with one rank
	// there is nothing to expose. Which layouts each mode accepts beyond that is the
	// release's (rules packs).
	switch m := pl.DataParallelLoadBalancing; {
	case !m.Valid():
		p.Field(at+".parallel.data_parallel_load_balancing", "%q is not one of %v", m,
			[]DPLoadBalancing{DPLBInternal, DPLBHybrid, DPLBExternal})
	case m.Mode() != DPLBInternal && pl.DP <= 1:
		p.Field(at+".parallel.data_parallel_load_balancing", "%s load balancing needs dp > 1, got dp %d", m, pl.DP)
	}
	if pl.DPLocal < 0 {
		// The engine constrains it ge=0; zero is unset here, as there.
		p.Field(at+".parallel.dp_local", "must not be negative, got %d", pl.DPLocal)
	}
	if pl.DPLocal > pl.DP && pl.DP > 0 {
		p.Field(at+".parallel.dp_local", "%d exceeds dp %d", pl.DPLocal, pl.DP)
	}
	// Expert parallelism without a mixture-of-experts model shards nothing. That
	// cross-check needs the model graph, so it is a rules-pack concern; what is
	// checkable here is that the flag and the derived width agree.
	if pl.EnableExpertParallel && pl.ExpertParallelWidth() < 2 {
		p.Warnf("%s.parallel: expert parallelism is enabled but the derived width is 1, so no expert sharding occurs", at)
	}
}

// validateDecodeContextParallel constrains decode-context-parallel width against the
// group it is carved out of. DCP has no ranks of its own: it reuses the tensor-parallel
// ranks when prefill-context parallelism is off, and the TP x PCP block when it is on.
// A width that does not partition that group describes no layout, so it is refused when
// the document loads rather than priced downstream.
//
// Two rules, mutually exclusive on whether PCP is enabled:
//
//   - PCP off: the DCP group must partition the TP group, so tp % dcp == 0.
//   - PCP on: DCP may be off, span the PCP axis, or span the whole TP x PCP block.
//     Nothing between, because any other width would straddle the two axes.
//
// Both are structural — they follow from how the group is built, need no coefficients
// and no model graph — which is why they are field checks here rather than a
// version-scoped rule. They mirror what the engine itself refuses at startup.
func validateDecodeContextParallel(p *validate.Problems, at string, pl Parallelism) {
	// Problems accumulates rather than aborting, so a width already reported above
	// arrives here unchanged. These rules are stated over a real rank group, and a
	// malformed width describes none, so running them anyway does active harm in two
	// ways: it files a second problem naming dcp when the fault is another field, and
	// — worse — it can ACCEPT a dcp that the corrected width would reject, so the
	// author sees a fresh error only on a later run. That second round is exactly
	// what an accumulating problem list exists to prevent, so every width the checks
	// above already reported is a reason to stop here rather than guess at intent.
	//
	// A negative pcp is the instructive case. Normalising it to 1 below would route
	// the deployment into the prefill-context-parallelism-OFF branch and assert, in
	// the message, that pcp is off — when the document asked for it and merely asked
	// malformedly. tp 8, pcp -2, dcp 4 is the masking half: 8 % 4 is 0 so nothing is
	// reported, while at the evident pcp of 2 the admissible set is {1, 2, 16} and 4
	// is not in it.
	if pl.TP < 1 || pl.PCP < 0 || pl.DCP < 0 {
		return
	}
	// Both widths are omitempty, so an absent field arrives as 0 meaning "not enabled".
	// The rules are stated over enabled widths, so normalise before applying them —
	// otherwise tp % dcp divides by zero for every deployment that simply omits dcp.
	// Only a zero reaches this point: a negative returned above.
	pcp, dcp := pl.PCP, pl.DCP
	if pcp < 1 {
		pcp = 1
	}
	if dcp < 1 {
		dcp = 1
	}
	if pcp == 1 {
		if pl.TP%dcp != 0 {
			p.Field(at+".parallel.dcp",
				"decode-context parallelism reuses the tensor-parallel ranks when prefill-context parallelism is off, so tp %d must be divisible by dcp %d",
				pl.TP, dcp)
		}
		return
	}
	// Name the admissible set rather than only the violation: the reader's next
	// question is which widths would work.
	//
	// The full TP x PCP block is compared exactly. The engine's own product is Python's,
	// which does not overflow, and an int product would wrap: tp MaxInt, pcp 3 wraps to
	// MaxInt-2 and would admit a dcp the engine rejects.
	block := new(big.Int).Mul(big.NewInt(int64(pl.TP)), big.NewInt(int64(pcp)))
	if dcp != 1 && dcp != pcp && block.Cmp(big.NewInt(int64(dcp))) != 0 {
		p.Field(at+".parallel.dcp",
			"with prefill-context parallelism enabled, decode-context parallelism must be disabled, span the pcp axis, or span the full tp x pcp axis; got tp %d, pcp %d, dcp %d, so the admissible widths are %v",
			pl.TP, pcp, dcp, admissibleDCP(pcp, block))
	}
}

// admissibleDCP returns the sorted, deduplicated widths the PCP-enabled rule allows:
// off, the pcp axis, or the full tp x pcp block. Deduplication matters because at tp 1
// the block IS the pcp axis, so a literal three-element list would repeat a width and
// read as though it were three distinct choices. The block is exact, so the set printed
// is the set the engine would name.
func admissibleDCP(pcp int, block *big.Int) []*big.Int {
	out := []*big.Int{big.NewInt(1)}
	for _, w := range []*big.Int{big.NewInt(int64(pcp)), block} {
		dup := false
		for _, have := range out {
			if have.Cmp(w) == 0 {
				dup = true
			}
		}
		if !dup {
			out = append(out, w)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Cmp(out[j]) < 0 })
	return out
}

func (d *Deployment) validateOffload(p *validate.Problems) {
	o := d.Offload
	if o == nil {
		return
	}
	if len(o.Tiers) == 0 {
		p.Field("offload.tiers", "an offload block with no tiers offloads nowhere")
	}
	seen := map[string]bool{}
	for i, t := range o.Tiers {
		at := fmt.Sprintf("offload.tiers[%d]", i)
		if t.Device == "" {
			p.Field(at+".tier", "required: names a catalog device class")
		} else if seen[t.Device] {
			p.Field(at+".tier", "duplicate tier %q", t.Device)
		}
		seen[t.Device] = true
		if t.Bytes < 1 {
			p.Field(at+".bytes", "must be positive")
		}
	}
	if o.PrefetchDepth < 0 {
		p.Field("offload.prefetch_depth", "must not be negative")
	}
}
