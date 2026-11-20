package deployment

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// pdDeployment is a prefill/decode-disaggregated deployment of the kind the design
// documents work through: two pools, different backends, expert parallelism. Its pool
// node counts sum to 60, which is the cluster ValidateAgainstCluster is tested against.
func pdDeployment() *Deployment {
	return &Deployment{
		Kind: "Deployment", Name: "glm-5.3-60node-pd-h200",
		Pools: []Pool{
			{Role: RolePrefill, Nodes: 36,
				Parallel: Parallelism{TP: 1, PP: 1, DP: 8, DPLocal: 8,
					EnableExpertParallel: true},
				Engine: engineYAML("{all2all_backend: deepep_high_throughput, allreduce_backend: nccl, " +
					"cudagraph_mode: PIECEWISE, block_size: 64, max_num_batched_tokens: 8192, " +
					"max_num_seqs: 256, cache_dtype: fp8, gpu_memory_utilization: 0.9}")},
			{Role: RoleDecode, Nodes: 24,
				Parallel: Parallelism{TP: 1, PP: 1, DP: 16, DPLocal: 8,
					EnableExpertParallel: true},
				Engine: engineYAML("{all2all_backend: deepep_low_latency, allreduce_backend: nccl, " +
					"cudagraph_mode: FULL_AND_PIECEWISE, block_size: 64, cache_dtype: fp8, " +
					"gpu_memory_utilization: 0.9, eplb: {enable_eplb: true, num_redundant_experts: 32}}")},
		},
		PDTransfer: &PDTransfer{Connector: "nixl"},
	}
}

// engineYAML is a pool's engine block as a document states it, held unopened.
func engineYAML(doc string) EngineBlock {
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(doc), &n); err != nil {
		panic(err)
	}
	return NewEngineBlock(n.Content[0])
}

// singleNode is the colocated shape the published Granite runs use: eight GPUs,
// tensor parallel eight, one node.
func singleNodeDeployment() *Deployment {
	return &Deployment{
		Kind: "Deployment", Name: "granite-230b-h100-tp8",
		Pools: []Pool{{Role: RoleColocated, Nodes: 1,
			Parallel: Parallelism{TP: 8, PP: 1, DP: 1},
			Engine: engineYAML("{cache_dtype: fp8, max_num_batched_tokens: 32768, " +
				"gpu_memory_utilization: 0.9, block_size: 16}")}},
	}
}

func TestValidDeploymentsPass(t *testing.T) {
	for _, d := range []*Deployment{pdDeployment(), singleNodeDeployment()} {
		if p := d.Validate(); !p.OK() {
			t.Fatalf("%s rejected:\n%s", d.Name, p.Error())
		}
	}
}

// TestExpertParallelWidthIsDerived pins the rule that EP has no field: the width is
// a function of the other three, and no document can disagree with it.
func TestExpertParallelWidthIsDerived(t *testing.T) {
	cases := []struct {
		name string
		p    Parallelism
		want int
	}{
		{"off", Parallelism{TP: 8, DP: 1}, 1},
		{"tp1 dp8", Parallelism{TP: 1, DP: 8, EnableExpertParallel: true}, 8},
		{"tp1 dp16", Parallelism{TP: 1, DP: 16, EnableExpertParallel: true}, 16},
		{"tp2 dp8", Parallelism{TP: 2, DP: 8, EnableExpertParallel: true}, 16},
		{"pcp wider than dp", Parallelism{TP: 2, DP: 1, PCP: 4, EnableExpertParallel: true}, 8},
		{"zero guards", Parallelism{EnableExpertParallel: true}, 1},
		// The llm-d glm-5.3 canary: the engine's expert group spans DP x PCP x TP, which
		// its manifest records as EP32. The max form would have said 8.
		{"pcp with dp", Parallelism{TP: 1, DP: 4, PCP: 8, EnableExpertParallel: true}, 32},
		{"tp pcp and dp", Parallelism{TP: 2, DP: 4, PCP: 2, EnableExpertParallel: true}, 16},
		{"saturates", Parallelism{TP: math.MaxInt, DP: 2, EnableExpertParallel: true}, math.MaxInt},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.p.ExpertParallelWidth(); got != c.want {
				t.Errorf("width = %d, want %d", got, c.want)
			}
		})
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Deployment)
	}{
		{"wrong kind", func(d *Deployment) { d.Kind = "Scenario" }},
		{"no name", func(d *Deployment) { d.Name = "" }},
		{"no pools", func(d *Deployment) { d.Pools = nil }},
		{"unknown role", func(d *Deployment) { d.Pools[0].Role = "warmup" }},
		{"zero tp", func(d *Deployment) { d.Pools[0].Parallel.TP = 0 }},
		{"dp_local exceeds dp", func(d *Deployment) {
			d.Pools[0].Parallel.DPLocal = 99
		}},
		{"prefill without decode", func(d *Deployment) {
			d.Pools = d.Pools[:1]
		}},
		{"decode without prefill", func(d *Deployment) {
			d.Pools = d.Pools[1:]
		}},
		{"disaggregated without pd_transfer", func(d *Deployment) { d.PDTransfer = nil }},
		{"offload with no tiers", func(d *Deployment) { d.Offload = &Offload{} }},
		{"offload tier with zero bytes", func(d *Deployment) {
			d.Offload = &Offload{Tiers: []Tier{{Device: "cpu_dram", Bytes: 0}}}
		}},
		{"duplicate offload tier", func(d *Deployment) {
			d.Offload = &Offload{Tiers: []Tier{
				{Device: "cpu_dram", Bytes: 1}, {Device: "cpu_dram", Bytes: 2}}}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := pdDeployment()
			tc.mutate(d)
			if p := d.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// TestColocatedAlongsideDisaggregated covers the mixed-role case separately, since
// it needs three pools rather than a one-field mutation.
func TestColocatedAlongsideDisaggregated(t *testing.T) {
	d := pdDeployment()
	d.Pools = append(d.Pools, Pool{Role: RoleColocated, Nodes: 4,
		Parallel: Parallelism{TP: 1, PP: 1, DP: 1}})
	if p := d.Validate(); p.OK() {
		t.Fatal("expected rejection of a mixed-role deployment")
	}
}

// would supply.
func TestValidateAgainstCluster(t *testing.T) {
	// The pdDeployment pools sum to 60 nodes of 8 GPUs, which the matching cluster
	// declares, and every dp_local of 8 divides the node.
	if p := pdDeployment().ValidateAgainstCluster(ClusterConstraints{Nodes: 60, GPUsPerNode: 8}); !p.OK() {
		t.Fatalf("a deployment that fills its cluster should pass:\n%s", p.Error())
	}
	// Pool node counts that do not sum to the cluster are rejected.
	if p := pdDeployment().ValidateAgainstCluster(ClusterConstraints{Nodes: 59, GPUsPerNode: 8}); p.OK() {
		t.Error("pool nodes summing to 60 against a 59-node cluster should fail")
	}
	// A data-parallel-local width that does not divide the node is rejected.
	d := pdDeployment()
	d.Pools[0].Parallel.DPLocal = 3
	if p := d.ValidateAgainstCluster(ClusterConstraints{Nodes: 60, GPUsPerNode: 8}); p.OK() {
		t.Error("dp_local of 3 does not divide an 8-GPU node and should fail")
	}
}

// TestValidateAgainstClusterStorage covers the coupling check that an offload tier may
// only name a storage class the cluster's inventory declares. The inventory lives on the
// Scenario, so the composition layer supplies it; a cluster that lists no storage
// constrains nothing, which keeps a deployment that offloads without a declared
// inventory valid as it was before the inventory existed.
func TestValidateAgainstClusterStorage(t *testing.T) {
	withOffload := func(devices ...string) *Deployment {
		d := singleNodeDeployment()
		tiers := make([]Tier, len(devices))
		for i, dev := range devices {
			tiers[i] = Tier{Device: dev, Bytes: 1}
		}
		d.Offload = &Offload{Tiers: tiers}
		return d
	}
	inventory := []string{"cpu_dram", "nvme_gen4"}

	// A tier drawn from the declared inventory passes.
	if p := withOffload("cpu_dram").ValidateAgainstCluster(ClusterConstraints{Nodes: 1, GPUsPerNode: 8, Storage: inventory}); !p.OK() {
		t.Errorf("a tier in the inventory should pass:\n%s", p.Error())
	}
	// A tier naming a class the cluster does not list is rejected.
	if p := withOffload("optane").ValidateAgainstCluster(ClusterConstraints{Nodes: 1, GPUsPerNode: 8, Storage: inventory}); p.OK() {
		t.Error("a tier outside the cluster storage inventory should fail")
	}
	// A cluster that declares no storage inventory constrains nothing: the same offload
	// that would fail above passes, preserving pre-inventory behavior.
	if p := withOffload("optane").ValidateAgainstCluster(ClusterConstraints{Nodes: 1, GPUsPerNode: 8}); !p.OK() {
		t.Errorf("an undeclared inventory should not constrain offload:\n%s", p.Error())
	}
}

// TestDecodeContextParallelWidth covers the two rules that constrain decode-context
// parallelism against the rank group it is carved out of. Every case below is a
// Parallelism block alone, because that is all the rules read: no cluster, no model,
// no coefficients. The accepted cases are as much the point as the rejected ones —
// dcp is omitempty, so an absent field must stay valid rather than divide by zero.
func TestDecodeContextParallelWidth(t *testing.T) {
	cases := []struct {
		name   string
		p      Parallelism
		reject bool
	}{
		// PCP off: the dcp group must partition the tp group it reuses.
		{"dcp absent", Parallelism{TP: 8, PP: 1, DP: 1}, false},
		{"dcp disabled", Parallelism{TP: 8, PP: 1, DP: 1, DCP: 1}, false},
		{"dcp divides tp", Parallelism{TP: 8, PP: 1, DP: 1, DCP: 2}, false},
		{"dcp half of tp", Parallelism{TP: 8, PP: 1, DP: 1, DCP: 4}, false},
		{"dcp equals tp", Parallelism{TP: 8, PP: 1, DP: 1, DCP: 8}, false},
		{"dcp does not divide tp", Parallelism{TP: 8, PP: 1, DP: 1, DCP: 3}, true},
		{"dcp wider than tp", Parallelism{TP: 8, PP: 1, DP: 1, DCP: 16}, true},

		// PCP on: dcp may be off, span the pcp axis, or span the whole tp x pcp block.
		{"pcp on, dcp disabled", Parallelism{TP: 4, PP: 1, DP: 1, PCP: 2, DCP: 1}, false},
		{"pcp on, dcp spans pcp", Parallelism{TP: 4, PP: 1, DP: 1, PCP: 2, DCP: 2}, false},
		{"pcp on, dcp spans tp x pcp", Parallelism{TP: 4, PP: 1, DP: 1, PCP: 2, DCP: 8}, false},
		{"pcp on, dcp absent", Parallelism{TP: 4, PP: 1, DP: 1, PCP: 2}, false},
		{"pcp on, dcp between the admissible widths", Parallelism{TP: 4, PP: 1, DP: 1, PCP: 2, DCP: 3}, true},
		// 4 divides tp, which the pcp-off rule would accept; with pcp on it is not in
		// {1, 2, 8}. This is the case that fails if the branch is written as one rule.
		{"pcp on, dcp divides tp but is not admissible", Parallelism{TP: 4, PP: 1, DP: 1, PCP: 2, DCP: 4}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := singleNodeDeployment()
			d.Pools[0].Parallel = c.p
			p := d.Validate()
			if !c.reject {
				if !p.OK() {
					t.Fatalf("expected %+v to validate:\n%s", c.p, p.Error())
				}
				return
			}
			if p.OK() {
				t.Fatalf("expected rejection of %+v, got none", c.p)
			}
			// The rejection must be attributed to dcp, not merely present. The path
			// is the load-bearing half of the diagnostic: it is what tells an author
			// which field to edit, and asserting only OK() would hold just as well if
			// the problem were filed against tp or the pool. Without this, both
			// p.Field paths in validateDecodeContextParallel can be renamed and the
			// whole suite stays green.
			const want = "pools[0].parallel.dcp"
			for _, problem := range p.All() {
				if problem.Path == want {
					return
				}
			}
			t.Errorf("expected a problem at %s, got:\n%s", want, p.Error())
		})
	}
}

// TestDecodeContextParallelWithInvalidTP pins the guard ordering. Problems accumulates
// rather than aborting, so validation continues past the tp check with tp unchanged:
// the divisibility rules must not be reached there, or a deployment already reported
// for its tp gains a second problem about dcp that points a reader at the wrong field.
//
// Each case below is one the guard is the only thing preventing. The first two are
// chosen because they DISTINGUISH a guarded implementation from an unguarded one:
// delete the guard and each gains a dcp problem. A tp of 0 with dcp 4 and pcp absent
// does not — 0 % 4 is 0, so the modulus branch is silent either way — which is why it
// cannot be the only case here, though it is kept as the plain omitted-pcp shape.
func TestDecodeContextParallelWithInvalidTP(t *testing.T) {
	cases := []struct {
		name string
		p    Parallelism
	}{
		// Reaches the modulus branch and reports a negative group as divisible.
		{"negative tp, pcp off", Parallelism{TP: -3, PP: 1, DP: 1, DCP: 2}},
		// Reaches the membership branch and names an admissible set containing 0.
		{"zero tp, pcp on", Parallelism{TP: 0, PP: 1, DP: 1, PCP: 2, DCP: 3}},
		// Silent in either implementation; kept so the ordinary shape is covered.
		{"zero tp, pcp off", Parallelism{TP: 0, PP: 1, DP: 1, DCP: 4}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := singleNodeDeployment()
			d.Pools[0].Parallel = c.p
			// Validate must return rather than panic, which is itself the assertion:
			// a panic here fails the test.
			p := d.Validate()
			if p.OK() {
				t.Fatal("a tp below 1 should be rejected")
			}
			sawTP := false
			for _, problem := range p.All() {
				if strings.Contains(problem.Path, "dcp") {
					t.Errorf("a deployment already reported for tp should gain no dcp problem, got %s", problem)
				}
				if strings.HasSuffix(problem.Path, ".tp") {
					sawTP = true
				}
			}
			// The one real fault must still be named; a guard that suppressed
			// everything would pass the check above for the wrong reason.
			if !sawTP {
				t.Errorf("expected the tp problem to be reported:\n%s", p.Error())
			}
		})
	}
}

// TestNegativeContextParallelWidth covers the other way a width can arrive outside the
// enabled range. A negative one is reported by the non-negativity check, and the rules
// then decline to run: a malformed width describes no rank group, so any divisibility
// verdict over it would be invented.
//
// The pcp cases are the ones that matter, and they are regressions. Normalising a
// negative pcp to 1 — rather than returning — routes the deployment into the
// prefill-context-parallelism-OFF branch and files a dcp problem whose message asserts
// pcp is off, when the document asked for it. The last case is the masking half: at
// pcp -2 the off-branch finds 8 % 4 == 0 and says nothing, while at the evident pcp of
// 2 the admissible set is {1, 2, 16} and dcp 4 is not in it — so the author would meet
// that error only on a second run, after fixing pcp.
func TestNegativeContextParallelWidth(t *testing.T) {
	cases := []struct {
		name string
		p    Parallelism
	}{
		{"negative dcp", Parallelism{TP: 8, PP: 1, DP: 1, DCP: -1}},
		{"negative pcp", Parallelism{TP: 8, PP: 1, DP: 1, PCP: -1}},
		{"negative pcp with an indivisible dcp", Parallelism{TP: 8, PP: 1, DP: 1, PCP: -1, DCP: 3}},
		{"negative pcp masking an inadmissible dcp", Parallelism{TP: 8, PP: 1, DP: 1, PCP: -2, DCP: 4}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := singleNodeDeployment()
			d.Pools[0].Parallel = c.p
			p := d.Validate()
			if p.OK() {
				t.Fatalf("a negative context-parallel width should be rejected: %+v", c.p)
			}
			sawWidth := false
			for _, problem := range p.All() {
				if strings.Contains(problem.Path, "dcp") {
					t.Errorf("a malformed width should yield no dcp verdict, got %s", problem)
				}
				if problem.Path == "pools[0].parallel" {
					sawWidth = true
				}
			}
			// The real fault must still be named, or a guard that suppressed
			// everything would satisfy the check above for the wrong reason.
			if !sawWidth {
				t.Errorf("expected the non-negativity problem to be reported:\n%s", p.Error())
			}
		})
	}
}

// TestPCPOnMessageNamesTheAdmissibleSet pins the set a reader is told they may use,
// read from the problem Validate reports. Issue #39 asks for the admissible widths
// rather than only the violation, because a reader's next question after "3 is wrong"
// is "then what works".
//
// The set must be ascending and must not repeat a width. The tp 1 row is the one that
// makes repetition possible: the full TP x PCP block IS the pcp axis there, so a naive
// three-element list would print "[1 2 2]" and read as three distinct choices.
func TestPCPOnMessageNamesTheAdmissibleSet(t *testing.T) {
	cases := []struct {
		tp, pcp, dcp int
		want         string
	}{
		{4, 2, 3, "[1 2 8]"},
		{8, 2, 3, "[1 2 16]"},
		{1, 2, 3, "[1 2]"}, // the block equals the axis: two widths, not three
		{2, 3, 4, "[1 3 6]"},
	}
	for _, c := range cases {
		t.Run(fmt.Sprintf("tp%d_pcp%d", c.tp, c.pcp), func(t *testing.T) {
			d := singleNodeDeployment()
			d.Pools[0].Parallel = Parallelism{TP: c.tp, PP: 1, DP: 1, PCP: c.pcp, DCP: c.dcp}
			for _, problem := range d.Validate().All() {
				if problem.Path == "pools[0].parallel.dcp" {
					if !strings.Contains(problem.Message, "admissible widths are "+c.want) {
						t.Errorf("want the admissible set %s, got: %s", c.want, problem.Message)
					}
					return
				}
			}
			t.Fatalf("tp %d pcp %d dcp %d should be rejected at parallel.dcp", c.tp, c.pcp, c.dcp)
		})
	}
}

// TestDecodeContextParallelMessagesNameTheFacts pins the load-bearing CONTENT of the two
// new problems, which no assertion on a verdict or a path can reach. A message reduced
// to "invalid" would satisfy every other test in this file while telling an author
// nothing they can act on.
//
// It asserts substrings rather than whole wordings deliberately: the contract from issue
// #39 is that the message names the offending width and, for the PCP-on branch, the set
// that would be accepted. Pinning the full sentence would couple these tests to
// rephrasings that change nothing a reader depends on.
func TestDecodeContextParallelMessagesNameTheFacts(t *testing.T) {
	find := func(t *testing.T, pl Parallelism) string {
		t.Helper()
		d := singleNodeDeployment()
		d.Pools[0].Parallel = pl
		for _, problem := range d.Validate().All() {
			if problem.Path == "pools[0].parallel.dcp" {
				return problem.Message
			}
		}
		t.Fatalf("no dcp problem for %+v", pl)
		return ""
	}

	// PCP off: the message must name both sides of the divisibility it is asserting,
	// since "8 must be divisible by 3" is actionable and "invalid" is not.
	msg := find(t, Parallelism{TP: 8, PP: 1, DP: 1, DCP: 3})
	for _, want := range []string{"8", "3"} {
		if !strings.Contains(msg, want) {
			t.Errorf("pcp-off message does not name %q: %s", want, msg)
		}
	}

	// PCP on: the message must name the admissible set, which is the half of the
	// contract that tells a reader what to change the width TO.
	msg = find(t, Parallelism{TP: 4, PP: 1, DP: 1, PCP: 2, DCP: 3})
	for _, want := range []string{"[1 2 8]", "3"} {
		if !strings.Contains(msg, want) {
			t.Errorf("pcp-on message does not name %q: %s", want, msg)
		}
	}
}

// TestPCPWithDPIsNotAFieldProblem: whether prefill-context parallelism may be combined
// with data parallelism depends on the engine release, so field validation must admit
// it. The engine supports it from e6dc16cebd and llm-d deploys it — the glm-5.3 canary
// runs DP 4 x PCP 8 x DCP 8 at TP 1 — so a field check refusing it would reject a real
// deployment for every engine version. v0.29.0's refusal lives in its rules pack.
func TestPCPWithDPIsNotAFieldProblem(t *testing.T) {
	d := singleNodeDeployment()
	d.Pools[0].Nodes = 4
	d.Pools[0].Parallel = Parallelism{TP: 1, PP: 1, DP: 4, PCP: 8, DCP: 8, DPLocal: 1,
		EnableExpertParallel: true}
	if p := d.Validate(); !p.OK() {
		t.Fatalf("DP 4 x PCP 8 should be admitted by field validation:\n%s", p.Error())
	}
	if p := d.ValidateAgainstCluster(ClusterConstraints{Nodes: 4, GPUsPerNode: 8}); !p.OK() {
		t.Fatalf("DP 4 x PCP 8 on four 8-GPU nodes should fit:\n%s", p.Error())
	}
	// The DCP rules still apply with data parallelism present.
	d.Pools[0].Parallel.DCP = 3
	if p := d.Validate(); p.OK() {
		t.Fatal("dcp 3 is not in {1, 8} and should be rejected whatever dp is")
	}
}

// rankFit runs only the cluster-coupling checks over one pool, so a case states the
// layout and the room it has and nothing else. The deployment around it is the ordinary
// single-pool shape, and the cluster is sized to that pool, which keeps the node-sum
// check out of the way of what is being tested.
func rankFit(pl Parallelism, nodes, gpusPerNode int) *validate.Problems {
	d := singleNodeDeployment()
	d.Pools[0].Nodes = nodes
	d.Pools[0].Parallel = pl
	return d.ValidateAgainstCluster(ClusterConstraints{Nodes: nodes, GPUsPerNode: gpusPerNode})
}

// TestRankFit is the rule that an engine cannot need more GPUs than it was given. Every
// rank is one device, and before this nothing related a layout's rank count to the
// inventory it ran on: tp 8 with pcp 4 on one 8-GPU node needs 32 devices and validated.
//
// The accepted cases are as much the point as the rejected ones. A pool is the set of
// nodes serving a role and may hold several engines, so a rank count BELOW the pool's
// GPUs is ordinary, and the bound is an inequality — an equality would reject the
// published 36-node prefill pool whose layout is dp 8.
func TestRankFit(t *testing.T) {
	cases := []struct {
		name        string
		pl          Parallelism
		nodes, gpus int
		reject      bool
	}{
		// The case that motivated the rule.
		{"tp8 pcp4 on one 8-GPU node", Parallelism{TP: 8, PP: 1, DP: 1, PCP: 4}, 1, 8, true},

		// Each axis counts. One case per factor of the product, so dropping any one
		// from the arithmetic fails exactly one of these.
		{"tp alone fills the node", Parallelism{TP: 8, PP: 1, DP: 1}, 1, 8, false},
		{"tp over the node", Parallelism{TP: 16, PP: 1, DP: 1}, 1, 8, true},
		{"pp counts", Parallelism{TP: 4, PP: 4, DP: 1}, 1, 8, true},
		{"pp within the node", Parallelism{TP: 4, PP: 2, DP: 1}, 1, 8, false},
		{"dp counts", Parallelism{TP: 2, PP: 1, DP: 8}, 1, 8, true},
		{"dp within the node", Parallelism{TP: 2, PP: 1, DP: 4}, 1, 8, false},
		{"pcp counts", Parallelism{TP: 2, PP: 1, DP: 1, PCP: 8}, 1, 8, true},
		{"pcp within the node", Parallelism{TP: 2, PP: 1, DP: 1, PCP: 4}, 1, 8, false},

		// Neither DCP nor expert parallelism adds a rank: DCP reuses the tensor-parallel
		// ranks and an expert group is carved across ranks that already exist. Rejecting
		// either would refuse layouts the engine runs.
		{"dcp adds no ranks", Parallelism{TP: 8, PP: 1, DP: 1, DCP: 8}, 1, 8, false},
		{"expert parallelism adds no ranks",
			Parallelism{TP: 8, PP: 1, DP: 1, EnableExpertParallel: true}, 1, 8, false},

		// An engine may span nodes; it needs the pool to own enough of them.
		{"two-node engine on two nodes", Parallelism{TP: 8, PP: 2, DP: 1}, 2, 8, false},
		{"two-node engine on one node", Parallelism{TP: 8, PP: 2, DP: 1}, 1, 8, true},

		// A pool holds several engines, so the layout may be far smaller than the pool.
		{"one dp-8 engine in a 36-node pool", Parallelism{TP: 1, PP: 1, DP: 8, DPLocal: 8}, 36, 8, false},
		{"unset pcp counts as one", Parallelism{TP: 8, PP: 1, DP: 1, PCP: 0}, 1, 8, false},

		// A width that would wrap must read as too many, not as a small number.
		{"overflowing product", Parallelism{TP: math.MaxInt, PP: 2, DP: 1}, 1, 8, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := rankFit(c.pl, c.nodes, c.gpus)
			if !c.reject {
				if !p.OK() {
					t.Fatalf("%+v on %d x %d should fit:\n%s", c.pl, c.nodes, c.gpus, p.Error())
				}
				return
			}
			if p.OK() {
				t.Fatalf("%+v on %d x %d needs more GPUs than it has and should be rejected",
					c.pl, c.nodes, c.gpus)
			}
			// Attributed to the parallelism block, where the author edits the layout.
			for _, problem := range p.All() {
				if problem.Path == "pools[0].parallel" {
					return
				}
			}
			t.Errorf("expected a problem at pools[0].parallel, got:\n%s", p.Error())
		})
	}
}

// TestRankFitMessageNamesTheArithmetic pins the content a reader acts on: the needed
// count, each factor it was built from, and what the pool provides. "does not fit"
// alone would leave them to redo the product to find which axis to shrink.
func TestRankFitMessageNamesTheArithmetic(t *testing.T) {
	p := rankFit(Parallelism{TP: 8, PP: 1, DP: 1, PCP: 4}, 1, 8)
	var msg string
	for _, problem := range p.All() {
		if problem.Path == "pools[0].parallel" {
			msg = problem.Message
		}
	}
	if msg == "" {
		t.Fatalf("no rank-fit problem:\n%s", p.Error())
	}
	for _, want := range []string{"32", "pp 1", "tp 8", "pcp 4", "dp 1", "provide 8"} {
		if !strings.Contains(msg, want) {
			t.Errorf("message does not contain %q: %s", want, msg)
		}
	}
}

// perNodeRefusal returns the per-node problem, if any. There are two messages — the last
// local replica starts past the node, or no split the pool allows leaves its share on the
// node — and they are matched on content, because the older dp_local checks share a path.
func perNodeRefusal(p *validate.Problems) string {
	for _, problem := range p.Errors() {
		if strings.Contains(problem.Message, "would start at device") ||
			strings.Contains(problem.Message, "no node count up to") {
			return problem.Message
		}
	}
	return ""
}

// TestRankFitPerNode covers the bound the total cannot see. The engine places local
// data-parallel replica i at devices [i x world, i x world + share), where world is
// pp x tp x pcp and share is that replica's part on one node, fixed by the engine's
// node count. A layout is refused when no node count up to the pool's lets every
// local replica's devices lie on the node.
func TestRankFitPerNode(t *testing.T) {
	cases := []struct {
		name        string
		pl          Parallelism
		nodes, gpus int
		reject      bool
	}{
		// The total cannot see this one: 16 ranks on 16 GPUs, last replica at device 12.
		{"fits the pool but not a node",
			Parallelism{TP: 4, PP: 1, DP: 4, DPLocal: 4}, 2, 8, true},
		{"replicas fill a node exactly",
			Parallelism{TP: 4, PP: 1, DP: 4, DPLocal: 2}, 2, 8, false},
		{"pp and pcp count toward the replica",
			Parallelism{TP: 2, PP: 2, DP: 4, PCP: 1, DPLocal: 4}, 2, 8, true},
		{"last replica would start exactly at the node's end",
			Parallelism{TP: 4, PP: 1, DP: 2, DPLocal: 2}, 2, 4, true},
		{"wide replica with two local replicas",
			Parallelism{TP: 16, PP: 1, DP: 4, DPLocal: 2}, 8, 8, true},
		{"wide replica with one local replica, split over the pool",
			Parallelism{TP: 8, PP: 2, DP: 1, DPLocal: 1}, 2, 8, false},

		// From review: tp 6 with two local replicas. On three nodes each replica splits
		// 2 / 2 / 2 and the second takes devices [6, 8). On two nodes no split works —
		// one node puts the second at [6, 12), two nodes at [6, 9) — and the engine
		// raises. Only the node count differs.
		{"tp6 dp_local 2 on three nodes", Parallelism{TP: 6, PP: 1, DP: 2, DPLocal: 2}, 3, 8, false},
		{"tp6 dp_local 2 on two nodes", Parallelism{TP: 6, PP: 1, DP: 2, DPLocal: 2}, 2, 8, true},
		// One local replica of 12 on 8-GPU nodes must split over two nodes, which with two
		// node groups (dp 2 / dp_local 1) needs an engine of four: the pool has three.
		{"split needs more nodes than the pool has",
			Parallelism{TP: 12, PP: 1, DP: 2, DPLocal: 1}, 3, 8, true},
		{"the same split with the nodes it needs",
			Parallelism{TP: 12, PP: 1, DP: 2, DPLocal: 1}, 4, 8, false},
		// A split must divide the replica: tp 9 cannot be shared evenly by two nodes.
		{"no even split exists", Parallelism{TP: 9, PP: 1, DP: 1, DPLocal: 1}, 2, 8, true},
		{"three nodes split it evenly", Parallelism{TP: 9, PP: 1, DP: 1, DPLocal: 1}, 3, 8, false},

		// llm-d shapes, from internal llm-d deployment manifests: one vLLM per pod, dp_local
		// replicas of tp each on every node. All must fit.
		{"llm-d wide-EP decode, LWS size 2", Parallelism{TP: 1, PP: 1, DP: 16, DPLocal: 8,
			EnableExpertParallel: true}, 2, 8, false},
		{"llm-d wide-EP prefill, 3 x LWS size 2", Parallelism{TP: 1, PP: 1, DP: 16, DPLocal: 8,
			EnableExpertParallel: true}, 6, 8, false},
		{"llm-d glm-5.2 decode, LWS size 4", Parallelism{TP: 1, PP: 1, DP: 32, DPLocal: 8,
			EnableExpertParallel: true}, 4, 8, false},
		{"llm-d canary DP4 x PCP8", Parallelism{TP: 1, PP: 1, DP: 4, PCP: 8, DCP: 8, DPLocal: 1,
			EnableExpertParallel: true}, 4, 8, false},
		{"llm-d decode DP16 x TP2", Parallelism{TP: 2, PP: 1, DP: 16, DPLocal: 4,
			EnableExpertParallel: true}, 4, 8, false},

		// dp_local unset: the engine infers it from the launch, so no verdict is given.
		{"dp_local unset", Parallelism{TP: 4, PP: 1, DP: 4}, 2, 8, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := rankFit(c.pl, c.nodes, c.gpus)
			got := perNodeRefusal(p)
			if c.reject && got == "" {
				t.Fatalf("%+v on %d x %d: expected the per-node bound to fire, got:\n%s",
					c.pl, c.nodes, c.gpus, p.Error())
			}
			if !c.reject && !p.OK() {
				t.Fatalf("%+v on %d x %d should fit, got:\n%s", c.pl, c.nodes, c.gpus, p.Error())
			}
		})
	}
}

// TestRankFitPerNodeSplitMessage pins what the split-case message tells a reader: the
// node count it searched up to, the replica size, and the room the last replica has.
func TestRankFitPerNodeSplitMessage(t *testing.T) {
	msg := perNodeRefusal(rankFit(Parallelism{TP: 6, PP: 1, DP: 2, DPLocal: 2}, 2, 8))
	for _, want := range []string{"up to the pool's 2", "2 local replicas of 6 GPUs", "8-GPU node", "leaves 2 GPUs"} {
		if !strings.Contains(msg, want) {
			t.Errorf("message does not contain %q: %s", want, msg)
		}
	}
}

// TestPerNodeDeclinesWhenThePoolBoundFired: an engine that does not fit its pool at all
// is told so once. The per-node bound would otherwise add a second refusal of the same
// layout, pointing at dp_local when the fix may be any width or the node count.
func TestPerNodeDeclinesWhenThePoolBoundFired(t *testing.T) {
	p := rankFit(Parallelism{TP: 16, PP: 1, DP: 4, DPLocal: 2}, 1, 8)
	if p.OK() {
		t.Fatal("64 ranks on 8 GPUs must be refused")
	}
	if got := perNodeRefusal(p); got != "" {
		t.Errorf("per-node bound fired after the pool bound: %s", got)
	}
}

// TestRankFitPerNodeMessage pins the arithmetic a reader acts on: where the last local
// replica would start and how big a node is.
func TestRankFitPerNodeMessage(t *testing.T) {
	p := rankFit(Parallelism{TP: 16, PP: 1, DP: 4, DPLocal: 2}, 8, 8)
	for _, problem := range p.All() {
		if problem.Path == "pools[0].parallel.dp_local" {
			for _, want := range []string{"2 local replicas", "16 GPUs", "start at device 16", "a node has 8"} {
				if !strings.Contains(problem.Message, want) {
					t.Errorf("message does not contain %q: %s", want, problem.Message)
				}
			}
			return
		}
	}
	t.Fatalf("no per-node problem:\n%s", p.Error())
}

// TestRankFitDoesNotRecountAReportedDPLocal: dp_local above dp is already reported, and
// states no replica count, so the per-node bound must not file a second problem that
// counts replicas which do not exist.
func TestRankFitDoesNotRecountAReportedDPLocal(t *testing.T) {
	d := singleNodeDeployment()
	d.Pools[0].Parallel = Parallelism{TP: 2, PP: 1, DP: 2, DPLocal: 8}
	if p := d.Validate(); p.OK() {
		t.Fatal("dp_local above dp should be reported by Validate")
	}
	for _, problem := range d.ValidateAgainstCluster(ClusterConstraints{Nodes: 1, GPUsPerNode: 8}).All() {
		if strings.Contains(problem.Message, "would start at device") {
			t.Errorf("per-node bound ran over a dp_local already reported: %s", problem)
		}
	}
}

// TestNegativeDPLocal: the engine constrains data_parallel_size_local ge=0, and a
// negative one was silently accepted here.
func TestNegativeDPLocal(t *testing.T) {
	d := singleNodeDeployment()
	d.Pools[0].Parallel = Parallelism{TP: 8, PP: 1, DP: 1, DPLocal: -1}
	p := d.Validate()
	for _, problem := range p.Errors() {
		if problem.Path == "pools[0].parallel.dp_local" && strings.Contains(problem.Message, "negative") {
			return
		}
	}
	t.Fatalf("a negative dp_local should be rejected, got:\n%s", p.Error())
}

// TestRankFitIsExactAtExtremes: with int arithmetic that saturated, both sides clamped to
// the same maximum and compared equal, so an engine needing 3 x MaxInt GPUs "fit" a pool
// of 2 x MaxInt. Counts are exact, so the comparison and the printed figures are right.
func TestRankFitIsExactAtExtremes(t *testing.T) {
	p := rankFit(Parallelism{TP: math.MaxInt, PP: 1, DP: 3}, math.MaxInt, 2)
	var msg string
	for _, problem := range p.All() {
		if problem.Path == "pools[0].parallel" {
			msg = problem.Message
		}
	}
	if msg == "" {
		t.Fatalf("3 x MaxInt ranks on 2 x MaxInt GPUs must be refused, got:\n%s", p.Error())
	}
	// 3 x (2^63-1) and 2 x (2^63-1), printed in full rather than clamped.
	for _, want := range []string{"27670116110564327421", "18446744073709551614"} {
		if !strings.Contains(msg, want) {
			t.Errorf("message should print the exact count %s: %s", want, msg)
		}
	}
}

// TestRankFitDeclinesOnMalformedInput pins that the rule stays out of the way of a
// document already reported for something else. A rank count over a width that is
// itself an error would be invented, and filing it would point the author at the wrong
// field.
//
// One honest caveat about what these cases can discriminate. A single zero width makes
// the product zero and a single negative one makes it negative, and neither can exceed
// the pool — so the width guards are individually redundant for a lone malformed width,
// and the first four cases pass with or without them. The guard earns its place when two
// widths are negative: their product is POSITIVE, so tp -2 and pp -50 would claim 100
// GPUs are needed by a layout that is not a layout at all. The "two negative" cases are
// the ones that fail without it.
func TestRankFitDeclinesOnMalformedInput(t *testing.T) {
	cases := []struct {
		name        string
		pl          Parallelism
		nodes, gpus int
	}{
		{"zero tp", Parallelism{TP: 0, PP: 99, DP: 1}, 1, 8},
		{"zero pp", Parallelism{TP: 99, PP: 0, DP: 1}, 1, 8},
		{"zero dp", Parallelism{TP: 99, PP: 1, DP: 0}, 1, 8},
		{"negative pcp", Parallelism{TP: 99, PP: 1, DP: 1, PCP: -1}, 1, 8},
		// A product of negatives is positive and large.
		{"two negative widths", Parallelism{TP: -2, PP: -50, DP: 1}, 1, 8},
		{"negative tp and dp", Parallelism{TP: -9, PP: 1, DP: -9}, 1, 8},
		{"negative pp and pcp", Parallelism{TP: 1, PP: -9, DP: 1, PCP: -9}, 1, 8},
		// A cluster or pool that states no room constrains nothing.
		{"cluster states no gpus per node", Parallelism{TP: 99, PP: 1, DP: 1}, 1, 0},
		{"pool with no nodes", Parallelism{TP: 99, PP: 1, DP: 1}, 0, 8},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, problem := range rankFit(c.pl, c.nodes, c.gpus).All() {
				if strings.Contains(problem.Message, "needs") && strings.Contains(problem.Message, "GPUs") {
					t.Errorf("rank-fit fired over input it should decline: %s", problem)
				}
			}
		})
	}
}

// TestDCPCompareIsExact: the engine's tp x pcp is Python arithmetic and cannot overflow,
// so neither may ours. With int arithmetic, tp MaxInt and pcp 3 wrap to MaxInt-2, and a
// dcp of MaxInt-2 would be admitted as "the full block" while the engine rejects it.
func TestDCPCompareIsExact(t *testing.T) {
	d := singleNodeDeployment()
	d.Pools[0].Parallel = Parallelism{TP: math.MaxInt, PP: 1, DP: 1, PCP: 3, DCP: math.MaxInt - 2}
	var msg string
	for _, problem := range d.Validate().Errors() {
		if problem.Path == "pools[0].parallel.dcp" {
			msg = problem.Message
		}
	}
	if msg == "" {
		t.Fatal("dcp MaxInt-2 is not tp x pcp = 3 x MaxInt and must be rejected")
	}
	// The set printed is the engine's: 1, 3, and the exact block.
	if want := "[1 3 27670116110564327421]"; !strings.Contains(msg, want) {
		t.Errorf("admissible set should be %s: %s", want, msg)
	}
	// And the exact block itself is not mistaken for anything else: tp 2^31, pcp 2^31
	// has a block of 2^62, which fits an int, and a dcp of that width is admitted.
	d.Pools[0].Parallel = Parallelism{TP: 1 << 31, PP: 1, DP: 1, PCP: 1 << 31, DCP: 1 << 62}
	for _, problem := range d.Validate().Errors() {
		if problem.Path == "pools[0].parallel.dcp" {
			t.Errorf("dcp equal to the full block must be admitted: %s", problem)
		}
	}
}

// TestRankFitAtTheEdgesOfInt pins two layouts from review whose counts exceed a machine
// word. The pool's capacity is an exact product and may exceed int64, so a layout that
// fits it may have a replica that does too; no step may assume otherwise.
func TestRankFitAtTheEdgesOfInt(t *testing.T) {
	cases := []struct {
		name        string
		pl          Parallelism
		nodes, gpus int
	}{
		// One replica of MaxInt ranks on one node of MaxInt GPUs: it fits exactly. The
		// ceiling of world/room once overflowed to a split of 0 and divided by it.
		{"replica exactly fills a MaxInt node", Parallelism{TP: math.MaxInt, PP: 1, DP: 1, DPLocal: 1}, 1, math.MaxInt},
		// A replica of 2 x MaxInt ranks split over two MaxInt nodes fits exactly: a split
		// of 2 leaves each node MaxInt. Truncating world to int64 once made it -2, and the
		// layout was refused.
		{"replica of 2 x MaxInt split over two nodes", Parallelism{TP: math.MaxInt, PP: 2, DP: 1, DPLocal: 1}, 2, math.MaxInt},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if p := rankFit(c.pl, c.nodes, c.gpus); !p.OK() {
				t.Fatalf("%+v on %d x %d fits exactly and must be accepted:\n%s", c.pl, c.nodes, c.gpus, p.Error())
			}
		})
	}
	// And the same replica one node short is refused, with the exact figures printed.
	p := rankFit(Parallelism{TP: math.MaxInt, PP: 2, DP: 1, DPLocal: 1}, 1, math.MaxInt)
	if !strings.Contains(p.Error(), "needs 18446744073709551614 GPUs") {
		t.Fatalf("2 x MaxInt ranks on one MaxInt node must be refused exactly:\n%s", p.Error())
	}
}

// TestPlacementIsDecidedExactlyAtScale covers layouts whose divisor search is far too
// wide to scan. Placement fits exactly when the replica has a divisor in a range, and
// that is decided from the replica's exact factorisation, not by sampling the range.
func TestPlacementIsDecidedExactlyAtScale(t *testing.T) {
	const p, q = 2147483647, 2147483629 // both prime, near 2^31
	cases := []struct {
		name        string
		pl          Parallelism
		nodes, gpus int
		reject      bool
	}{
		// From review: a prime replica of 1,000,000,007 on 1,100,000 nodes of 1,100,000
		// GPUs. A split must divide the prime and lie in [910, 1,100,000]; none does, so
		// no launch can place it. Both ranges are far past any scan.
		{"prime replica, no divisor in range", Parallelism{TP: 1_000_000_007, PP: 1, DP: 1, DPLocal: 1},
			1_100_000, 1_100_000, true},
		// 2^61 split 2^21 ways leaves 2^40 per node: placeable.
		{"power-of-two replica, divisor in range", Parallelism{TP: 1 << 61, PP: 1, DP: 1, DPLocal: 1},
			1 << 30, 1 << 40, false},
		// p x q split q ways leaves p per node; it fits when a node holds p.
		{"semiprime replica, a factor fits", Parallelism{TP: p, PP: q, DP: 1, DPLocal: 1}, q, p, false},
		// One GPU short of p per node, and q is the only split the pool allows besides
		// 1: refused.
		{"semiprime replica, no factor fits", Parallelism{TP: p, PP: q, DP: 1, DPLocal: 1}, q + 1, p - 1, true},
		// The same two replicas as a single width, which is itself a semiprime near 2^62:
		// deciding these factors the width with rho rather than reading two primes.
		{"one semiprime width, a factor fits", Parallelism{TP: p * q, PP: 1, DP: 1, DPLocal: 1}, q, p, false},
		{"one semiprime width, no factor fits", Parallelism{TP: p * q, PP: 1, DP: 1, DPLocal: 1}, q + 1, p - 1, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			pr := rankFit(c.pl, c.nodes, c.gpus)
			if refused := perNodeRefusal(pr) != ""; refused != c.reject {
				t.Fatalf("%+v on %d x %d: refused=%v, want %v\n%s", c.pl, c.nodes, c.gpus, refused, c.reject, pr.Error())
			}
			if !c.reject && len(pr.All()) != 0 {
				t.Fatalf("a decided placement must carry no finding:\n%s", pr.Error())
			}
		})
	}
}

// TestUndecidedPlacementIsReported: the walk is bounded so no input can hang validation.
// No layout a cluster could hold reaches the bound, so the bound is lowered here to
// reach the path. An undecided layout must be neither refused without proof nor admitted
// in silence: it carries a warning saying its per-node fit is unverified.
func TestUndecidedPlacementIsReported(t *testing.T) {
	saved := placeableSearchLimit
	placeableSearchLimit = 1
	defer func() { placeableSearchLimit = saved }()

	// 2 x 3 x 5 x 7 = 210, with no divisor in [11, 13]: deciding that takes more than
	// one step.
	pr := rankFit(Parallelism{TP: 210, PP: 1, DP: 1, DPLocal: 1}, 13, 20)
	if perNodeRefusal(pr) != "" {
		t.Fatalf("an undecided layout must not be refused:\n%s", pr.Error())
	}
	warned := false
	for _, f := range pr.All() {
		if f.Severity == validate.SeverityWarning && strings.Contains(f.Message, "unverified") {
			warned = true
		}
	}
	if !warned {
		t.Fatalf("an undecided layout must be reported as unverified, got:\n%s", pr.Error())
	}
}

// TestUnfactoredPlacementIsReported: factoring is budgeted so validation cannot hang. If
// the budget ever runs out, the layout is reported as unverified — like an exhausted
// search, never refused without proof and never admitted in silence. The budget is
// lowered here so the semiprime below, whose factors are near 2^31, cannot be split.
func TestUnfactoredPlacementIsReported(t *testing.T) {
	saved := rhoSteps
	rhoSteps = 1
	defer func() { rhoSteps = saved }()

	// One width that is itself a semiprime, so splitting it needs rho. (Two prime widths
	// would not: each is recognised as prime directly.)
	const p, q = 2147483647, 2147483629
	pr := rankFit(Parallelism{TP: p * q, PP: 1, DP: 1, DPLocal: 1}, q+1, p-1)
	if perNodeRefusal(pr) != "" {
		t.Fatalf("an unfactored layout must not be refused:\n%s", pr.Error())
	}
	for _, f := range pr.All() {
		if f.Severity == validate.SeverityWarning && strings.Contains(f.Message, "unverified") {
			return
		}
	}
	t.Fatalf("an unfactored layout must be reported as unverified, got:\n%s", pr.Error())
}
