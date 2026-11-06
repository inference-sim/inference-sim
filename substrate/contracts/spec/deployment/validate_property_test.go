package deployment

import (
	"math"
	"math/big"
	"math/rand"
	"strings"
	"testing"
)

// These tests state the validation rules as properties over many inputs, through the
// public validators only. An oracle here is an independent model of what the ENGINE
// does — never a copy of the code under test — and a metamorphic test relates two
// documents whose verdicts must agree or move one way, without needing an oracle at all.
// Every verdict is read from the problems a user would see: the path a problem is filed
// at, and for the rank-fit rule the message, since its per-node bound shares a path with
// older dp_local checks.
//
// Random inputs come from fixed seeds, so a failure reproduces exactly.

// --- what a user sees -------------------------------------------------------------

func dcpRejected(pl Parallelism) bool {
	d := singleNodeDeployment()
	d.Pools[0].Parallel = pl
	for _, p := range d.Validate().Errors() {
		if p.Path == "pools[0].parallel.dcp" {
			return true
		}
	}
	return false
}

// fitVerdict reports which of the two rank-fit bounds refused a layout. The per-node
// bound speaks only when the pool bound did not, so a test that relates two layouts
// should compare refused(), which is what a user sees.
func fitVerdict(pl Parallelism, nodes, gpusPerNode int) (pool, node bool) {
	p := rankFit(pl, nodes, gpusPerNode)
	for _, problem := range p.Errors() {
		// The split refusal shares this path, so match the pool bound's own opening.
		if problem.Path == "pools[0].parallel" && strings.HasPrefix(problem.Message, "needs ") {
			pool = true
		}
	}
	node = perNodeRefusal(p) != ""
	return pool, node
}

func refused(pl Parallelism, nodes, gpusPerNode int) bool {
	pool, node := fitVerdict(pl, nodes, gpusPerNode)
	return pool || node
}

// --- engine models ------------------------------------------------------------------

// engineAcceptsDCP is the tp/pcp/dcp block of vllm/config/parallel.py, transcribed from
// the engine's source (identical at v0.29.0 and 119937c6f1, apart from the PCP/DP line
// the later release dropped). Python integers do not overflow, so the block is exact.
func engineAcceptsDCP(tp, pcp, dcp int) bool {
	if pcp == 1 {
		return tp%dcp == 0 // "DCP reuses the TP ranks when PCP is disabled."
	}
	block := new(big.Int).Mul(big.NewInt(int64(tp)), big.NewInt(int64(pcp)))
	return dcp == 1 || dcp == pcp || block.Cmp(big.NewInt(int64(dcp))) == 0
}

// engineCanPlace simulates how the engine places one node's local replicas, for every
// engine node count n the pool allows, device range by device range:
//
//   - nnodes_within_dp is 1 when n is 1, else n // (dp // dp_local). v0.29.0 takes the
//     floor (and a split of 0 cannot run); later releases raise unless it divides.
//   - The multiprocessing executor asserts the replica divides evenly over that split.
//   - Local replica i takes [i*world, i*world + world/split).
//
// It shares no arithmetic with the rule, which reasons about the smallest split instead.
func engineCanPlace(world, dp, dpLocal, nodes, gpusPerNode int, v029 bool) bool {
	for n := 1; n <= nodes; n++ {
		split := 1
		if n > 1 {
			groups := dp / dpLocal
			if !v029 && n%groups != 0 {
				continue
			}
			split = n / groups
		}
		if split == 0 || world%split != 0 {
			continue
		}
		share, fits := world/split, true
		for i := 0; i < dpLocal && fits; i++ {
			fits = i*world+share <= gpusPerNode
		}
		if fits {
			return true
		}
	}
	return false
}

// engineExpertGroup counts the ranks in rank 0's expert group by building the engine's
// rank layout — ranks ordered DP x PP x PCP x TP — and collecting every rank that shares
// rank 0's pipeline stage, which is how initialize_model_parallel groups them.
func engineExpertGroup(tp, pp, pcp, dp int) int {
	n := 0
	for r := 0; r < dp*pp*pcp*tp; r++ {
		if (r/(pcp*tp))%pp == 0 { // same pipeline stage as rank 0
			n++
		}
	}
	return n
}

// --- differential: the rules agree with the engine ---------------------------------

// TestDCPMatchesTheEngine compares every verdict over a grid wider than any deployment
// uses, not a sample: tp 1-16, pcp 1-8, dcp 1-64.
func TestDCPMatchesTheEngine(t *testing.T) {
	cases, mismatches := 0, 0
	for tp := 1; tp <= 16; tp++ {
		for pcp := 1; pcp <= 8; pcp++ {
			for dcp := 1; dcp <= 64; dcp++ {
				cases++
				got := !dcpRejected(Parallelism{TP: tp, PP: 1, DP: 1, PCP: pcp, DCP: dcp})
				if want := engineAcceptsDCP(tp, pcp, dcp); got != want {
					mismatches++
					if mismatches <= 5 {
						t.Errorf("tp %d pcp %d dcp %d: validator accepts=%v, engine accepts=%v",
							tp, pcp, dcp, got, want)
					}
				}
			}
		}
	}
	if cases != 8192 {
		t.Fatalf("grid covered %d cases, want 8192", cases)
	}
	// Widths whose block overflows an int, where wrapping would invent a match.
	r := rand.New(rand.NewSource(5))
	edge := func() int {
		switch r.Intn(3) {
		case 0:
			return math.MaxInt - r.Intn(4)
		case 1:
			return 1 << (31 + r.Intn(32))
		default:
			return 1 + r.Intn(8)
		}
	}
	for i := 0; i < 20000; i++ {
		tp, pcp := edge(), 2+r.Intn(6)
		dcp := []int{1, pcp, tp * pcp, edge(), tp}[r.Intn(5)] // tp*pcp may wrap: that is the point
		if dcp < 1 {
			continue
		}
		got := !dcpRejected(Parallelism{TP: tp, PP: 1, DP: 1, PCP: pcp, DCP: dcp})
		if want := engineAcceptsDCP(tp, pcp, dcp); got != want {
			t.Fatalf("tp %d pcp %d dcp %d: validator accepts=%v, engine accepts=%v", tp, pcp, dcp, got, want)
		}
	}
}

// TestExpertParallelWidthMatchesTheEngine compares the derived width with the size of
// the expert group the engine builds, over every small layout.
func TestExpertParallelWidthMatchesTheEngine(t *testing.T) {
	for tp := 1; tp <= 8; tp++ {
		for pp := 1; pp <= 3; pp++ {
			for pcp := 1; pcp <= 8; pcp++ {
				for dp := 1; dp <= 8; dp++ {
					pl := Parallelism{TP: tp, PP: pp, PCP: pcp, DP: dp, EnableExpertParallel: true}
					if got, want := pl.ExpertParallelWidth(), engineExpertGroup(tp, pp, pcp, dp); got != want {
						t.Fatalf("%+v: width %d, engine group %d", pl, got, want)
					}
				}
			}
		}
	}
}

// TestExpertParallelWidthUnchangedWhereItWasValid: the width used to be tp x max(dp, pcp).
// Wherever one of dp and pcp is at most 1 — every layout field validation admitted while
// the two could not be combined — the product must give exactly that, so no existing
// deployment's width moves.
func TestExpertParallelWidthUnchangedWhereItWasValid(t *testing.T) {
	old := func(p Parallelism) int {
		wide := max(p.DP, p.PCP, 1)
		return max(p.TP, 1) * wide
	}
	for tp := 0; tp <= 16; tp++ {
		for wide := 0; wide <= 64; wide++ {
			for _, pl := range []Parallelism{
				{TP: tp, DP: wide, PCP: 0}, {TP: tp, DP: wide, PCP: 1},
				{TP: tp, DP: 0, PCP: wide}, {TP: tp, DP: 1, PCP: wide},
			} {
				pl.EnableExpertParallel = true
				if got, want := pl.ExpertParallelWidth(), old(pl); got != want {
					t.Fatalf("%+v: width %d, previously %d", pl, got, want)
				}
			}
		}
	}
}

// TestPerNodeBoundMatchesEnginePlacement: where the pool bound holds, the per-node bound
// refuses a layout exactly when no engine node count up to the pool's lets the engine
// place every local replica on one node. It is checked against the simulation under both
// v0.29.0's and later semantics, which must agree — the rule claims the set of placeable
// layouts did not change.
func TestPerNodeBoundMatchesEnginePlacement(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	refusedN, considered := 0, 0
	const n = 20000
	for i := 0; i < n; i++ {
		gpn := []int{1, 2, 4, 8, 16}[r.Intn(5)]
		dp := 1 + r.Intn(16)
		pl := Parallelism{TP: 1 + r.Intn(16), PP: 1 + r.Intn(3), DP: dp, PCP: r.Intn(3),
			DPLocal: 1 + r.Intn(dp)}
		nodes := 1 + r.Intn(12)
		world := pl.TP * pl.PP * max(pl.PCP, 1)
		pool, node := fitVerdict(pl, nodes, gpn)
		if pool {
			if node {
				t.Fatalf("%+v on %d x %d: per-node bound spoke after the pool bound", pl, nodes, gpn)
			}
			continue
		}
		considered++
		head := engineCanPlace(world, dp, pl.DPLocal, nodes, gpn, false)
		v029 := engineCanPlace(world, dp, pl.DPLocal, nodes, gpn, true)
		if head != v029 {
			t.Fatalf("%+v on %d x %d (world %d): releases disagree, later=%v v0.29.0=%v",
				pl, nodes, gpn, world, head, v029)
		}
		if node {
			refusedN++
		}
		if node != !head {
			t.Fatalf("%+v on %d x %d (world %d): validator refuses=%v, engine can place=%v",
				pl, nodes, gpn, world, node, head)
		}
	}
	if refusedN == 0 || refusedN == considered {
		t.Fatalf("degenerate sample: %d of %d refused", refusedN, considered)
	}
}

// TestPoolBoundMatchesExactRankCount: the pool bound refuses exactly the layouts whose
// rank count exceeds the pool's GPUs, computed in arbitrary precision — including widths
// and pools large enough to overflow a 64-bit product on either side.
func TestPoolBoundMatchesExactRankCount(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	width := func() int {
		switch r.Intn(5) {
		case 0:
			return math.MaxInt - r.Intn(3)
		case 1:
			return 1 + r.Intn(300)
		default:
			return 1 + r.Intn(8)
		}
	}
	refused := 0
	const n = 20000
	for i := 0; i < n; i++ {
		pl := Parallelism{TP: width(), PP: width(), DP: width(), PCP: r.Intn(5)}
		nodes, gpn := width(), []int{1, 2, 4, 8}[r.Intn(4)]
		need := big.NewInt(1)
		for _, f := range []int{pl.TP, pl.PP, max(pl.PCP, 1), pl.DP} {
			need.Mul(need, big.NewInt(int64(f)))
		}
		have := new(big.Int).Mul(big.NewInt(int64(nodes)), big.NewInt(int64(gpn)))
		want := need.Cmp(have) > 0
		got, _ := fitVerdict(pl, nodes, gpn)
		if got {
			refused++
		}
		if got != want {
			t.Fatalf("%+v on %d x %d: validator refuses=%v, needs %s of %s", pl, nodes, gpn, got, need, have)
		}
	}
	if refused == 0 || refused == n {
		t.Fatalf("degenerate sample: %d of %d refused", refused, n)
	}
}

// --- metamorphic: relations between documents --------------------------------------

// TestOmittedWidthIsTheSameAsOne: pcp and dcp are omitempty, so an absent one arrives as
// zero meaning "not enabled". A document that omits either must be judged exactly as one
// that states 1 — the same problems, word for word — under both validators.
func TestOmittedWidthIsTheSameAsOne(t *testing.T) {
	report := func(pl Parallelism, nodes int) string {
		d := singleNodeDeployment()
		d.Pools[0].Nodes = nodes
		d.Pools[0].Parallel = pl
		return d.Validate().Error() + "\n--\n" +
			d.ValidateAgainstCluster(ClusterConstraints{Nodes: nodes, GPUsPerNode: 8}).Error()
	}
	for tp := 1; tp <= 16; tp++ {
		for dp := 1; dp <= 4; dp++ {
			for nodes := 1; nodes <= 4; nodes++ {
				for _, dcp := range []int{0, 1, 2, 3, 4, 8} {
					omitted := report(Parallelism{TP: tp, PP: 1, DP: dp, PCP: 0, DCP: dcp}, nodes)
					stated := report(Parallelism{TP: tp, PP: 1, DP: dp, PCP: 1, DCP: dcp}, nodes)
					if omitted != stated {
						t.Fatalf("tp %d dp %d dcp %d on %d nodes: pcp omitted and pcp 1 differ:\n%s\nvs\n%s",
							tp, dp, dcp, nodes, omitted, stated)
					}
				}
				for _, pcp := range []int{0, 1, 2, 4} {
					omitted := report(Parallelism{TP: tp, PP: 1, DP: dp, PCP: pcp, DCP: 0}, nodes)
					stated := report(Parallelism{TP: tp, PP: 1, DP: dp, PCP: pcp, DCP: 1}, nodes)
					if omitted != stated {
						t.Fatalf("tp %d dp %d pcp %d on %d nodes: dcp omitted and dcp 1 differ:\n%s\nvs\n%s",
							tp, dp, pcp, nodes, omitted, stated)
					}
				}
			}
		}
	}
}

// randomLayout draws a well-formed layout and room for it, biased so that both verdicts
// are common.
func randomLayout(r *rand.Rand) (Parallelism, int, int) {
	dp := 1 + r.Intn(16)
	pl := Parallelism{TP: 1 + r.Intn(16), PP: 1 + r.Intn(4), DP: dp, PCP: r.Intn(4)}
	if r.Intn(2) == 0 {
		pl.DPLocal = 1 + r.Intn(dp)
	}
	return pl, 1 + r.Intn(16), []int{1, 2, 4, 8}[r.Intn(4)]
}

// TestMoreRoomNeverAddsARankFitProblem: whatever fits a pool fits one with more nodes, and
// one with nodes twice as large. (Doubling keeps every dp_local that divided a node
// dividing it, so the older divisibility check cannot interfere.)
func TestMoreRoomNeverAddsARankFitProblem(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	for i := 0; i < 20000; i++ {
		pl, nodes, gpn := randomLayout(r)
		if refused(pl, nodes, gpn) {
			continue
		}
		for _, room := range [][2]int{{nodes + 1 + r.Intn(8), gpn}, {nodes, gpn * 2}} {
			if refused(pl, room[0], room[1]) {
				t.Fatalf("%+v: fits %d x %d but not %d x %d", pl, nodes, gpn, room[0], room[1])
			}
		}
	}
}

// TestWiderLayoutsNeverLoseARankFitProblem: whatever does not fit still does not fit when
// a width grows. Growing dp alone changes how replicas split over nodes (dp / dp_local
// node groups), so it is only checked against the pool bound, which it can only worsen.
func TestWiderLayoutsNeverLoseARankFitProblem(t *testing.T) {
	r := rand.New(rand.NewSource(13))
	grow := map[string]func(*Parallelism){
		"tp":  func(p *Parallelism) { p.TP++ },
		"pp":  func(p *Parallelism) { p.PP++ },
		"pcp": func(p *Parallelism) { p.PCP = max(p.PCP, 1) + 1 },
	}
	for i := 0; i < 20000; i++ {
		pl, nodes, gpn := randomLayout(r)
		pool, _ := fitVerdict(pl, nodes, gpn)
		if pool {
			wider := pl
			wider.DP++
			if p2, _ := fitVerdict(wider, nodes, gpn); !p2 {
				t.Fatalf("%+v exceeds %d x %d, but growing dp made it fit", pl, nodes, gpn)
			}
		}
		if !refused(pl, nodes, gpn) {
			continue
		}
		for axis, g := range grow {
			wider := pl
			g(&wider)
			// A wider replica can only start later and need more: if no split placed the
			// narrower one, none places this — unless the wider width admits a split the
			// narrower did not (tp 9 has no split of 2; tp 10 does). That is placement,
			// not room, so only the pool bound is monotone here.
			if pool {
				if p2, _ := fitVerdict(wider, nodes, gpn); !p2 {
					t.Fatalf("%+v exceeds %d x %d, but growing %s made it fit", pl, nodes, gpn, axis)
				}
			}
		}
	}
}

// TestDCPAndExpertParallelismAddNoRanks: neither changes how many devices a layout
// needs — DCP reuses the TP ranks, an expert group spans ranks that already exist — so
// neither may change a rank-fit verdict.
func TestDCPAndExpertParallelismAddNoRanks(t *testing.T) {
	r := rand.New(rand.NewSource(17))
	for i := 0; i < 20000; i++ {
		pl, nodes, gpn := randomLayout(r)
		pool, node := fitVerdict(pl, nodes, gpn)
		for _, v := range []Parallelism{
			func() Parallelism { q := pl; q.DCP = q.TP; return q }(),
			func() Parallelism { q := pl; q.DCP = 1 + r.Intn(64); return q }(),
			func() Parallelism { q := pl; q.EnableExpertParallel = !q.EnableExpertParallel; return q }(),
		} {
			if p2, n2 := fitVerdict(v, nodes, gpn); p2 != pool || n2 != node {
				t.Fatalf("%+v vs %+v on %d x %d: verdict changed from (%v, %v) to (%v, %v)",
					pl, v, nodes, gpn, pool, node, p2, n2)
			}
		}
	}
}

// TestReplicaAxesAreInterchangeable: a replica's footprint is pp x tp x pcp, so permuting
// the three widths among themselves cannot change either verdict.
func TestReplicaAxesAreInterchangeable(t *testing.T) {
	r := rand.New(rand.NewSource(19))
	for i := 0; i < 20000; i++ {
		pl, nodes, gpn := randomLayout(r)
		pl.PCP = 1 + r.Intn(4) // a stated width, so it can trade places with the others
		pool, node := fitVerdict(pl, nodes, gpn)
		for _, perm := range [][3]int{{pl.PP, pl.TP, pl.PCP}, {pl.PCP, pl.PP, pl.TP}, {pl.TP, pl.PCP, pl.PP}} {
			q := pl
			q.TP, q.PP, q.PCP = perm[0], perm[1], perm[2]
			if p2, n2 := fitVerdict(q, nodes, gpn); p2 != pool || n2 != node {
				t.Fatalf("%+v vs %+v on %d x %d: verdict changed under permutation", pl, q, nodes, gpn)
			}
		}
	}
}

// TestScalingReplicasWithTheirNodesPreservesFit: k times the replicas on k times the
// nodes needs exactly k times the GPUs it is given, so the pool verdict cannot change.
func TestScalingReplicasWithTheirNodesPreservesFit(t *testing.T) {
	r := rand.New(rand.NewSource(23))
	for i := 0; i < 20000; i++ {
		pl, nodes, gpn := randomLayout(r)
		pl.DPLocal = 0 // the per-node bound is about one node; scaling is about the pool
		pool, _ := fitVerdict(pl, nodes, gpn)
		k := 2 + r.Intn(7)
		scaled := pl
		scaled.DP *= k
		if p2, _ := fitVerdict(scaled, nodes*k, gpn); p2 != pool {
			t.Fatalf("%+v on %d x %d fits=%v, but x%d replicas on x%d nodes fits=%v",
				pl, nodes, gpn, !pool, k, k, !p2)
		}
	}
}

// engineCanPlaceExact is engineCanPlace in arbitrary precision, for layouts whose counts
// exceed a machine word. It is the same simulation — every engine node count, the split
// it implies, the divisibility the executor asserts, and the last replica's device range
// — over big integers, so it shares nothing with the rule's search either.
func engineCanPlaceExact(world *big.Int, dp, dpLocal, nodes int, gpusPerNode *big.Int) bool {
	for n := 1; n <= nodes; n++ {
		split := int64(1)
		if n > 1 {
			groups := dp / dpLocal
			if n%groups != 0 {
				continue
			}
			split = int64(n / groups)
		}
		k := big.NewInt(split)
		share, rem := new(big.Int).QuoRem(world, k, new(big.Int))
		if rem.Sign() != 0 {
			continue
		}
		end := new(big.Int).Mul(big.NewInt(int64(dpLocal-1)), world)
		end.Add(end, share)
		if end.Cmp(gpusPerNode) <= 0 {
			return true
		}
	}
	return false
}

// TestPerNodeBoundIsExactAtTheEdgesOfInt draws widths and node sizes around the largest
// int, where any machine-word step wraps, and compares the per-node verdict with the
// exact simulation wherever the pool bound holds.
func TestPerNodeBoundIsExactAtTheEdgesOfInt(t *testing.T) {
	r := rand.New(rand.NewSource(29))
	edge := func() int {
		switch r.Intn(4) {
		case 0:
			return math.MaxInt - r.Intn(3)
		case 1:
			return math.MaxInt/(2+r.Intn(4)) + r.Intn(3)
		case 2:
			return 1 << (r.Intn(62) + 1)
		default:
			return 1 + r.Intn(8)
		}
	}
	considered, refusedN := 0, 0
	for i := 0; i < 20000; i++ {
		dp := 1 + r.Intn(4)
		pl := Parallelism{TP: edge(), PP: 1 + r.Intn(3), DP: dp, PCP: r.Intn(3), DPLocal: 1 + r.Intn(dp)}
		nodes, gpn := 1+r.Intn(8), edge()
		pool, node := fitVerdict(pl, nodes, gpn)
		if pool {
			continue
		}
		considered++
		world := big.NewInt(int64(pl.TP))
		world.Mul(world, big.NewInt(int64(pl.PP)))
		world.Mul(world, big.NewInt(int64(max(pl.PCP, 1))))
		want := !engineCanPlaceExact(world, dp, pl.DPLocal, nodes, big.NewInt(int64(gpn)))
		if node {
			refusedN++
		}
		if node != want {
			t.Fatalf("%+v on %d x %d (world %s): validator refuses=%v, engine cannot place=%v",
				pl, nodes, gpn, world, node, want)
		}
	}
	if considered == 0 || refusedN == 0 || refusedN == considered {
		t.Fatalf("degenerate sample: %d of %d refused", refusedN, considered)
	}
}

// FuzzRankFit holds two properties for any widths and room at all: validation returns
// rather than panicking, and wherever the inputs are small enough to simulate, the verdict
// agrees with the engine's placement. The seeds are the layouts review found at the edges
// of int. `go test` runs the seeds; `go test -fuzz=FuzzRankFit` explores beyond them.
func FuzzRankFit(f *testing.F) {
	f.Add(math.MaxInt, 1, 1, 0, 1, 1, math.MaxInt)
	f.Add(math.MaxInt, 2, 1, 0, 1, 2, math.MaxInt)
	f.Add(math.MaxInt, 1, 3, 0, 2, 1, 8)
	f.Add(6, 1, 2, 0, 2, 2, 8)
	f.Add(6, 1, 2, 0, 2, 3, 8)
	f.Add(-2, -50, 1, 0, 0, 1, 8)
	f.Add(1, 1, 4, 8, 1, 4, 8)
	f.Add(1_000_000_007, 1, 1, 0, 1, 1_100_000, 1_100_000)
	f.Fuzz(func(t *testing.T, tp, pp, dp, pcp, dpLocal, nodes, gpn int) {
		pl := Parallelism{TP: tp, PP: pp, DP: dp, PCP: pcp, DPLocal: dpLocal}
		pool, node := fitVerdict(pl, nodes, gpn) // must not panic
		small := tp >= 1 && tp <= 64 && pp >= 1 && pp <= 4 && dp >= 1 && dp <= 32 &&
			pcp >= 0 && pcp <= 8 && dpLocal >= 1 && dpLocal <= dp &&
			nodes >= 1 && nodes <= 16 && gpn >= 1 && gpn <= 64
		if !small || pool {
			return
		}
		world := tp * pp * max(pcp, 1)
		if want := !engineCanPlace(world, dp, dpLocal, nodes, gpn, false); node != want {
			t.Fatalf("%+v on %d x %d: validator refuses=%v, engine cannot place=%v", pl, nodes, gpn, node, want)
		}
	})
}

// FuzzParallelismValidation holds, for any widths and any room: field validation and the
// cluster checks return rather than panic, the derived expert width is at least one, and
// wherever the engine's DCP rule is defined (every width at least one) the validator's
// DCP verdict is the engine's, computed exactly.
func FuzzParallelismValidation(f *testing.F) {
	f.Add(math.MaxInt, 1, 1, 3, math.MaxInt-2, 0, true, 1, 8)
	f.Add(math.MaxInt, math.MaxInt, math.MaxInt, math.MaxInt, math.MaxInt, math.MaxInt, true, math.MaxInt, math.MaxInt)
	f.Add(math.MinInt, math.MinInt, math.MinInt, math.MinInt, math.MinInt, math.MinInt, true, math.MinInt, math.MinInt)
	f.Add(1, 1, 4, 8, 8, 1, true, 4, 8)
	f.Add(0, 0, 0, 0, 0, 0, false, 0, 0)
	f.Fuzz(func(t *testing.T, tp, pp, dp, pcp, dcp, dpLocal int, ep bool, nodes, gpn int) {
		pl := Parallelism{TP: tp, PP: pp, DP: dp, PCP: pcp, DCP: dcp, DPLocal: dpLocal,
			EnableExpertParallel: ep}
		d := singleNodeDeployment()
		d.Pools[0].Nodes = nodes
		d.Pools[0].Parallel = pl
		d.Validate()
		d.ValidateAgainstCluster(ClusterConstraints{Nodes: nodes, GPUsPerNode: gpn})
		if w := pl.ExpertParallelWidth(); w < 1 {
			t.Fatalf("%+v: expert width %d", pl, w)
		}
		if tp >= 1 && pp >= 1 && dp >= 1 && pcp >= 1 && dcp >= 1 {
			if got, want := !dcpRejected(pl), engineAcceptsDCP(tp, pcp, dcp); got != want {
				t.Fatalf("%+v: validator accepts=%v, engine accepts=%v", pl, got, want)
			}
		}
	})
}

// TestPlacementDecisionMatchesKnownFactorisations builds replicas from primes the test
// chooses, so the oracle knows every divisor without factoring anything: it enumerates
// the divisors of the product it built and checks the range directly. Replicas run to
// near 2^63 and ranges far wider than any scan, so the rule's own factorisation is what
// is under test.
func TestPlacementDecisionMatchesKnownFactorisations(t *testing.T) {
	primes := []int{2, 3, 5, 7, 11, 13, 101, 65537, 1_000_000_007, 2147483647}
	r := rand.New(rand.NewSource(31))
	placed, refusedN := 0, 0
	for i := 0; i < 3000; i++ {
		// A replica of up to four chosen primes, kept below 2^62 so tp is an int.
		chosen, world := []int{}, 1
		for j := 0; j < 1+r.Intn(4); j++ {
			q := primes[r.Intn(len(primes))]
			if world > (1<<62)/q {
				break
			}
			chosen, world = append(chosen, q), world*q
		}
		divisors := []int{1}
		for _, q := range chosen {
			next := []int{}
			for _, d := range divisors {
				next = append(next, d, d*q)
			}
			divisors = next
		}
		// Room drawn up to the replica's own size, and half the time a pool of up to 2^30
		// nodes, so the range a split must fall in is often far wider than any scan.
		nodes, gpn := 1+r.Intn(world), 1+r.Intn(world)
		if r.Intn(2) == 0 {
			nodes = 1 + r.Intn(1<<30)
		}
		// The pool bound must hold, so the per-node bound is what decides.
		if big.NewInt(0).Mul(big.NewInt(int64(nodes)), big.NewInt(int64(gpn))).Cmp(big.NewInt(int64(world))) < 0 {
			continue
		}
		lo := (world + gpn - 1) / gpn
		want := false
		for _, d := range divisors {
			if d >= lo && d <= nodes {
				want = true
			}
		}
		pr := rankFit(Parallelism{TP: world, PP: 1, DP: 1, DPLocal: 1}, nodes, gpn)
		got := perNodeRefusal(pr) == ""
		if got != want {
			t.Fatalf("replica %d = %v on %d x %d: validator places=%v, divisors say %v\n%s",
				world, chosen, nodes, gpn, got, want, pr.Error())
		}
		if len(pr.All()) != 0 && got {
			t.Fatalf("replica %d on %d x %d: placed but with a finding:\n%s", world, nodes, gpn, pr.Error())
		}
		if got {
			placed++
		} else {
			refusedN++
		}
	}
	if placed == 0 || refusedN == 0 {
		t.Fatalf("degenerate sample: %d placed, %d refused", placed, refusedN)
	}
}
