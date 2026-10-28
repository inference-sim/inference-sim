// Moved from blis-latency-kernel/properties_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"math/rand"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Properties that must hold for EVERY deployment the catalog can price, checked across a
// spread of batch shapes rather than on one fixture at one shape.
//
// These are metamorphic and invariant tests: they relate one call's output to another's
// without knowing either absolute value. That is the only kind of assertion available here
// — the right step time for a deployment is an empirical question answered by the scoring
// corpora, not by a unit test — and it is the kind that survives a recoefficienting. A test
// pinning a microsecond figure breaks whenever the registry is refitted and tells you
// nothing about whether the model still behaves like a cost model.
//
// They complement methods_test.go, whose assertions are pinned to one fixture. A term that
// is wired up correctly for minimax-m2.5 on H200 and wrong for a dense model on one GPU
// passes there and fails here.

// propertyFixtures are deployments spanning the shapes that change which terms fire:
// dense vs MoE, expert-parallel vs tensor-parallel only, single- vs multi-node, a
// recurrent stack, and speculative decoding.
//
// Named rather than discovered so the list is reviewable, and kept small because each
// builds a kernel against the real catalog.
func propertyFixtures(t *testing.T) map[string]*Kernel {
	t.Helper()
	out := map[string]*Kernel{}
	for _, name := range []string{
		"glm5-h200-tp8.yaml",               // MoE, tensor-parallel only, one node
		"minimax-m25-h200-tp8.yaml",        // MoE, one node, no expert parallelism
		"nemotron3-ultra-h100-agg.yaml",    // three replicas, fabric declared
		"kimi-k3-h100-nospec.yaml",         // multi-node, speculation disabled
		"direct/m27-h200-puretp4-fpm.yaml", // four GPUs per node
		v41,                                // stated caches, replicated nodes, a drafter
	} {
		out[name] = fixture(t, name)
	}
	return out
}

// batchShapes spans the regimes a step can be in: empty, pure decode at several widths and
// contexts, pure prefill, and mixed.
func batchShapes() map[string]kernel.Batch {
	return map[string]kernel.Batch{
		"empty":                {},
		"1 decode ctx 1":       decodeBatch(1, 1, 1),
		"8 decode ctx 1k":      decodeBatch(8, 1, 1024),
		"256 decode ctx 8k":    decodeBatch(256, 1, 8192),
		"512 decode ctx 32k":   decodeBatch(512, 1, 32768),
		"1 prefill 2k":         decodeBatch(1, 2048, 2048),
		"4 prefill 8k":         decodeBatch(4, 8192, 8192),
		"speculative q=2":      decodeBatch(64, 2, 4096),
		"speculative q=4 wide": decodeBatch(256, 4, 16384),
	}
}

// TestOverlapBracketsNoOverlapEverywhere is the band's defining relation: the per-stage max
// cannot exceed the per-resource sum, for any deployment and any batch.
//
// If it ever inverts, the estimate reports a band the wrong way round and every consumer
// reading "between Overlap and NoOverlap" is reading a lie.
//
// Cheap and currently redundant: stepTime clamps overlap to noOverlap before returning, so
// mutating that clamp leaves this green — the relation holds structurally rather than
// numerically. Kept because the clamp is one line that a refactor can drop, and because
// the property is the contract whether or not today's implementation enforces it by
// construction.
func TestOverlapBracketsNoOverlapEverywhere(t *testing.T) {
	for name, k := range propertyFixtures(t) {
		for shape, b := range batchShapes() {
			if got, limit := k.StepTime(b).Overlap, k.StepTime(b).NoOverlap; got > limit {
				t.Errorf("%s / %s: Overlap %v exceeds NoOverlap %v", name, shape, got, limit)
			}
		}
	}
}

// TestStepTimeIsMonotoneInWork is the metamorphic core: strictly more work cannot cost
// strictly less time.
//
// Three independent axes, each holding the others fixed, so a failure names which one
// regressed. Monotonicity is the weakest useful statement about a cost model and the one a
// miswired term breaks first — a term that divides where it should multiply, or reads a
// per-rank count where it wants a global one, usually shows up as a curve that bends the
// wrong way rather than as an implausible absolute number.
func TestStepTimeIsMonotoneInWork(t *testing.T) {
	for name, k := range propertyFixtures(t) {
		// More requests at one context.
		t.Run(name+"/requests", func(t *testing.T) {
			prev := k.StepTime(decodeBatch(1, 1, 4096)).NoOverlap
			for _, n := range []int{2, 8, 32, 128, 512} {
				got := k.StepTime(decodeBatch(n, 1, 4096)).NoOverlap
				if got < prev {
					t.Errorf("%d requests cost %v, fewer than the step before it (%v)",
						n, got, prev)
				}
				prev = got
			}
		})
		// Longer context at one batch. Charged to HBM, where KV reads live; the overall
		// step can be flat in context at small batch because collective floors dominate,
		// which is correct and is why this reads the term rather than the total.
		//
		// STRICTLY increasing, not merely non-decreasing. A KV read is proportional to the
		// resident context, so a term that is flat across a 128x span is not conservative
		// — it has lost the context factor altogether, which is the exact shape of a
		// miswiring that charges per request instead of per token. Mutation-checked:
		// replacing the context-scaled KV read with its per-token constant leaves this
		// test's non-strict form green and is caught by the strict one.
		t.Run(name+"/context", func(t *testing.T) {
			prev := k.StepTime(decodeBatch(64, 1, 1024)).PerResource[kernel.ResourceHBM]
			for _, ctx := range []int{2048, 8192, 32768, 131072} {
				got := k.StepTime(decodeBatch(64, 1, ctx)).PerResource[kernel.ResourceHBM]
				if got <= prev {
					t.Errorf("context %d reads %v from HBM, not more than the shorter "+
						"context before it (%v); a KV read scales with resident context",
						ctx, got, prev)
				}
				prev = got
			}
		})
		// More scheduled tokens per request, which is what speculation adds.
		t.Run(name+"/scheduled", func(t *testing.T) {
			prev := k.StepTime(decodeBatch(32, 1, 8192)).NoOverlap
			for _, q := range []int{2, 3, 5, 8} {
				got := k.StepTime(decodeBatch(32, q, 8192)).NoOverlap
				if got < prev {
					t.Errorf("q=%d costs %v, less than the narrower draft before it (%v)",
						q, got, prev)
				}
				prev = got
			}
		})
	}
}

// TestAnEmptyStepIsNotFree pins the one absolute the model does assert: a step that
// schedules nothing still pays host work.
//
// A zero here would mean a simulator could run infinitely many empty steps in no time,
// which is the degenerate case a discrete-event loop is most likely to hit.
func TestAnEmptyStepIsNotFree(t *testing.T) {
	for name, k := range propertyFixtures(t) {
		e := k.StepTime(kernel.Batch{})
		if e.Overlap <= 0 || e.NoOverlap <= 0 {
			t.Errorf("%s: an empty step costs %v/%v; it still dispatches host work",
				name, e.Overlap, e.NoOverlap)
		}
		if e.Bottleneck != kernel.ResourceHost {
			t.Errorf("%s: an empty step's bottleneck is %v; with no device work it is the "+
				"host", name, e.Bottleneck)
		}
	}
}

// TestStepTimeIsDeterministic is purity restated as a property over many shapes.
//
// The interface promises a pure function so a simulator can call it at any simulated
// instant from any goroutine. Determinism is the half that a cached or memoised term would
// break, and it would break it silently: a wrong number that is stable looks like a
// modelling choice.
func TestStepTimeIsDeterministic(t *testing.T) {
	for name, k := range propertyFixtures(t) {
		for shape, b := range batchShapes() {
			first := k.StepTime(b)
			for range 8 {
				again := k.StepTime(b)
				if again.Overlap != first.Overlap || again.NoOverlap != first.NoOverlap {
					t.Fatalf("%s / %s: repeated calls disagree: %v/%v then %v/%v",
						name, shape, first.Overlap, first.NoOverlap,
						again.Overlap, again.NoOverlap)
				}
			}
		}
	}
}

// TestPerResourceSumsToNoOverlap is the accounting identity.
//
// NoOverlap is defined as the sum over resources, so the breakdown a caller inspects to
// answer "why" must add up to the figure it explains.
//
// Weaker than it looks, and worth saying so: NoOverlap is COMPUTED from the same
// per-resource array the breakdown is published from, so scaling or dropping a term moves
// both sides and this still holds. Mutation-checked — halving the host term leaves it
// green. What it does catch is a term published to the map but not summed, or summed twice,
// or charged to a resource the map does not expose: a divergence between the two views
// rather than an error in either.
func TestPerResourceSumsToNoOverlap(t *testing.T) {
	for name, k := range propertyFixtures(t) {
		for shape, b := range batchShapes() {
			e := k.StepTime(b)
			var sum int64
			for _, d := range e.PerResource {
				sum += d.Nanoseconds()
			}
			// Nanosecond tolerance: the total and the parts are each rounded from seconds.
			if diff := sum - e.NoOverlap.Nanoseconds(); diff > 1 || diff < -1 {
				t.Errorf("%s / %s: per-resource sums to %dns, NoOverlap is %dns",
					name, shape, sum, e.NoOverlap.Nanoseconds())
			}
		}
	}
}

// TestBottleneckNamesTheLargestResource checks the field against its own definition: the
// resource that did the most work over the step.
//
// It is what distinguishes "add GPUs" from "raise the batch size" as the next action, so a
// bottleneck that names the wrong resource sends a reader to the wrong fix.
func TestBottleneckNamesTheLargestResource(t *testing.T) {
	for name, k := range propertyFixtures(t) {
		for shape, b := range batchShapes() {
			e := k.StepTime(b)
			worst := e.PerResource[e.Bottleneck]
			for res, d := range e.PerResource {
				if d > worst {
					t.Errorf("%s / %s: bottleneck is %v at %v, but %v did more work (%v)",
						name, shape, e.Bottleneck, worst, res, d)
				}
			}
		}
	}
}

// TestStepTimeIntoAgreesWithAllocatingForm is the metamorphic relation between the two
// entry points: the caller-supplied-map form exists only to avoid an allocation, so it
// must be observationally identical.
//
// Checked with a DIRTY map, because the failure this guards is the interesting one: a
// breakdown that keeps a resource from a previous step reports work that did not happen.
func TestStepTimeIntoAgreesWithAllocatingForm(t *testing.T) {
	for name, k := range propertyFixtures(t) {
		for shape, b := range batchShapes() {
			want := k.StepTime(b)

			dirty := map[kernel.Resource]time.Duration{
				kernel.ResourceSM:   99 * time.Second,
				kernel.ResourceHBM:  99 * time.Second,
				kernel.ResourceHost: 99 * time.Second,
				kernel.ResourceNIC:  99 * time.Second,
			}
			got := k.StepTimeInto(b, dirty)

			if got.Overlap != want.Overlap || got.NoOverlap != want.NoOverlap {
				t.Errorf("%s / %s: StepTimeInto gives %v/%v, StepTime %v/%v",
					name, shape, got.Overlap, got.NoOverlap, want.Overlap, want.NoOverlap)
			}
			for res, d := range got.PerResource {
				if d == 99*time.Second {
					t.Errorf("%s / %s: %v kept its stale value; a supplied map must be "+
						"cleared of resources this step did not use", name, shape, res)
				}
			}
		}
	}
}

// TestScalingTheBatchDoesNotScaleTheFloor is a metamorphic check on the floor/rate split.
//
// A collective is priced as a fixed floor plus bytes over a rate. So doubling the work must
// LESS than double the step wherever a floor contributes at all — if cost were exactly
// proportional, the floor is being charged per byte rather than per operation, which is the
// error that makes a small-batch step look like a large one.
func TestScalingTheBatchDoesNotScaleTheFloor(t *testing.T) {
	for name, k := range propertyFixtures(t) {
		small := k.StepTime(decodeBatch(16, 1, 4096)).NoOverlap.Seconds()
		large := k.StepTime(decodeBatch(32, 1, 4096)).NoOverlap.Seconds()
		if small <= 0 {
			t.Fatalf("%s: a 16-request step costs nothing", name)
		}
		if large >= 2*small {
			t.Errorf("%s: doubling the batch took the step from %.3fms to %.3fms, at least "+
				"2x; a per-operation floor cannot scale with bytes",
				name, small*1e3, large*1e3)
		}
	}
}

// TestRandomBatchesNeverProduceNonsense is a light fuzz over the shape space.
//
// Deterministically seeded so a failure is reproducible. It asserts only the invariants
// that must hold for ANY shape — finite, positive, ordered band, a named bottleneck —
// because that is what a randomly generated batch can be held to without an oracle. Its
// value is reaching combinations the tables above do not enumerate.
func TestRandomBatchesNeverProduceNonsense(t *testing.T) {
	for name, k := range propertyFixtures(t) {
		rng := rand.New(rand.NewSource(1))
		for i := range 200 {
			reqs := 1 + rng.Intn(512)
			q := 1 + rng.Intn(8)
			ctx := q + rng.Intn(131072)
			b := decodeBatch(reqs, q, ctx)
			e := k.StepTime(b)

			switch {
			case e.Overlap <= 0 || e.NoOverlap <= 0:
				t.Fatalf("%s: case %d (%d reqs, q=%d, ctx=%d) costs %v/%v",
					name, i, reqs, q, ctx, e.Overlap, e.NoOverlap)
			case e.Overlap > e.NoOverlap:
				t.Fatalf("%s: case %d (%d reqs, q=%d, ctx=%d) inverts the band: %v > %v",
					name, i, reqs, q, ctx, e.Overlap, e.NoOverlap)
			case e.Bottleneck == "":
				t.Fatalf("%s: case %d (%d reqs, q=%d, ctx=%d) names no bottleneck",
					name, i, reqs, q, ctx)
			}
		}
	}
}

// A pad-only step is a dummy forward, not an empty step, and the capture ceiling is met at
// the padded count: 500 scheduled tokens and a 13-token pad run past the 512-token ceiling
// with no graph.
func TestPadTokensRunTheForwardAndMeetTheCeiling(t *testing.T) {
	k := fixture(t, memoryFixture)
	if idle, padded := k.StepTime(kernel.Batch{}), k.StepTime(kernel.Batch{PadTokens: 64}); padded.NoOverlap <= idle.NoOverlap {
		t.Errorf("a 64-token pad prices %v, no more than an empty step's %v", padded.NoOverlap,
			idle.NoOverlap)
	}
	in := fixtureInputs(t, memoryFixture)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.CUDAGraphMode = "NONE" })
	eager := build(t, in)
	b := decodeBatch(1, 500, 500)
	b.PadTokens = 13
	if a, e := k.StepTime(b).PerResource[kernel.ResourceHost],
		eager.StepTime(b).PerResource[kernel.ResourceHost]; a != e {
		t.Errorf("513 padded tokens priced %v of host time, want NONE's %v", a, e)
	}
}
