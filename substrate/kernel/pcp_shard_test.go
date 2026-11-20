package latencykernel

import (
	"math"
	"testing"
)

// vllmPCPRankTokens is _iter_rank_chunks transcribed for one rank
// (vllm/v1/worker/gpu/pcp_manager.py:235-273 at v0.31.0): each prefill is cut into 2*pcp
// chunks of ceil(sched/(2*pcp)), and rank r takes chunks r and 2*pcp-1-r. It is the ORACLE
// the closed form must match.
func vllmPCPRankTokens(sched, pcp, rank int) int {
	numChunks := 2 * pcp
	chunk := (sched + numChunks - 1) / numChunks
	total := 0
	for _, idx := range [...]int{rank, numChunks - 1 - rank} {
		lo := idx * chunk
		hi := lo + chunk
		if hi > sched {
			hi = sched
		}
		if hi > lo {
			total += hi - lo
		}
	}
	return total
}

// The busiest rank binds, because the ranks gather before the next layer and the step waits
// for the slowest. And the shards must conserve the chunk: a split that lost or duplicated
// tokens would misprice every prefill.
//
// Mutation-checked: a nominal sched/pcp, a floor instead of a ceiling, and 2*pcp in place
// of pcp each fail here.
func TestPCPLocalTokensMatchesTheBusiestRankAndConservesTheChunk(t *testing.T) {
	for _, pcp := range []int{2, 4, 8} {
		for sched := 1; sched < 600; sched++ {
			want, sum := 0, 0
			for rank := 0; rank < pcp; rank++ {
				v := vllmPCPRankTokens(sched, pcp, rank)
				if v > want {
					want = v
				}
				sum += v
			}
			if sum != sched {
				t.Fatalf("the oracle does not conserve: pcp=%d sched=%d shards sum to %d",
					pcp, sched, sum)
			}
			if got := pcpLocalTokens(sched, pcp, 1); got != want {
				t.Fatalf("pcp=%d sched=%d: got %d, want %d (the busiest rank's share)",
					pcp, sched, got, want)
			}
		}
	}
}

// Replication of a short prefill happens ONLY ALONGSIDE DCP, and the condition is not the
// one the mechanism suggests.
//
// replicated_requests (pcp_manager.py:222-233 at v0.31.0) computes `drops_a_chunk` inside
// `if self.dcp_world_size > 1`, so with PCP alone a prefill is partitioned however short
// it is, and a rank that owns no chunk simply contributes nothing. An earlier draft of
// this test asserted the opposite -- that any prefill under 2*pcp tokens is replicated at
// every width -- and it failed against correct code, which is recorded here because the
// gate is the kind of detail a reader reconstructs wrongly from the mechanism alone.
//
// The condition is `(2*pcp - 1) * ceil(sched/(2*pcp)) >= sched`, which detects a partition
// with an empty chunk. It is NOT "shorter than 2*pcp": at pcp 8 it holds at 17 tokens and
// fails at 16.
func TestPCPReplicatesOnlyAlongsideDCPAndOnTheEnginesOwnCondition(t *testing.T) {
	for _, pcp := range []int{2, 4, 8} {
		numChunks := 2 * pcp
		for sched := 1; sched < 80; sched++ {
			chunk := (sched + numChunks - 1) / numChunks
			replicates := (numChunks-1)*chunk >= sched

			// With DCP off there is no replication branch: the busiest rank holds its
			// own two chunks, each truncated where it runs past the query. Taken from
			// the oracle rather than as 2*chunk, which is right only when every chunk
			// is full -- at two tokens over two ranks the chunks are one token each and
			// the busiest rank holds ONE, not two.
			withoutDCP := pcpLocalTokens(sched, pcp, 1)
			wantWithout := 0
			for rank := 0; rank < pcp; rank++ {
				if v := vllmPCPRankTokens(sched, pcp, rank); v > wantWithout {
					wantWithout = v
				}
			}
			if withoutDCP != wantWithout {
				t.Errorf("pcp=%d sched=%d dcp=1: got %d, want %d", pcp, sched,
					withoutDCP, wantWithout)
			}

			// With DCP on, a partition that would drop a chunk is replicated instead.
			withDCP := pcpLocalTokens(sched, pcp, 2)
			if replicates {
				if withDCP != sched {
					t.Errorf("pcp=%d sched=%d dcp=2: got %d, want the whole query %d "+
						"(this partition drops a chunk, so the engine replicates)",
						pcp, sched, withDCP, sched)
				}
			} else if withDCP != wantWithout {
				t.Errorf("pcp=%d sched=%d dcp=2: got %d, want %d (no chunk is dropped, "+
					"so the query is partitioned)", pcp, sched, withDCP, wantWithout)
			}
		}
	}
	// The documented boundary: at pcp 8 the condition holds at 17 and fails at 16, which
	// is what rules out reading it as "shorter than 2*pcp".
	if got := pcpLocalTokens(17, 8, 2); got != 17 {
		t.Errorf("17 tokens over 8 prefill-context ranks alongside DCP: got %d, want 17", got)
	}
	if got := pcpLocalTokens(16, 8, 2); got != 2 {
		t.Errorf("16 tokens over 8 prefill-context ranks alongside DCP: got %d, want 2 "+
			"(sixteen chunks of one, two per rank)", got)
	}
	// Absent or degenerate widths read as "no split", never as a division by zero.
	for _, pcp := range []int{1, 0, -2} {
		if got := pcpLocalTokens(4096, pcp, 1); got != -1 {
			t.Errorf("pcp=%d should read as absent: got %d, want -1", pcp, got)
		}
	}
	if got := pcpLocalTokens(0, 4, 1); got != -1 {
		t.Errorf("an empty chunk should read as absent: got %d", got)
	}
}

// The causal count must reproduce the unsharded figure's own CONTINUUM convention, so an
// inactive split changes nothing and a 2*pcp-divisible one is exactly the whole count over
// pcp.
//
// The repository's unsharded term is s*c + s^2/2, which is half a token below the exact
// per-position sum s*c + s(s+1)/2. windowedCausalFLOPs already subtracts that half-token
// for the same reason. Getting it wrong here would have made every PCP prefill 0.005%
// expensive at a 4,096-token chunk and, worse, made an exactly-divisible split disagree
// with the figure it replaces -- which is how this was caught.
func TestPCPCausalCountKeepsTheUnshardedContinuumConvention(t *testing.T) {
	for _, c := range []struct{ sched, prefix int }{
		{4096, 8192}, {2048, 0}, {8192, 131072}, {1024, 203760},
	} {
		s := float64(c.sched)
		whole := 2 * 2 * (s*float64(c.prefix) + s*s*0.5)
		for _, pcp := range []int{2, 4, 8} {
			if c.sched%(2*pcp) != 0 {
				continue // only the exactly-divisible case has a closed expectation
			}
			got := pcpCausalPairs(c.sched, c.prefix, pcp, 1)
			if want := whole / float64(pcp); math.Abs(got-want)/want > 1e-12 {
				t.Errorf("sched=%d prefix=%d pcp=%d: %v pairs, want %v (the unsharded "+
					"count over the split)", c.sched, c.prefix, pcp, got, want)
			}
		}
	}
	// A RAGGED PARTITION COSTS MORE THAN THE NOMINAL SHARE, and the exact count is what
	// says by how much. This is the case that separates counting the engine's partition
	// from dividing the whole figure by pcp: where 2*pcp divides the query the two agree
	// exactly, and where it does not the busiest rank carries measurably more, because a
	// ceil-based chunking hands one rank two chunks from the expensive end.
	//
	// Pinned at both ends so neither form can pass: exact agreement at 4,096 tokens over
	// 8 ranks, and a strict excess at 1,001 over 8 (1.4%) and 100 over 8 (25.4%).
	for _, c := range []struct {
		sched, prefix, pcp int
		wantRatio          float64
		tol                float64
	}{
		{4096, 0, 8, 1.000000, 1e-12}, // 2*pcp divides the query: the two forms agree
		{1001, 0, 8, 1.014035, 1e-5},  // ragged
		{1001, 8192, 4, 1.007398, 1e-5},
		{100, 0, 8, 1.254400, 1e-5}, // a query barely longer than the chunk count
	} {
		s := float64(c.sched)
		nominal := 2 * 2 * (s*float64(c.prefix) + s*s*0.5) / float64(c.pcp)
		got := pcpCausalPairs(c.sched, c.prefix, c.pcp, 1)
		if ratio := got / nominal; math.Abs(ratio-c.wantRatio) > c.tol {
			t.Errorf("sched=%d prefix=%d pcp=%d: the busiest rank carries %.6f of the "+
				"nominal share, want %.6f; the count must follow the engine's partition "+
				"rather than dividing the whole figure",
				c.sched, c.prefix, c.pcp, ratio, c.wantRatio)
		}
	}

	// A ragged partition costs the busiest rank MORE than the ideal share, and the
	// overhead is bounded: the engine's chunking is ceil-based, so one rank carries two
	// chunks from the expensive end. Asserted as a bound rather than a value because the
	// figure depends on where the remainder falls.
	sched, prefix, pcp := 1001, 0, 8
	s := float64(sched)
	ideal := 2 * 2 * (s*float64(prefix) + s*s*0.5) / float64(pcp)
	got := pcpCausalPairs(sched, prefix, pcp, 1)
	if got < ideal {
		t.Errorf("a ragged split gave %v pairs, below the ideal share %v; the busiest "+
			"rank cannot carry less than an even share", got, ideal)
	}
	if got > ideal*1.05 {
		t.Errorf("a ragged split gave %v pairs, more than 5%% above the ideal share %v",
			got, ideal)
	}
}

// THE ZIGZAG IS WHAT MAKES A SINGLE DIVISOR DEFENSIBLE for the causal term, and this test
// is the evidence rather than the claim.
//
// Pairing chunk r with chunk 2*pcp-1-r pairs a cheap early chunk, with few keys to its
// left, against an expensive late one. Where 2*pcp divides the chunk the per-rank causal
// pair counts come out EXACTLY equal, so dividing the batch-wide count by pcp is the right
// per-rank figure and not an approximation. A contiguous 1/pcp slice would not balance at
// all: the last rank would carry the whole upper triangle.
//
// Asserted on the engine's own partition, over prefixes too, since a chunked prefill is the
// case the repository's causal count was corrected for.
func TestPCPZigzagBalancesTheCausalWorkExactly(t *testing.T) {
	pairs := func(sched, prefix, pcp, rank int) float64 {
		numChunks := 2 * pcp
		chunk := (sched + numChunks - 1) / numChunks
		var total float64
		for _, idx := range [...]int{rank, numChunks - 1 - rank} {
			lo := idx * chunk
			hi := lo + chunk
			if hi > sched {
				hi = sched
			}
			for j := lo; j < hi; j++ {
				total += float64(prefix + j + 1)
			}
		}
		return total
	}
	for _, c := range []struct{ sched, prefix int }{
		{2048, 0}, {4096, 8192}, {1024, 131072}, {8192, 0}, {2048, 203760},
	} {
		for _, pcp := range []int{2, 4, 8} {
			var ideal float64
			for j := 0; j < c.sched; j++ {
				ideal += float64(c.prefix + j + 1)
			}
			ideal /= float64(pcp)
			lo, hi := math.Inf(1), 0.0
			for rank := 0; rank < pcp; rank++ {
				v := pairs(c.sched, c.prefix, pcp, rank)
				lo, hi = math.Min(lo, v), math.Max(hi, v)
			}
			// Exactly equal across ranks, so the ideal share IS each rank's share.
			if hi != lo {
				t.Errorf("sched=%d prefix=%d pcp=%d: ranks carry between %.0f and %.0f "+
					"causal pairs; the zigzag must balance them exactly",
					c.sched, c.prefix, pcp, lo, hi)
			}
			if math.Abs(hi-ideal)/ideal > 1e-12 {
				t.Errorf("sched=%d prefix=%d pcp=%d: each rank carries %.0f pairs against "+
					"an ideal share of %.0f", c.sched, c.prefix, pcp, hi, ideal)
			}
		}
	}
}
