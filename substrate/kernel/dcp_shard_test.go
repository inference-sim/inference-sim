package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// vllmLocalSeqLen is get_dcp_local_seq_lens transcribed for one rank
// (vllm/v1/attention/backends/utils.py:1143-1156 at v0.31.0). It is the ORACLE: the
// kernel's own form must equal the maximum of this across ranks, because the decode
// combine is a collective and every rank waits for the one holding the most tokens.
func vllmLocalSeqLen(seqLen, dcp, interleave, rank int) int {
	base := seqLen / interleave / dcp * interleave
	rem := seqLen - base*dcp - rank*interleave
	if rem < 0 {
		rem = 0
	}
	if rem > interleave {
		rem = interleave
	}
	return base + rem
}

// The headline arithmetic property. Charging the mean would price a step cheaper than any
// rank can deliver, and dropping the interleave remainder would price it cheaper still.
//
// Mutation-checked: replacing the body with ctx/dcp (floor), with ceil over the batch
// total, or with base alone each fails here.
func TestDCPLocalTokensMatchesTheSlowestRankNotTheMean(t *testing.T) {
	for _, dcp := range []int{2, 4, 8} {
		for _, interleave := range []int{1, 16, 32} {
			for seqLen := 0; seqLen < 600; seqLen++ {
				want, sum := 0, 0
				for rank := 0; rank < dcp; rank++ {
					v := vllmLocalSeqLen(seqLen, dcp, interleave, rank)
					if v > want {
						want = v
					}
					sum += v
				}
				// The engine's own form must conserve the context. If this ever fails the
				// oracle is wrong, not the kernel.
				if sum != seqLen {
					t.Fatalf("oracle does not conserve: dcp=%d interleave=%d seqLen=%d "+
						"shards sum to %d", dcp, interleave, seqLen, sum)
				}
				if seqLen > 0 && want < 1 {
					want = 1
				}
				got := dcpLocalTokens([]int{seqLen}, dcp, interleave)
				if got != float64(want) {
					t.Fatalf("dcp=%d interleave=%d seqLen=%d: got %v, want %d "+
						"(the slowest rank's share)", dcp, interleave, seqLen, got, want)
				}
			}
		}
	}
}

// At the engine's default interleave of 1 -- what an unstated size runs at unless NIXL pins
// it -- the form must reduce to ceil(context/dcp). Stated separately because it is the
// common case, and because a reader checking this file against vLLM will look for it.
func TestDCPLocalTokensReducesToCeilingAtTheDefaultInterleave(t *testing.T) {
	for _, seqLen := range []int{1, 7, 100, 999, 8192, 131072} {
		for _, dcp := range []int{2, 3, 8} {
			want := float64((seqLen + dcp - 1) / dcp)
			if got := dcpLocalTokens([]int{seqLen}, dcp, 1); got != want {
				t.Errorf("seqLen=%d dcp=%d: got %v, want %v", seqLen, dcp, got, want)
			}
		}
	}
	// The documented worked example: striping contributes a bounded additive term, not a
	// multiplicative one. 1000 tokens over 8 ranks in runs of 32 gives the slowest rank
	// 128 against a nominal 125 -- under 3%. Bounded is not the same as small, though: the
	// term is at most one run per request, so on a context short against the run it
	// dominates, and 100 tokens over 8 ranks in runs of 64 put 64 on the slowest rank
	// against a nominal 13.
	if got := dcpLocalTokens([]int{1000}, 8, 32); got != 128 {
		t.Errorf("interleave 32 at 1000 tokens over 8 ranks: got %v, want 128", got)
	}
	if got := dcpLocalTokens([]int{100}, 8, 64); got != 64 {
		t.Errorf("interleave 64 at 100 tokens over 8 ranks: got %v, want 64", got)
	}
}

// The shards must sum to the whole context: tokens divide freely, with NO replication
// floor. This is where DCP differs from the KV-head division next door, which floors at
// one head because a head is never split -- and copying that floor here would be the
// natural mistake, since the two functions sit feet apart.
func TestDCPLocalTokensShardsSumToTheWholeContextWithNoReplication(t *testing.T) {
	for _, dcp := range []int{2, 4, 8} {
		for _, seqLen := range []int{16, 17, 100, 1024, 32768} {
			var sum int
			for rank := 0; rank < dcp; rank++ {
				sum += vllmLocalSeqLen(seqLen, dcp, 1, rank)
			}
			if sum != seqLen {
				t.Errorf("dcp=%d seqLen=%d: shards sum to %d", dcp, seqLen, sum)
			}
			// A floored form would exceed the context once summed. The kernel's figure is
			// the maximum rather than the sum, so the check is that it never exceeds the
			// context and never falls below the fair share.
			got := dcpLocalTokens([]int{seqLen}, dcp, 1)
			if got > float64(seqLen) {
				t.Errorf("dcp=%d seqLen=%d: a shard of %v exceeds the whole context",
					dcp, seqLen, got)
			}
			if got < float64(seqLen)/float64(dcp) {
				t.Errorf("dcp=%d seqLen=%d: a shard of %v is below the fair share %v",
					dcp, seqLen, got, float64(seqLen)/float64(dcp))
			}
		}
	}
}

// A context shorter than the group still occupies a token somewhere. Zero here would make
// a decode read free, which is the silent-and-cheap direction this repository refuses.
func TestDCPLocalTokensFloorsAtOneTokenPerRank(t *testing.T) {
	for _, dcp := range []int{2, 8, 64} {
		if got := dcpLocalTokens([]int{1}, dcp, 1); got != 1 {
			t.Errorf("one token over %d ranks: got %v, want 1", dcp, got)
		}
	}
	// Defensive arithmetic, since a library cannot assume its caller validated: a
	// non-positive width is absent, not a division by zero. blis-schemas owns the
	// admissibility rules (tp % dcp == 0, and dcp in {1, pcp, tp*pcp} when pcp is on);
	// this only refuses to divide by zero, following the max(TP, 1) pattern elsewhere.
	for _, dcp := range []int{0, -4} {
		if got := dcpLocalTokens([]int{4096}, dcp, 1); got != -1 {
			t.Errorf("dcp=%d should read as absent: got %v, want -1", dcp, got)
		}
	}
	if got := dcpLocalTokens([]int{4096}, 4, 0); got != 1024 {
		t.Errorf("a zero interleave should fall back to the engine default of 1: "+
			"got %v, want 1024", got)
	}
}

// Per request, not over the batch total. The remainder term is per sequence, so
// sum-then-shard and shard-then-sum disagree whenever a context is not a multiple of
// interleave*dcp -- which in a real batch is the normal case.
func TestDCPLocalTokensBoundsEachRequestNotTheBatchTotal(t *testing.T) {
	// Four requests of 1000 at dcp=3: each pays its own ceiling, 4*334 = 1336. Sharding
	// the 4000-token total once would give 1334.
	perRequest := dcpLocalTokens([]int{1000, 1000, 1000, 1000}, 3, 1)
	batchTotal := dcpLocalTokens([]int{4000}, 3, 1)
	if perRequest <= batchTotal {
		t.Errorf("per-request sharding gave %v and batch-total sharding %v; the "+
			"per-request figure must be the larger, since each sequence pays its own "+
			"remainder", perRequest, batchTotal)
	}
	if perRequest != 1336 || batchTotal != 1334 {
		t.Errorf("got per-request %v and batch-total %v; want 1336 and 1334",
			perRequest, batchTotal)
	}
}

// Which kinds shard, and which the engine refuses to shard at all.
func TestDCPShardsOnlyTheKindsTheEngineShards(t *testing.T) {
	for _, c := range []struct {
		kind model.AttentionKind
		want bool
		why  string
	}{
		{model.AttentionGQA, true, "full attention: AttentionSpec.dcp_sharded is True"},
		{model.AttentionMLA, true, "latent: the only axis that can shard a one-head cache"},
		{model.AttentionSparseMLA, true, "latent and selective: still sharded by token"},
		{model.AttentionSWA, false, "refused: \"DCP not support sliding window\""},
		{model.AttentionKind("something-new"), false, "unrecognized over-prices rather than under-prices"},
	} {
		if got := dcpShardsKV(c.kind); got != c.want {
			t.Errorf("%s: got %v, want %v (%s)", c.kind, got, c.want, c.why)
		}
	}
}
