package kvtransfer

import (
	"testing"
)

// byteReader hands out bounded values from a fuzz corpus, returning zeros once
// exhausted so the decode is total (never panics on short input).
type byteReader struct {
	data []byte
	pos  int
}

func (r *byteReader) u8() int {
	if r.pos >= len(r.data) {
		return 0
	}
	b := r.data[r.pos]
	r.pos++
	return int(b)
}

// fuzzOp is one decoded operation.
type fuzzOp struct {
	submit    bool // true = Submit, false = Poll
	tier      int
	dir       Direction
	bytes     int64
	tickDelta int64 // non-negative; ticks are cumulative and non-decreasing
}

// decodeScenario turns raw fuzz bytes into a validated station config and a
// sequence of operations. All fields are constrained to the station's documented
// contract (non-negative bytes, valid tier/direction, non-decreasing ticks) so
// the fuzzer explores adversarial-but-legal inputs rather than API misuse.
func decodeScenario(data []byte) (Config, []fuzzOp) {
	r := &byteReader{data: data}
	nTiers := 1 + r.u8()%2 // 1 or 2 tiers
	tiers := make([]TierConfig, nTiers)
	for i := range tiers {
		nRead := r.u8() % 4    // 0..3
		nWrite := r.u8() % 4   // 0..3
		if nRead+nWrite == 0 { // ensure ≥ 1 server
			nWrite = 1
		}
		tiers[i] = TierConfig{
			NRead:             nRead,
			NWrite:            nWrite,
			ReadBaseTicks:     int64(r.u8() % 5),
			WriteBaseTicks:    int64(r.u8() % 5),
			ReadBytesPerTick:  1 + float64(r.u8()%8),
			WriteBytesPerTick: 1 + float64(r.u8()%8),
			MaxQueueDepth:     r.u8() % 6, // 0..5; 0 = unbounded, small values force rejections
		}
	}

	nOps := r.u8() // up to 255 ops
	ops := make([]fuzzOp, 0, nOps)
	for i := 0; i < nOps; i++ {
		kind := r.u8()
		op := fuzzOp{
			submit:    kind%3 != 0, // ~2/3 submits, ~1/3 polls
			tier:      r.u8() % nTiers,
			dir:       Direction(r.u8() % 2),
			tickDelta: int64(r.u8() % 4),
		}
		// Byte sizes span the adversarial range: 0-byte, 1-block-ish, and very
		// large (up to ~1 GiB). The overflow-guard clamp for astronomically large
		// bytes is covered separately by TestServiceTicks_OverflowGuard; the fuzz
		// keeps sizes large-but-realistic so the drain horizon below is finite.
		switch r.u8() % 4 {
		case 0:
			op.bytes = 0
		case 1:
			op.bytes = int64(1 + r.u8())
		case 2:
			op.bytes = int64(r.u8()) * 1024
		default:
			op.bytes = int64(r.u8()%64+1) << 24 // very large: up to ~1 GiB
		}
		ops = append(ops, op)
	}
	return Config{Tiers: tiers}, ops
}

// runScenario executes the ops against a fresh station and returns the full
// completion sequence plus the set of accepted job ids. It enforces BC-S1 after
// every operation.
func runScenario(t *testing.T, cfg Config, ops []fuzzOp) (completed []JobID, accepted map[JobID]bool) {
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	accepted = map[JobID]bool{}
	tick := int64(0)
	activeTotal := func() int {
		total := 0
		for ti := range cfg.Tiers {
			total += s.ActiveJobs(ti, Read) + s.ActiveJobs(ti, Write)
		}
		return total
	}
	checkBCS1 := func() {
		for ti := range cfg.Tiers {
			inService := s.ActiveJobs(ti, Read) + s.ActiveJobs(ti, Write)
			limit := cfg.Tiers[ti].NRead + cfg.Tiers[ti].NWrite
			if inService > limit {
				t.Fatalf("BC-S1 violated: tier %d has %d in service, limit %d", ti, inService, limit)
			}
		}
	}
	for _, op := range ops {
		tick += op.tickDelta
		if op.submit {
			id, ok := s.Submit(TransferJob{Tier: op.tier, Direction: op.dir, Bytes: op.bytes, SubmitTick: tick})
			if ok {
				accepted[id] = true
			}
		} else {
			completed = append(completed, s.Poll(tick)...)
		}
		checkBCS1()
	}
	// Drain to completion: advance in large steps until nothing is in service.
	// This also asserts work-conservation — no accepted job is ever stranded.
	for iter := 0; activeTotal() > 0; iter++ {
		if iter > 2000 {
			t.Fatalf("station failed to drain: %d jobs still in service", activeTotal())
		}
		tick += 1 << 40
		completed = append(completed, s.Poll(tick)...)
		checkBCS1()
	}
	return completed, accepted
}

// FuzzStation drives the station with adversarial-but-legal scenarios and asserts
// the invariants that must hold on ALL inputs: BC-S1 (bounded concurrency),
// conservation (every accepted job completes exactly once; nothing else does),
// and BC-S4 determinism (a fresh replay of the same scenario yields an identical
// completion sequence).
func FuzzStation(f *testing.F) {
	// Seed corpus: empty, a small mix, and a config likely to force rejections.
	f.Add([]byte{})
	f.Add([]byte{1, 2, 1, 1, 1, 3, 3, 2, 0, 10, 5, 7, 2, 3, 1, 9, 4, 2, 0, 1})
	f.Add([]byte{2, 1, 0, 2, 0, 1, 1, 1, 1, 3, 40, 3, 1, 0, 1, 2, 1, 0, 3, 1, 2, 1, 1})

	f.Fuzz(func(t *testing.T, data []byte) {
		cfg, ops := decodeScenario(data)

		completed, accepted := runScenario(t, cfg, ops)

		// Conservation: every accepted job completed exactly once; no unknown or
		// duplicate completions.
		seen := map[JobID]int{}
		for _, id := range completed {
			seen[id]++
		}
		if len(seen) != len(accepted) {
			t.Fatalf("conservation: %d distinct completions, %d accepted", len(seen), len(accepted))
		}
		for id := range accepted {
			if seen[id] != 1 {
				t.Fatalf("conservation: job %d completed %d times, want 1", id, seen[id])
			}
		}

		// Determinism (BC-S4): a fresh replay of the same scenario produces an
		// identical completion sequence.
		completed2, _ := runScenario(t, cfg, ops)
		if !equalIDs(completed, completed2) {
			t.Fatalf("non-deterministic completion sequence across identical runs")
		}
	})
}
