// Original to blis-vllm.

package offloading

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// refEvict is what eviction must choose, stated without a heap: the n evictable keys that are not
// protected, least recent first by rank and then by key bytes (the order of vLLM's (rank, key)
// heap entries, lru.py:16, 70-74@4ba787ec). nil when fewer than n qualify.
func refEvict(p *lruPolicy, n int, protected map[Key]bool) []Key {
	var cands []Key
	for k := range p.evictable {
		if !protected[k] {
			cands = append(cands, k)
		}
	}
	if len(cands) < n {
		return nil
	}
	slices.SortFunc(cands, func(a, b Key) int {
		ra, rb := p.ranks[a], p.ranks[b]
		switch {
		case ra.less(rb):
			return -1
		case rb.less(ra):
			return 1
		}
		return compareKeys(a, b)
	})
	return append([]Key{}, cands[:n]...)
}

func compareKeys(a, b Key) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// The evictions of the lazy heap are the evictions of a full sort of the evictable chunks, over
// seeded random sequences of stores, loads, touches, request finishes and resets that give chunks
// provisional request ranks and equal ranks (a request's keys at one position in two KV cache
// groups share a rank, so the key bytes decide), pin and unpin chunks, and probe an eviction with
// and without protected keys, including ones that cannot be met.
func TestLazyHeapEvictsAsAFullSortWould(t *testing.T) {
	for seed := int64(1); seed <= 40; seed++ {
		t.Run(fmt.Sprint("seed", seed), func(t *testing.T) {
			rng := rand.New(rand.NewSource(seed))
			m := newCPUManager(30 + rng.Intn(40))
			type req struct {
				a   *reqAccess
				pos int
			}
			newReq := func() *req { return &req{a: newReqAccess()} }
			reqs := []*req{newReq(), newReq(), newReq()}
			var stores, loads []Key
			probes, hitsNil := 0, 0
			for step := 0; step < 1500; step++ {
				r := reqs[rng.Intn(len(reqs))]
				switch op := rng.Intn(12); {
				case op < 4: // a store batch: the next positions of a request, in two groups
					var keys []Key
					for i, n := 0, 1+rng.Intn(5); i < n; i++ {
						r.pos++
						h := []byte{byte(rng.Intn(6)), byte(r.pos % 7)}
						for g := 0; g < 2; g++ {
							k := makeKey(h, g)
							if _, ok := r.a.positions[k]; !ok {
								r.a.positions[k] = 16 * r.pos
							}
							keys = append(keys, k)
						}
					}
					if r.a.finished {
						continue
					}
					stored, _, _, ok := m.prepareStore(keys, r.a)
					if ok {
						stores = append(stores, stored...)
					}
				case op == 4 && len(stores) > 0: // a store completes, or fails
					i := rng.Intn(len(stores))
					m.completeStore([]Key{stores[i]}, rng.Intn(10) != 0)
					stores = slices.Delete(stores, i, i+1)
				case op == 5: // a load of ready chunks
					var keys []Key
					for k, c := range m.lru.chunks {
						if c.ready() && len(keys) < 1+rng.Intn(3) && !r.a.finished {
							keys = append(keys, k)
						}
					}
					slices.SortFunc(keys, compareKeys)
					if len(keys) > 0 {
						m.prepareLoad(keys, r.a)
						loads = append(loads, keys...)
					}
				case op == 6 && len(loads) > 0:
					i := rng.Intn(len(loads))
					m.completeLoad([]Key{loads[i]})
					loads = slices.Delete(loads, i, i+1)
				case op == 7: // a request finishes and is replaced
					m.requestFinished(r.a)
					for i := range reqs {
						if reqs[i] == r {
							reqs[i] = newReq()
						}
					}
				case op == 8: // a recency touch of cached keys
					var keys []Key
					for k := range m.lru.chunks {
						if rng.Intn(4) == 0 {
							keys = append(keys, k)
						}
					}
					slices.SortFunc(keys, compareKeys)
					m.touch(keys)
				case op == 9 && rng.Intn(40) == 0: // reset_cache
					m.resetCache()
					stores, loads = nil, nil
					reqs = []*req{newReq(), newReq(), newReq()}
				default: // an eviction probe
					n := rng.Intn(6)
					protected := map[Key]bool{}
					for k := range m.lru.chunks {
						if rng.Intn(5) == 0 {
							protected[k] = true
						}
					}
					if rng.Intn(3) == 0 {
						protected = nil
					}
					want := refEvict(m.lru, n, protected)
					got := m.lru.evict(n, protected)
					probes++
					if (want == nil) != (got == nil) {
						t.Fatalf("step %d: evict(%d) = %v, a full sort says %v", step, n, got, want)
					}
					if got == nil {
						hitsNil++
						continue
					}
					var keys []Key
					for _, e := range got {
						keys = append(keys, e.key)
					}
					if n > 0 && !slices.Equal(keys, want) {
						t.Fatalf("step %d: evict(%d) = %x, a full sort says %x", step, n, keys, want)
					}
					for _, e := range got {
						m.numEvictable--
						m.freeList = append(m.freeList, e.chunk.id)
					}
				}
				for k := range m.lru.evictable {
					if c := m.lru.chunks[k]; c == nil || c.refCnt != 0 {
						t.Fatalf("step %d: %x is evictable but not an unpinned cached chunk", step, []byte(k))
					}
				}
			}
			if probes == 0 {
				t.Fatal("no eviction was probed")
			}
		})
	}
}

// An eviction of k chunks compares O(k log n) heap entries however many chunks the tier holds; the
// cost of a sort of every evictable chunk grows with n log n. At 10^6 chunks one eviction of ten,
// and a long run of single evictions between recency updates, stay within a small multiple of
// k log2 n comparisons.
func TestEvictionCostGrowsWithTheEvictedChunksNotTheTier(t *testing.T) {
	const n = 1_000_000
	p := newLRU()
	for i := 0; i < n; i++ {
		p.insert(makeKey([]byte{byte(i >> 16), byte(i >> 8), byte(i)}, 0), &chunk{refCnt: 0, id: i}, nil)
	}
	logn := math.Log2(n)
	p.heap.compared = 0
	const k = 10
	out := p.evict(k, nil)
	if len(out) != k {
		t.Fatalf("evicted %d chunks, want %d", len(out), k)
	}
	if bound := int(4 * k * logn); p.heap.compared > bound {
		t.Fatalf("evicting %d of %d chunks compared %d heap entries, want at most %d (O(k log n))", k, n, p.heap.compared, bound)
	}
	// The oldest chunks went first.
	for i, e := range out {
		if want := makeKey([]byte{0, 0, byte(i)}, 0); e.key != want {
			t.Fatalf("eviction %d is %x, want the %dth chunk inserted, %x", i, []byte(e.key), i, []byte(want))
		}
	}
	// Steady state: a touch, an insert and an eviction of one chunk, a thousand times.
	p.heap.compared = 0
	const cycles = 1000
	for c := 0; c < cycles; c++ {
		p.touch([]Key{makeKey([]byte{0, byte(c >> 8), byte(c)}, 0)})
		p.insert(makeKey([]byte{1, byte(c >> 8), byte(c)}, 1), &chunk{refCnt: 0, id: n + c}, nil)
		if len(p.evict(1, nil)) != 1 {
			t.Fatal("no chunk evicted")
		}
	}
	if bound := int(8 * cycles * logn); p.heap.compared > bound {
		t.Errorf("%d touch, insert and evict cycles at %d chunks compared %d heap entries, want at most %d", cycles, n, p.heap.compared, bound)
	}
}
