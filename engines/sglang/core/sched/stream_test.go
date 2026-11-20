// Original to blis-sglang.

package sched

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-sglang/core/cache"
)

func shuffled(seed int64, name string, n int) []int {
	xs := make([]int, n)
	for i := range xs {
		xs[i] = i
	}
	NewStream(seed, name).Shuffle(n, func(i, j int) { xs[i], xs[j] = xs[j], xs[i] })
	return xs
}

// The same seed and name give the same shuffle; a different seed or name gives another.
func TestStreamIsRepeatableAndNamed(t *testing.T) {
	a := shuffled(7, "q", 12)
	if !reflect.DeepEqual(a, shuffled(7, "q", 12)) {
		t.Fatal("the same seed and name shuffled two ways")
	}
	if reflect.DeepEqual(a, shuffled(8, "q", 12)) || reflect.DeepEqual(a, shuffled(7, "other", 12)) {
		t.Fatal("a different seed or name gave the same shuffle")
	}
}

// Every arrangement of three items comes up about equally often.
func TestStreamShuffleIsUniform(t *testing.T) {
	counts := map[[3]int]int{}
	const trials = 6000
	for seed := int64(0); seed < trials; seed++ {
		x := shuffled(seed, "uniform", 3)
		counts[[3]int{x[0], x[1], x[2]}]++
	}
	if len(counts) != 6 {
		t.Fatalf("%d of the 6 arrangements occurred", len(counts))
	}
	for k, c := range counts {
		if c < trials/6*85/100 || c > trials/6*115/100 {
			t.Fatalf("arrangement %v occurred %d times of %d, far from %d", k, c, trials, trials/6)
		}
	}
}

// Without a supplied order, the random policy shuffles the queue with the scheduler's own stream:
// two schedulers with one seed schedule alike, and some seed schedules otherwise than first-come.
func TestRandomPolicyUsesTheSeededStream(t *testing.T) {
	order := func(seed int) []string {
		cfg := baseConfig(t)
		cfg.Policy = "random"
		cfg.RandomSeed = seed
		s, err := build(cfg, 13)
		if err != nil {
			t.Fatal(err)
		}
		var queue []*Req
		for _, id := range []string{"a", "b", "c", "d", "e", "f"} {
			queue = append(queue, &Req{C: cache.Req{ID: id}})
		}
		s.calcPriority(queue, &Batch{})
		var out []string
		for _, r := range queue {
			out = append(out, r.RID())
		}
		return out
	}
	if !reflect.DeepEqual(order(3), order(3)) {
		t.Fatal("one seed ordered the queue two ways")
	}
	moved := false
	for seed := 0; seed < 8; seed++ {
		if !reflect.DeepEqual(order(seed), []string{"a", "b", "c", "d", "e", "f"}) {
			moved = true
		}
	}
	if !moved {
		t.Fatal("the random policy never reordered the queue")
	}
}
