// Original to blis-sglang.

package metrics

import (
	"math"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-sglang/core/sched"
)

// These tests pin two observable behaviours of the registry (M3.1, blis-sglang#107): the cache hit
// rate of a prefill that moved no tokens, and the order of Series.

func metricsConfig(priority bool) sched.Config {
	return sched.Config{PageSize: 1, MaxTotalNumTokens: 100, ContextLen: 64, EnablePriorityScheduling: priority}
}

func find(t *testing.T, ss []Series, name string) Series {
	t.Helper()
	for _, s := range ss {
		if s.Name == name {
			return s
		}
	}
	t.Fatalf("no series %s", name)
	return Series{}
}

// report_prefill_stats sets cache_hit_rate to 0 when the batch's effective input and hit are both
// zero (metrics_reporter.py:719-899@297e54fd); dividing 0 by 0 would give NaN.
func TestCacheHitRateIsZeroWhenAPrefillMovedNoTokens(t *testing.T) {
	c := New(metricsConfig(false))
	// Every counted token was reprocessed, so the effective input and hit are 0 and 0.
	c.Observe([]sched.Report{{Logged: true, Prefill: &sched.PrefillStats{LogInputTokens: 3, ReprocessedInput: 3, LogHitTokens: 2, ReprocessedHit: 2}}})
	got := find(t, c.Series(), "sglang:cache_hit_rate").Value
	if got != 0 || math.IsNaN(got) {
		t.Fatalf("cache_hit_rate = %v, want 0", got)
	}
}

func TestCacheHitRateIsHitOverInputPlusHit(t *testing.T) {
	c := New(metricsConfig(false))
	c.Observe([]sched.Report{{Logged: true, Prefill: &sched.PrefillStats{LogInputTokens: 10, LogHitTokens: 30}}})
	if got := find(t, c.Series(), "sglang:cache_hit_rate").Value; got != 0.75 {
		t.Fatalf("cache_hit_rate = %v, want 0.75", got)
	}
}

func labelString(l map[string]string) string {
	keys := make([]string, 0, len(l))
	for k := range l {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		b.WriteString(k + "=" + l[k] + ",")
	}
	return b.String()
}

// Series are sorted by name, then by labels, and no two share both.
func TestSeriesAreSortedByNameThenLabelsWithoutDuplicates(t *testing.T) {
	c := New(metricsConfig(true))
	c.Observe([]sched.Report{{
		Logged:  true,
		Prefill: &sched.PrefillStats{LogInputTokens: 5, Running: sched.Queue{Total: 1, ByPriority: map[int]int{1: 1}}},
		Waiting: sched.Queue{Total: 4, ByPriority: map[int]int{2: 1, 0: 3}},
	}})
	ss := c.Series()
	for i := 1; i < len(ss); i++ {
		a, b := ss[i-1], ss[i]
		if a.Name > b.Name || (a.Name == b.Name && labelString(a.Labels) >= labelString(b.Labels)) {
			t.Fatalf("series %d %s{%s} is not strictly before %s{%s}", i, a.Name, labelString(a.Labels), b.Name, labelString(b.Labels))
		}
	}
	var prios []string
	for _, s := range ss {
		if s.Name == "sglang:num_queue_reqs" {
			prios = append(prios, s.Labels["priority"])
		}
	}
	if got := strings.Join(prios, "|"); got != "|0|1|2" {
		t.Fatalf("num_queue_reqs priorities in order = %q, want total then 0, 1, 2", got)
	}
}
