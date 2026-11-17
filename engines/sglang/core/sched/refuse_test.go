// Original to blis-sglang.

package sched

import (
	"strings"
	"testing"
)

// What the normal loop does not model is refused by name, citing the open issue (X28), before any
// request runs; an input SGLang aborts at ingest is refused when it arrives.

func baseConfig(t *testing.T) Config {
	t.Helper()
	var tr trace
	names := loadGoldens(t)
	readJSON(t, names["retract1"], &tr)
	return tr.Config
}

func TestUnmodelledSettingsAreRefusedByName(t *testing.T) {
	cases := []struct {
		name string
		edit func(*Config)
		want string
	}{
		{"overlap", func(c *Config) { c.EnableOverlap = true }, "overlap"},
		{"disagg", func(c *Config) { c.DisaggregationMode = "prefill" }, "disaggregation"},
		{"interval", func(c *Config) { c.PrefillDecodeInterval = 4 }, "prefill-decode interval"},
		{"pp", func(c *Config) { c.PPSize = 2 }, "pipeline parallelism"},
		{"unknown", func(c *Config) { c.Policy = "nope" }, "unknown schedule_policy"},
		{"tree", func(c *Config) { c.TreeCache = "HiRadixCache" }, "HiRadixCache"},
		{"allocator", func(c *Config) { c.Allocator = "SWATokenToKVPoolAllocator" }, "SWATokenToKVPoolAllocator"},
		{"spf without cache", func(c *Config) { c.Policy = "shortest-prefill-first"; c.Disable = true }, "without prefix caching"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseConfig(t)
			tc.edit(&cfg)
			_, err := build(cfg, 13)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error naming %q", err, tc.want)
			}
		})
	}
}

// A prompt SGLang aborts at ingest is not refused: it runs as the single token SGLang shrinks it to
// and finishes with an abort, and the pool is whole afterwards.
func TestInputTooLongIsAbortedNotRefused(t *testing.T) {
	cfg := baseConfig(t)
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	before := s.alloc.AvailableSize()
	long := make([]int64, cfg.MaxReqInputLen)
	var finished []string
	for i := 0; i < 5; i++ {
		var arr []Arrival
		if i == 0 {
			arr = []Arrival{{RID: "x", Prompt: long, MaxNewTokens: 4, IgnoreEOS: true}}
		}
		rec, err := s.Step(arr)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		finished = append(finished, rec.Finished...)
	}
	if len(finished) != 1 || finished[0] != "x" {
		t.Fatalf("finished %v, want the aborted request alone", finished)
	}
	if got := s.alloc.AvailableSize() + s.tree.EvictableSize(); got != before {
		t.Fatalf("pool holds %d tokens after the abort, had %d", got, before)
	}
}
