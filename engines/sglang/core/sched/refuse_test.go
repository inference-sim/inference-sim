// Original to blis-sglang.

package sched

import (
	"errors"
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
		{"priority", func(c *Config) { c.EnablePriorityScheduling = true }, "priority scheduling"},
		{"overlap", func(c *Config) { c.EnableOverlap = true }, "overlap"},
		{"disagg", func(c *Config) { c.DisaggregationMode = "prefill" }, "disaggregation"},
		{"interval", func(c *Config) { c.PrefillDecodeInterval = 4 }, "prefill-decode interval"},
		{"pp", func(c *Config) { c.PPSize = 2 }, "pipeline parallelism"},
		{"random", func(c *Config) { c.Policy = "random" }, "random"},
		{"dfs", func(c *Config) { c.Policy = "dfs-weight" }, "dfs-weight"},
		{"routing", func(c *Config) { c.Policy = "routing-key" }, "routing-key"},
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

func TestInputSGLangAbortsIsRefusedOnArrival(t *testing.T) {
	cfg := baseConfig(t)
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	long := make([]int64, cfg.MaxReqInputLen)
	_, err = s.Step([]Arrival{{RID: "x", Prompt: long, MaxNewTokens: 1}})
	if !errors.Is(err, ErrRefused) || !strings.Contains(err.Error(), Refusal) {
		t.Fatalf("got %v, want a refusal citing %s", err, Refusal)
	}
}
