package engine

import (
	"strings"
	"testing"
)

func ip(n int) *int { return &n }

// An event of an adapter's run, of a salted run, of a sliding-window group and a removal all
// validate; each rule below has an event that breaks it and the message says which.
func TestKVEventValidate(t *testing.T) {
	good := map[string]KVEvent{
		"base block":       {Hash: "h1", Tokens: []int32{1, 2}},
		"adapter block":    {Hash: "h1", LoRAID: ip(3), LoRAName: "sql", ExtraKeys: []any{[]any{"lora", "sql", "/loras/sql"}}},
		"salted block":     {Hash: "h1", CacheSalt: "tenant-a"},
		"sliding window":   {Hash: "h1", GroupIdx: ip(1), SpecKind: "sliding_window", SlidingWindow: ip(1024)},
		"removal":          {Removed: true, Hash: "h1", GroupIdx: ip(1)},
		"cleared":          {Cleared: true},
		"cleared one tier": {Cleared: true, Tier: "cpu"},
	}
	for name, e := range good {
		if err := e.Validate(); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	bad := map[string]struct {
		e    KVEvent
		want string
	}{
		"cleared with a hash":     {KVEvent{Cleared: true, Hash: "h1"}, "names no block"},
		"cleared with an adapter": {KVEvent{Cleared: true, LoRAName: "sql"}, "carries no identity"},
		"no hash":                 {KVEvent{Tokens: []int32{1}}, "names a block hash"},
		"removal with tokens":     {KVEvent{Removed: true, Hash: "h1", Tokens: []int32{1}}, "removal carries no tokens"},
		"removal with an adapter": {KVEvent{Removed: true, Hash: "h1", LoRAName: "sql"}, "removal carries no tokens"},
		"salt and extra keys":     {KVEvent{Hash: "h1", CacheSalt: "s", ExtraKeys: []any{"x"}}, "not both"},
		"negative group":          {KVEvent{Hash: "h1", GroupIdx: ip(-1)}, "negative"},
		"zero window":             {KVEvent{Hash: "h1", SpecKind: "sliding_window", SlidingWindow: ip(0)}, "not positive"},
		"window on a full group":  {KVEvent{Hash: "h1", SpecKind: "full_attention", SlidingWindow: ip(128)}, "which has none"},
	}
	for name, c := range bad {
		err := c.e.Validate()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %v, want an error containing %q", name, err, c.want)
		}
	}
}
