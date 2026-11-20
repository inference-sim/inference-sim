// Original to blis-sglang.

package sched

import "testing"

// A request's cache salt isolates its prefix-cache entries: a request reads the prefix of an earlier
// request with the same salt, and none of a request with another salt or none.
func TestACacheSaltIsolatesThePrefixCache(t *testing.T) {
	cfg := baseConfig(t)
	cfg.ChunkedPrefillSize = nil
	s, err := build(cfg, 13)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int64{1, 2, 3, 4, 5, 6}
	prefix := func(rid, salt string) int {
		t.Helper()
		rec := mustStep(t, s, Arrival{RID: rid, Prompt: prompt, MaxNewTokens: 2, CacheSalt: salt})
		for _, f := range rec.Forwards {
			if f.Mode == Extend && len(f.RIDs) > 0 && f.RIDs[0] == rid {
				return f.ExtendPrefixLen[0]
			}
		}
		t.Fatalf("%s: no prefill pass in %+v", rid, rec.Forwards)
		return -1
	}
	if got := prefix("a", "s"); got != 0 {
		t.Fatalf("the first request read a prefix of %d", got)
	}
	if got := prefix("b", "s"); got != 5 {
		t.Errorf("a request with the same salt read a prefix of %d, want 5", got)
	}
	if got := prefix("c", "t"); got != 0 {
		t.Errorf("a request with another salt read a prefix of %d, want 0", got)
	}
	if got := prefix("d", ""); got != 0 {
		t.Errorf("a request with no salt read a prefix of %d, want 0", got)
	}
}
