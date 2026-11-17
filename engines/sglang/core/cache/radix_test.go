// Original to blis-sglang.

package cache

import (
	"strings"
	"testing"
)

// What the FULL radix cache does not model is refused at construction, by name, citing its open
// issue (SGL-6, X28); each refusal has an accepted partner.
func TestTheRadixCacheRefusesWhatItDoesNotModelByName(t *testing.T) {
	pool := &stubPool{rows: map[int]map[int]int64{}}
	for _, tc := range []struct {
		cfg  RadixConfig
		want string
	}{
		{RadixConfig{PageSize: 1, EvictionPolicy: "tlru"}, "tlru"},
		{RadixConfig{PageSize: 1, SessionRadixCache: true}, "--enable-session-radix-cache"},
		{RadixConfig{PageSize: 1, EvictionPolicy: "nope"}, "unknown eviction policy"},
	} {
		_, err := NewRadixCache(tc.cfg, &stubAllocator{page: 1}, pool)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%+v: %v, want a refusal naming %q", tc.cfg, err, tc.want)
		}
		if tc.want != "unknown eviction policy" && !strings.Contains(err.Error(), RefusalIssue) {
			t.Errorf("%+v: the refusal cites no issue: %v", tc.cfg, err)
		}
	}
	for _, p := range []string{"", "lru", "LFU", "fifo", "mru", "filo", "priority", "slru"} {
		if _, err := NewRadixCache(RadixConfig{PageSize: 1, EvictionPolicy: p}, &stubAllocator{page: 1}, pool); err != nil {
			t.Errorf("policy %q refused: %v", p, err)
		}
	}
	// A disabled cache neither builds nor checks its eviction policy (unified_radix_cache.py:242-253).
	if _, err := NewRadixCache(RadixConfig{PageSize: 1, Disable: true, EvictionPolicy: "tlru"}, &stubAllocator{page: 1}, pool); err != nil {
		t.Errorf("a disabled cache refused its policy: %v", err)
	}
}
