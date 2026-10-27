// Moved from inference-sim/blis-schemas/spec/deployment/prefixcaching_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package engineconfig

import (
	"testing"

	"gopkg.in/yaml.v3"
)

// Prefix caching is tri-state, and the three states must survive a round trip through
// YAML distinguishably.
//
// This is not a formality. vLLM's default is ON, so an omitted field describes a
// deployment WITH caching while an explicit false describes one without -- and the two
// run different amounts of prefill work for the same prompt, because a matched prefix
// arrives as already-computed tokens. A benchmark launched with
// --no-enable-prefix-caching therefore cannot be expressed by omission, which is the
// reason the field is a pointer rather than a bool.
func TestPrefixCachingTriStateRoundTrips(t *testing.T) {
	yes, no := true, false
	for _, c := range []struct {
		name string
		set  *bool
		// wantKey is whether the field appears in the emitted YAML at all.
		wantKey bool
	}{
		{"unstated takes the engine default", nil, false},
		{"explicitly disabled", &no, true},
		{"explicitly enabled", &yes, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			out, err := yaml.Marshal(Engine{EnablePrefixCaching: c.set})
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var back Engine
			if err := yaml.Unmarshal(out, &back); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			switch {
			case c.set == nil && back.EnablePrefixCaching != nil:
				t.Errorf("unstated became %v; a deployment that says nothing must stay "+
					"distinguishable from one that says false",
					*back.EnablePrefixCaching)
			case c.set != nil && back.EnablePrefixCaching == nil:
				t.Errorf("explicit %v was lost in the round trip", *c.set)
			case c.set != nil && *back.EnablePrefixCaching != *c.set:
				t.Errorf("round trip changed %v to %v", *c.set, *back.EnablePrefixCaching)
			}
			if got := containsKey(string(out), "enable_prefix_caching"); got != c.wantKey {
				t.Errorf("key present = %v, want %v; emitted:\n%s", got, c.wantKey, out)
			}
		})
	}
}

func containsKey(doc, key string) bool {
	for _, line := range splitLines(doc) {
		if len(line) > len(key) && line[:len(key)] == key && line[len(key)] == ':' {
			return true
		}
	}
	return false
}

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}
