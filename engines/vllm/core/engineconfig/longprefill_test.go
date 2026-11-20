// Moved from inference-sim/blis-schemas/spec/deployment/longprefill_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package engineconfig

import (
	"math/rand"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestLongPrefillTokenThreshold(t *testing.T) {
	for _, c := range []struct {
		n  int
		ok bool
	}{{0, true}, {8192, true}, {-1, false}} {
		p := engineOnly(Engine{LongPrefillTokenThreshold: c.n}).Validate()
		if p.OK() != c.ok {
			t.Errorf("long_prefill_token_threshold %d: ok=%v, want %v\n%s", c.n, p.OK(), c.ok, p.Error())
		}
	}
	// Load, marshal, load: a stated threshold comes back as itself, and the default
	// (zero, off) is omitted rather than written.
	r := rand.New(rand.NewSource(5))
	for i := 0; i < 500; i++ {
		e := Engine{LongPrefillTokenThreshold: r.Intn(1 << 20)}
		out, err := yaml.Marshal(e)
		if err != nil {
			t.Fatal(err)
		}
		var back Engine
		if err := yaml.Unmarshal(out, &back); err != nil || back.LongPrefillTokenThreshold != e.LongPrefillTokenThreshold {
			t.Fatalf("%d round-tripped through %q to %d (%v)", e.LongPrefillTokenThreshold, out, back.LongPrefillTokenThreshold, err)
		}
		if e.LongPrefillTokenThreshold == 0 && containsKey(string(out), "long_prefill_token_threshold") {
			t.Fatalf("the default was written: %s", out)
		}
	}
}
