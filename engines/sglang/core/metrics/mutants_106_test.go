// Original to blis-sglang.

package metrics

import (
	"encoding/hex"
	"strings"
	"testing"

	"github.com/inference-sim/blis-sglang/core/cache"
)

// These tests pin the msgpack width boundaries of the KV-event encoder (M3.1, blis-sglang#106).
// The expected bytes come from the msgpack specification (smallest form: positive fixint to 127,
// uint8/16/32/64 above, negative fixint to -32, int8/16/32/64 below), which is what msgspec writes
// (disaggregation/kv_events.py:593-615@297e54fd); each case sits on the last value of one form or the
// first of the next.

// removedPrefix is the start of the batch of one BlockRemoved event with no medium, timestamp 0.0:
// array of 3, float64, array of 1, map of 2, "type": "BlockRemoved", "block_hashes".
const removedPrefix = "93cb0000000000000000" + "91" + "82a474797065ac426c6f636b52656d6f766564ac626c6f636b5f686173686573"

func TestEncodeIntegersAtEveryWidthBoundary(t *testing.T) {
	cases := []struct {
		v    int64
		want string
	}{
		{0, "00"}, {127, "7f"}, {128, "cc80"}, {255, "ccff"}, {256, "cd0100"},
		{65535, "cdffff"}, {65536, "ce00010000"}, {4294967295, "ceffffffff"},
		{4294967296, "cf0000000100000000"},
		{-1, "ff"}, {-32, "e0"}, {-33, "d0df"}, {-128, "d080"}, {-129, "d1ff7f"},
		{-32768, "d18000"}, {-32769, "d2ffff7fff"}, {-2147483648, "d280000000"},
		{-2147483649, "d3ffffffff7fffffff"},
	}
	for _, c := range cases {
		got := hex.EncodeToString(EncodeKVEventBatch(0, []cache.Event{{Type: "BlockRemoved", BlockHashes: []int64{c.v}}}, 1))
		if want := removedPrefix + "91" + c.want + "01"; got != want {
			t.Errorf("hash %d encoded\n got  %s\n want %s", c.v, got, want)
		}
	}
}

func TestEncodeArrayHeaderAtEveryWidthBoundary(t *testing.T) {
	cases := []struct {
		n      int
		header string
	}{{0, "90"}, {15, "9f"}, {16, "dc0010"}, {65535, "dcffff"}, {65536, "dd00010000"}}
	for _, c := range cases {
		got := hex.EncodeToString(EncodeKVEventBatch(0, []cache.Event{{Type: "BlockRemoved", BlockHashes: make([]int64, c.n)}}, 1))
		if want := removedPrefix + c.header + strings.Repeat("00", c.n) + "01"; got != want {
			t.Errorf("%d hashes: header and body differ (got %d hex chars, want %d); got head %.40s", c.n, len(got), len(want), got[len(removedPrefix):])
		}
	}
}

func TestEncodeStringHeaderAtEveryWidthBoundary(t *testing.T) {
	cases := []struct {
		n      int
		header string
	}{{0, "a0"}, {31, "bf"}, {32, "d920"}, {255, "d9ff"}, {256, "da0100"}, {65535, "daffff"}, {65536, "db00010000"}}
	for _, c := range cases {
		medium := strings.Repeat("a", c.n)
		got := hex.EncodeToString(EncodeKVEventBatch(0, []cache.Event{{Type: "BlockRemoved", Medium: medium}}, 1))
		// A set medium makes the event a map of 3; block_hashes is the empty array.
		want := "93cb0000000000000000" + "91" + "83a474797065ac426c6f636b52656d6f766564ac626c6f636b5f68617368657390" +
			"a6" + hex.EncodeToString([]byte("medium")) + c.header + strings.Repeat("61", c.n) + "01"
		if c.n == 0 {
			// An empty medium is the default and is omitted.
			want = removedPrefix + "90" + "01"
		}
		if got != want {
			t.Errorf("medium of %d bytes: header or body differ (got %d hex chars, want %d); got %.60s", c.n, len(got), len(want), got)
		}
	}
}

// A map of 16 fields takes the 16-bit header; no KV event has that many fields (a BlockStored has
// at most ten), so the header is checked on its own.
func TestMapHeaderAtTheFixmapBoundary(t *testing.T) {
	for _, c := range []struct {
		n    int
		want string
	}{{1, "81"}, {15, "8f"}, {16, "de0010"}} {
		if got := hex.EncodeToString(appendMapHeader(nil, c.n)); got != c.want {
			t.Errorf("map of %d: got %s, want %s", c.n, got, c.want)
		}
	}
}
