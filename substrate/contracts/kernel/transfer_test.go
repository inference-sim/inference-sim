package kernel

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"
	"time"
)

func TestAloneTimeIsSlowestLinkOrFloor(t *testing.T) {
	nic := LinkUse{Link: "node0/nic/egress", BytesPerSecond: 50e9}
	nvl := LinkUse{Link: "node0/nvlink", BytesPerSecond: 450e9}
	cases := []struct {
		name string
		d    TransferDemand
		want time.Duration
	}{
		{"no transfer", TransferDemand{}, 0},
		{"no link costs the floor", TransferDemand{Floor: 7 * time.Microsecond}, 7 * time.Microsecond},
		{"bandwidth bound", TransferDemand{Bytes: 50e9, Floor: time.Millisecond, Links: []LinkUse{nic}}, time.Second},
		{"floor bound", TransferDemand{Bytes: 50, Floor: time.Millisecond, Links: []LinkUse{nic}}, time.Millisecond},
		// The payload crosses every link in series, so the slowest binds whatever the order.
		{"slowest link binds", TransferDemand{Bytes: 50e9, Links: []LinkUse{nvl, nic}}, time.Second},
		{"zero bytes on a link costs the floor", TransferDemand{Floor: 3, Links: []LinkUse{nic}}, 3},
		{"saturates", TransferDemand{Bytes: math.MaxInt64, Links: []LinkUse{{Link: "slow", BytesPerSecond: 1e-3}}}, time.Duration(math.MaxInt64)},
	}
	for _, c := range cases {
		if got := c.d.AloneTime(); got != c.want {
			t.Errorf("%s: AloneTime() = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestCheckRejectsMalformedDemands(t *testing.T) {
	ok := LinkUse{Link: "a", BytesPerSecond: 1}
	cases := map[string]TransferDemand{
		"negative":       {Bytes: -1},
		"negative floor": {Floor: -1},
		"unnamed link":   {Links: []LinkUse{{BytesPerSecond: 1}}},
		"named twice":    {Links: []LinkUse{ok, ok}},
		"zero rate":      {Links: []LinkUse{{Link: "a"}}},
		"NaN rate":       {Links: []LinkUse{{Link: "a", BytesPerSecond: math.NaN()}}},
		"infinite rate":  {Links: []LinkUse{{Link: "a", BytesPerSecond: math.Inf(1)}}},
	}
	for name, d := range cases {
		if d.Check() == nil {
			t.Errorf("%s: Check() accepted %+v", name, d)
		}
	}
	if err := (TransferDemand{Bytes: 1, Links: []LinkUse{ok}}).Check(); err != nil {
		t.Errorf("a well-formed demand was rejected: %v", err)
	}
	if err := (TransferDemand{Links: []LinkUse{ok, ok}}).Check(); err == nil || !strings.Contains(err.Error(), "twice") {
		t.Errorf("duplicate link: got %v", err)
	}
	if err := (TransferDemand{Bytes: 1 << 20, Floor: time.Millisecond}).Check(); err == nil || !strings.Contains(err.Error(), "cross no link") {
		t.Errorf("a payload on no link: got %v", err)
	}
	if err := (TransferDemand{Floor: time.Millisecond}).Check(); err != nil {
		t.Errorf("a demand with no payload needs no link: %v", err)
	}
}

func randomDemand(r *rand.Rand) TransferDemand {
	d := TransferDemand{Bytes: r.Int63n(1 << 40), Floor: time.Duration(r.Int63n(int64(time.Millisecond)))}
	for i, n := 0, r.Intn(4); i < n; i++ {
		d.Links = append(d.Links, LinkUse{Link: LinkID(fmt.Sprint("l", i)), BytesPerSecond: 1e6 + r.Float64()*1e12})
	}
	return d
}

// AloneTime's properties as a price, over random demands: it is never below the floor;
// more bytes never cost less; a faster slowest link never costs more; the order links are
// listed in does not matter; and adding a link at least as fast as every other changes
// nothing, while adding a slower one never helps.
func TestAloneTimeProperties(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for i := 0; i < 5000; i++ {
		d := randomDemand(r)
		got := d.AloneTime()
		if got < d.Floor {
			t.Fatalf("%+v costs %v, below its floor", d, got)
		}
		more := d
		more.Bytes += r.Int63n(1 << 30)
		if more.AloneTime() < got {
			t.Fatalf("more bytes cost less: %+v", d)
		}
		if len(d.Links) == 0 {
			if got != d.Floor {
				t.Fatalf("no link should cost the floor alone: %+v", d)
			}
			continue
		}
		faster := d
		faster.Links = append([]LinkUse(nil), d.Links...)
		for j := range faster.Links {
			faster.Links[j].BytesPerSecond *= 2
		}
		if faster.AloneTime() > got {
			t.Fatalf("faster links cost more: %+v", d)
		}
		shuffled := d
		shuffled.Links = append([]LinkUse(nil), d.Links...)
		r.Shuffle(len(shuffled.Links), func(a, b int) {
			shuffled.Links[a], shuffled.Links[b] = shuffled.Links[b], shuffled.Links[a]
		})
		if shuffled.AloneTime() != got {
			t.Fatalf("link order changed the price: %+v", d)
		}
		fastest := 0.0
		for _, l := range d.Links {
			fastest = math.Max(fastest, l.BytesPerSecond)
		}
		withFast := d
		withFast.Links = append(append([]LinkUse(nil), d.Links...), LinkUse{Link: "fast", BytesPerSecond: fastest * 2})
		if withFast.AloneTime() != got {
			t.Fatalf("a link faster than every other changed the price: %+v", d)
		}
		withSlow := d
		withSlow.Links = append(append([]LinkUse(nil), d.Links...), LinkUse{Link: "slow", BytesPerSecond: 1})
		if withSlow.AloneTime() < got {
			t.Fatalf("a slower link lowered the price: %+v", d)
		}
	}
}

// A transfer that pulls over several rails at once gets its share of a link, and no more.
func TestStreamsDivideALinkAmongTheTransfersOwnFlows(t *testing.T) {
	uplink := LinkUse{Link: "uplink", BytesPerSecond: 400e9, Streams: 8}
	nic := LinkUse{Link: "nic", BytesPerSecond: 50e9}
	d := TransferDemand{Bytes: 50e9, Links: []LinkUse{nic, uplink}}
	if got := d.AloneTime(); got != time.Second {
		t.Errorf("8 flows over a 400 GB/s uplink and 50 GB/s rails: %v, want 1s (the rail binds)", got)
	}
	uplink.Streams = 16
	d.Links[1] = uplink
	if got := d.AloneTime(); got != 2*time.Second {
		t.Errorf("16 flows over the same uplink: %v, want 2s (25 GB/s each)", got)
	}
	uplink.Streams = 0
	d.Links[1] = uplink
	if got := d.AloneTime(); got != time.Second {
		t.Errorf("no stated streams is one: %v, want 1s", got)
	}
	uplink.Streams = -1
	d.Links[1] = uplink
	if err := d.Check(); err == nil || !strings.Contains(err.Error(), "streams") {
		t.Errorf("a negative stream count: %v", err)
	}
}

// Moves that run together share the links they name: two nodes behind one uplink load it twice,
// and a node's own NIC carries only its own flow. The move ends with its slowest entry.
func TestConcurrentAddsSharedLinksAndKeepsPrivateOnes(t *testing.T) {
	const nic, uplink = 50e9, 100e9
	node := func(n string) TransferDemand {
		return TransferDemand{Bytes: 1 << 30, Floor: time.Millisecond, Links: []LinkUse{
			{Link: LinkID("nic/" + n), BytesPerSecond: nic},
			{Link: "uplink", BytesPerSecond: uplink},
		}}
	}
	one := node("a")
	both := Concurrent(node("a"), node("b"))
	if err := both.Check(); err != nil {
		t.Fatal(err)
	}
	if both.Bytes != one.Bytes || both.Floor != one.Floor {
		t.Errorf("payload %d and floor %v; want the largest entry's (%d, %v)", both.Bytes, both.Floor, one.Bytes, one.Floor)
	}
	streams := map[LinkID]int{}
	for _, l := range both.Links {
		streams[l.Link] = max(l.Streams, 1)
	}
	if streams["uplink"] != 2 || streams["nic/a"] != 1 || streams["nic/b"] != 1 || len(both.Links) != 3 {
		t.Errorf("links %+v; want the uplink twice and each NIC once", both.Links)
	}
	// Alone each entry is bound by its NIC (50 GB/s); together the uplink gives each 50 GB/s,
	// which is no slower, and a third entry makes the uplink bind.
	if both.AloneTime() != one.AloneTime() {
		t.Errorf("two entries take %v, one takes %v; the uplink's half is still a NIC's rate", both.AloneTime(), one.AloneTime())
	}
	three := Concurrent(node("a"), node("b"), node("c"))
	if three.AloneTime() <= one.AloneTime() {
		t.Errorf("three entries take %v, one takes %v; three through one uplink must be slower", three.AloneTime(), one.AloneTime())
	}
	if got := Concurrent(); got.Bytes != 0 || len(got.Links) != 0 {
		t.Errorf("Concurrent of nothing = %+v", got)
	}
	if got := Concurrent(TransferDemand{}, one); got.Bytes != one.Bytes || len(got.Links) != 2 {
		t.Errorf("an empty entry changed the demand: %+v", got)
	}
}
