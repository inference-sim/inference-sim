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
