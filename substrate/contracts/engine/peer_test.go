package engine

import (
	"strings"
	"testing"
)

func TestAPeerMessageNoSimulatorCanDeliverIsRefused(t *testing.T) {
	ok := PeerMessage{From: "a:1", To: "b:2", Seq: 1, Bytes: 64}
	if err := ok.Validate(); err != nil {
		t.Fatalf("a complete message is refused: %v", err)
	}
	for name, c := range map[string]struct {
		m    PeerMessage
		want string
	}{
		"no sender":   {PeerMessage{To: "b:2", Seq: 1, Bytes: 64}, "no sender"},
		"no receiver": {PeerMessage{From: "a:1", Seq: 1, Bytes: 64}, "no receiver"},
		"to itself":   {PeerMessage{From: "a:1", To: "a:1", Seq: 1, Bytes: 64}, "its sender"},
		"no sequence": {PeerMessage{From: "a:1", To: "b:2", Bytes: 64}, "sequence"},
		"no size":     {PeerMessage{From: "a:1", To: "b:2", Seq: 1}, "at least one"},
	} {
		t.Run(name, func(t *testing.T) {
			err := c.m.Validate()
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("Validate(%+v) = %v, want an error naming %q", c.m, err, c.want)
			}
		})
	}
}
