package llmd

import (
	"context"
	"errors"
	"sort"
	"testing"

	"github.com/inference-sim/blis-schemas/embed"
)

// TestSupportedTypesAreHostedByTheFork: every type BLIS's signal policy allows is one the fork can
// host. Each is built alone, with every type allowed; the fork refusing it for a hosting reason
// (an *embed.RefusalError naming it) means BLIS allows a type the fork has not audited. Other
// errors (a plugin needing parameters, peers or pods) are not hosting refusals.
//
// It also holds the constructor to drawing no randomness (embed.Constructor), which BLIS relies on
// to seed the streams with the pool's identity after New returns: the runtime here has no scope
// yet, so a draw while any supported type is built panics (desRuntime.Rand), failing the test.
func TestSupportedTypesAreHostedByTheFork(t *testing.T) {
	types := make([]string, 0, len(supported))
	for typ := range supported {
		types = append(types, typ)
	}
	sort.Strings(types)
	for _, typ := range types {
		drew, err := build(typ)
		if drew != nil {
			// The constructor may hold a process-wide lock it never released: stop here.
			t.Fatalf("building %s drew randomness or panicked during construction: %v", typ, drew)
		}
		if err != nil {
			t.Errorf("BLIS allows %s, but the fork refuses to host it: %v", typ, err)
		}
	}
}

// build is hostingRefusal(typ), with any panic during construction returned as drew.
func build(typ string) (drew any, err error) {
	defer func() { drew = recover() }()
	return nil, hostingRefusal(typ)
}

// hostingRefusal builds a router whose configuration names one plugin of type typ, with every type
// allowed, and returns the fork's refusal of typ, if any.
func hostingRefusal(typ string) error {
	config := "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nplugins:\n- type: " + typ + "\n"
	rt := newDESRuntime(idleHost{}, 1, 0)
	r, err := newRouter(context.Background(), rt, embed.Config{
		Flags:  append(DefaultEPPFlags(), "--allow-experimental-plugins", "--config-text="+config),
		Allow:  func(string) (bool, string) { return true, "" },
		Logger: nil,
	})
	if err == nil {
		r.Stop()
		rt.co.drain()
		return nil
	}
	var refusal *embed.RefusalError
	if errors.As(err, &refusal) && refusal.Type == typ {
		return err
	}
	return nil
}

// idleHost is a Host at time zero that never runs what is scheduled: the router is only built.
type idleHost struct{}

func (idleHost) Now() int64                               { return 0 }
func (idleHost) Schedule(int64, int, bool, func()) func() { return func() {} }
