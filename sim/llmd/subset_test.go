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
func TestSupportedTypesAreHostedByTheFork(t *testing.T) {
	types := make([]string, 0, len(supported))
	for typ := range supported {
		types = append(types, typ)
	}
	sort.Strings(types)
	for _, typ := range types {
		t.Run(typ, func(t *testing.T) {
			if err := hostingRefusal(typ); err != nil {
				t.Errorf("BLIS allows %s, but the fork refuses to host it: %v", typ, err)
			}
		})
	}
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
