package llmd

import (
	"errors"
	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/engine"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// unindexable is an engine that says a router cannot index its KV events (blis-vllm's v0.32.0
// engine for DeepSeek V4's packed full-attention group, Engine.KVEventsRefusal).
type unindexable struct{ engine.Engine }

func (unindexable) EnableKVEvents()        {}
func (unindexable) KVEventsRefusal() error { return errors.New("packed group at block size 256") }

// TestPrecisePrefix_RefusesAnEngineItCannotIndex: a router configuration that indexes KV events
// is refused, naming the reason, over an engine whose events it cannot key; a configuration that
// indexes none runs.
func TestPrecisePrefix_RefusesAnEngineItCannotIndex(t *testing.T) {
	for _, precise := range []bool{true, false} {
		dep := testDeployment(1)
		inner := compatEngineFactory(0)
		dep.NewEngine = func(i int, ic sim.InstanceConfig) (*sim.EngineDriver, error) {
			d, err := inner(i, ic)
			if err != nil {
				return nil, err
			}
			return sim.NewEngineDriver(unindexable{d.Engine()}, sim.EngineDriverConfig{Horizon: ic.Horizon}), nil
		}
		cs := newCluster(dep, cluster.NewSliceRequestSource(testRequests(2, 7, 200, 0)), nil)
		flags := DefaultEPPFlags()
		if precise {
			flags = append(flags, "--config-text="+preciseConfig(""))
		}
		cs.SetGateway(New(Options{Model: "test-model", EPPFlags: flags}))
		err := cs.Run()
		if precise && (err == nil || !strings.Contains(err.Error(), "packed group at block size 256")) {
			t.Fatalf("precise prefix over an unindexable engine: Run() = %v, want the engine's reason", err)
		}
		if !precise && err != nil {
			t.Fatalf("a router indexing no KV events: Run() = %v", err)
		}
	}
}
