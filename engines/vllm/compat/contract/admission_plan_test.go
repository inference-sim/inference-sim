// Original to blis-vllm.

package contract

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
)

func engineWithFrontend(t *testing.T, f cengine.FrontendConfig) *Engine {
	t.Helper()
	cfg := cengine.SimConfig{
		Horizon:             1 << 40,
		KVCacheConfig:       cengine.NewKVCacheConfig(64, 16, 0, 0, 0, 0),
		BatchConfig:         cengine.NewBatchConfig(16, 2048, 0),
		LatencyModel:        fixedLatency{},
		ModelHardwareConfig: cengine.ModelHardwareConfig{GPU: "h100", TP: 1, DP: 1},
		Frontend:            f,
	}
	e, err := New("instance_0", cfg)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// The plan states the queue a request waits in before the engine admits it, from the engine's own
// settings: renderer_num_workers threads on each of api_server_count servers. The simulator
// queues requests on that shape and knows neither setting.
func TestThePlanStatesTheAdmissionQueueFromTheFrontendSettings(t *testing.T) {
	for _, c := range []struct {
		name string
		f    cengine.FrontendConfig
		want plan.AdmissionQueue
	}{
		{"vLLM default: one thread, one server", cengine.FrontendConfig{RendererNumWorkers: 1}, plan.AdmissionQueue{WorkersPerServer: 1}},
		{"two threads", cengine.FrontendConfig{RendererNumWorkers: 2}, plan.AdmissionQueue{WorkersPerServer: 2}},
		{"three servers of two", cengine.FrontendConfig{RendererNumWorkers: 2, APIServerCount: 3}, plan.AdmissionQueue{Servers: 3, WorkersPerServer: 2}},
		{"no render stage", cengine.FrontendConfig{}, plan.AdmissionQueue{}},
	} {
		p := engineWithFrontend(t, c.f).Plan()
		if err := p.Validate(); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if p.Host.Admission != c.want {
			t.Errorf("%s: admission queue %+v, want %+v", c.name, p.Host.Admission, c.want)
		}
	}
}
