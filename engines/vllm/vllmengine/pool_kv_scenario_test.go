package vllmengine

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// A prefill pool runs NIXL beside a CPU tier (the deployment's MultiConnector) while the decode pool
// runs NIXL alone: each pool is a server with its own --kv-transfer-config (vllm/engine/arg_utils.py:789,
// 1881 at v0.33.0). The decode pool gets no CPU tier, and its connector is the pool's own.
func TestAPrefillPoolOffloadsWhileTheDecodePoolRunsNixlAlone(t *testing.T) {
	Register()
	r := roots(t, "muse-h100-roce-1p1d-prefill-offload.yaml")

	kt, off := prepareKV(t, r, 0)
	if kt.Connector != "MultiConnector" || len(kt.Extra.Connectors) != 2 ||
		kt.Extra.Connectors[0].Connector != "NixlConnector" || kt.Extra.Connectors[0].Role != "kv_producer" ||
		kt.Extra.Connectors[1].Connector != "OffloadingConnector" || kt.Extra.Connectors[1].Extra.CPUBytes != 512<<30 {
		t.Errorf("prefill pool kv_transfer %+v, want MultiConnector over NixlConnector (producer) then the 512 GiB tier", kt)
	}
	if off.BytesPerBlock == 0 {
		t.Errorf("prefill pool kv_offload %+v, want what vLLM derives from the model", off)
	}

	kt, off = prepareKV(t, r, 1)
	if kt.Connector != "NixlConnector" || kt.Role != "kv_consumer" || len(kt.Extra.Connectors) != 0 {
		t.Errorf("decode pool kv_transfer %+v, want NixlConnector alone as kv_consumer", kt)
	}
	if off != (kvOffload{}) {
		t.Errorf("decode pool kv_offload %+v, want none: it has no CPU tier", off)
	}
	for i := 0; i < 2; i++ {
		if _, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0", Roots: r, Pool: i}); err != nil {
			t.Fatalf("pool %d does not build: %v", i, err)
		}
	}
}

// The same deployment with the decode pool inheriting the deployment's MultiConnector is still refused,
// and the refusal names the pool and the way out: a MultiConnector over NIXL alone is not modelled.
func TestADecodePoolInheritingMultiConnectorWithoutATierIsRefusedByName(t *testing.T) {
	Register()
	_, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0",
		Roots: roots(t, "muse-h100-roce-1p1d-prefill-offload-inherited.yaml"), Pool: 1})
	if err == nil {
		t.Fatal("a decode pool running a MultiConnector over NIXL alone was built")
	}
	for _, want := range []string{"decode pool (pools[1])", "MultiConnector", "NixlConnector"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal %q does not name %q", err, want)
		}
	}
	// The prefill pool of the same deployment is fine.
	if _, err := engine.New(engine.Spec{Kind: "vllm", UpstreamVersion: "0.33.0",
		Roots: roots(t, "muse-h100-roce-1p1d-prefill-offload-inherited.yaml"), Pool: 0}); err != nil {
		t.Errorf("prefill pool: %v", err)
	}
}
