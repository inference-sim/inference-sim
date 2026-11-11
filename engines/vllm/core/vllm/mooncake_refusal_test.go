// Original to blis-vllm.

package vllm

import (
	"strings"
	"testing"
)

// The inertness proofs of MC_ENABLE_DEST_DEVICE_AFFINITY, MC_SLICE_SIZE and MC_WORKERS_PER_CTX
// (oracle/inertness/MC_*.json) rest on one fact besides the scan: vLLM loads the Mooncake
// transfer-engine library only through a Mooncake connector, and a deployment this engine accepts
// cannot select one. There are two ways to select it in v0.32.0, and each is refused here.
//
//   - By connector name: MooncakeConnector and MooncakeStoreConnector are registered in
//     vllm/distributed/kv_transfer/kv_connector/factory.py:225-234@5df63a72. The engine models
//     NixlConnector and NixlPullConnector and refuses every other name (buildConnector).
//   - Through the NIXL connector: its backends come from kv_connector_extra_config
//     (vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:524-526@5df63a72), which can
//     name MOONCAKE. The engine block's kv_transfer has three keys (connectorConfig: connector,
//     kv_role, kv_load_failure_policy) and the block is decoded with KnownFields(true), so the
//     extra-config key cannot be stated.
//
// If either test fails because the engine now takes such a deployment, the three proofs no longer
// hold: the variables become reads of a library the engine loads, and the fields are classified again.
// inertness_test.go checks that each MC_ proof names these tests and that they still exist.

func TestAMooncakeConnectorNameIsRefused(t *testing.T) {
	for _, name := range []string{"MooncakeConnector", "MooncakeStoreConnector"} {
		_, err := build(t, "kv_transfer: {connector: "+name+", kv_role: kv_producer}\n")()
		if err == nil {
			t.Errorf("%s: built, want refused", name)
			continue
		}
		if !strings.Contains(err.Error(), "not modelled") || !strings.Contains(err.Error(), name) {
			t.Errorf("%s: refused for another reason: %v", name, err)
		}
	}
}

func TestTheConnectorBlockCannotStateWhichBackendNixlLoads(t *testing.T) {
	for _, role := range []string{"kv_producer", "kv_consumer"} {
		_, err := build(t, "kv_transfer: {connector: NixlConnector, kv_role: "+role+
			", kv_connector_extra_config: {backends: [MOONCAKE]}}\n")()
		if err == nil {
			t.Errorf("%s: kv_connector_extra_config was accepted, so NIXL's backends could name MOONCAKE", role)
			continue
		}
		if !strings.Contains(err.Error(), "kv_connector_extra_config") {
			t.Errorf("%s: refused for another reason than the unknown key: %v", role, err)
		}
	}
	// The same key beside the engine block's own keys is refused as well (an unknown key).
	if _, err := build(t, producer+"kv_connector_extra_config: {backends: [MOONCAKE]}\n")(); err == nil {
		t.Error("kv_connector_extra_config was accepted at the top of the engine block")
	}
}
