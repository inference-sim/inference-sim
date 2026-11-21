package cluster

import (
	"encoding/json"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"

	"github.com/inference-sim/inference-sim/sim"
)

// tagged is a peered engine that keeps the kv_transfer_params of every request it is given.
type tagged struct {
	*talker
	params map[string]string
}

func (e *tagged) Add(r engine.Request, now engine.Time) error {
	e.params[r.ID] = r.Tags[sim.KVTransferParamsTag]
	return e.FIFO.Add(r, now)
}

// The talker's messages are not needed here: this engine only takes requests.
func (e *tagged) Schedule(now engine.Time) (engine.Batch, error) { return e.FIFO.Schedule(now) }

// A P/D handoff between two peered instances goes through vLLM's p2p tier (inference-sim/inference-sim#2439): the
// prefill sub-request is told to keep its KV for the remote decoder, and the decode sub-request, an
// ordinary request to its engine, to pull it from the prefill instance at its address, under one
// transfer id. The request completes. Before, both sub-requests went untagged and the decode was
// handed to the P/D connector's pull.
func TestAP2PHandoffTagsBothSubRequestsWithTheirKVTransferParams(t *testing.T) {
	var engines []*tagged
	cfg := newTestDisaggDeploymentConfig(2, 1, 1)
	cfg.PeerEndpoint = func(id InstanceID) plan.Endpoint { return testInstanceEndpoint(id) }
	cfg.NewEngine = func(index int, _ sim.InstanceConfig) (*sim.EngineDriver, error) {
		fifo, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 256, InFlight: 1, StepBaseUs: 10})
		if err != nil {
			return nil, err
		}
		addr := []string{"prefill:5710", "decode:5711"}[index]
		e := &tagged{talker: &talker{FIFO: fifo, addr: addr, arrivals: new([]arrival), placed: map[string]plan.Endpoint{}},
			params: map[string]string{}}
		engines = append(engines, e)
		return sim.NewEngineDriver(e, sim.EngineDriverConfig{Horizon: 1 << 40}), nil
	}
	reqs := newTestRequests(1)
	cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	if len(cs.parentRequests) != 1 {
		t.Fatalf("%d parent requests, want 1", len(cs.parentRequests))
	}
	var parent *ParentRequest
	for _, p := range cs.parentRequests {
		parent = p
	}
	if parent.CompletionTime == 0 {
		t.Fatal("the request did not complete")
	}
	var pre, dec map[string]map[string]any
	if err := json.Unmarshal([]byte(engines[0].params[parent.PrefillSubReqID]), &pre); err != nil {
		t.Fatalf("the prefill sub-request's kv_transfer_params %q: %v", engines[0].params[parent.PrefillSubReqID], err)
	}
	if err := json.Unmarshal([]byte(engines[1].params[parent.DecodeSubReqID]), &dec); err != nil {
		t.Fatalf("the decode sub-request %s: kv_transfer_params %q: %v", parent.DecodeSubReqID, engines[1].params[parent.DecodeSubReqID], err)
	}
	id := pre["remote_decoder"]["kv_request_id"]
	if id == nil || id == "" || len(pre) != 1 {
		t.Errorf("prefill kv_transfer_params %v, want only remote_decoder with a kv_request_id", pre)
	}
	rp := dec["remote_prefiller"]
	if len(dec) != 1 || rp["kv_request_id"] != id || rp["remote_host"] != "prefill" || rp["remote_port"] != float64(5710) {
		t.Errorf("decode kv_transfer_params %v, want remote_prefiller{kv_request_id: %v, remote_host: prefill, remote_port: 5710}", dec, id)
	}
}

// A handoff whose instances are not both peered is left to the P/D connector: no kv_transfer_params.
func TestAHandoffBetweenUnpeeredInstancesSetsNoKVTransferParams(t *testing.T) {
	cs := newTestCluster(newTestDisaggDeploymentConfig(2, 1, 1), NewSliceRequestSource(newTestRequests(1)), nil)
	mustRun(t, cs)
	for _, p := range cs.parentRequests {
		if _, ok := cs.p2pPrefillParams(p); ok {
			t.Errorf("parent %s between unpeered instances is given p2p params", p.ID)
		}
	}
}
