// Original to blis-vllm.

package vllm

import (
	"fmt"
	"sort"

	"github.com/inference-sim/blis-schemas/engine"

	core "github.com/inference-sim/blis-vllm/core/engine"
	"github.com/inference-sim/blis-vllm/core/connector/nixl"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// P/D through engine.Engine, with vLLM's NIXL connector (core/connector/nixl) on the scheduler.
//
// A prefill engine (kv_producer) runs a request the simulator marks RemoteDecode with
// do_remote_decode: it computes the prompt and samples one token, and when it finishes the
// connector keeps its blocks (request_finished). The engine reports them in Result.Handoffs, and
// the simulator ends the hold with TransferDone once the decode side has read them, which is the
// worker's finished_sending.
//
// A decode engine (kv_consumer) runs a request the simulator marks RemotePrefill with
// do_remote_prefill: the scheduler admits it into WAITING_FOR_REMOTE_KVS with its blocks
// allocated and the prompt's tokens external (get_num_new_matched_tokens). The engine reports
// that admission as a TransferLoad among the batch's TransfersStarted, and the simulator reports
// the pull's end with TransferDone, which is the worker's finished_recving (or, failed,
// failed_recving). The outcome reaches the scheduler with the next applied output, as vLLM's
// worker reports it in the next step's kv_connector_output (vllm/v1/outputs.py:215-233@5df63a72).

// kvRole is a pool's kv_transfer_config.kv_role.
type kvRole string

const (
	roleProducer kvRole = "kv_producer"
	roleConsumer kvRole = "kv_consumer"
	roleBoth     kvRole = "kv_both"
)

// connectorConfig is the engine block's kv_transfer: the connector a P/D pool runs.
type connectorConfig struct {
	// Connector is kv_connector, the connector's registered name.
	Connector string `yaml:"connector"`
	// Role is kv_role.
	Role kvRole `yaml:"kv_role"`
	// LoadFailurePolicy is kv_load_failure_policy, "fail" unstated
	// (vllm/config/kv_transfer.py:88@5df63a72).
	LoadFailurePolicy string `yaml:"kv_load_failure_policy,omitempty"`
}

// nixlNames are the names vLLM's connector factory resolves to the NIXL pull scheduler
// (vllm/distributed/kv_transfer/kv_connector/factory.py:177-189@5df63a72), as registry/connectors.yaml
// lists them.
var nixlNames = map[string]bool{"NixlConnector": true, "NixlPullConnector": true}

// pd is an engine's P/D side: its connector, the transfers in flight and the outcome the next
// applied output reports.
type pd struct {
	role kvRole
	nixl *nixl.Scheduler
	// matched is each request's external tokens, as the connector last reported them to the
	// scheduler (get_num_new_matched_tokens): what its load moves.
	matched map[string]int
	next    engine.TransferID
	// pulls are the decode side's loads in flight, and holds the prefill side's finished
	// requests whose blocks wait to be read, by transfer.
	pulls map[engine.TransferID]string
	holds map[engine.TransferID]string
	// pulling are the requests whose load has started, so a schedule reports each once.
	pulling map[string]bool
	// remote are the decode side's remote-prefilled requests, and decode the prefill side's
	// remote-decode ones.
	remote, decode map[string]bool
	// outcome is what the worker reports with the next applied output.
	outcome sched.TransferOutcome
}

// buildConnector installs the engine block's connector on the scheduler, refusing one it does
// not model and what vLLM's NIXL worker refuses at start-up for the parallel layout.
func buildConnector(s *sched.Scheduler, c *connectorConfig, pp int, spec sched.Spec, hybrid bool) (*pd, error) {
	if c == nil {
		return nil, nil
	}
	if !nixlNames[c.Connector] {
		return nil, fmt.Errorf("vllm: kv_transfer.connector %q is not modelled through the engine (modelled: NixlConnector, "+
			"NixlPullConnector; registry/connectors.yaml)", c.Connector)
	}
	switch c.Role {
	case roleProducer, roleConsumer:
	case roleBoth:
		return nil, fmt.Errorf("vllm: kv_transfer.kv_role kv_both is not modelled: a P/D pool is a prefiller or a decoder")
	default:
		return nil, fmt.Errorf("vllm: kv_transfer.kv_role %q: want kv_producer or kv_consumer", c.Role)
	}
	recompute := false
	switch c.LoadFailurePolicy {
	case "", "fail":
	case "recompute":
		recompute = true
	default:
		return nil, fmt.Errorf("vllm: kv_transfer.kv_load_failure_policy %q: want fail or recompute", c.LoadFailurePolicy)
	}
	if err := nixl.CheckWorker(pp, hybrid, c.Role == roleConsumer); err != nil {
		return nil, fmt.Errorf("vllm: %w", err)
	}
	cfg := nixl.Default()
	cfg.PrefillBackoff = nixl.Backoff(spec)
	p := &pd{role: c.Role, nixl: nixl.New(cfg), matched: map[string]int{}, pulls: map[engine.TransferID]string{},
		holds: map[engine.TransferID]string{}, pulling: map[string]bool{}, remote: map[string]bool{}, decode: map[string]bool{}}
	if err := s.SetConnector(metered{p.nixl, p.matched}, sched.ConnectorConfig{RecomputeLoadFailures: recompute}); err != nil {
		return nil, fmt.Errorf("vllm: %w", err)
	}
	return p, nil
}

// metered is the NIXL scheduler with the tokens it reports matched recorded, so a load is sized
// by what the scheduler allocated for it. It keeps every capability the scheduler states.
type metered struct {
	*nixl.Scheduler
	matched map[string]int
}

func (m metered) MatchedTokens(r *sched.Request, localComputed int) (int, bool) {
	n, async := m.Scheduler.MatchedTokens(r, localComputed)
	if async {
		m.matched[r.ID] = n
	}
	return n, async
}

// transferParams are the kv_transfer_params a P/D proxy hands each side: do_remote_decode to the
// prefiller, and do_remote_prefill with the prefiller's blocks and address to the decoder.
func (p *pd) transferParams(r engine.Request, blockSize int) (*sched.TransferParams, error) {
	switch {
	case r.RemoteDecode && r.RemotePrefill:
		return nil, fmt.Errorf("vllm: request %s is both a remote prefill and a remote decode", r.ID)
	case r.RemoteDecode:
		if p == nil || p.role != roleProducer {
			return nil, fmt.Errorf("vllm: request %s is a disaggregated prefill, and this engine is not a kv_producer", r.ID)
		}
		p.decode[r.ID] = true
		return &sched.TransferParams{DoRemoteDecode: true}, nil
	case r.RemotePrefill:
		if p == nil || p.role != roleConsumer {
			return nil, fmt.Errorf("vllm: request %s is a disaggregated decode, and this engine is not a kv_consumer", r.ID)
		}
		p.remote[r.ID] = true
		return &sched.TransferParams{DoRemotePrefill: true, HasRemote: true,
			RemoteBlocks: (len(r.Prompt) + blockSize - 1) / blockSize}, nil
	}
	return nil, nil
}

// started reports the loads a schedule began: the remote-prefilled requests that entered
// WAITING_FOR_REMOTE_KVS, in queue order, each once.
func (p *pd) started(s *sched.Scheduler) []engine.Transfer {
	if p == nil || p.role != roleConsumer {
		return nil
	}
	var out []engine.Transfer
	for _, id := range s.WaitingForRemoteKV() {
		if p.pulling[id] || !p.remote[id] {
			continue
		}
		p.pulling[id] = true
		p.next++
		p.pulls[p.next] = id
		out = append(out, engine.Transfer{ID: p.next, Request: id, Kind: engine.TransferLoad, Tokens: p.matched[id],
			Blocks: len(s.KV.Blocks(id))})
	}
	p.nixl.BuildMeta()
	return out
}

// handoffs reports the prefill side's finished requests whose blocks the connector holds, each a
// transfer the simulator ends once the decode side has read them, in finish order.
func (p *pd) handoffs(s *sched.Scheduler, fins []sched.Finished) []engine.Handoff {
	if p == nil || p.role != roleProducer {
		return nil
	}
	var out []engine.Handoff
	for _, f := range fins {
		if !p.decode[f.ID] {
			continue
		}
		delete(p.decode, f.ID)
		if s.Request(f.ID) == nil {
			continue // freed at once: nothing computed to read, or not a normal finish
		}
		p.next++
		p.holds[p.next] = f.ID
		// The prompt's blocks: a full-attention group's, which hold every token, rather than a
		// sliding window's, whose freed slots are the null block (a hybrid model's group 0 may be
		// either).
		g := 0
		for i := 0; i < s.KV.NumGroups(); i++ {
			if s.KV.PrefixIndexable(uint32(i)) {
				g = i
				break
			}
		}
		var blocks []int64
		for _, b := range s.KV.GroupBlocks(g, f.ID) {
			if !b.Null {
				blocks = append(blocks, int64(b.ID))
			}
		}
		out = append(out, engine.Handoff{ID: f.ID, Transfer: p.next, Blocks: blocks})
	}
	return out
}

// done records a transfer's end for the next applied output: a hold read (finished_sending) or a
// load landed (finished_recving). A failed load returns its request's ID: the simulator gives up
// on the pull and the request leaves this engine (engine.Engine's TransferDone, failed), so the
// engine aborts it and frees what the load reserved, rather than reporting it as vLLM's
// failed_recving, whose request the simulator no longer tracks.
func (p *pd) done(id engine.TransferID, ok bool) (failed string, err error) {
	if p == nil {
		return "", engine.ErrUnknownTransfer
	}
	if rid, held := p.holds[id]; held {
		delete(p.holds, id)
		p.outcome.FinishedSending = append(p.outcome.FinishedSending, rid)
		return "", nil
	}
	rid, pulling := p.pulls[id]
	if !pulling {
		return "", engine.ErrUnknownTransfer
	}
	delete(p.pulls, id)
	delete(p.pulling, rid)
	delete(p.matched, rid)
	delete(p.remote, rid)
	// Either way the worker's load is over: finished_recving lets the scheduler free what an
	// aborted request's load held (finish_requests delays the free until it, scheduler.py:2690-
	// 2700@5df63a72).
	p.outcome.FinishedRecving = append(p.outcome.FinishedRecving, rid)
	if !ok {
		return rid, nil
	}
	return "", nil
}

// take is the outcome the worker reports with the applied output, once.
func (p *pd) take() sched.TransferOutcome {
	if p == nil {
		return sched.TransferOutcome{}
	}
	o := p.outcome
	p.outcome = sched.TransferOutcome{}
	sort.Strings(o.FinishedSending)
	return o
}

// pending reports whether a transfer outcome is waiting for an output to carry it.
func (p *pd) pending() bool {
	return p != nil && len(p.outcome.FinishedRecving)+len(p.outcome.FinishedSending) > 0
}

// aborted forgets a request Finish ended: a load in flight for it still ends with TransferDone,
// which the scheduler then ignores, as vLLM ignores a finished_recving for a freed request.
func (p *pd) aborted(id string) {
	if p == nil {
		return
	}
	delete(p.remote, id)
	delete(p.decode, id)
	if !p.pulling[id] {
		delete(p.matched, id)
	}
}

// Transfers implements core/engine.TransferGPU: the worker's kv_connector_output for the step
// being applied, which carries every transfer that ended since the last. A step that ran no
// model still reports it: with a connector, vLLM's runner returns the connector's output without
// a forward pass (kv_connector_no_forward, vllm/v1/worker/gpu_model_runner.py@5df63a72).
func (g *batchOutput) Transfers(core.Batch) sched.TransferOutcome { return g.pd.take() }

var _ core.TransferGPU = (*batchOutput)(nil)
