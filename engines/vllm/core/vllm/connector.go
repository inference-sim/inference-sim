// Original to blis-vllm.

package vllm

import (
	"fmt"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"

	"github.com/inference-sim/blis-vllm/core/connector/multi"
	"github.com/inference-sim/blis-vllm/core/connector/nixl"
	core "github.com/inference-sim/blis-vllm/core/engine"
	"github.com/inference-sim/blis-vllm/core/kv"
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

// producer and consumer are KVTransferConfig.is_kv_producer and is_kv_consumer
// (vllm/config/kv_transfer.py:129-135@5df63a72): kv_both is both. A kv_both engine is no fixed side
// of a transfer: which side a request is comes from its own kv_transfer_params, do_remote_decode
// for a prefill leg and do_remote_prefill for a decode leg, which the host sets for each request.
func (r kvRole) producer() bool { return r == roleProducer || r == roleBoth }
func (r kvRole) consumer() bool { return r == roleConsumer || r == roleBoth }

// connectorConfig is the engine block's kv_transfer: the connector a P/D pool runs.
type connectorConfig struct {
	// Connector is kv_connector, the connector's registered name.
	Connector string `yaml:"connector"`
	// Role is kv_role.
	Role kvRole `yaml:"kv_role"`
	// LoadFailurePolicy is kv_load_failure_policy, "fail" unstated
	// (vllm/config/kv_transfer.py:88@5df63a72).
	LoadFailurePolicy string `yaml:"kv_load_failure_policy,omitempty"`
	// ExtraConfig is kv_connector_extra_config. Only MultiConnector reads one here: its "connectors"
	// list of child KVTransferConfig objects, in the order vLLM tries them
	// (multi_connector.py:226-249@5df63a72). multiconnector.go models the pair llm-d's P/D with
	// offload runs, a NIXL child and the OffloadingConnector.
	ExtraConfig *multiExtraConfig `yaml:"kv_connector_extra_config,omitempty"`
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
	// next is the last transfer ID started; an engine with a KV offload shares it with the
	// offload side, so the IDs of both are one sequence (newEngine).
	next *engine.TransferID
	// notif are the transfers that only tell the prefiller its blocks are no longer wanted,
	// by transfer, with their request (see started).
	notif map[engine.TransferID]string
	// multi is the MultiConnector the NIXL connector is a child of, at index child; nil when it
	// runs alone.
	multi *multi.Connector
	child int
	// stride and lane make the IDs of a data-parallel group's ranks distinct: rank r of n states
	// its k-th transfer as (k-1)*n + r + 1, so the group finds the rank of a TransferDone from the
	// ID alone. An engine alone has stride 1 and lane 0, and its IDs are 1, 2, 3 ...
	stride, lane int
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
	// at is the node and rack this engine's workers sit on, the end of a pull a decode engine receives
	// at; peer is each remote-prefilled request's prefill, the end the pull reads from
	// (engine.Request.Peer).
	at   plan.Endpoint
	peer map[string]plan.Endpoint
}

// buildConnector installs the engine block's connector on the scheduler, refusing one it does
// not model and what vLLM's NIXL worker refuses at start-up for the parallel layout.
//
// install is how the connector reaches the scheduler: SetConnector for an engine that runs it
// alone, or the MultiConnector's list (multiconnector.go) beside an offload.
func buildConnector(s *sched.Scheduler, c *connectorConfig, pp int, spec sched.Spec, hybrid bool, at plan.Endpoint, install installer) (*pd, error) {
	if c == nil {
		return nil, nil
	}
	if !nixlNames[c.Connector] {
		return nil, fmt.Errorf("vllm: kv_transfer.connector %q is not modelled through the engine (modelled: NixlConnector, "+
			"NixlPullConnector; registry/connectors.yaml)", c.Connector)
	}
	switch c.Role {
	case roleProducer, roleConsumer, roleBoth:
	default:
		return nil, fmt.Errorf("vllm: kv_transfer.kv_role %q: want kv_producer, kv_consumer or kv_both", c.Role)
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
	cfg.PrefillBackoff = nixl.Backoff(spec, s.KV.PrefixReplayTokens() > 0)
	for g := 0; g < s.KV.NumGroups(); g++ {
		// blocks_per_sw reads the window of a SlidingWindowSpec (SlidingWindowMLASpec is one) and the
		// group's own block size (base_scheduler.py:143-153@5df63a72).
		grp := nixl.Group{BlockSize: s.KV.GroupBlockSize(g)}
		if spec := s.KV.GroupSpec(g); spec.Kind == kv.SlidingWindow || spec.Kind == kv.SlidingWindowMLA {
			grp.Window = spec.Window
		}
		cfg.Groups = append(cfg.Groups, grp)
	}
	p := &pd{role: c.Role, nixl: nixl.New(cfg), next: new(engine.TransferID), notif: map[engine.TransferID]string{},
		matched: map[string]int{}, pulls: map[engine.TransferID]string{},
		holds: map[engine.TransferID]string{}, pulling: map[string]bool{}, remote: map[string]bool{}, decode: map[string]bool{},
		at: at, peer: map[string]plan.Endpoint{}, stride: 1}
	// is_kv_consumer and is_kv_producer (vllm/config/kv_transfer.py:130-136@5df63a72) set
	// defer_block_free and requires_kv_delivery beside more than one batch in flight
	// (vllm/v1/core/sched/scheduler.py:184-194@5df63a72).
	if err := install(metered{p.nixl, p.matched}, sched.ConnectorConfig{RecomputeLoadFailures: recompute,
		Consumer: c.Role.consumer(), Producer: c.Role.producer()}); err != nil {
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
		if p == nil || !p.role.producer() {
			return nil, fmt.Errorf("vllm: request %s is a disaggregated prefill, and this engine is not a kv_producer", r.ID)
		}
		p.decode[r.ID] = true
		return &sched.TransferParams{DoRemoteDecode: true}, nil
	case r.RemotePrefill:
		if p == nil || !p.role.consumer() {
			return nil, fmt.Errorf("vllm: request %s is a disaggregated decode, and this engine is not a kv_consumer", r.ID)
		}
		if r.Peer == nil {
			return nil, fmt.Errorf("vllm: request %s is a disaggregated decode and states no peer to pull from", r.ID)
		}
		p.remote[r.ID] = true
		p.peer[r.ID] = *r.Peer
		return &sched.TransferParams{DoRemotePrefill: true, HasRemote: true,
			RemoteBlocks: (len(r.Prompt) + blockSize - 1) / blockSize}, nil
	}
	return nil, nil
}

// newID is the next transfer's ID (stride, lane).
func (p *pd) newID() engine.TransferID {
	*p.next++
	return (*p.next-1)*engine.TransferID(p.stride) + engine.TransferID(p.lane) + 1
}

// started reports the loads a schedule began: the remote-prefilled requests that entered
// WAITING_FOR_REMOTE_KVS, in queue order, each once.
//
// Beside an offload (MultiConnector), the child that loads a request is the first that reported
// tokens. When that is the offload, the request waits for the offload's load, and NIXL, given no
// external tokens, only tells the prefiller to release its blocks (pull_scheduler.py:157-168,
// base_worker's notification for a pull of nothing): a pull of no tokens, reported so the
// simulator can end the prefiller's handoff, which lands no KV and reports no finished_recving.
func (p *pd) started(s *sched.Scheduler) []engine.Transfer {
	if p == nil || !p.role.consumer() {
		return nil
	}
	var out []engine.Transfer
	for _, id := range s.WaitingForRemoteKV() {
		if p.pulling[id] || !p.remote[id] {
			continue
		}
		p.pulling[id] = true
		tid := p.newID()
		tokens, release := p.matched[id], false
		if p.multi != nil {
			if c, ok := p.multi.Chosen(id); !ok || c != p.child {
				tokens, release = 0, true
			}
		}
		if release {
			p.notif[tid] = id
		} else {
			p.pulls[tid] = id
		}
		// A pull from a peer: it reads the prefill's node and rack (engine.Request.Peer) and lands on this
		// engine's, and uses no tier.
		blocks := 0
		if !release {
			counts, _ := p.nixl.Pulled(id)
			for _, n := range counts {
				blocks += n
			}
		}
		out = append(out, engine.Transfer{ID: tid, Request: id, Kind: engine.TransferLoad, Tokens: tokens,
			Blocks: blocks, Move: plan.Transfer{Tokens: units.Tokens(tokens), From: p.peer[id], To: p.at,
				Kind: plan.TransferPeer}})
	}
	p.nixl.BuildMeta()
	return out
}

// handoffs reports the prefill side's finished requests whose blocks the connector holds, each a
// transfer the simulator ends once the decode side has read them, in finish order.
func (p *pd) handoffs(s *sched.Scheduler, fins []sched.Finished) []engine.Handoff {
	if p == nil || !p.role.producer() {
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
		tid := p.newID()
		p.holds[tid] = f.ID
		// The blocks held for the read are every transfer group's, clipped as the connector clips the
		// remote_block_ids it advertises (get_exchange_clipped_blocks, nixl/base_scheduler.py:262-300
		// @5df63a72): all of a full-attention or MLA group's, a sliding window's last blocks, a
		// ring's one. A sliding window's freed slots (the null block) hold nothing. Block ids are
		// the pool's, so a block a prefix cache shares appears in every handoff that holds it.
		ids := make([][]int, s.KV.NumGroups())
		for g := range ids {
			for _, b := range s.KV.GroupBlocks(g, f.ID) {
				id := b.ID
				if b.Null {
					id = -1
				}
				ids[g] = append(ids[g], id)
			}
		}
		var blocks []int64
		for _, grp := range p.nixl.Clip(ids) {
			for _, id := range grp {
				if id >= 0 {
					blocks = append(blocks, int64(id))
				}
			}
		}
		out = append(out, engine.Handoff{ID: f.ID, Transfer: tid, Blocks: blocks})
	}
	return out
}

// done records a transfer's end for the next applied output: a hold read (finished_sending) or a
// load landed (finished_recving). A failed load is reported as vLLM's NIXL worker reports one, in
// finished_recving and failed_recving both (nixl/base_worker.py:2880-2889, 2991@5df63a72), and the
// scheduler's kv_load_failure_policy decides what follows: "fail" (the default) finishes the
// request as an error, which the simulator sees in a result; "recompute" recomputes its prompt here
// (scheduler.py:2057-2061, 3132-3153, 3341-3354@5df63a72).
func (p *pd) done(id engine.TransferID, ok bool) error {
	if p == nil {
		return engine.ErrUnknownTransfer
	}
	if rid, held := p.holds[id]; held {
		delete(p.holds, id)
		p.outcome.FinishedSending = append(p.outcome.FinishedSending, rid)
		return nil
	}
	if rid, ok := p.notif[id]; ok {
		// The prefiller's release notification landed; the offload's load, not this, brings the KV.
		delete(p.notif, id)
		delete(p.pulling, rid)
		delete(p.matched, rid)
		delete(p.remote, rid)
		return nil
	}
	rid, pulling := p.pulls[id]
	if !pulling {
		return engine.ErrUnknownTransfer
	}
	delete(p.pulls, id)
	delete(p.pulling, rid)
	p.nixl.Forget(rid)
	delete(p.matched, rid)
	delete(p.remote, rid)
	p.outcome.FinishedRecving = append(p.outcome.FinishedRecving, rid)
	if !ok {
		p.outcome.FailedRecving = append(p.outcome.FailedRecving, rid)
	}
	return nil
}

// take is the outcome the worker reports with the applied output, once.
func (p *pd) take() sched.TransferOutcome {
	if p == nil {
		return sched.TransferOutcome{}
	}
	o := p.outcome
	p.outcome = sched.TransferOutcome{}
	sort.Strings(o.FinishedSending)
	sort.Strings(o.FailedRecving)
	return o
}

// pending reports whether a transfer outcome is waiting for an output to carry it.
func (p *pd) pending() bool {
	return p != nil && len(p.outcome.FinishedRecving)+len(p.outcome.FinishedSending)+len(p.outcome.FailedRecving) > 0
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
		p.nixl.Forget(id)
	}
}

// Transfers implements core/engine.TransferGPU: the worker's kv_connector_output for the step
// being applied, which carries every transfer that ended since the last. A step that ran no
// model still reports it: with a connector, vLLM's runner returns the connector's output without
// a forward pass (kv_connector_no_forward, vllm/v1/worker/gpu_model_runner.py@5df63a72).
func (g *batchOutput) Transfers(b core.Batch) sched.TransferOutcome {
	o := g.pd.take()
	if g.off != nil {
		// The offload workers' completed jobs (kv_connector_worker_meta) and completed loads
		// (finished_recving), offload.go.
		jobs, recv := g.off.take(b.ID)
		if len(jobs) > 0 {
			o.WorkerMeta = g.off.workerMeta(jobs)
		}
		o.FinishedRecving = append(o.FinishedRecving, recv...)
	}
	return o
}

var _ core.TransferGPU = (*batchOutput)(nil)
