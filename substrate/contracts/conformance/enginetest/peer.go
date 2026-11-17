package enginetest

import (
	"fmt"
	"reflect"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// PeerStep is one input of a CheckPeered script: a message delivered to the engine at At. The
// message is the engine's own encoding of what a peer would send, so the caller writes it.
type PeerStep struct {
	At  engine.Time
	Msg engine.PeerMessage
}

// CheckPeered checks the laws of engine.Peered on engines newEngine builds, by running script
// on two of them and draining each one's Outbox after every Deliver:
//
//   - the engine is a Peered with a PeerAddr, and each scripted message is addressed to it;
//   - every message it sends is valid (engine.PeerMessage.Validate) and comes from its PeerAddr;
//   - its Seq numbers run 1, 2, 3, ... across drains, in send order;
//   - an Outbox drained again at once is empty;
//   - a message whose To is not its PeerAddr is refused with an error;
//   - the two engines send the same messages, in the same order, for the same script;
//   - an engine that is a PeerPlaced refuses an endpoint with a negative coordinate and accepts
//     a valid one.
//
// It returns the first law broken, nil when none is.
func CheckPeered(newEngine func() (engine.Engine, error), script []PeerStep) error {
	var runs [2][]engine.PeerMessage
	for r := range runs {
		e, err := newEngine()
		if err != nil {
			return fmt.Errorf("enginetest: building the engine: %w", err)
		}
		p, ok := e.(engine.Peered)
		if !ok {
			return fmt.Errorf("enginetest: %T is not an engine.Peered", e)
		}
		addr := p.PeerAddr()
		if addr == "" {
			return fmt.Errorf("enginetest: %T has no PeerAddr, so it takes no peer messages", e)
		}
		if pp, ok := e.(engine.PeerPlaced); ok {
			if err := pp.PlacePeer("peer:9", plan.Endpoint{Node: -1}); err == nil {
				return fmt.Errorf("enginetest: PlacePeer accepted an endpoint with a negative node")
			}
			if err := pp.PlacePeer("peer:9", plan.Endpoint{Node: 1, Rack: 0}); err != nil {
				return fmt.Errorf("enginetest: PlacePeer refused a valid endpoint: %w", err)
			}
		}
		var next uint64 = 1
		drain := func(after string) error {
			for _, m := range p.Outbox() {
				if err := m.Validate(); err != nil {
					return fmt.Errorf("enginetest: after %s: %w", after, err)
				}
				if m.From != addr {
					return fmt.Errorf("enginetest: after %s: a message from %q, but the engine's PeerAddr is %q", after, m.From, addr)
				}
				if m.Seq != next {
					return fmt.Errorf("enginetest: after %s: message Seq %d, want %d (Seq runs 1, 2, 3, ... in send order)", after, m.Seq, next)
				}
				next++
				runs[r] = append(runs[r], m)
			}
			if again := p.Outbox(); len(again) != 0 {
				return fmt.Errorf("enginetest: after %s: Outbox returned %d more messages when drained again at once", after, len(again))
			}
			return nil
		}
		if err := drain("construction"); err != nil {
			return err
		}
		for i, s := range script {
			if s.Msg.To != addr {
				return fmt.Errorf("enginetest: script step %d is addressed to %q, not the engine's PeerAddr %q", i, s.Msg.To, addr)
			}
			if err := p.Deliver(s.Msg, s.At); err != nil {
				return fmt.Errorf("enginetest: script step %d: Deliver: %w", i, err)
			}
			if err := drain(fmt.Sprintf("step %d", i)); err != nil {
				return err
			}
		}
		if len(script) > 0 {
			stray := script[0].Msg
			stray.To = addr + "/not-this-engine"
			if err := p.Deliver(stray, script[len(script)-1].At); err == nil {
				return fmt.Errorf("enginetest: Deliver accepted a message addressed to %q, which is not the engine's PeerAddr %q", stray.To, addr)
			}
			if err := drain("the stray message"); err != nil {
				return err
			}
		}
	}
	if !reflect.DeepEqual(runs[0], runs[1]) {
		return fmt.Errorf("enginetest: two engines given the same script sent different messages:\n%v\n%v", runs[0], runs[1])
	}
	return nil
}
