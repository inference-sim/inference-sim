package engine

import "fmt"

// Peered is implemented by an engine that exchanges control messages with peer engines, such
// as vLLM's p2p KV tier, whose sessions talk over a side channel between instances. It is
// optional. A simulator that does not look for it runs no peer, and an engine whose
// configuration needs one refuses that configuration when the simulator cannot carry it.
//
// The engine names a peer only by the address its configuration gives it. The simulator maps
// each address to the instance whose PeerAddr it is, and so to that instance's node. A message
// carries no time: the simulator decides when it arrives, so the engine still reads no clock.
// Delivery stays deterministic: the simulator never reorders the messages from one sender to
// one receiver, and orders arrivals at one instant by (From, Seq).
type Peered interface {
	// PeerAddr is the address peers reach this engine at, such as vLLM's "host:port". It is
	// fixed at construction; "" means the engine takes no peer messages.
	PeerAddr() string
	// Outbox returns the messages the engine has sent since the last Outbox, in the engine's
	// send order, and forgets them. The simulator drains it after every call that can change
	// engine state (Add, Schedule, Complete, TransferDone, Finish, Deliver and a wake firing),
	// so a reply made inside Deliver is not lost.
	Outbox() []PeerMessage
	// Deliver hands the engine a message that reached it at now. Its instant never decreases,
	// as for TransferDone, and a Schedule may follow at the same instant. A message whose To is
	// not this engine's PeerAddr is an error, as an unknown transfer is for TransferDone. The
	// simulator drops a message to an engine that has stopped, or to an address no engine
	// holds, without calling Deliver, and counts it.
	Deliver(m PeerMessage, now Time) error
}

// PeerMessage is one control message from one engine to another.
type PeerMessage struct {
	// From and To are PeerAddrs.
	From, To string
	// Seq is the sender's send sequence: 1 for its first message, one more for each after.
	Seq uint64
	// Bytes is the message's size on the wire, which a simulator may price; at least 1.
	Bytes int64
	// Payload is the engine's own encoding, opaque to the simulator.
	Payload []byte
}

// Validate reports a message no simulator can deliver: one without a sender, a receiver, a
// sequence number or a size, or one an engine sends to itself.
func (m PeerMessage) Validate() error {
	switch {
	case m.From == "" || m.To == "":
		return fmt.Errorf("engine: peer message %d names no sender or no receiver (from %q, to %q)", m.Seq, m.From, m.To)
	case m.From == m.To:
		return fmt.Errorf("engine: peer message %d from %q is addressed to its sender", m.Seq, m.From)
	case m.Seq == 0:
		return fmt.Errorf("engine: peer message from %q to %q has no sequence number", m.From, m.To)
	case m.Bytes < 1:
		return fmt.Errorf("engine: peer message %d from %q has %d bytes; a message has at least one", m.Seq, m.From, m.Bytes)
	}
	return nil
}
