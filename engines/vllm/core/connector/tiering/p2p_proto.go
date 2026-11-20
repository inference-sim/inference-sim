// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/p2p/session/protocol.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// Also p2p/control/base.py and p2p/data/base.py.

package tiering

import "sort"

// The p2p session protocol's message types (protocol.py:133-335@4ba787ec).
const (
	p2pConnect      = "connect"
	p2pConnectAck   = "connect_ack"
	p2pDisconnect   = "disconnect"
	p2pFetch        = "fetch"
	p2pLookup       = "lookup"
	p2pLookupResp   = "lookup_resp"
	p2pTransferDone = "transfer_done"
	p2pAbortFetch   = "abort_fetch"
	p2pAbortAck     = "abort_ack"
)

// P2PMsg is one control message of the p2p session protocol (protocol.py:116-341): a msgpack dict
// on vLLM's wire, a value here. Type names the message; the other fields are those its type
// carries. A message is built by its sender and is never changed after Send, so the host may hand
// the same value to the receiver.
//
// vLLM validates each message it receives (ConnectMsg.validate and the like) and disconnects a
// peer that sends a malformed one. Every message here is built by this port, so none is malformed
// and the port does not validate them.
type P2PMsg struct {
	Type string
	// PeerID is a connect or connect_ack's sender identity (host:port).
	PeerID string
	// A connect's NIXL metadata and memory layout.
	AgentMetadata []byte
	BaseAddr      int64
	NumBlocks     int
	BlockLen      int64
	Fingerprint   string
	HashSeed      string
	// KVRequestID is the request a fetch, lookup, lookup_resp, transfer_done, abort_fetch or
	// abort_ack is about.
	KVRequestID  string
	Keys         []Key
	BlockIndexes []int
	RoundSeq     int
	Hits         []bool
	Success      bool
}

// P2PConn is ControlConnection (control/base.py:55-113): a bidirectional channel to one peer.
type P2PConn interface {
	PeerID() string
	// Alive is false after MarkDead or Close.
	Alive() bool
	// Send queues a message for the peer; it fails on a connection that is not alive.
	Send(m P2PMsg) error
	// Recv drains the messages received since the last call.
	Recv() []P2PMsg
	MarkDead()
	Close()
}

// P2PControl is ControlTransport (control/base.py:116-176), one per instance.
type P2PControl interface {
	// Connect opens an outbound connection to peer ("host:port").
	Connect(peer string) (P2PConn, error)
	// Poll processes the pending I/O and returns the newly accepted inbound connections, each
	// with its first messages already buffered.
	Poll() []P2PConn
	Close()
}

// P2PPollResult is PollResult (data/base.py:111-125). PeerFault "" is None.
type P2PPollResult struct {
	Done, Failed []int
	PeerFault    string
}

// P2PRegistration is the Future add_remote_peer_async returns (data/base.py:213-258): a peer
// registration that resolves later.
type P2PRegistration interface {
	Done() bool
	// Err is the failed registration's cause; nil is success. Read only once Done.
	Err() error
	Cancel()
}

// P2PData is DataTransport (data/base.py:128-351), one per instance: block WRITEs from the local
// primary region to registered peers. It moves bytes; the tier only decides which.
type P2PData interface {
	AgentMetadata() []byte
	BaseAddr() int64
	NumBlocks() int
	BlockLen() int64
	Fingerprint() string
	AddRemotePeerAsync(peer string, metadata []byte, baseAddr int64, numBlocks int, blockLen int64) P2PRegistration
	RemoveRemotePeer(peer string)
	// WriteBlocks submits one WRITE of local blocks to the peer's blocks; ok false is None (the peer
	// is not registered, or the submission failed).
	WriteBlocks(peer string, local, remote []int) (id int, ok bool)
	// Poll reports the peer's transfers that ended since the last poll.
	Poll(peer string) P2PPollResult
	// Cancel is cancel(ids, mode): wait true is mode="wait", which returns the ids still in flight;
	// otherwise "immediate".
	Cancel(ids []int, wait bool) []int
	// SetDraining marks the span of drain_jobs, inside which a transfer's state check blocks until
	// the transfer ends (the oracle's NIXL agent finishes every transfer there).
	SetDraining(on bool)
	Close()
}

// omap is a Python dict: a map that iterates in insertion order. Setting an existing key keeps its
// place; deleting a key and setting it again moves it to the end.
type omap[K comparable, V any] struct {
	order []K
	m     map[K]V
}

func (o *omap[K, V]) get(k K) (V, bool) {
	v, ok := o.m[k]
	return v, ok
}

func (o *omap[K, V]) has(k K) bool { _, ok := o.m[k]; return ok }

func (o *omap[K, V]) set(k K, v V) {
	if o.m == nil {
		o.m = map[K]V{}
	}
	if _, ok := o.m[k]; !ok {
		o.order = append(o.order, k)
	}
	o.m[k] = v
}

// setDefault is dict.setdefault.
func (o *omap[K, V]) setDefault(k K, v V) V {
	if old, ok := o.m[k]; ok {
		return old
	}
	o.set(k, v)
	return v
}

func (o *omap[K, V]) pop(k K) (V, bool) {
	v, ok := o.m[k]
	if !ok {
		return v, false
	}
	delete(o.m, k)
	for i, x := range o.order {
		if x == k {
			o.order = append(o.order[:i:i], o.order[i+1:]...)
			break
		}
	}
	return v, true
}

func (o *omap[K, V]) len() int { return len(o.m) }

// keys is a snapshot of the keys in order, safe to range over while the map changes.
func (o *omap[K, V]) keys() []K { return append([]K(nil), o.order...) }

func (o *omap[K, V]) values() []V {
	out := make([]V, 0, len(o.order))
	for _, k := range o.order {
		out = append(out, o.m[k])
	}
	return out
}

func (o *omap[K, V]) clear() { o.order, o.m = nil, nil }

// intSet is a Python set of small non-negative ints (job and transfer ids). CPython iterates such a
// set in ascending order while every member is below the hash table's size (8 slots until a 5th
// member), so the port iterates in ascending order. A set with larger members can iterate
// otherwise in vLLM; no sequence holds one.
type intSet map[int]bool

func (s intSet) sorted() []int {
	out := make([]int, 0, len(s))
	for i := range s {
		out = append(out, i)
	}
	sort.Ints(out)
	return out
}

// strSet is a Python set of strings. CPython iterates it in hash order, which follows
// PYTHONHASHSEED; the port iterates in insertion order. The sequences do not depend on the order.
type strSet = omap[string, struct{}]
