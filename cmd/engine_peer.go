package cmd

// peerMessageLatencyUs is the time a peer message takes from sender to receiver, in microseconds: the
// named substitution peer_message_delivery_latency. vLLM's p2p tier sends them over ZMQ between
// instances; nothing in the data this repository may use measures that round trip, so the run assumes
// one value for every pair and states it in the Result (peer_message_delivery_latency_us) beside the
// messages it carried and dropped. It moves when a measurement replaces it.
const peerMessageLatencyUs = 100
