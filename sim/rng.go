package sim

// Named random substreams (design §8.4, DEV-6). Every random draw the simulator makes comes
// from a substream named by component, purpose and entity, for example component
// "workload/client/3", purpose "input-length", entity "request/17". The substream's seed is the
// SHA-256 of the scenario seed and that name, so request 17 of client 3 draws the same prompt
// length whatever requests 1 to 16 drew, in every configuration (common random numbers). Draws
// that depend on how the system responds (an autoscaler's actuation delay, a router tie-break)
// use substreams of their own, so they never shift a workload draw.

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/rand"
	randv2 "math/rand/v2"
	"strings"
)

// StreamGenerator names the generator behind every substream, and its version. It belongs to a
// run's experimental frame (§8.6): a different generator changes every draw. The seed of the
// substream named (component, purpose, entity) under scenario seed s is the SHA-256 of s as
// eight big-endian bytes followed by the name's encoding (StreamName.encode); the first sixteen
// bytes of the digest, as two big-endian words, seed Go's math/rand/v2 PCG (128-bit state, DXSM
// output), which a math/rand Rand reads through its Source64 interface.
const StreamGenerator = "sha256-seeded math/rand/v2 PCG-DXSM, v1"

// StreamName names one random substream.
type StreamName struct {
	// Component is the part of the model that draws, for example "workload/client/3" or
	// "router".
	Component string
	// Purpose is what the draws are for, for example "input-length" or "arrivals".
	Purpose string
	// Entity is the object the draws belong to, for example "request/17"; empty when the
	// component draws one sequence for the purpose.
	Entity string
}

// String is the name as component|purpose|entity, for messages.
func (n StreamName) String() string { return n.Component + "|" + n.Purpose + "|" + n.Entity }

// encode is the name's bytes in the seed: the three parts separated by NUL, which no part may
// contain, so two different names never encode alike.
func (n StreamName) encode() []byte {
	return []byte(n.Component + "\x00" + n.Purpose + "\x00" + n.Entity)
}

// NewStream returns the substream named n under the scenario seed. Two calls with the same seed
// and name return generators that produce the same sequence. It panics on a name with an empty
// component or purpose, or with a NUL byte in any part: either is a programming error.
func NewStream(seed int64, n StreamName) *rand.Rand {
	if n.Component == "" || n.Purpose == "" {
		panic(fmt.Sprintf("sim.NewStream: stream %q needs a component and a purpose", n.String()))
	}
	if strings.ContainsRune(n.Component+n.Purpose+n.Entity, 0) {
		panic(fmt.Sprintf("sim.NewStream: stream %q contains a NUL byte", n.String()))
	}
	h := sha256.New()
	var s [8]byte
	binary.BigEndian.PutUint64(s[:], uint64(seed))
	h.Write(s[:])
	h.Write(n.encode())
	sum := h.Sum(nil)
	pcg := randv2.NewPCG(binary.BigEndian.Uint64(sum[0:8]), binary.BigEndian.Uint64(sum[8:16]))
	return rand.New(pcgSource{pcg}) //nolint:gosec // simulation RNG, not cryptographic
}

// pcgSource adapts a PCG to math/rand's Source64, so samplers keep taking a *rand.Rand.
type pcgSource struct{ p *randv2.PCG }

func (s pcgSource) Uint64() uint64 { return s.p.Uint64() }
func (s pcgSource) Int63() int64   { return int64(s.p.Uint64() >> 1) }

// Seed panics: a named substream's seed is its name, and reseeding it would hide a draw from an
// unnamed stream.
func (s pcgSource) Seed(int64) {
	panic("sim: a named random substream cannot be reseeded")
}

// Streams is the family of substreams of one entity under a scenario seed: every purpose the
// entity draws for is one substream (Stream), and a part of the entity is a child family
// (Child). The zero value names nothing, and its Stream panics.
type Streams struct {
	Seed      int64
	Component string
	Entity    string
}

// Stream returns the entity's substream for purpose.
func (s Streams) Stream(purpose string) *rand.Rand {
	return NewStream(s.Seed, StreamName{Component: s.Component, Purpose: purpose, Entity: s.Entity})
}

// Child returns the family of a part of the entity: its entity is this one's followed by "/"
// and part.
func (s Streams) Child(part string) Streams {
	if s.Entity != "" {
		part = s.Entity + "/" + part
	}
	return Streams{Seed: s.Seed, Component: s.Component, Entity: part}
}
