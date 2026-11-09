package sim

import (
	"crypto/sha256"
	"encoding/binary"
	randv2 "math/rand/v2"
	"strings"
	"testing"
)

// A substream is a function of the seed and its name alone: two generators of one name produce
// one sequence, and any change to the seed or to any part of the name gives another (DEV-6).
func TestNewStream_IsAFunctionOfSeedAndName(t *testing.T) {
	name := StreamName{Component: "workload/client/3", Purpose: "input-length", Entity: "request/17"}
	draws := func(seed int64, n StreamName) [4]int64 {
		r := NewStream(seed, n)
		return [4]int64{r.Int63(), r.Int63(), r.Int63(), r.Int63()}
	}
	base := draws(42, name)
	if again := draws(42, name); again != base {
		t.Fatalf("same seed and name gave %v then %v", base, again)
	}
	others := map[string][4]int64{
		"seed":      draws(43, name),
		"component": draws(42, StreamName{Component: "workload/client/4", Purpose: name.Purpose, Entity: name.Entity}),
		"purpose":   draws(42, StreamName{Component: name.Component, Purpose: "output-length", Entity: name.Entity}),
		"entity":    draws(42, StreamName{Component: name.Component, Purpose: name.Purpose, Entity: "request/18"}),
		// The separator keeps "a"+"bc" and "ab"+"c" apart.
		"boundary": draws(42, StreamName{Component: name.Component + "i", Purpose: strings.TrimPrefix(name.Purpose, "i"), Entity: name.Entity}),
	}
	for what, d := range others {
		if d == base {
			t.Errorf("changing the %s left the draws unchanged: %v", what, d)
		}
	}
}

// The seed is the SHA-256 of the scenario seed (eight big-endian bytes) and the name's encoding,
// as StreamGenerator states, so another implementation can reproduce every draw.
func TestNewStream_SeedIsTheStatedDigest(t *testing.T) {
	n := StreamName{Component: "c", Purpose: "p", Entity: "e"}
	seed := int64(-7)
	var s [8]byte
	binary.BigEndian.PutUint64(s[:], uint64(seed))
	sum := sha256.Sum256(append(s[:], []byte("c\x00p\x00e")...))
	want := randv2.NewPCG(binary.BigEndian.Uint64(sum[0:8]), binary.BigEndian.Uint64(sum[8:16]))
	got := NewStream(seed, n)
	for i := 0; i < 8; i++ {
		if g, w := got.Uint64(), want.Uint64(); g != w {
			t.Fatalf("draw %d: %d, want %d from the stated digest", i, g, w)
		}
	}
	if !strings.Contains(StreamGenerator, "sha256") || !strings.Contains(StreamGenerator, "PCG") {
		t.Errorf("StreamGenerator %q does not name the digest and the generator", StreamGenerator)
	}
}

// A name without a component or a purpose, or with a NUL byte, is a programming error.
func TestNewStream_RefusesAnIllFormedName(t *testing.T) {
	for _, n := range []StreamName{
		{Purpose: "p"},
		{Component: "c"},
		{Component: "c\x00d", Purpose: "p"},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("NewStream(%q) did not panic", n.String())
				}
			}()
			NewStream(1, n)
		}()
	}
}

// A named substream cannot be reseeded: a reseed would make its draws depend on an unnamed
// integer.
func TestNewStream_CannotBeReseeded(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Seed on a named stream did not panic")
		}
	}()
	NewStream(1, StreamName{Component: "c", Purpose: "p"}).Seed(2)
}

// Streams.Child names a part of an entity under the same component.
func TestStreams_ChildNamesAPart(t *testing.T) {
	s := Streams{Seed: 5, Component: "workload/session/s1"}
	r := s.Child("round/2")
	if r.Entity != "round/2" || r.Component != s.Component || r.Seed != 5 {
		t.Fatalf("Child = %+v", r)
	}
	if g := r.Child("x").Entity; g != "round/2/x" {
		t.Fatalf("grandchild entity = %q, want round/2/x", g)
	}
}
