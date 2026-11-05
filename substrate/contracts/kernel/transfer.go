package kernel

import (
	"fmt"
	"math"
	"time"
)

// LinkID names one shared resource a transfer occupies: a node's NIC egress or ingress,
// an on-node NVLink domain, a rack-scale NVLink domain. Two demands that name the same
// LinkID share that link's bandwidth while both are in flight.
//
// An ID is opaque. Its only contract is identity: within one Kernel, and across the
// kernels built from one deployment, the same physical link always has the same ID and
// different links have different IDs. IDs derive from Placement, which is what makes
// them comparable across the prefill and decode kernels of a disaggregated deployment,
// since both price handoffs over the same fabric.
type LinkID string

// LinkUse is one link a transfer crosses, and the full rate that link offers: the rate a
// transfer gets when it is alone on it.
type LinkUse struct {
	Link           LinkID
	BytesPerSecond float64
}

// TransferDemand is the work one transfer asks of the fabric: its payload, the fixed
// floor no amount of bandwidth removes, and every link it crosses.
//
// It separates what concurrency changes from what it does not. Sharing a link divides
// the rate a transfer sees on it, so the bandwidth-bound part stretches. The floor is the
// least time a transfer takes however much bandwidth it has, and it is not added to the
// bandwidth-bound part: alone a transfer costs the larger of the two, so the floor
// overlaps the transfer rather than preceding it, and a second transfer on the same link
// does not lengthen it. A duration alone cannot be split back into the two, which is why
// a simulator arbitrating concurrent handoffs needs this rather than PDTransferTime.
//
// Which links a transfer crosses is a pricing fact, derived from the placements and the
// fabric, so the kernel states it. How a link shared by several in-flight transfers is
// divided among them (fair share, re-rating as transfers start and finish) is
// simulation, so the simulator decides it and this type does not prescribe it.
type TransferDemand struct {
	// Bytes is the payload, rounded up to a whole byte.
	Bytes int64
	// Floor is the time the transfer takes however much bandwidth it is given.
	Floor time.Duration
	// Links are the links the payload crosses, each named once. A payload crosses at
	// least one, so Links is empty only for a demand with no payload, which costs its
	// floor alone.
	Links []LinkUse
}

// AloneTime is the time the demand takes with every link to itself:
//
//	max(Floor, Bytes / min_i Links[i].BytesPerSecond)
//
// or Floor when Links is empty. The slowest link binds because the payload crosses all
// of them in series.
//
// It is the definition PDTransferTime is held to (see Kernel.PDTransferDemand), so the
// conversion to a Duration is fixed here rather than left to each implementation: the
// seconds are truncated to whole nanoseconds, and a time too long for a Duration
// saturates at the largest one. An implementation that computes PDTransferTime as
// PDTransferDemand(...).AloneTime() satisfies the law by construction.
func (d TransferDemand) AloneTime() time.Duration {
	if len(d.Links) == 0 {
		return d.Floor
	}
	rate := math.Inf(1)
	for _, l := range d.Links {
		rate = math.Min(rate, l.BytesPerSecond)
	}
	if d.Bytes <= 0 {
		return d.Floor
	}
	if !(rate > 0) {
		return time.Duration(math.MaxInt64)
	}
	ns := float64(d.Bytes) / rate * float64(time.Second)
	if ns >= math.MaxInt64 {
		return time.Duration(math.MaxInt64)
	}
	if t := time.Duration(ns); t > d.Floor {
		return t
	}
	return d.Floor
}

// Check reports the first way the demand is malformed, or nil: a negative payload or
// floor, a payload that crosses no link, a link with no ID, a link named twice, or a rate
// that is not a positive finite number. A payload on no link leaves an arbiter nothing to
// share, so a kernel could pass every concurrent transfer at its solo price; a link named
// twice would be counted twice by an arbiter dividing its bandwidth; and a zero rate
// describes a link nothing can cross.
func (d TransferDemand) Check() error {
	if d.Bytes < 0 {
		return fmt.Errorf("bytes %d is negative", d.Bytes)
	}
	if d.Floor < 0 {
		return fmt.Errorf("floor %v is negative", d.Floor)
	}
	if d.Bytes > 0 && len(d.Links) == 0 {
		return fmt.Errorf("%d bytes cross no link", d.Bytes)
	}
	seen := make(map[LinkID]bool, len(d.Links))
	for i, l := range d.Links {
		if l.Link == "" {
			return fmt.Errorf("links[%d] has no id", i)
		}
		if seen[l.Link] {
			return fmt.Errorf("links[%d]: %q is named twice", i, l.Link)
		}
		seen[l.Link] = true
		if !(l.BytesPerSecond > 0) || math.IsInf(l.BytesPerSecond, 1) {
			return fmt.Errorf("links[%d] %q: rate %v must be positive and finite",
				i, l.Link, l.BytesPerSecond)
		}
	}
	return nil
}
