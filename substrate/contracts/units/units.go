// Package units gives quantities their units as types (design SUB-16).
//
// Bytes, seconds, tokens, FLOPs, bytes per second and dimensionless ratios are distinct
// types, and so are per-rank, per-node and per-group amounts, so adding a byte count to a
// token count, or a per-rank amount to a per-group one, does not compile. A conversion is
// a named function that says what it does: a rate times a time is bytes, a per-rank amount
// times the ranks is a per-group amount.
//
// The package is additive. Existing fields keep their types until a contract-change review
// set moves them; new contracts (engine.PoolPlan, engine.StepShape) are written with these
// types from the start.
package units

import (
	"fmt"
	"math"
	"time"
)

// Bytes is a byte count. It is signed so that a difference can be negative; a size that must
// not be negative is checked where it is read.
type Bytes int64

// Tokens is a token count.
type Tokens int64

// FLOPs is a count of floating-point operations.
type FLOPs float64

// Seconds is a span of time. It is a float because a rate times a size is not a whole
// number of nanoseconds; Duration converts at the end, rounding once.
type Seconds float64

// BytesPerSecond is a data rate.
type BytesPerSecond float64

// FLOPsPerSecond is a compute rate.
type FLOPsPerSecond float64

// Ratio is a dimensionless quotient, such as a utilisation or a hit rate. It carries no
// range; Efficiency is the type for a value that lies in (0, 1].
type Ratio float64

// Over returns the time b takes at rate r. A rate that is not positive and finite is an
// error, never an infinite time that a later sum would hide.
func (b Bytes) Over(r BytesPerSecond) (Seconds, error) {
	if !(r > 0) || math.IsInf(float64(r), 0) {
		return 0, fmt.Errorf("units: %d bytes over a rate of %v bytes per second", int64(b), float64(r))
	}
	return Seconds(float64(b) / float64(r)), nil
}

// Over returns the time f takes at rate r, with the same refusal as Bytes.Over.
func (f FLOPs) Over(r FLOPsPerSecond) (Seconds, error) {
	if !(r > 0) || math.IsInf(float64(r), 0) {
		return 0, fmt.Errorf("units: %v FLOPs over a rate of %v FLOP/s", float64(f), float64(r))
	}
	return Seconds(float64(f) / float64(r)), nil
}

// In returns the bytes rate r moves in s.
func (r BytesPerSecond) In(s Seconds) Bytes { return Bytes(math.Round(float64(r) * float64(s))) }

// Duration converts to a time.Duration, rounding to the nearest nanosecond once.
func (s Seconds) Duration() time.Duration { return time.Duration(math.Round(float64(s) * 1e9)) }

// FromDuration is the inverse of Duration.
func FromDuration(d time.Duration) Seconds { return Seconds(d.Seconds()) }

// Scale multiplies a byte count by a dimensionless ratio, rounding to the nearest byte.
func (b Bytes) Scale(r Ratio) Bytes { return Bytes(math.Round(float64(b) * float64(r))) }

// Efficiency is a fraction of a peak that was achieved: it lies in (0, 1]. Zero would be a
// part that does no work, and above one would beat its own datasheet, so neither is an
// efficiency (SUB-5, SUB-16).
type Efficiency float64

// NewEfficiency returns v as an Efficiency, or an error if it is not in (0, 1].
func NewEfficiency(v float64) (Efficiency, error) {
	if !(v > 0 && v <= 1) {
		return 0, fmt.Errorf("units: efficiency %v is not in (0, 1]", v)
	}
	return Efficiency(v), nil
}

// OfBytes returns the rate an efficiency achieves of a peak rate.
func (e Efficiency) OfBytes(peak BytesPerSecond) BytesPerSecond {
	return BytesPerSecond(float64(e) * float64(peak))
}

// OfFLOPs returns the rate an efficiency achieves of a peak rate.
func (e Efficiency) OfFLOPs(peak FLOPsPerSecond) FLOPsPerSecond {
	return FLOPsPerSecond(float64(e) * float64(peak))
}

// Quantity is what a basis wraps: an amount with a unit.
type Quantity interface {
	~int64 | ~float64
}

// PerRank is an amount held by each rank of a group.
type PerRank[T Quantity] struct{ v T }

// PerNode is an amount held once by each node, whatever the number of ranks on it.
type PerNode[T Quantity] struct{ v T }

// PerGroup is an amount for a whole group of ranks.
type PerGroup[T Quantity] struct{ v T }

// Rank, Node and Group wrap a value in its basis.
func Rank[T Quantity](v T) PerRank[T]   { return PerRank[T]{v} }
func Node[T Quantity](v T) PerNode[T]   { return PerNode[T]{v} }
func Group[T Quantity](v T) PerGroup[T] { return PerGroup[T]{v} }

// Value returns the wrapped amount. Reading it drops the basis, so it is for the edge of
// the program: printing, serialising, handing a value to a library.
func (a PerRank[T]) Value() T  { return a.v }
func (a PerNode[T]) Value() T  { return a.v }
func (a PerGroup[T]) Value() T { return a.v }

// Add sums two amounts of one basis and one unit. Mixing bases or units does not compile.
func (a PerRank[T]) Add(b PerRank[T]) PerRank[T]    { return PerRank[T]{a.v + b.v} }
func (a PerNode[T]) Add(b PerNode[T]) PerNode[T]    { return PerNode[T]{a.v + b.v} }
func (a PerGroup[T]) Add(b PerGroup[T]) PerGroup[T] { return PerGroup[T]{a.v + b.v} }

// Across returns the amount for a group of n ranks that each hold a. A group of fewer than
// one rank is an error.
func (a PerRank[T]) Across(ranks int) (PerGroup[T], error) {
	if ranks < 1 {
		return PerGroup[T]{}, fmt.Errorf("units: a group of %d ranks", ranks)
	}
	return PerGroup[T]{a.v * T(ranks)}, nil
}

// OnNodes returns the amount for a group spread over n nodes that each hold a. A shared
// per-node figure counts once per node, whatever the ranks on it.
func (a PerNode[T]) OnNodes(nodes int) (PerGroup[T], error) {
	if nodes < 1 {
		return PerGroup[T]{}, fmt.Errorf("units: a group on %d nodes", nodes)
	}
	return PerGroup[T]{a.v * T(nodes)}, nil
}
