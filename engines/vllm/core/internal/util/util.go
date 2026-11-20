// Moved from inference-sim/sim/internal/util/util.go at ca532abca209af41425c007974a5f3ef336f5981.

// Package util provides generic utility functions shared across sim/ sub-packages.
package util

// Len64 returns the length of a slice as int64.
func Len64[T any](v []T) int64 { return int64(len(v)) }
