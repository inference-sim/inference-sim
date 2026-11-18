package sim

import (
	"slices"
	"testing"
)

// A series reads as the flat slice its segments make, however they were joined, and a later append to
// a slice it was given cannot reach it.
func TestIntSeriesReadsAsTheFlatSequence(t *testing.T) {
	src := make([]int64, 2, 8) // spare capacity: a request's slice after more tokens arrive
	src[0], src[1] = 10, 20
	var a IntSeries
	a.Add(src)
	a.Add(nil)
	a.Add([]int64{30})
	_ = append(src, 99) // a later append writes inside the spare capacity
	if got, want := a.Flatten(), []int64{10, 20, 30}; !slices.Equal(got, want) || a.Len() != 3 || a.Segments() != 2 {
		t.Fatalf("series %v (len %d, %d segments), want %v in 2 segments", got, a.Len(), a.Segments(), want)
	}
	var b IntSeries
	b.Add([]int64{40, 50})
	var merged IntSeries
	merged.AddAll(a)
	merged.AddAll(b)
	if got, want := slices.Collect(merged.All), []int64{10, 20, 30, 40, 50}; !slices.Equal(got, want) {
		t.Errorf("merged series %v, want %v", got, want)
	}
	if (IntSeries{}).Flatten() != nil || SeriesOf().Len() != 0 {
		t.Error("an empty series must flatten to nil and have no entries")
	}
}

// Owners count entries across segment boundaries: the measured series keeps the entries of the owners
// in the window, in order, whatever the segments were.
func TestMeasuredOwnedCountsAcrossSegments(t *testing.T) {
	var s IntSeries
	s.Add([]int64{1, 2, 3})
	s.Add([]int64{4})
	s.Add([]int64{5, 6})
	owners := []StepOwner{{CompletionUs: 1, Steps: 2}, {CompletionUs: 100, Steps: 3}, {CompletionUs: 2, Steps: 1}}
	got := measuredStepIntervals(s, owners, func(us int64) bool { return us < 50 })
	if want := []int64{1, 2, 6}; !slices.Equal(got, want) {
		t.Errorf("measured entries %v, want %v (owner 1 skips 3,4,5)", got, want)
	}
}
