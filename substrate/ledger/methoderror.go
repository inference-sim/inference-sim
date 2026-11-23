package ledger

import (
	"fmt"
	"math"
	"sort"
)

// Method error (design §9.3, VAL-5): the typical error of each less trusted method, measured by
// leaving one chip out. For every chip that has a measured value, each method's estimate is
// computed as if that measurement did not exist, and compared with it. The error is published
// per method and per primitive family, as aggregates over at least MinAggregateUnits points
// (D7), so a figure names no unit.
//
// The points are the validate-role units' measured values; the seed-holding error job reads
// them and calls MethodError. This file holds no seed and no role: it is the computation.

// Point is one measured value of a coefficient family on one chip.
type Point struct {
	// Family is the primitive family the value belongs to, such as "attention_decode_rate".
	Family string
	// Chip is the catalog chip the value was measured on.
	Chip string
	// Key tells apart values of one family and chip that measure different things, such as the
	// rank count of a collective. Points with equal Family and Key are one series across chips.
	Key string
	// Value is the measured value, positive and finite.
	Value float64
}

// Estimator computes a method's estimate of a point from the series' values on the other
// chips, never from the held-out chip's own. ok is false when the method gives no estimate:
// no sibling to read from, or no formula for the family.
type Estimator struct {
	// Method is the evidence term it stands for (vocab.Method).
	Method string
	// Estimate is given the series' values by chip with the held-out group removed.
	Estimate func(family, key, heldChip string, others map[string]float64) (est float64, ok bool)
}

// MedianOfOthers is the estimator that reads the median of the series on the remaining chips.
// It is the baseline a method must beat: it reads no datasheet and no sibling structure.
func MedianOfOthers(method string) Estimator {
	return Estimator{Method: method, Estimate: func(_, _, _ string, others map[string]float64) (float64, bool) {
		if len(others) == 0 {
			return 0, false
		}
		vs := make([]float64, 0, len(others))
		for _, v := range others {
			vs = append(vs, v)
		}
		sort.Float64s(vs)
		n := len(vs)
		if n%2 == 1 {
			return vs[n/2], true
		}
		return (vs[n/2-1] + vs[n/2]) / 2, true
	}}
}

// MethodErrorRow is one published figure: a method's typical error in one family.
type MethodErrorRow struct {
	Method string
	Family string
	// Points is how many held-out points the figure covers, at least MinAggregateUnits.
	Points int
	// MedianRatio is the median over the points of max(estimate/truth, truth/estimate): 1.0 is
	// exact and 2.0 is a factor of two either way. A ratio is symmetric in over- and
	// under-estimation, which an error in percent is not.
	MedianRatio float64
	// WorstRatio is the largest ratio.
	WorstRatio float64
}

// MethodError leaves one chip out for every chip with a point, for each estimator, and returns
// the typical error per method and family. group maps a chip to the group of near-copies it
// belongs to; the whole group is left out together, because a chip whose measured values equal
// its sibling's (two parts on one die) would otherwise predict itself from its twin and report
// an error of one. A chip absent from group is its own group.
//
// A family with fewer than MinAggregateUnits held-out points for a method is merged into the
// family "other" for that method; if "other" is still too small it is not reported (D7). A
// point an estimator gives no estimate for does not count. Nothing about a candidate enters:
// the figures depend on the measured values alone.
func MethodError(points []Point, estimators []Estimator, group map[string]string) ([]MethodErrorRow, error) {
	for i, p := range points {
		if p.Family == "" || p.Chip == "" {
			return nil, fmt.Errorf("methoderror: point %d names no family or chip", i)
		}
		if !(p.Value > 0) || math.IsInf(p.Value, 0) {
			return nil, fmt.Errorf("methoderror: point %d (%s, %s) has value %v; a measured value is positive and finite", i, p.Family, p.Chip, p.Value)
		}
	}
	for _, e := range estimators {
		if e.Method == "" || e.Estimate == nil {
			return nil, fmt.Errorf("methoderror: an estimator names no method or has no estimate")
		}
	}
	grp := func(chip string) string {
		if g, ok := group[chip]; ok && g != "" {
			return g
		}
		return chip
	}
	type seriesKey struct{ family, key string }
	series := map[seriesKey]map[string]float64{}
	for _, p := range points {
		k := seriesKey{p.Family, p.Key}
		if series[k] == nil {
			series[k] = map[string]float64{}
		}
		if _, dup := series[k][p.Chip]; dup {
			return nil, fmt.Errorf("methoderror: %s %q has two points on %s", p.Family, p.Key, p.Chip)
		}
		series[k][p.Chip] = p.Value
	}
	type acc struct{ ratios []float64 }
	byMethodFamily := map[[2]string]*acc{}
	keys := make([]seriesKey, 0, len(series))
	for k := range series {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].family != keys[j].family {
			return keys[i].family < keys[j].family
		}
		return keys[i].key < keys[j].key
	})
	for _, k := range keys {
		s := series[k]
		chips := make([]string, 0, len(s))
		for c := range s {
			chips = append(chips, c)
		}
		sort.Strings(chips)
		for _, held := range chips {
			others := map[string]float64{}
			for c, v := range s {
				if grp(c) != grp(held) {
					others[c] = v
				}
			}
			for _, e := range estimators {
				est, ok := e.Estimate(k.family, k.key, held, others)
				if !ok {
					continue
				}
				if !(est > 0) || math.IsInf(est, 0) {
					return nil, fmt.Errorf("methoderror: %s estimated %s %q on %s as %v", e.Method, k.family, k.key, held, est)
				}
				truth := s[held]
				r := est / truth
				if r < 1 {
					r = 1 / r
				}
				mk := [2]string{e.Method, k.family}
				if byMethodFamily[mk] == nil {
					byMethodFamily[mk] = &acc{}
				}
				byMethodFamily[mk].ratios = append(byMethodFamily[mk].ratios, r)
			}
		}
	}
	mfs := make([][2]string, 0, len(byMethodFamily))
	for mf := range byMethodFamily {
		mfs = append(mfs, mf)
	}
	sort.Slice(mfs, func(i, j int) bool {
		if mfs[i][0] != mfs[j][0] {
			return mfs[i][0] < mfs[j][0]
		}
		return mfs[i][1] < mfs[j][1]
	})
	var rows []MethodErrorRow
	other := map[string][]float64{}
	var otherMethods []string
	for _, mf := range mfs {
		rs := byMethodFamily[mf].ratios
		if len(rs) < MinAggregateUnits {
			if other[mf[0]] == nil {
				otherMethods = append(otherMethods, mf[0])
			}
			other[mf[0]] = append(other[mf[0]], rs...)
			continue
		}
		rows = append(rows, rowOf(mf[0], mf[1], rs))
	}
	for _, m := range otherMethods {
		if rs := other[m]; len(rs) >= MinAggregateUnits {
			rows = append(rows, rowOf(m, "other", rs))
		}
	}
	return rows, nil
}

func rowOf(method, family string, ratios []float64) MethodErrorRow {
	rs := append([]float64(nil), ratios...)
	sort.Float64s(rs)
	n := len(rs)
	med := rs[n/2]
	if n%2 == 0 {
		med = (rs[n/2-1] + rs[n/2]) / 2
	}
	return MethodErrorRow{Method: method, Family: family, Points: n, MedianRatio: med, WorstRatio: rs[n-1]}
}

// MethodErrorReport turns the rows into the aggregates a seed-holding job publishes: one
// figure per method and family, in the factor-of-error form, over the points it covers.
func MethodErrorReport(candidate string, rows []MethodErrorRow) *Report {
	rep := &Report{Kind: "AggregateReport", Job: "error", Candidate: candidate, Aggregates: []Aggregate{}}
	for _, r := range rows {
		rep.Aggregates = append(rep.Aggregates,
			Aggregate{Lane: r.Family, Metric: r.Method + "_median_ratio", Units: r.Points, Value: r.MedianRatio},
			Aggregate{Lane: r.Family, Metric: r.Method + "_worst_ratio", Units: r.Points, Value: r.WorstRatio})
	}
	rep.Evaluated = len(rep.Aggregates) > 0
	return rep
}

// SameGeneration is the estimator the registry's by-generation writer uses
// (registry/scripts/extrapolate_by_generation.py): the median of the series on the other chips
// of the held-out chip's architecture generation, and no estimate when it has none. It refuses
// the global median on purpose; MedianOfOthers is the baseline it is compared with.
func SameGeneration(method string, generation map[string]string) Estimator {
	return Estimator{Method: method, Estimate: func(family, key, held string, others map[string]float64) (float64, bool) {
		g, ok := generation[held]
		if !ok || g == "" {
			return 0, false
		}
		same := map[string]float64{}
		for c, v := range others {
			if generation[c] == g {
				same[c] = v
			}
		}
		return MedianOfOthers(method).Estimate(family, key, held, same)
	}}
}
