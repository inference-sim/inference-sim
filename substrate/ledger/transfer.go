package ledger

import (
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/substrate/contracts/vocab"
)

// The cross-generation transfer rule (Table DC, VAL-5, design §9.2) and its validation by
// leaving one generation out.
//
// A rate that is limited by a datasheet quantity (memory bandwidth, link bandwidth) carries over
// to another chip by keeping the efficiency, the measured value over the datasheet figure, and
// multiplying by the target's figure. The rule reads one series (the same family, key and scope)
// measured on the source generation. Roofline (vocab.MethodRoofline) is what such a value would
// otherwise rank as: the target's datasheet figure times a default efficiency taken from the
// same family on every chip outside the target's generation, whichever series it came from.
//
// The rule is used only if it beats roofline on the same held-out points: both are scored on
// the points for which both have an estimate. A direction with too few such points to publish
// (D7), or with a rule that does not win, ranks as roofline.

// TransferSpec says which families a datasheet quantity limits and what the quantity is.
type TransferSpec struct {
	// Axis names the datasheet quantity that limits a family, as a class of families
	// ("memory_bandwidth"), or reports false when no quantity does. A family with none has no
	// roofline and no rule, and is compared on neither.
	Axis func(family string) (axis string, ok bool)
	// Datasheet is the quantity by axis and chip, positive.
	Datasheet map[string]map[string]float64
	// Generation is the architecture generation of each chip.
	Generation map[string]string
}

// TransferRow is one published figure: the transfer from one generation to another in one class
// of families, scored on the held-out points both methods estimate.
type TransferRow struct {
	// From is the source generation the rule reads; To is the generation left out.
	From string `json:"from"`
	To   string `json:"to"`
	// Class is the axis of the families the row covers.
	Class string `json:"class"`
	// Points is how many held-out points the figure covers, at least MinAggregateUnits.
	Points int `json:"points"`
	// RuleMedian and RuleWorst are the median and largest of max(estimate/truth,
	// truth/estimate) for the rule; RooflineMedian and RooflineWorst the same for roofline.
	RuleMedian     float64 `json:"rule_median_ratio"`
	RuleWorst      float64 `json:"rule_worst_ratio"`
	RooflineMedian float64 `json:"roofline_median_ratio"`
	RooflineWorst  float64 `json:"roofline_worst_ratio"`
	// UseRule is true when the rule's median ratio is below roofline's. A tie is roofline.
	UseRule bool `json:"use_rule"`
}

type transferKey struct{ from, to, class string }

// TransferError leaves each generation out in turn. For every point on a chip of the held-out
// generation it computes the rule's estimate from each other generation's chips in the same
// series, and roofline's from the family on all chips outside the held-out generation, and
// compares both with the measured value. Nothing from the held-out generation enters an
// estimate. A row is published for a source, a target and a class only if both methods
// estimate at least MinAggregateUnits points.
func TransferError(points []Point, spec TransferSpec) ([]TransferRow, error) {
	for axis, byChip := range spec.Datasheet {
		for chip, v := range byChip {
			if !(v > 0) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("transfer: datasheet %s of %s is %v; a datasheet figure is positive and finite", axis, chip, v)
			}
		}
	}
	type seriesKey struct{ family, key string }
	series := map[seriesKey]map[string]float64{}
	byFamily := map[string][]Point{}
	for i, p := range points {
		if p.Family == "" || p.Chip == "" {
			return nil, fmt.Errorf("transfer: point %d names no family or chip", i)
		}
		if !(p.Value > 0) || math.IsInf(p.Value, 0) {
			return nil, fmt.Errorf("transfer: point %d (%s, %s) has value %v; a measured value is positive and finite", i, p.Family, p.Chip, p.Value)
		}
		k := seriesKey{p.Family, p.Key}
		if series[k] == nil {
			series[k] = map[string]float64{}
		}
		if _, dup := series[k][p.Chip]; dup {
			return nil, fmt.Errorf("transfer: %s %q has two points on %s", p.Family, p.Key, p.Chip)
		}
		series[k][p.Chip] = p.Value
		byFamily[p.Family] = append(byFamily[p.Family], p)
	}
	gen := func(chip string) string { return spec.Generation[chip] }
	median := func(vs []float64) float64 {
		sort.Float64s(vs)
		n := len(vs)
		if n%2 == 1 {
			return vs[n/2]
		}
		return (vs[n/2-1] + vs[n/2]) / 2
	}
	ratio := func(est, truth float64) float64 {
		if r := est / truth; r >= 1 {
			return r
		}
		return truth / est
	}

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

	type pair struct{ rule, roof []float64 }
	acc := map[transferKey]*pair{}
	for _, k := range keys {
		axis, ok := spec.Axis(k.family)
		if !ok {
			continue
		}
		ds := spec.Datasheet[axis]
		s := series[k]
		chips := make([]string, 0, len(s))
		for c := range s {
			chips = append(chips, c)
		}
		sort.Strings(chips)
		for _, held := range chips {
			to := gen(held)
			if to == "" || ds[held] == 0 {
				continue
			}
			var pool []float64
			for _, p := range byFamily[k.family] {
				if gen(p.Chip) != to && ds[p.Chip] > 0 {
					pool = append(pool, p.Value/ds[p.Chip])
				}
			}
			if len(pool) == 0 {
				continue
			}
			roof := median(pool) * ds[held]
			bySource := map[string][]float64{}
			for _, c := range chips {
				if g := gen(c); g != "" && g != to && ds[c] > 0 {
					bySource[g] = append(bySource[g], s[c]*ds[held]/ds[c])
				}
			}
			for from, ests := range bySource {
				tk := transferKey{from, to, axis}
				if acc[tk] == nil {
					acc[tk] = &pair{}
				}
				acc[tk].rule = append(acc[tk].rule, ratio(median(ests), s[held]))
				acc[tk].roof = append(acc[tk].roof, ratio(roof, s[held]))
			}
		}
	}

	var rows []TransferRow
	for tk, p := range acc {
		if len(p.rule) < MinAggregateUnits {
			continue
		}
		rule, roof := rowOf("rule", tk.class, p.rule), rowOf("roofline", tk.class, p.roof)
		rows = append(rows, TransferRow{
			From: tk.from, To: tk.to, Class: tk.class, Points: len(p.rule),
			RuleMedian: rule.MedianRatio, RuleWorst: rule.WorstRatio,
			RooflineMedian: roof.MedianRatio, RooflineWorst: roof.WorstRatio,
			UseRule: rule.MedianRatio < roof.MedianRatio,
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.From != b.From {
			return a.From < b.From
		}
		if a.To != b.To {
			return a.To < b.To
		}
		return a.Class < b.Class
	})
	return rows, nil
}

// TransferMethod is the method a value moved from generation from to generation to ranks as,
// for a family of the given class. A transfer within one generation is transferred. A transfer
// across generations is transferred only if a published row for that direction and class has
// the rule beating roofline; otherwise, including when nothing was published, it is roofline.
func TransferMethod(rows []TransferRow, from, to, class string) vocab.Method {
	if from == to {
		return vocab.MethodTransferred
	}
	for _, r := range rows {
		if r.From == from && r.To == to && r.Class == class && r.UseRule {
			return vocab.MethodTransferred
		}
	}
	return vocab.MethodRoofline
}
