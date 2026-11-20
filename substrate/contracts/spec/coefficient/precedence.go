package coefficient

import (
	"fmt"
	"sort"
	"strings"
)

// Point is one deployment's coordinates on every scope dimension: the values a coefficient's
// scope is matched against.
type Point struct {
	Hardware     string
	Model        string
	TP           int
	EP           int
	NodesSpanned int
	// Fabric is the cluster's inter-node network, or "" for a single-node cluster.
	Fabric string
}

// Admits reports whether the scope covers the point. A dimension the scope does not state
// admits any value. A stated dimension must list the point's value, so a coefficient scoped
// to hardware alone applies at every width, and one scoped to tp [8] does not apply at tp 4.
// A fabric-scoped entry was measured over one interconnect, so a point on another fabric, or
// on none, is outside it.
func (s Scope) Admits(p Point) bool {
	if len(s.Hardware) > 0 && !containsFold(s.Hardware, p.Hardware) {
		return false
	}
	if len(s.Model) > 0 && !containsFold(s.Model, p.Model) {
		return false
	}
	if len(s.TP) > 0 && !containsInt(s.TP, p.TP) {
		return false
	}
	if len(s.EP) > 0 && !containsInt(s.EP, p.EP) {
		return false
	}
	if len(s.NodesSpanned) > 0 && !containsInt(s.NodesSpanned, p.NodesSpanned) {
		return false
	}
	if len(s.Fabric) > 0 && (p.Fabric == "" || !containsFold(s.Fabric, p.Fabric)) {
		return false
	}
	return true
}

// PrecedenceOrder is the written order in which scope dimensions decide between two entries
// of one name and one method rank that match the same deployment (design VAL-4). The first
// dimension on which the two scopes differ decides, and a dimension differs only when one
// entry states it and the other leaves it open:
//
//   - the entry that states the dimension is chosen over the one that leaves it open;
//   - two entries that both state it with the same values do not differ on it;
//   - two entries that both state it with different values are not ordered, because a list
//     that merely overlaps another is no narrower than it: refuse, and let the author make the
//     lists disjoint or identical.
//
// The model comes first, as the cost model's lookup has always had it: a value fitted on one
// model is a narrower claim than a part-wide one, whichever part it names. Then the chip, the
// layout widths (tp, ep), the nodes spanned, and the fabric. Two entries equal on every
// dimension are not ordered either.
var PrecedenceOrder = []string{"model", "hardware", "tp", "ep", "nodes_spanned", "fabric"}

// ComparePrecedence orders two scopes by PrecedenceOrder. It returns -1 when a is chosen
// over b, +1 when b is chosen over a, and ordered=false when the order does not choose.
func ComparePrecedence(a, b Scope) (cmp int, ordered bool) {
	for _, dim := range PrecedenceOrder {
		var sa, sb []string
		switch dim {
		case "model":
			sa, sb = a.Model, b.Model
		case "hardware":
			sa, sb = a.Hardware, b.Hardware
		case "tp":
			sa, sb = intsAsStrings(a.TP), intsAsStrings(b.TP)
		case "ep":
			sa, sb = intsAsStrings(a.EP), intsAsStrings(b.EP)
		case "nodes_spanned":
			sa, sb = intsAsStrings(a.NodesSpanned), intsAsStrings(b.NodesSpanned)
		case "fabric":
			sa, sb = a.Fabric, b.Fabric
		}
		switch {
		case len(sa) == 0 && len(sb) == 0:
			continue
		case len(sa) == 0:
			return 1, true
		case len(sb) == 0:
			return -1, true
		case subsetFold(sa, sb) && subsetFold(sb, sa):
			continue
		}
		return 0, false
	}
	return 0, false
}

func intsAsStrings(vs []int) []string {
	out := make([]string, len(vs))
	for i, v := range vs {
		out[i] = fmt.Sprint(v)
	}
	return out
}

// subsetFold reports whether every value of a is in b, ignoring case.
func subsetFold(a, b []string) bool {
	for _, x := range a {
		if !containsFold(b, x) {
			return false
		}
	}
	return true
}

// Candidate is one entry that matches a lookup, with where it came from.
type Candidate struct {
	Set   string
	Entry Entry
}

// Describe renders the candidate for an error or a provenance record.
func (c Candidate) Describe() string {
	return fmt.Sprintf("%s: %s [%s; %s]", c.Set, c.Entry.Name, c.Entry.Method, c.Entry.Scope.Describe())
}

// beats reports whether a is chosen over b by the written precedence of design VAL-4:
//
//  1. The entry with the smaller method rank (design §9.2) is used.
//  2. Between entries of one rank, or when either has no rank (not_charged), the scope
//     precedence of ComparePrecedence decides.
//
// Entries that neither rule orders are not ordered, and a lookup that finds them is refused.
func beats(a, b Entry) bool {
	ra, aok := a.Method.Rank()
	rb, bok := b.Method.Rank()
	if aok && bok && ra != rb {
		return ra < rb
	}
	cmp, ordered := ComparePrecedence(a.Scope, b.Scope)
	return ordered && cmp < 0
}

// Select picks the one entry a lookup uses among the entries that match it, whatever order
// they were loaded in, and returns the others it displaced. It refuses when no single
// candidate is chosen over every other, naming the candidates the precedence does not order.
// Candidates must all share one entry name.
func Select(cands []Candidate) (winner Candidate, displaced []Candidate, err error) {
	switch len(cands) {
	case 0:
		return Candidate{}, nil, fmt.Errorf("no candidate")
	case 1:
		return cands[0], nil, nil
	}
	var winners []int
	for i := range cands {
		all := true
		for j := range cands {
			if i != j && !beats(cands[i].Entry, cands[j].Entry) {
				all = false
				break
			}
		}
		if all {
			winners = append(winners, i)
		}
	}
	if len(winners) == 1 {
		for j, c := range cands {
			if j != winners[0] {
				displaced = append(displaced, c)
			}
		}
		sortCandidates(displaced)
		return cands[winners[0]], displaced, nil
	}
	desc := make([]string, len(cands))
	for i, c := range cands {
		desc[i] = c.Describe()
	}
	sort.Strings(desc)
	return Candidate{}, nil, fmt.Errorf("coefficient %q: %d entries match one deployment and the "+
		"scope precedence does not choose one of them (VAL-4); narrow one scope or remove one "+
		"entry:\n  %s", cands[0].Entry.Name, len(cands), strings.Join(desc, "\n  "))
}

func sortCandidates(cs []Candidate) {
	sort.Slice(cs, func(i, j int) bool { return cs[i].Describe() < cs[j].Describe() })
}

func containsFold(list []string, want string) bool {
	for _, s := range list {
		if strings.EqualFold(s, want) {
			return true
		}
	}
	return false
}

func containsInt(list []int, want int) bool {
	for _, v := range list {
		if v == want {
			return true
		}
	}
	return false
}
