package scenario

import (
	"fmt"
	"math"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// Rule names for the ways a cluster's switch groups can be wrong. Each is the Rule of the
// problem that reports it, so a caller tells them apart without reading the message.
const (
	RuleSwitchGroupEmpty          = "switch_group_empty"
	RuleSwitchGroupUnknownNode    = "switch_group_unknown_node"
	RuleSwitchGroupOverlap        = "switch_group_overlap"
	RuleSwitchGroupMissingNode    = "switch_group_missing_node"
	RuleSwitchGroupNeedsNodes     = "switch_group_needs_two_nodes"
	RuleSwitchGroupNeedsRatio     = "switch_group_needs_oversubscription"
	RuleOversubscriptionNoGroups  = "oversubscription_without_groups"
	RuleOversubscriptionNonFinite = "oversubscription_not_finite"
	RuleOversubscriptionNonpos    = "oversubscription_nonpositive"
	RuleOversubscriptionBelowOne  = "oversubscription_below_one"
)

// Who owns a cluster's switch groups (design §2.2, SUB-3).
//
// A switch group is the set of nodes under one leaf switch. The scenario's cluster states
// which nodes sit under which switch and how oversubscribed each group's uplink is: that is
// one cluster's wiring, not a reusable fabric class, so the catalog's network entry cannot
// carry it. Switching is the one function that reads the two fields (X15); the kernel and
// the Result's inputs read what it returns.

// Switching is a cluster's resolved switch groups.
type Switching struct {
	// Groups lists the nodes of each group, as the scenario stated them.
	Groups [][]int
	// Oversubscription is the ratio r, at least 1.
	Oversubscription float64

	groupOf map[int]int
}

// Switching returns the cluster's switch groups, or nil when the scenario states none. It
// refuses groups that Validate refuses, so a caller that holds a non-nil result holds a
// partition of the nodes.
func (c Cluster) Switching() (*Switching, error) {
	p := &validate.Problems{}
	c.validateSwitching(p)
	if !p.OK() {
		return nil, p
	}
	if len(c.SwitchGroups) == 0 {
		return nil, nil
	}
	sw := &Switching{Oversubscription: c.UplinkOversubscription, groupOf: map[int]int{}}
	for g, nodes := range c.SwitchGroups {
		sw.Groups = append(sw.Groups, append([]int(nil), nodes...))
		for _, n := range nodes {
			sw.groupOf[n] = g
		}
	}
	return sw, nil
}

// GroupOf returns the group a node sits in. A node outside the partition reports false.
func (s *Switching) GroupOf(node int) (int, bool) {
	if s == nil {
		return 0, false
	}
	g, ok := s.groupOf[node]
	return g, ok
}

// GroupNodes returns how many nodes group g holds.
func (s *Switching) GroupNodes(g int) int { return len(s.Groups[g]) }

// validateSwitching records every way the stated groups fail to partition the cluster's
// nodes, and every way the ratio is unusable. It needs no catalog entry.
func (c Cluster) validateSwitching(p *validate.Problems) {
	if len(c.SwitchGroups) == 0 {
		if c.UplinkOversubscription != 0 {
			refuse(p, "uplink_oversubscription", RuleOversubscriptionNoGroups,
				"is %v, but the cluster states no switch_groups: there is no uplink to oversubscribe",
				c.UplinkOversubscription)
		}
		return
	}
	if c.Nodes < 2 {
		refuse(p, "switch_groups", RuleSwitchGroupNeedsNodes,
			"a cluster of %d node has no uplink between groups: state switch_groups for two or more nodes",
			c.Nodes)
	}
	owner := map[int]int{}
	for g, nodes := range c.SwitchGroups {
		at := fmt.Sprintf("switch_groups[%d]", g)
		if len(nodes) == 0 {
			refuse(p, at, RuleSwitchGroupEmpty, "lists no node: a group holds at least one")
		}
		for _, n := range nodes {
			switch prev, seen := owner[n]; {
			case n < 0 || n >= c.Nodes:
				refuse(p, at, RuleSwitchGroupUnknownNode,
					"names node %d, but the cluster has nodes 0 to %d", n, c.Nodes-1)
			case seen:
				refuse(p, at, RuleSwitchGroupOverlap,
					"names node %d, which switch_groups[%d] already holds: a node sits under one switch",
					n, prev)
			default:
				owner[n] = g
			}
		}
	}
	for n := 0; n < c.Nodes; n++ {
		if _, ok := owner[n]; !ok {
			refuse(p, "switch_groups", RuleSwitchGroupMissingNode,
				"leaves node %d under no switch: the groups partition all %d nodes", n, c.Nodes)
			break
		}
	}
	r := c.UplinkOversubscription
	switch {
	case r == 0:
		refuse(p, "uplink_oversubscription", RuleSwitchGroupNeedsRatio,
			"required with switch_groups: state how oversubscribed each group's uplink is (at least 1)")
	case math.IsNaN(r) || math.IsInf(r, 0):
		refuse(p, "uplink_oversubscription", RuleOversubscriptionNonFinite,
			"must be a finite number, got %v", r)
	case r < 0:
		refuse(p, "uplink_oversubscription", RuleOversubscriptionNonpos,
			"must be positive, got %v", r)
	case r < 1:
		refuse(p, "uplink_oversubscription", RuleOversubscriptionBelowOne,
			"is %v: an uplink faster than the group's NICs is not oversubscribed; state at least 1", r)
	}
}

// refuse records a problem under a rule at a field path. validate.Problems.At returns a
// detached collector, so the path is joined here and merged back.
func refuse(p *validate.Problems, path, rule, format string, args ...any) {
	one := &validate.Problems{}
	one.RuleErrorf(rule, format, args...)
	p.Merge(path, one)
}
