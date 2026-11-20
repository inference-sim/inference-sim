package scenario

import (
	"strings"
	"testing"
)

func grouped(groups [][]int, r float64) *Scenario {
	s := multiNode()
	s.Cluster.Nodes = 6
	s.Cluster.SwitchGroups = groups
	s.Cluster.UplinkOversubscription = r
	return s
}

func rulesOf(s *Scenario) map[string]string {
	got := map[string]string{}
	for _, pr := range s.Validate().Errors() {
		got[pr.Rule] = pr.Path
	}
	return got
}

func TestSwitchGroupsAcceptAPartition(t *testing.T) {
	s := grouped([][]int{{0, 1, 2}, {3, 4}, {5}}, 3.4)
	if p := s.Validate(); !p.OK() {
		t.Fatalf("a partition was refused:\n%s", p.Error())
	}
	sw, err := s.Cluster.Switching()
	if err != nil {
		t.Fatal(err)
	}
	for node, want := range map[int]int{0: 0, 2: 0, 3: 1, 4: 1, 5: 2} {
		if g, ok := sw.GroupOf(node); !ok || g != want {
			t.Errorf("node %d: group %d (found %v), want %d", node, g, ok, want)
		}
	}
	if sw.GroupNodes(1) != 2 || sw.Oversubscription != 3.4 {
		t.Errorf("group 1 holds %d nodes at ratio %v, want 2 at 3.4", sw.GroupNodes(1), sw.Oversubscription)
	}
}

func TestNoGroupsResolveToNothing(t *testing.T) {
	sw, err := multiNode().Cluster.Switching()
	if sw != nil || err != nil {
		t.Fatalf("a cluster with no groups resolved to %v, %v", sw, err)
	}
	if _, ok := sw.GroupOf(0); ok {
		t.Error("a nil Switching placed node 0 in a group")
	}
}

// Each way the groups or the ratio can be wrong is refused under its own rule, at the
// field that is wrong.
func TestSwitchGroupsRefusals(t *testing.T) {
	cases := []struct {
		name string
		s    *Scenario
		rule string
		path string
	}{
		{"overlap", grouped([][]int{{0, 1, 2}, {2, 3, 4, 5}}, 2), RuleSwitchGroupOverlap, "switch_groups[1]"},
		{"unknown node", grouped([][]int{{0, 1, 2}, {3, 4, 9}}, 2), RuleSwitchGroupUnknownNode, "switch_groups[1]"},
		{"negative node", grouped([][]int{{0, 1, 2, -1}, {3, 4, 5}}, 2), RuleSwitchGroupUnknownNode, "switch_groups[0]"},
		{"missing node", grouped([][]int{{0, 1, 2}, {3, 4}}, 2), RuleSwitchGroupMissingNode, "switch_groups"},
		{"empty group", grouped([][]int{{0, 1, 2}, {}, {3, 4, 5}}, 2), RuleSwitchGroupEmpty, "switch_groups[1]"},
		{"zero ratio", grouped([][]int{{0, 1, 2}, {3, 4, 5}}, 0), RuleSwitchGroupNeedsRatio, "uplink_oversubscription"},
		{"negative ratio", grouped([][]int{{0, 1, 2}, {3, 4, 5}}, -2), RuleOversubscriptionNonpos, "uplink_oversubscription"},
		{"ratio below one", grouped([][]int{{0, 1, 2}, {3, 4, 5}}, 0.5), RuleOversubscriptionBelowOne, "uplink_oversubscription"},
		{"ratio without groups", func() *Scenario { s := multiNode(); s.Cluster.UplinkOversubscription = 3; return s }(),
			RuleOversubscriptionNoGroups, "uplink_oversubscription"},
		{"groups on one node", func() *Scenario {
			s := grouped([][]int{{0}}, 2)
			s.Cluster.Nodes = 1
			return s
		}(), RuleSwitchGroupNeedsNodes, "switch_groups"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := rulesOf(tc.s)
			if path, ok := got[tc.rule]; !ok || !strings.HasSuffix(path, tc.path) {
				t.Errorf("want rule %q at %q, got %v", tc.rule, tc.path, got)
			}
			if _, err := tc.s.Cluster.Switching(); err == nil {
				t.Error("Switching resolved groups that Validate refuses")
			}
		})
	}
}
