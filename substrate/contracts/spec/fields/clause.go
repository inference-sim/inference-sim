package fields

import (
	"regexp"
	"strconv"
	"strings"
)

// Clause is one way a stand-in can be exact (design §11.2). It holds when every key it sets
// holds, so a clause is a conjunction. At least one key is set.
type Clause struct {
	// ValueIn: the recorded value equals one of the strings, ignoring case.
	ValueIn []string `json:"value_in,omitempty" yaml:"value_in,omitempty"`
	// ValueListHasAny: the recorded value, split on commas, has at least one of the strings
	// as an item.
	ValueListHasAny []string `json:"value_list_has_any,omitempty" yaml:"value_list_has_any,omitempty"`
	// ValueListHasNone: the recorded value, split on commas, has none of the strings as an item.
	ValueListHasNone []string `json:"value_list_has_none,omitempty" yaml:"value_list_has_none,omitempty"`
	// ValueNotPrefixed: the recorded value starts with none of the strings.
	ValueNotPrefixed []string `json:"value_not_prefixed,omitempty" yaml:"value_not_prefixed,omitempty"`
	// TPGroupNodes: the number of nodes the pool's tensor-parallel group spans satisfies
	// every comparison.
	TPGroupNodes *NodeComparison `json:"tp_group_nodes,omitempty" yaml:"tp_group_nodes,omitempty"`
	// ComputeCapability: the GPU part's compute capability satisfies every comparison.
	ComputeCapability *CapabilityComparison `json:"compute_capability,omitempty" yaml:"compute_capability,omitempty"`
}

// NodeComparison compares a node count. It holds when every comparison it sets holds; at
// least one is set.
type NodeComparison struct {
	Eq  *int `json:"eq,omitempty" yaml:"eq,omitempty"`
	Gt  *int `json:"gt,omitempty" yaml:"gt,omitempty"`
	Gte *int `json:"gte,omitempty" yaml:"gte,omitempty"`
	Lt  *int `json:"lt,omitempty" yaml:"lt,omitempty"`
	Lte *int `json:"lte,omitempty" yaml:"lte,omitempty"`
}

// CapabilityComparison compares a compute capability, each side a "major.minor" string such
// as "10.0". It holds when every comparison it sets holds; at least one is set.
type CapabilityComparison struct {
	Eq  *string `json:"eq,omitempty" yaml:"eq,omitempty"`
	Gt  *string `json:"gt,omitempty" yaml:"gt,omitempty"`
	Gte *string `json:"gte,omitempty" yaml:"gte,omitempty"`
	Lt  *string `json:"lt,omitempty" yaml:"lt,omitempty"`
	Lte *string `json:"lte,omitempty" yaml:"lte,omitempty"`
}

// capability is a compute capability as the pair the comparison orders.
type capability struct{ major, minor int }

func (a capability) cmp(b capability) int {
	switch {
	case a.major != b.major:
		return sign(a.major - b.major)
	default:
		return sign(a.minor - b.minor)
	}
}

func sign(x int) int {
	switch {
	case x < 0:
		return -1
	case x > 0:
		return 1
	}
	return 0
}

var capabilityForm = regexp.MustCompile(`^([0-9]+)\.([0-9]+)$`)

// parseCapability reads "major.minor". It refuses any other spelling, and a part too large
// for an int.
func parseCapability(s string) (capability, bool) {
	m := capabilityForm.FindStringSubmatch(strings.TrimSpace(s))
	if m == nil {
		return capability{}, false
	}
	major, err1 := strconv.Atoi(m[1])
	minor, err2 := strconv.Atoi(m[2])
	return capability{major, minor}, err1 == nil && err2 == nil
}
