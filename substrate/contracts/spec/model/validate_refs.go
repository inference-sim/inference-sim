package model

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// resolveAlias finds the one node an aliases reference names, with its path, or says why
// it names none. It is the one resolver, so validation and pricing cannot disagree about
// which weights an alias reads.
func (g *Graph) resolveAlias(ref string) (Node, string, error) {
	// Split at the last dot, so a layer-kind id may contain dots and a role may not.
	i := strings.LastIndex(ref, ".")
	if i <= 0 || i == len(ref)-1 {
		return Node{}, "", fmt.Errorf("%q must be <layer kind>.<role>, head.<role> or embedding.<role>", ref)
	}
	scope, role := ref[:i], ref[i+1:]
	li := -1
	for k, lk := range g.LayerKinds {
		if lk.ID == scope {
			li = k
		}
	}
	var pool []Node
	var poolAt string
	switch {
	case li >= 0 && (scope == "head" || scope == "embedding"):
		return Node{}, "", fmt.Errorf("%q is ambiguous: a layer kind is also named %q", ref, scope)
	case scope == "head":
		pool, poolAt = g.Head, "head"
	case scope == "embedding":
		pool, poolAt = g.Embedding, "embedding"
	case li >= 0:
		// A layer kind instantiated more than once holds one copy of its weights per
		// instance, so "the node in that kind" names no single weight.
		if c := g.instances()[scope]; c != 1 {
			return Node{}, "", fmt.Errorf("layer kind %q runs %d times, so it holds no single weight to alias", scope, c)
		}
		pool, poolAt = g.LayerKinds[li].Nodes, fmt.Sprintf("layer_kinds[%d].nodes", li)
	default:
		return Node{}, "", fmt.Errorf("%q names no layer kind, and is neither head nor embedding", scope)
	}
	var hits []int
	for k, m := range pool {
		if m.Role == role {
			hits = append(hits, k)
		}
	}
	if len(hits) != 1 {
		return Node{}, "", fmt.Errorf("%q matches %d nodes; it must match exactly one", ref, len(hits))
	}
	return pool[hits[0]], fmt.Sprintf("%s[%d]", poolAt, hits[0]), nil
}

// instances counts how many times each layer kind runs across every stack.
func (g *Graph) instances() map[string]int {
	out := map[string]int{}
	stacks := []Stack{g.Stack}
	if s := g.Speculator; s != nil {
		stacks = append(stacks, s.Stack)
		if s.ContextStack != nil {
			stacks = append(stacks, *s.ContextStack)
		}
	}
	for _, st := range stacks {
		for _, id := range st.Expand() {
			out[id]++
		}
	}
	return out
}

// validateAliases resolves every aliases reference to exactly one weight-holding node of
// the same shape. An alias that resolved to nothing, to several nodes, or to a node that
// is itself an alias would leave the residency it claims is counted elsewhere counted
// nowhere, or twice. An alias reads its target's weights in the target's format, so it
// states no format of its own: one that did could only repeat the target's or contradict it.
func (g *Graph) validateAliases(p *validate.Problems) {
	g.EachNode(func(at string, n Node) {
		if n.Aliases == "" {
			return
		}
		if n.WeightDType != "" || n.WeightScale != nil {
			p.Field(at+".aliases", "an alias reads its target's weights in the target's format; remove its weight_dtype and weight_scale")
		}
		target, targetAt, err := g.resolveAlias(n.Aliases)
		switch {
		case err != nil:
			p.Field(at+".aliases", "%v", err)
		case targetAt == at:
			p.Field(at+".aliases", "a node cannot alias itself")
		case target.Aliases != "":
			p.Field(at+".aliases", "%s is itself an alias; name the node that holds the weights", targetAt)
		case !sameWeights(n, target):
			p.Field(at+".aliases", "%s holds a %s; it cannot be the weights of this %s", targetAt, weightShape(target), weightShape(n))
		}
	})
}

// sameWeights reports whether two nodes can read one weight tensor: two GEMMs of equal n
// and k, two tables of equal rows and row width, or a GEMM over a table's rows, which is
// a tied LM head reading its embedding. The last compares the vocabulary only: a row's
// bytes are k times an element width the graph does not state for a table.
func sameWeights(a, b Node) bool {
	switch {
	case a.Op == OpGEMM && b.Op == OpGEMM:
		return a.N == b.N && a.K == b.K
	case a.Op == OpGather && b.Op == OpGather:
		return a.Rows == b.Rows && a.RowBytes == b.RowBytes
	case a.Op == OpGEMM && b.Op == OpGather:
		return int64(a.N) == b.Rows
	case a.Op == OpGather && b.Op == OpGEMM:
		return a.Rows == int64(b.N)
	}
	return false
}

func weightShape(n Node) string {
	switch n.Op {
	case OpGEMM:
		return fmt.Sprintf("GEMM of %d x %d", n.N, n.K)
	case OpGather:
		return fmt.Sprintf("table of %d rows of %d B", n.Rows, n.RowBytes)
	}
	return string(n.Op) + " node"
}

// validateArchOverrides checks each override is well formed and that every role and
// cache it names exists. An override naming nothing would read as a statement about the
// served model that changes nothing. Each miss is reported at its own entry, so a graph
// with several reports all of them.
func (g *Graph) validateArchOverrides(p *validate.Problems) {
	if len(g.ArchOverrides) > 0 && len(g.BaseCapabilityFamilies) == 0 {
		p.Field("base_capability_families", "required with arch_overrides: say which families the base graph describes")
	}
	base := map[int]bool{}
	for i, f := range g.BaseCapabilityFamilies {
		at := fmt.Sprintf("base_capability_families[%d]", i)
		switch {
		case f < 1:
			p.Field(at, "must be a positive compute-capability major version, got %d", f)
		case base[f]:
			p.Field(at, "family %d is listed twice", f)
		}
		base[f] = true
	}
	seen := map[int]bool{}
	for i, o := range g.ArchOverrides {
		at := fmt.Sprintf("arch_overrides[%d]", i)
		if o.CapabilityFamily < 1 {
			p.Field(at+".capability_family", "must be a positive compute-capability major version, got %d", o.CapabilityFamily)
		} else if seen[o.CapabilityFamily] {
			p.Field(at+".capability_family", "family %d is overridden twice", o.CapabilityFamily)
		} else if base[o.CapabilityFamily] {
			p.Field(at+".capability_family", "family %d is listed as the base; it cannot also be overridden", o.CapabilityFamily)
		}
		seen[o.CapabilityFamily] = true
		if len(o.Nodes) == 0 && len(o.Caches) == 0 {
			p.Field(at, "overrides nothing; remove it")
		}
		for j, no := range o.Nodes {
			nat := fmt.Sprintf("%s.nodes[%d]", at, j)
			if no.Role == "" {
				p.Field(nat+".role", "required")
			}
			if no.WeightDType == "" && no.WeightScale == nil {
				p.Field(nat, "restates nothing; give a weight_dtype or a weight_scale")
			}
			if no.WeightDType != "" && !no.WeightDType.Valid() {
				p.Field(nat+".weight_dtype", "%q is not a recognized dtype", no.WeightDType)
			}
			validateWeightScale(p, nat+".weight_scale", no.WeightScale)
			matched := false
			g.EachNode(func(nodeAt string, n Node) {
				if n.Role != no.Role {
					return
				}
				matched = true
				if n.Op != OpGEMM && n.Op != OpGroupedGEMM {
					p.Field(nat+".role", "%q matches %s, a %s, which holds no parameters", no.Role, nodeAt, n.Op)
				}
			})
			if no.Role != "" && !matched {
				p.Field(nat+".role", "no node has role %q", no.Role)
			}
		}
		for j, co := range o.Caches {
			cat := fmt.Sprintf("%s.caches[%d]", at, j)
			if co.Name == "" {
				p.Field(cat+".name", "required")
			}
			if co.BytesPerEntry < 0 || co.PageAlignment < 0 {
				p.Field(cat, "sizes must not be negative")
			}
			if co.BytesPerEntry == 0 && co.PageAlignment == 0 && len(co.BytesByCacheDType) == 0 {
				p.Field(cat, "restates nothing; give a size")
			}
			for _, k := range sortedKeys(co.BytesByCacheDType) {
				if k == "" {
					p.Field(cat+".bytes_by_cache_dtype", "a cache dtype name is empty")
				} else if v := co.BytesByCacheDType[k]; v < 1 {
					p.Field(cat+".bytes_by_cache_dtype."+k, "must be positive, got %d", v)
				}
			}
			matched := false
			g.EachNode(func(_ string, n Node) {
				for _, c := range n.Caches {
					if c.Name == co.Name {
						matched = true
					}
				}
			})
			if co.Name != "" && !matched {
				p.Field(cat+".name", "no cache is named %q", co.Name)
			}
		}
	}
}
