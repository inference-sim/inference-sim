package model

import "fmt"

// ArchOverride restates facts about the SERVED model that differ by GPU architecture:
// the width a weight is resident at, and the size of a cache record. A graph is
// otherwise architecture-independent, and these are the exceptions the engine forces.
//
// DeepSeek-V4.1-Flash on the engine is the case that needs them. Its attention output
// projection wo_a is an MXFP8 batched matmul, and DeepGEMM's MXFP8 BMM "requires
// Blackwell", so on SM90 the emulation path dequantizes it to BF16 at load and it is
// resident at twice its checkpoint width (model_executor/kernels/linear/mxfp8/
// deep_gemm.py, emulation.py). Its KV record is 584 B with 576 B page alignment on SM90
// and 528 B with 512 B alignment on SM100 (models/deepseek_v41/attention.py,
// _use_v41_mxfp8_kv_record). Both are properties of what is resident, which is why they
// live with the graph rather than with an engine rule.
//
// What an override does NOT carry is kernel selection and launch fusion -- SM100's mega
// attention fusing Q RoPE, attention and the inverse RoPE into one launch, or its mHC
// all-reduce fusion at TP 4. Those change the cost of nodes the graph already states
// rather than what is resident, and they turn on the layout and on the build as well as
// the chip, so they are the kernel's to resolve with its coefficients.
//
// The base graph describes the architecture the catalog prices by default; each
// override names the capability family it applies to.
type ArchOverride struct {
	// CapabilityFamily is the CUDA compute-capability major version the override applies
	// to: 10 for SM100 and SM103 (B200, GB200, B300, GB300), 9 for SM90 (H100, H200).
	// It matches the engine's is_device_capability_family(100), which compares major versions
	// and so excludes SM120.
	CapabilityFamily int `yaml:"capability_family"`
	// Nodes restates per-node weight facts, by role.
	Nodes []NodeOverride `yaml:"nodes,omitempty"`
	// Caches restates per-cache record facts, by cache name.
	Caches []CacheOverride `yaml:"caches,omitempty"`
}

// NodeOverride restates the resident weight format of every node with Role, wherever in
// the graph it appears. A field left empty keeps the base graph's value.
type NodeOverride struct {
	Role        string       `yaml:"role"`
	WeightDType DType        `yaml:"weight_dtype,omitempty"`
	WeightScale *WeightScale `yaml:"weight_scale,omitempty"`
}

// CacheOverride restates the record of every cache named Name. A field left zero keeps
// the base graph's value; BytesByCacheDType entries are merged over the base's.
// BytesPerEntry and BytesByCacheDType are restated on readers too, since a reader states
// the entry it reads and must keep matching its owner; PageAlignment describes how the
// owner pages the cache and reaches the owners only.
type CacheOverride struct {
	Name              string         `yaml:"name"`
	BytesPerEntry     int            `yaml:"bytes_per_entry,omitempty"`
	PageAlignment     int            `yaml:"page_alignment,omitempty"`
	BytesByCacheDType map[string]int `yaml:"bytes_by_cache_dtype,omitempty"`
}

// ForCapability returns the graph as served on a chip of the given compute-capability
// major version: a copy with the matching override applied, no overrides left, and that
// family as its only base. A base family gets the base graph. The receiver is not
// modified.
//
// A graph that distinguishes architectures (it states overrides or base families)
// resolves only for a family it describes: a base family gets the base graph, an
// override family its override, and any other family -- including the zero a chip with
// no ComputeCapability yields -- is an error rather than the base's facts, which would
// give a B200 SM90 records or an SM120 part a cache layout it does not run. A graph that
// states neither is the same on every chip and resolves for any major version.
//
// It is the one implementation of the overlay, so the catalog, the kernel and a
// simulator cannot apply it three different ways.
func (g *Graph) ForCapability(major int) (*Graph, error) {
	if len(g.ArchOverrides) > 0 || len(g.BaseCapabilityFamilies) > 0 {
		described := false
		for _, f := range g.BaseCapabilityFamilies {
			described = described || f == major
		}
		for _, o := range g.ArchOverrides {
			described = described || o.CapabilityFamily == major
		}
		if !described {
			return nil, fmt.Errorf("the graph describes compute-capability families %v as its base and %v by override, not %d",
				g.BaseCapabilityFamilies, overrideFamilies(g), major)
		}
	}
	out := g.clone()
	out.ArchOverrides = nil
	// The result is the graph as served on this family alone, so it says so: resolving
	// it again for the same family is the identity, and for any other is refused.
	if len(g.ArchOverrides) > 0 || len(g.BaseCapabilityFamilies) > 0 {
		out.BaseCapabilityFamilies = []int{major}
	}
	for _, o := range g.ArchOverrides {
		if o.CapabilityFamily != major {
			continue
		}
		if err := out.apply(o); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (g *Graph) apply(o ArchOverride) error {
	for _, no := range o.Nodes {
		hit := false
		g.eachNodePtr(func(n *Node) {
			if n.Role != no.Role {
				return
			}
			hit = true
			if no.WeightDType != "" {
				n.WeightDType = no.WeightDType
			}
			if no.WeightScale != nil {
				s := *no.WeightScale
				n.WeightScale = &s
			}
		})
		if !hit {
			return fmt.Errorf("capability family %d: no node has role %q", o.CapabilityFamily, no.Role)
		}
	}
	for _, co := range o.Caches {
		hit := false
		g.eachNodePtr(func(n *Node) {
			for i := range n.Caches {
				c := &n.Caches[i]
				if c.Name != co.Name {
					continue
				}
				hit = true
				if co.BytesPerEntry != 0 {
					c.BytesPerEntry = co.BytesPerEntry
				}
				if co.PageAlignment != 0 && c.Owned {
					c.PageAlignment = co.PageAlignment
				}
				if len(co.BytesByCacheDType) > 0 {
					m := make(map[string]int, len(c.BytesByCacheDType)+len(co.BytesByCacheDType))
					for k, v := range c.BytesByCacheDType {
						m[k] = v
					}
					for k, v := range co.BytesByCacheDType {
						m[k] = v
					}
					c.BytesByCacheDType = m
				}
			}
		})
		if !hit {
			return fmt.Errorf("capability family %d: no cache is named %q", o.CapabilityFamily, co.Name)
		}
	}
	return nil
}

// clone copies the graph deeply: every slice, map and pointer it holds, so no write to
// the copy -- by an override or by a caller -- reaches the original.
func (g *Graph) clone() *Graph {
	out := *g
	cloneNodes := func(ns []Node) []Node {
		if ns == nil {
			return nil
		}
		cp := make([]Node, len(ns))
		for i, n := range ns {
			if n.Caches != nil {
				n.Caches = append([]Cache(nil), n.Caches...)
				for j := range n.Caches {
					if m := n.Caches[j].BytesByCacheDType; m != nil {
						c := make(map[string]int, len(m))
						for k, v := range m {
							c[k] = v
						}
						n.Caches[j].BytesByCacheDType = c
					}
				}
			}
			if n.WeightScale != nil {
				s := *n.WeightScale
				n.WeightScale = &s
			}
			if n.Context != nil {
				c := *n.Context
				n.Context = &c
			}
			if n.ShardedAtDecode != nil {
				d := *n.ShardedAtDecode
				n.ShardedAtDecode = &d
			}
			cp[i] = n
		}
		return cp
	}
	cloneStack := func(s Stack) Stack {
		s.Prologue = append([]string(nil), s.Prologue...)
		s.Pattern = append([]string(nil), s.Pattern...)
		s.Epilogue = append([]string(nil), s.Epilogue...)
		return s
	}
	out.Stack = cloneStack(g.Stack)
	out.LayerKinds = make([]LayerKind, len(g.LayerKinds))
	for i, lk := range g.LayerKinds {
		lk.Nodes = cloneNodes(lk.Nodes)
		lk.Edges = append([][2]int(nil), lk.Edges...)
		out.LayerKinds[i] = lk
	}
	out.Embedding = cloneNodes(g.Embedding)
	out.BaseCapabilityFamilies = append([]int(nil), g.BaseCapabilityFamilies...)
	out.Head = cloneNodes(g.Head)
	if g.Speculator != nil {
		s := *g.Speculator
		s.Stack = cloneStack(s.Stack)
		if s.ContextStack != nil {
			c := cloneStack(*s.ContextStack)
			s.ContextStack = &c
		}
		s.Embedding = cloneNodes(s.Embedding)
		s.Head = cloneNodes(s.Head)
		s.SequentialHead = cloneNodes(s.SequentialHead)
		s.AuxHiddenStateLayers = append([]int(nil), s.AuxHiddenStateLayers...)
		out.Speculator = &s
	}
	if g.Global.WeightScale != nil {
		s := *g.Global.WeightScale
		out.Global.WeightScale = &s
	}
	return &out
}

func overrideFamilies(g *Graph) []int {
	var out []int
	for _, o := range g.ArchOverrides {
		out = append(out, o.CapabilityFamily)
	}
	return out
}
