package model

import "fmt"

// EachNode calls fn for every node the graph declares, with its path in the document:
// the embedding, every layer kind's nodes, the head, and the speculator's embedding and
// heads. A layer
// kind is visited once however many times the stack uses it.
func (g *Graph) EachNode(fn func(at string, n Node)) {
	g.eachNode(func(at string, n *Node) { fn(at, *n) })
}

func (g *Graph) eachNodePtr(fn func(n *Node)) {
	g.eachNode(func(_ string, n *Node) { fn(n) })
}

func (g *Graph) eachNode(fn func(at string, n *Node)) {
	list := func(prefix string, ns []Node) {
		for i := range ns {
			fn(fmt.Sprintf("%s[%d]", prefix, i), &ns[i])
		}
	}
	list("embedding", g.Embedding)
	for i := range g.LayerKinds {
		list(fmt.Sprintf("layer_kinds[%d].nodes", i), g.LayerKinds[i].Nodes)
	}
	list("head", g.Head)
	if g.Speculator != nil {
		list("speculator.embedding", g.Speculator.Embedding)
		list("speculator.head", g.Speculator.Head)
		list("speculator.sequential_head", g.Speculator.SequentialHead)
	}
}
