package llmd

import (
	"container/list"

	"github.com/inference-sim/inference-sim/sim"
)

// canonMemo is what canonicalSession remembers of a session's last prompt: its tokens, the
// canonical form it returned, and for each block the chain key and the chain it named.
type canonMemo struct {
	session string
	root    string
	tokens  []sim.TokenID
	out     []uint32
	keys    []string
	chains  []*chain
	elem    *list.Element
}

// canonicalSession is canonicalUnder for the prompt of a session. A session's next prompt usually
// extends its last one, so the blocks the two share are not hashed again: their chain keys are
// kept, and so are the canonical tokens of a chain that has not been forgotten since. The result,
// and every change to the chain table and its LRU order, are exactly those of canonicalUnder:
// each block still records a use of its chain, in order. An empty session id uses no memo.
func (c *promptCanon) canonicalSession(root, session string, prompt []sim.TokenID) []uint32 {
	if session == "" || c.memoCap <= 0 || len(prompt) > c.memoCap {
		return c.canonicalUnder(root, prompt)
	}
	var m *canonMemo
	reuse := 0 // leading blocks whose tokens are those of the memo's prompt
	if old := c.memos[session]; old != nil {
		if old.root == root {
			m = old
			reuse = commonPrefix(old.tokens, prompt) / c.blockSize
		}
		c.dropMemo(old)
	}
	nblocks := (len(prompt) + c.blockSize - 1) / c.blockSize
	out := make([]uint32, len(prompt))
	keys := make([]string, nblocks)
	chains := make([]*chain, nblocks)
	key := root
	for b := 0; b < nblocks; b++ {
		start := b * c.blockSize
		end := min(start+c.blockSize, len(prompt))
		var ch *chain
		if b < reuse {
			key = m.keys[b]
			if old := m.chains[b]; !old.dead {
				c.touch(old)
				ch = old
				copy(out[start:end], m.out[start:end])
			} else {
				ch = c.lookup(key)
			}
		} else {
			key = hashBlock(key, prompt[start:end])
			ch = c.lookup(key)
		}
		if !(b < reuse && ch == m.chains[b]) {
			for i := start; i < end; i++ {
				out[i] = c.token(ch.id, i-start)
			}
		}
		keys[b], chains[b] = key, ch
	}
	nm := &canonMemo{session: session, root: root, tokens: append([]sim.TokenID(nil), prompt...), out: out, keys: keys, chains: chains}
	nm.elem = c.memoLRU.PushFront(nm)
	c.memos[session] = nm
	c.memoTokens += len(prompt)
	for c.memoTokens > c.memoCap {
		c.dropMemo(c.memoLRU.Back().Value.(*canonMemo))
	}
	return out
}

func (c *promptCanon) dropMemo(m *canonMemo) {
	c.memoLRU.Remove(m.elem)
	delete(c.memos, m.session)
	c.memoTokens -= len(m.tokens)
}

// commonPrefix is the length of the longest common prefix of a and b.
func commonPrefix(a, b []sim.TokenID) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
