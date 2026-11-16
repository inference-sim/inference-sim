// Original to blis-sglang.

package cache

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// A Trace is the calls SGLang's own classes received at the six framework interfaces during one
// oracle run, with SGLang's answer to each: the format oracle/cache_interfaces.py writes. Kind is
// "recorded" (the scheduler made the calls) or "direct" (the repository's driver did).
type Trace struct {
	Provenance map[string]any `json:"provenance"`
	Kind       string         `json:"kind"`
	Calls      []Call         `json:"calls"`
}

// A Call is one call into an interface from outside it.
type Call struct {
	// Seq is the call's place in the run, across every interface.
	Seq int `json:"seq"`
	// Iface is the interface and Class the SGLang class that answered.
	Iface  Interface `json:"i"`
	Class  string    `json:"cls"`
	Method string    `json:"m"`
	// Args is each argument by its SGLang parameter name; Ret is the answer, absent when the
	// call raised Raise.
	Args  map[string]json.RawMessage `json:"args"`
	Ret   json.RawMessage            `json:"ret"`
	Raise string                     `json:"raise"`
	// In is the Seq of the call, at another interface, that made this one, or nil for a call
	// the scheduler or the driver made.
	In *int `json:"in"`
	// ReqAfter is each request argument as it stood when the call returned.
	ReqAfter []json.RawMessage `json:"req_after"`
}

// GoldenDir is where the framework's traces are committed.
const GoldenDir = "testdata/golden-fid5"

// traceDirs is each trace directory under GoldenDir and the kind of trace it holds: recorded
// holds the tier-2 scenarios of oracle/scenarios, t2 the cache's own scheduler scenarios of
// oracle/scenarios/cache, direct the driver's sequences of oracle/scenarios-cache-iface.
var traceDirs = [][2]string{{"recorded", "recorded"}, {"t2", "recorded"}, {"direct", "direct"}}

// LoadTraces reads every trace under dir's trace directories, keyed "<directory>/<name>".
func LoadTraces(dir string) (map[string]Trace, error) {
	out := map[string]Trace{}
	for _, d := range traceDirs {
		sub, kind := d[0], d[1]
		paths, err := filepath.Glob(filepath.Join(dir, sub, "*.trace.json"))
		if err != nil {
			return nil, err
		}
		for _, p := range paths {
			raw, err := os.ReadFile(p)
			if err != nil {
				return nil, err
			}
			var tr Trace
			if err := json.Unmarshal(raw, &tr); err != nil {
				return nil, fmt.Errorf("%s: %w", p, err)
			}
			if tr.Kind != kind {
				return nil, fmt.Errorf("%s: a %q trace in the %s directory", p, tr.Kind, sub)
			}
			out[sub+"/"+strings.TrimSuffix(filepath.Base(p), ".trace.json")] = tr
		}
	}
	return out, nil
}

// Top returns the calls of iface that the scheduler or the driver made itself, in order: the
// calls a port of that interface is driven with.
func (t Trace) Top(iface Interface) []Call {
	var out []Call
	for _, c := range t.Calls {
		if c.Iface == iface && c.In == nil {
			out = append(out, c)
		}
	}
	return out
}

// Within returns the calls at iface that the call with sequence number seq made, directly: what
// a port answering that call must itself call, in order.
func (t Trace) Within(seq int, iface Interface) []Call {
	var out []Call
	for _, c := range t.Calls {
		if c.Iface == iface && c.In != nil && *c.In == seq {
			out = append(out, c)
		}
	}
	return out
}

// Arg decodes argument name of c into v.
func (c Call) Arg(name string, v any) error {
	raw, ok := c.Args[name]
	if !ok {
		return fmt.Errorf("%s.%s (seq %d): no argument %q", c.Iface, c.Method, c.Seq, name)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("%s.%s (seq %d) argument %s: %w", c.Iface, c.Method, c.Seq, name, err)
	}
	return nil
}

// Answer decodes SGLang's answer to c into v.
func (c Call) Answer(v any) error {
	if c.Raise != "" {
		return fmt.Errorf("%s.%s (seq %d) raised %s", c.Iface, c.Method, c.Seq, c.Raise)
	}
	if err := json.Unmarshal(c.Ret, v); err != nil {
		return fmt.Errorf("%s.%s (seq %d) answer: %w", c.Iface, c.Method, c.Seq, err)
	}
	return nil
}

// Nodes pairs SGLang's node ids with a port's as a replay meets them. Two trees agree when one
// consistent pairing maps every node either side names onto the other's: SGLang numbers nodes
// from a counter all its trees share, so the ids themselves differ.
type Nodes struct {
	toPort, toSGLang map[NodeID]NodeID
}

// NewNodes returns an empty pairing.
func NewNodes() *Nodes {
	return &Nodes{toPort: map[NodeID]NodeID{}, toSGLang: map[NodeID]NodeID{}}
}

// Pair records that SGLang's node s is the port's node p, and fails if either is already paired
// with another node.
func (n *Nodes) Pair(s, p NodeID) error {
	if q, ok := n.toPort[s]; ok && q != p {
		return fmt.Errorf("SGLang's node %d is the port's node %d, and now %d", s, q, p)
	}
	if q, ok := n.toSGLang[p]; ok && q != s {
		return fmt.Errorf("the port's node %d is SGLang's node %d, and now %d", p, q, s)
	}
	n.toPort[s], n.toSGLang[p] = p, s
	return nil
}

// Port is the port's node for SGLang's node s, which an earlier answer must have paired.
func (n *Nodes) Port(s NodeID) (NodeID, error) {
	p, ok := n.toPort[s]
	if !ok {
		return 0, fmt.Errorf("SGLang's node %d was never named in an answer", s)
	}
	return p, nil
}

// Methods returns the SGLang method names a trace calls at each interface, sorted.
func (t Trace) Methods() map[Interface][]string {
	seen := map[Interface]map[string]bool{}
	for _, c := range t.Calls {
		if seen[c.Iface] == nil {
			seen[c.Iface] = map[string]bool{}
		}
		seen[c.Iface][c.Method] = true
	}
	out := map[Interface][]string{}
	for i, ms := range seen {
		for m := range ms {
			out[i] = append(out[i], m)
		}
		sort.Strings(out[i])
	}
	return out
}

// GoName is the Go method that stands for SGLang method m: its snake_case name in CamelCase,
// with SGLang's abbreviations (kv, swa, lru, ...) upper case, and __init__ as New.
func GoName(m string) string {
	if m == "__init__" {
		return "New"
	}
	var b strings.Builder
	for _, part := range strings.Split(m, "_") {
		if part == "" {
			continue
		}
		if up, ok := initialisms[part]; ok {
			b.WriteString(up)
			continue
		}
		b.WriteString(strings.ToUpper(part[:1]) + part[1:])
	}
	return b.String()
}

var initialisms = map[string]string{"kv": "KV", "swa": "SWA", "lru": "LRU", "id": "ID", "ids": "IDs", "kvcache": "KVCache"}

// RowWrite is a ReqToTokenPool.write call (memory_pool.py:414-415@297e54fd) as the positions it
// sets: each (row, position) with its slot. SGLang indexes the pool two ways, (row, slice) for a
// run of one row and ([rows], [positions]) for one slot in each of several rows.
func (c Call) RowWrite() ([]RowSlot, error) {
	var idx []json.RawMessage
	if err := c.Arg("indices", &idx); err != nil {
		return nil, err
	}
	var values []int64
	if err := c.Arg("values", &values); err != nil {
		return nil, err
	}
	if len(idx) != 2 {
		return nil, fmt.Errorf("write (seq %d): %d indices, want 2", c.Seq, len(idx))
	}
	var row int
	if err := json.Unmarshal(idx[0], &row); err == nil {
		var s struct {
			T     string `json:"_t"`
			Start *int   `json:"start"`
			Stop  *int   `json:"stop"`
			Step  *int   `json:"step"`
		}
		if err := json.Unmarshal(idx[1], &s); err != nil || s.T != "slice" || s.Start == nil || s.Stop == nil ||
			(s.Step != nil && *s.Step != 1) {
			return nil, fmt.Errorf("write (seq %d): the second index is not a bounded slice: %s", c.Seq, idx[1])
		}
		if *s.Stop-*s.Start != len(values) {
			return nil, fmt.Errorf("write (seq %d): slice [%d:%d) of %d values", c.Seq, *s.Start, *s.Stop, len(values))
		}
		out := make([]RowSlot, len(values))
		for i, v := range values {
			out[i] = RowSlot{Row: row, Pos: *s.Start + i, Slot: v}
		}
		return out, nil
	}
	var rows, pos []int
	if err := json.Unmarshal(idx[0], &rows); err != nil {
		return nil, fmt.Errorf("write (seq %d): rows: %w", c.Seq, err)
	}
	if err := json.Unmarshal(idx[1], &pos); err != nil {
		return nil, fmt.Errorf("write (seq %d): positions: %w", c.Seq, err)
	}
	if len(rows) != len(pos) || len(pos) != len(values) {
		return nil, fmt.Errorf("write (seq %d): %d rows, %d positions, %d values", c.Seq, len(rows), len(pos), len(values))
	}
	out := make([]RowSlot, len(values))
	for i, v := range values {
		out[i] = RowSlot{Row: rows[i], Pos: pos[i], Slot: v}
	}
	return out, nil
}

// RowSlot is one position of a request's row and the slot written there.
type RowSlot struct {
	Row, Pos int
	Slot     int64
}
