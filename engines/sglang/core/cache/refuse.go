// Original to blis-sglang.

package cache

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// The allocators, KV pools and request pools of SGLang v0.5.23 (297e54fd) and which the port
// models (SGL-6). Every class below subclasses BaseTokenToKVPoolAllocator (allocator/base.py:41),
// KVCache (memory_pool.py:2005) or ReqToTokenPool (memory_pool.py:376), or stands in for one
// (DecodeReqToTokenPool, disaggregation/decode.py:171, borrows ReqToTokenPool's methods; MambaPool,
// memory_pool.py:505, is the request-indexed state pool). A configuration that selects any other is
// refused by the class's name and the setting that selects it, citing the open issue that tracks
// it (X28); config resolution (S08) makes the check at start-up. The HiCache host pools are HiCache's (inference-sim/blis-sglang#38).

// RefusedPoolsIssue is the open issue every refused allocator and pool cites.
const RefusedPoolsIssue = "inference-sim/blis-sglang#44"

// PoolKind is which framework interface a class implements.
type PoolKind string

const (
	AllocatorKind PoolKind = "allocator"
	KVPoolKind    PoolKind = "kv_pool"
	ReqPoolKind   PoolKind = "req_to_token"
)

// poolClassesJSON is every class above, each with its status, where SGLang defines it and, for a
// refused one, the setting or model that selects it. oracle/check_pool_classes.py checks it
// against SGLang's source at the pin: every subclass of the three base classes is listed, and each
// listed class is defined where it says.
//
//go:embed pool_classes.json
var poolClassesJSON []byte

type poolClassEntry struct {
	Class      string   `json:"class"`
	Kind       PoolKind `json:"kind"`
	Site       string   `json:"site"`
	Status     string   `json:"status"`
	Port       string   `json:"port"`
	SelectedBy string   `json:"selected_by"`
}

var poolClasses = func() map[string]poolClassEntry {
	var doc struct {
		Issue   string           `json:"issue"`
		Classes []poolClassEntry `json:"classes"`
	}
	if err := json.Unmarshal(poolClassesJSON, &doc); err != nil {
		panic("pool_classes.json: " + err.Error())
	}
	if doc.Issue != RefusedPoolsIssue {
		panic("pool_classes.json cites " + doc.Issue + ", not " + RefusedPoolsIssue)
	}
	out := map[string]poolClassEntry{}
	for _, c := range doc.Classes {
		if _, dup := out[c.Class]; dup || (c.Status != "modelled" && c.Status != "refused") {
			panic("pool_classes.json: class " + c.Class + " listed twice or with status " + c.Status)
		}
		out[c.Class] = c
	}
	return out
}()

// PoolRefusal is a start-up refusal of an allocator or pool class.
type PoolRefusal struct {
	Kind  PoolKind
	Class string
	// Site is where SGLang defines the class, or empty for a name that is no class of the release.
	Site string
	// SelectedBy is the setting or model that makes SGLang select the class, where known.
	SelectedBy string
	Issue      string
}

func (r *PoolRefusal) Error() string {
	if r.Site == "" {
		return fmt.Sprintf("%s %s is not an allocator or pool of SGLang v0.5.23", r.Kind, r.Class)
	}
	by := "the pool configurator's choice for this model and these settings"
	if r.SelectedBy != "" {
		by = r.SelectedBy
	}
	return fmt.Sprintf("%s %s (python/sglang/srt/%s@297e54fd), selected by %s, is not modelled; refused until it is (%s)",
		r.Kind, r.Class, r.Site, by, r.Issue)
}

// CheckPoolClass is nil when SGLang's class of the given kind is one the port models, and its
// refusal otherwise.
func CheckPoolClass(kind PoolKind, class string) error {
	c, ok := poolClasses[class]
	if ok && c.Kind == kind && c.Status == "modelled" {
		return nil
	}
	if ok && c.Kind == kind {
		return &PoolRefusal{Kind: kind, Class: class, Site: c.Site, SelectedBy: c.SelectedBy, Issue: RefusedPoolsIssue}
	}
	return &PoolRefusal{Kind: kind, Class: class}
}

// CheckPoolSelection checks the three classes a configuration selects, and refuses every one the
// port does not model, each by name. Config resolution (S08, inference-sim/blis-sglang#24) calls
// it at start-up with the classes SGLang's pool configurator would select.
func CheckPoolSelection(allocator, kvPool, reqPool string) error {
	var msgs []string
	for _, c := range []struct {
		kind  PoolKind
		class string
	}{{AllocatorKind, allocator}, {KVPoolKind, kvPool}, {ReqPoolKind, reqPool}} {
		if err := CheckPoolClass(c.kind, c.class); err != nil {
			msgs = append(msgs, err.Error())
		}
	}
	if len(msgs) == 0 {
		return nil
	}
	return fmt.Errorf("refused at start-up: %s", strings.Join(msgs, "; "))
}

// RefusedPoolClasses is every class refused by name, sorted.
func RefusedPoolClasses() []string {
	var out []string
	for name, c := range poolClasses {
		if c.Status == "refused" {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}
