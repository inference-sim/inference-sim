package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"sort"
	"strings"
)

// diffReport is what compare finds between two probe outputs.
type diffReport struct {
	Records int
	// Differs counts, per field, the records whose field differs; a field in orderOnly that holds
	// the same items in another order is counted under "<field> (order only)".
	Differs map[string]int
	// Example is one record (fixture and pool) for each entry of Differs.
	Example map[string]string
}

// failed reports whether anything differs other than the order of a field the caller allows to
// be reordered.
func (r diffReport) failed() bool {
	for k := range r.Differs {
		if !strings.HasSuffix(k, " (order only)") {
			return true
		}
	}
	return false
}

func readRecords(r io.Reader) (map[string]map[string]json.RawMessage, error) {
	out := map[string]map[string]json.RawMessage{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<30)
	for sc.Scan() {
		var rec map[string]json.RawMessage
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			return nil, err
		}
		key := string(rec["fixture"]) + "#" + string(rec["pool"])
		if _, dup := out[key]; dup {
			return nil, fmt.Errorf("record %s appears twice", key)
		}
		out[key] = rec
	}
	return out, sc.Err()
}

// sameItems reports whether two JSON values are equal once every list in them, at any depth, is
// read as a multiset: the same items, in any order.
func sameItems(a, b json.RawMessage) bool {
	var x, y interface{}
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	return reflect.DeepEqual(unordered(x), unordered(y))
}

// unordered rewrites every list in v as its items' canonical encodings, sorted.
func unordered(v interface{}) interface{} {
	switch t := v.(type) {
	case []interface{}:
		out := make([]string, len(t))
		for i, item := range t {
			c, _ := json.Marshal(unordered(item))
			out[i] = string(c)
		}
		sort.Strings(out)
		return out
	case map[string]interface{}:
		out := map[string]interface{}{}
		for k, item := range t {
			out[k] = unordered(item)
		}
		return out
	}
	return v
}

// equalJSON compares two JSON values by meaning, not by bytes.
func equalJSON(a, b json.RawMessage) bool {
	var x, y interface{}
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return string(a) == string(b)
	}
	return reflect.DeepEqual(x, y)
}

// compare reports every field that differs between the records of old and new, by (fixture,
// pool); a record in one and not the other is a difference ("record"). A provenance field may
// differ only in order where orderOnly names it.
func compare(old, new io.Reader, orderOnly map[string]bool) (diffReport, error) {
	a, err := readRecords(old)
	if err != nil {
		return diffReport{}, fmt.Errorf("old: %w", err)
	}
	b, err := readRecords(new)
	if err != nil {
		return diffReport{}, fmt.Errorf("new: %w", err)
	}
	r := diffReport{Differs: map[string]int{}, Example: map[string]string{}}
	note := func(field, key string) {
		r.Differs[field]++
		if _, ok := r.Example[field]; !ok {
			r.Example[field] = key
		}
	}
	keys := map[string]bool{}
	for k := range a {
		keys[k] = true
	}
	for k := range b {
		keys[k] = true
	}
	order := make([]string, 0, len(keys))
	for k := range keys {
		order = append(order, k)
	}
	sort.Strings(order)
	for _, k := range order {
		r.Records++
		x, y := a[k], b[k]
		if x == nil || y == nil {
			note("record", k)
			continue
		}
		fields := map[string]bool{}
		for f := range x {
			fields[f] = true
		}
		for f := range y {
			fields[f] = true
		}
		for f := range fields {
			if f == "fixture" || f == "pool" || equalJSON(x[f], y[f]) {
				continue
			}
			if orderOnly[f] && sameItems(x[f], y[f]) {
				note(f+" (order only)", k)
				continue
			}
			note(f, k)
		}
	}
	return r, nil
}

// runCompare is the -compare mode: it prints the report and exits 1 when anything but an
// allowed reordering differs.
func runCompare(oldPath, newPath, orderOnly string) int {
	o, err := os.Open(oldPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "equalityprobe:", err)
		return 2
	}
	defer o.Close()
	n, err := os.Open(newPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "equalityprobe:", err)
		return 2
	}
	defer n.Close()
	allowed := map[string]bool{}
	for _, f := range strings.Split(orderOnly, ",") {
		if f = strings.TrimSpace(f); f != "" {
			allowed[f] = true
		}
	}
	r, err := compare(o, n, allowed)
	if err != nil {
		fmt.Fprintln(os.Stderr, "equalityprobe:", err)
		return 2
	}
	fmt.Printf("records %d\n", r.Records)
	var fields []string
	for f := range r.Differs {
		fields = append(fields, f)
	}
	sort.Strings(fields)
	for _, f := range fields {
		fmt.Printf("%d %s, e.g. %s\n", r.Differs[f], f, r.Example[f])
	}
	if r.failed() {
		fmt.Println("DIFFERENT: a price, a figure or a refusal differs")
		return 1
	}
	fmt.Println("EQUAL: every price, figure and refusal matches")
	return 0
}
