// Command shard splits a list of tests, or of packages, across the legs of a CI matrix so that
// every item runs in exactly one leg, and checks that each leg ran what it was given.
//
// A leg of a matrix is told its index and the leg count. It reads the full list from standard
// input (go test -list for the tests of one package, go list for packages) and prints its own
// items. The split is a function of the list and the weights only, so every leg computes the
// same partition on its own and no leg depends on another.
//
//	shard select -of N -index I [-weights FILE]   print the items of leg I
//	shard cover  -of N [-weights FILE] [-min-items M]  fail unless the N legs partition the list, which has at least M items
//	shard verify -results FILE -mode test|package  fail unless the items on stdin ran, and only they
//
// The weights file holds lines "item seconds" (the item may hold spaces); an item it does not
// name weighs 1. The weights
// only balance the legs: a stale or missing one makes a leg slower, never makes an item run twice
// or not at all.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "shard:", err)
		os.Exit(1)
	}
}

func run(args []string, in io.Reader, out io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: shard select|cover|verify ...")
	}
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	of := fs.Int("of", 0, "number of legs")
	index := fs.Int("index", -1, "this leg, from 0")
	weights := fs.String("weights", "", "file of \"item seconds\" lines")
	results := fs.String("results", "", "go test -json output (verify)")
	minItems := fs.Int("min-items", 1, "cover: the fewest items the list may hold (an empty listing is a failure)")
	mode := fs.String("mode", "", "test or package (verify)")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	items, err := readItems(in)
	if err != nil {
		return err
	}
	switch args[0] {
	case "select", "cover":
		w, err := readWeights(*weights)
		if err != nil {
			return err
		}
		if args[0] == "select" {
			sel, err := Select(items, w, *of, *index)
			if err != nil {
				return err
			}
			for _, s := range sel {
				fmt.Fprintln(out, s)
			}
			return nil
		}
		if len(items) < *minItems {
			return fmt.Errorf("the list holds %d items, fewer than the %d expected: the listing found too little", len(items), *minItems)
		}
		n, err := Cover(items, w, *of)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "%d items are split across %d legs, each in exactly one\n", n, *of)
		return nil
	case "verify":
		f, err := os.Open(*results)
		if err != nil {
			return err
		}
		defer f.Close()
		n, err := Verify(items, f, *mode)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "all %d assigned items ran, and nothing else\n", n)
		return nil
	}
	return fmt.Errorf("unknown command %q", args[0])
}

func readItems(r io.Reader) ([]string, error) {
	var items []string
	seen := map[string]bool{}
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		s := strings.TrimSpace(sc.Text())
		if s == "" {
			continue
		}
		if seen[s] {
			return nil, fmt.Errorf("item %q is listed twice", s)
		}
		seen[s] = true
		items = append(items, s)
	}
	return items, sc.Err()
}

func readWeights(path string) (map[string]float64, error) {
	w := map[string]float64{}
	if path == "" {
		return w, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	for i, l := range strings.Split(string(raw), "\n") {
		l = strings.TrimSpace(l)
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		// The seconds are the last field; the item is everything before it, so an item may
		// hold spaces (a pytest id with an argument does).
		cut := strings.LastIndexAny(l, " \t")
		if cut < 0 {
			return nil, fmt.Errorf("%s:%d: want \"item seconds\"", path, i+1)
		}
		item, secs := strings.TrimSpace(l[:cut]), l[cut+1:]
		v, err := strconv.ParseFloat(secs, 64)
		if err != nil || v < 0 {
			return nil, fmt.Errorf("%s:%d: %q is not a weight", path, i+1, secs)
		}
		w[item] = v
	}
	return w, nil
}

// assign gives each item to a leg: heaviest first, each to the leg with the least load so far,
// the lowest index on a tie, and equal weights in name order. The result depends on the set of
// items and the weights only, not on the order the list arrived in.
func assign(items []string, w map[string]float64, of int) map[string]int {
	order := append([]string(nil), items...)
	weight := func(s string) float64 {
		if v, ok := w[s]; ok {
			return v
		}
		return 1
	}
	sort.Slice(order, func(i, j int) bool {
		a, b := weight(order[i]), weight(order[j])
		if a != b {
			return a > b
		}
		return order[i] < order[j]
	})
	load := make([]float64, of)
	leg := make(map[string]int, len(items))
	for _, s := range order {
		best := 0
		for l := 1; l < of; l++ {
			if load[l] < load[best] {
				best = l
			}
		}
		leg[s] = best
		load[best] += weight(s)
	}
	return leg
}

// Select returns the items of leg index of of, in the order of items.
func Select(items []string, w map[string]float64, of, index int) ([]string, error) {
	if of < 1 || index < 0 || index >= of {
		return nil, fmt.Errorf("leg %d of %d is not a leg", index, of)
	}
	leg := assign(items, w, of)
	var out []string
	for _, s := range items {
		if leg[s] == index {
			out = append(out, s)
		}
	}
	return out, nil
}

// Cover selects every leg and fails unless each item is in exactly one of them. It returns the
// number of items.
func Cover(items []string, w map[string]float64, of int) (int, error) {
	count := map[string]int{}
	for i := 0; i < of; i++ {
		sel, err := Select(items, w, of, i)
		if err != nil {
			return 0, err
		}
		for _, s := range sel {
			count[s]++
		}
	}
	for _, s := range items {
		if count[s] != 1 {
			return 0, fmt.Errorf("%q is in %d legs, not 1", s, count[s])
		}
	}
	if len(count) != len(items) {
		return 0, errors.New("a leg holds an item that is not in the list")
	}
	return len(items), nil
}

type event struct {
	Action  string
	Package string
	Test    string
}

// Verify reads go test -json output and checks the assigned items against it. In test mode the
// items are top-level test names and each must have finished (pass, fail or skip); in package
// mode they are import paths and each package must have finished. A finished item that was not
// assigned is also an error: the leg ran something another leg runs.
func Verify(items []string, results io.Reader, mode string) (int, error) {
	if mode != "test" && mode != "package" {
		return 0, fmt.Errorf("mode %q is not test or package", mode)
	}
	done := map[string]bool{}
	sc := bufio.NewScanner(results)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	for sc.Scan() {
		var e event
		// go test -json prints a build failure as plain text; it finishes nothing, and the
		// items it left unfinished are reported below.
		if json.Unmarshal(sc.Bytes(), &e) != nil {
			continue
		}
		switch e.Action {
		case "pass", "fail", "skip":
		default:
			continue
		}
		switch {
		case mode == "package" && e.Test == "":
			done[e.Package] = true
		case mode == "test" && e.Test != "" && !strings.Contains(e.Test, "/"):
			done[e.Test] = true
		}
	}
	if err := sc.Err(); err != nil {
		return 0, fmt.Errorf("reading results: %w", err)
	}
	want := map[string]bool{}
	for _, s := range items {
		want[s] = true
		if !done[s] {
			return 0, fmt.Errorf("%q was assigned to this leg and has no result", s)
		}
	}
	var extra []string
	for s := range done {
		if !want[s] {
			extra = append(extra, s)
		}
	}
	sort.Strings(extra)
	if len(extra) > 0 {
		return 0, fmt.Errorf("ran items that are not assigned to this leg: %s", strings.Join(extra, ", "))
	}
	return len(items), nil
}
