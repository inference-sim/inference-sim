package main

import (
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
)

func names(n int) []string {
	var out []string
	for i := 0; i < n; i++ {
		out = append(out, fmt.Sprintf("TestCase%03d", i))
	}
	return out
}

// Every item is in exactly one leg, whatever the leg count, the weights or the order of the list.
func TestEveryItemRunsInExactlyOneLeg(t *testing.T) {
	items := names(183)
	w := map[string]float64{"TestCase007": 246, "TestCase100": 48, "TestCase999": 5}
	for of := 1; of <= 9; of++ {
		if n, err := Cover(items, w, of); err != nil || n != len(items) {
			t.Fatalf("%d legs: %d items, %v", of, n, err)
		}
	}
	rev := slices.Clone(items)
	slices.Reverse(rev)
	for i := 0; i < 4; i++ {
		a, _ := Select(items, w, 4, i)
		b, _ := Select(rev, w, 4, i)
		slices.Sort(a)
		slices.Sort(b)
		if !slices.Equal(a, b) {
			t.Errorf("leg %d depends on the order of the list", i)
		}
	}
}

// A heavy item gets a leg to itself while there are legs to spare, so the legs are balanced.
func TestHeavyItemsAreSpread(t *testing.T) {
	items := names(60)
	w := map[string]float64{"TestCase001": 246, "TestCase002": 48}
	one, _ := Select(items, w, 3, 0)
	two, _ := Select(items, w, 3, 1)
	if !slices.Contains(one, "TestCase001") || slices.Contains(two, "TestCase001") || slices.Contains(one, "TestCase002") {
		t.Errorf("heavy items share a leg: %v | %v", one, two)
	}
	if len(one) != 1 {
		t.Errorf("the 246 s item should hold its leg alone while the others split 48 + 58, got %d items", len(one))
	}
}

func TestALegThatIsNotOneIsRefused(t *testing.T) {
	for _, c := range [][2]int{{3, 3}, {3, -1}, {0, 0}} {
		if _, err := Select(names(5), nil, c[0], c[1]); err == nil {
			t.Errorf("leg %d of %d accepted", c[1], c[0])
		}
	}
}

func TestVerifyFailsOnAMissingOrAnExtraItem(t *testing.T) {
	res := strings.Join([]string{
		`{"Action":"pass","Package":"p","Test":"TestA","Elapsed":1}`,
		`{"Action":"pass","Package":"p","Test":"TestA/sub","Elapsed":1}`,
		`{"Action":"skip","Package":"p","Test":"TestB"}`,
		`not json: build failure text`,
		`{"Action":"fail","Package":"p"}`,
	}, "\n")
	if n, err := Verify([]string{"TestA", "TestB"}, strings.NewReader(res), "test"); err != nil || n != 2 {
		t.Errorf("exact set refused: %d %v", n, err)
	}
	if _, err := Verify([]string{"TestA", "TestB", "TestC"}, strings.NewReader(res), "test"); err == nil || !strings.Contains(err.Error(), "TestC") {
		t.Errorf("a test with no result was accepted: %v", err)
	}
	if _, err := Verify([]string{"TestA"}, strings.NewReader(res), "test"); err == nil || !strings.Contains(err.Error(), "TestB") {
		t.Errorf("a test outside the leg was accepted: %v", err)
	}
	if _, err := Verify([]string{"p"}, strings.NewReader(res), "package"); err != nil {
		t.Errorf("package mode: %v", err)
	}
	if _, err := Verify([]string{"p", "q"}, strings.NewReader(res), "package"); err == nil {
		t.Error("a package with no result was accepted")
	}
}

func TestDuplicateItemsAndBadWeightsAreRefused(t *testing.T) {
	if _, err := readItems(strings.NewReader("A\nB\nA\n")); err == nil {
		t.Error("a duplicate item was accepted")
	}
}

// A listing that found nothing must not look like a leg with nothing to do.
func TestAnEmptyOrShortListingIsRefused(t *testing.T) {
	var out strings.Builder
	if err := run([]string{"cover", "-of", "3", "-min-items", "5"}, strings.NewReader("A\nB\n"), &out); err == nil {
		t.Error("a list of 2 items passed with a floor of 5")
	}
	if err := run([]string{"cover", "-of", "3"}, strings.NewReader(""), &out); err == nil {
		t.Error("an empty list passed")
	}
	if err := run([]string{"cover", "-of", "3", "-min-items", "2"}, strings.NewReader("A\nB\n"), &out); err != nil {
		t.Errorf("a list at the floor was refused: %v", err)
	}
}

// A weight line names its item up to the last field, so an item with spaces keeps them.
func TestWeightsAcceptItemsWithSpaces(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/w.txt"
	if err := os.WriteFile(path, []byte("# c\nf.py::t[a.py --x y] 30\nplain 4\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	w, err := readWeights(path)
	if err != nil || w["f.py::t[a.py --x y]"] != 30 || w["plain"] != 4 || len(w) != 2 {
		t.Fatalf("weights %v, %v", w, err)
	}
	if err := os.WriteFile(path, []byte("noweight\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readWeights(path); err == nil {
		t.Error("a line with no weight was accepted")
	}
}
